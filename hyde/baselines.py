"""ML baselines under exactly the HyDE protocol (same cache, columns, fit/val split, test, metrics).

Input matrix = MinMax(numeric, fit rows) ++ one-hot(categorical), NaN -> 0  (same as NonlinearEncoder).
Hyper-parameters are the multi-tester ones (SWEEP_RESULTS §6); not tuned, except the class-weight
exponent p in cfg.ml.class_weight_p, chosen on val macro-F1 for xgboost / lightgbm / RF / MLP.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .metrics import fast_f1
from .model import class_weights

MODELS = ["xgboost", "lightgbm", "random_forest", "decision_tree", "naive_bayes", "mlp"]
WEIGHTED = {"xgboost", "lightgbm", "random_forest", "mlp"}


class FeatureMatrix:
    """MinMax + one-hot, fitted on fit rows. Builds dense float32 matrices in chunks."""

    def fit(self, X_num, cat_widths):
        with np.errstate(all="ignore"):
            lo, hi = np.nanmin(X_num, 0), np.nanmax(X_num, 0)
        lo = np.where(np.isfinite(lo), lo, 0.0)
        hi = np.where(np.isfinite(hi), hi, 0.0)
        self.lo, self.span = lo.astype(np.float32), np.where(hi - lo > 0, hi - lo, 1.0).astype(np.float32)
        self.widths = list(cat_widths)
        return self

    def transform(self, X_num, X_cat, chunk: int = 1_000_000) -> np.ndarray:
        n = X_num.shape[0]
        F = X_num.shape[1] + sum(self.widths)
        out = np.zeros((n, F), np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            x = (X_num[s:e] - self.lo) / self.span
            np.nan_to_num(x, copy=False, nan=0.0)
            out[s:e, :X_num.shape[1]] = np.clip(x, 0.0, 1.0)
            off = X_num.shape[1]
            for j, w in enumerate(self.widths):
                codes = np.clip(X_cat[s:e, j], 0, w - 1)
                out[np.arange(s, e), off + codes] = 1.0
                off += w
        return out


def _make(name: str, seed: int, threads: int):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.05, tree_method="hist",
                             random_state=seed, n_jobs=threads)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=seed, deterministic=True,
                              force_col_wise=True, n_jobs=threads, verbose=-1)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=threads)
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed)
    if name == "naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    raise ValueError(name)


class MLP:
    """3 x 512 ReLU MLP (PyTorch), Adam, val macro-F1 checkpoint."""

    def __init__(self, cfg: dict, n_classes: int, seed: int, device: torch.device):
        self.c, self.C, self.seed, self.device = cfg, n_classes, seed, device

    def fit(self, X, y, X_val, y_val, sample_weight=None, cw=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        h, L = self.c["hidden"], self.c["layers"]
        layers, d = [], X.shape[1]
        for _ in range(L):
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            d = h
        self.net = torch.nn.Sequential(*layers, torch.nn.Linear(d, self.C)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.c["lr"])
        w = torch.as_tensor(cw, device=self.device) if cw is not None else None
        best, stale, state = -1.0, 0, None
        for epoch in range(self.c["epochs"]):
            self.net.train()
            order = rng.permutation(y.size)
            for s in range(0, y.size, self.c["batch"]):
                b = order[s:s + self.c["batch"]]
                xb = torch.as_tensor(X[b], device=self.device)
                yb = torch.as_tensor(y[b], device=self.device)
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(self.net(xb), yb, weight=w).backward()
                opt.step()
            macro, _ = fast_f1(y_val, self.predict_proba(X_val).argmax(1), self.C)
            if macro > best + 1e-12:
                best, stale = macro, 0
                state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.c["patience"]:
                    break
        self.net.load_state_dict(state)
        return self

    @torch.no_grad()
    def predict_proba(self, X, batch: int = 8192):
        self.net.eval()
        out = [torch.softmax(self.net(torch.as_tensor(X[s:s + batch], device=self.device)), 1).cpu().numpy()
               for s in range(0, X.shape[0], batch)]
        return np.concatenate(out)


def _proba(model, X, batch: int = 8192) -> np.ndarray:
    return np.concatenate([model.predict_proba(X[s:s + batch]) for s in range(0, X.shape[0], batch)])


def _full_proba(model, X, C):
    """predict_proba with columns for all C classes (sklearn drops classes absent from fit)."""
    p = _proba(model, X)
    cls = getattr(model, "classes_", None)
    if cls is None or len(cls) == C:
        return p
    out = np.zeros((p.shape[0], C), np.float32)
    out[:, np.asarray(cls)] = p
    return out


def fit_baseline(name: str, cfg: dict, X_fit, y_fit, X_val, y_val, C: int, device) -> tuple[object, dict]:
    """Fit one baseline; for weighted models pick p in cfg.ml.class_weight_p on val macro-F1."""
    seed, threads = cfg["seed"], cfg["threads"]
    ps = cfg["ml"]["class_weight_p"] if name in WEIGHTED else [0.0]
    best, info = None, {"candidates": []}
    for p in ps:
        t0 = time.time()
        cw = class_weights(y_fit, C, p, 100.0)
        sw = cw[y_fit] if cw is not None else None
        if name == "mlp":
            model = MLP(cfg["ml"]["mlp"], C, seed, device).fit(X_fit, y_fit, X_val, y_val, cw=cw)
        else:
            model = _make(name, seed, threads)
            model.fit(X_fit, y_fit, sample_weight=sw) if sw is not None else model.fit(X_fit, y_fit)
        train_sec = time.time() - t0
        macro, weighted = fast_f1(y_val, _full_proba(model, X_val, C).argmax(1), C)
        info["candidates"].append({"p": p, "val_macro_f1": macro, "val_weighted_f1": weighted,
                                   "train_sec": round(train_sec, 2)})
        print(f"  [{name}] p={p} val macro={macro:.4f} weighted={weighted:.4f} ({train_sec:.0f}s)", flush=True)
        if best is None or macro > best[0] + 1e-12:
            best = (macro, model, p, train_sec)
    info.update(class_weight_p=best[2], train_sec=round(best[3], 2))
    return best[1], info


def predict_proba(model, X, C) -> np.ndarray:
    return _full_proba(model, X, C)
