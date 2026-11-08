"""HyDE model: Step1 (normal-traffic prototypes) + Step2 (all-class classifier) + unknown routing.

    fit(prep_fit, y_fit, prep_val, y_val)
        Step2: MASS | AdamW readout (+ class weight, + hard negative), or SAMME boosting
        Step1: K-means on benign HVs -> K prototypes -> pull/push retrain
    scores(prep, n)  -> {"s1": max cos to normal prototypes, "s2": class scores, "s2cos": max class cos}
    predict(scores, theta1, theta2)   pure numpy, so thresholds can be swept without retraining
    save(path)       -> .npz consumed by infer.py

Labels inside the model are LOCAL class indices 0..C-1 (benign = 0); run.py maps them back.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from sklearn.cluster import KMeans

from .data import stratified_subset
from .io import save_npz
from .metrics import fast_f1

ENCODE_CHUNK = 8192


def _batches(n: int, size: int):
    for s in range(0, n, size):
        yield s, min(s + size, n)


def _cos(H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """(B, D) x (K, D) -> (B, K) cosine."""
    num = H @ P.T
    return num / (H.norm(dim=1, keepdim=True).clamp_min(1e-12) * P.norm(dim=1).clamp_min(1e-12)[None])


def class_weights(y: np.ndarray, C: int, p: float, cap: float) -> np.ndarray | None:
    """w_c = min((N/(C n_c))^p, cap), normalised so the mean per-sample weight is 1 (MT hdc.py)."""
    if p <= 0:
        return None
    counts = np.bincount(y, minlength=C).astype(np.float64)
    raw = np.zeros(C)
    present = counts > 0
    raw[present] = np.minimum((y.size / (C * counts[present])) ** p, cap)
    return (raw / ((counts * raw).sum() / y.size)).astype(np.float32)


class HyDE:
    def __init__(self, cfg: dict, encoder, n_classes: int, K: int, device: torch.device, benign: int = 0):
        self.cfg, self.enc, self.C, self.device, self.benign = cfg, encoder, n_classes, device, benign
        self.K = int(cfg["step1"]["K"] or K)
        self.rng = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        self.N = None            # Step1 prototypes (K, D)
        self.M = None            # Step2 prototypes / weight (C, D)
        self.bias = None         # AdamW bias (C,)
        self.ensemble = None     # SAMME: (T, C, D), alphas (T,)
        self.readout = None
        self.history: dict = {}

    # ------------------------------------------------------------------ helpers
    def _H(self, prep, idx) -> torch.Tensor:
        return self.enc.encode(prep, idx).float()

    def _iter_H(self, prep, idx: np.ndarray, chunk: int = ENCODE_CHUNK):
        for s, e in _batches(idx.size, chunk):
            yield s, e, self._H(prep, idx[s:e])

    def _sorted_scores(self, prep, idx, fn) -> np.ndarray:
        outs = [fn(H).cpu().numpy() for _, _, H in self._iter_H(prep, idx)]
        return np.concatenate(outs) if outs else np.zeros((0,))

    # ------------------------------------------------------------------ Step2 readouts
    def _logits(self, H: torch.Tensor, int8: bool = False) -> torch.Tensor:
        if self.readout == "samme":
            T, a = self.ensemble
            P = torch.sign(T) if int8 else T
            s = sum(float(a[t]) * _cos(H, P[t]) for t in range(P.shape[0]))
            return s / float(a.sum())
        M = torch.sign(self.M) if int8 else self.M
        if self.readout == "adamw":
            return (H / math.sqrt(H.shape[1])) @ M.T + self.bias
        return H @ M.T                                                   # mass: raw dot

    def _class_cos(self, H: torch.Tensor, int8: bool = False) -> torch.Tensor:
        if self.readout == "samme":
            return self._logits(H, int8)
        M = torch.sign(self.M) if int8 else self.M
        return _cos(H, M)

    def _val_score(self, prep_val, idx_val, y_val) -> tuple[float, float]:
        pred = self._sorted_scores(prep_val, idx_val, lambda H: self._logits(H).argmax(1))
        return fast_f1(y_val, pred.astype(np.int64), self.C)

    def _order(self, prep, idx_fit, y_fit, beta: float) -> np.ndarray:
        """Epoch sample order: uniform permutation, or hard-negative resampling (Robinson 2021)."""
        if beta <= 0:
            return self.rng.permutation(idx_fit.size)
        yt = torch.as_tensor(y_fit, device=self.device)
        hard = np.empty(idx_fit.size, np.float32)
        for s, e, H in self._iter_H(prep, idx_fit):
            sims = self._class_cos(H)                                    # always cosine
            rows = torch.arange(e - s, device=self.device)
            correct = sims[rows, yt[s:e]].clone()
            sims[rows, yt[s:e]] = -math.inf
            hard[s:e] = (sims.max(1).values - correct).cpu().numpy()
        logw = beta * (hard - hard.max())
        p = np.exp(logw.astype(np.float64))
        p /= p.sum()
        order = self.rng.choice(idx_fit.size, size=idx_fit.size, replace=True, p=p)
        self.history.setdefault("hn_unique_fraction", []).append(float(np.unique(order).size / idx_fit.size))
        return order

    def _train_epochs(self, prep, idx_fit, y_fit, prep_val, idx_val, y_val, step_fn, snapshot, restore):
        s2 = self.cfg["step2"]
        beta = float(self.cfg["hard_negative"]["beta"])
        best, best_epoch, stale, hist = -1.0, 0, 0, []
        key = 0 if s2["select_metric"] == "macro_f1" else 1
        for epoch in range(1, s2["epochs"] + 1):
            t0 = time.time()
            order = self._order(prep, idx_fit, y_fit, beta)
            step_fn(idx_fit[order], y_fit[order])
            macro, weighted = self._val_score(prep_val, idx_val, y_val)
            hist.append({"epoch": epoch, "val_macro_f1": macro, "val_weighted_f1": weighted,
                         "sec": round(time.time() - t0, 2)})
            score = (macro, weighted)[key]
            print(f"  [step2 {self.readout}] epoch {epoch:2d} val macro={macro:.4f} weighted={weighted:.4f}"
                  f" ({hist[-1]['sec']}s)", flush=True)
            if score > best + 1e-12:
                best, best_epoch, stale = score, epoch, 0
                snapshot()
            else:
                stale += 1
                if stale >= s2["patience"]:
                    break
        restore()
        self.history["step2_epochs"] = hist
        self.history["selected_epoch"] = best_epoch

    def _fit_mass(self, prep, idx_fit, y_fit, prep_val, idx_val, y_val, cw):
        lr, bs = self.cfg["step2"]["mass"]["lr"], self.cfg["step2"]["mass"]["batch"]
        self.M = torch.zeros(self.C, self.enc.D, device=self.device)
        cwt = torch.as_tensor(cw, device=self.device) if cw is not None else None
        eye = torch.eye(self.C, device=self.device)

        def step(idx, y):
            for s, e in _batches(idx.size, bs):
                H = self._H(prep, idx[s:e])
                yb = torch.as_tensor(y[s:e], device=self.device)
                R = eye[yb] - torch.softmax(torch.relu(H @ self.M.T), dim=1)
                if cwt is not None:
                    w = cwt[yb]
                    R = R * (w / w.mean().clamp_min(1e-12))[:, None]      # per-batch renormalisation
                self.M.add_(R.T @ H, alpha=lr)

        best = {}
        self._train_epochs(prep, idx_fit, y_fit, prep_val, idx_val, y_val, step,
                           lambda: best.__setitem__("M", self.M.clone()),
                           lambda: setattr(self, "M", best["M"]))

    def _centroids(self, prep, idx, y) -> torch.Tensor:
        sums = torch.zeros(self.C, self.enc.D, device=self.device)
        yt = torch.as_tensor(y, device=self.device)
        for s, e, H in self._iter_H(prep, idx):
            sums.index_add_(0, yt[s:e], H)
        return sums

    def _fit_adamw(self, prep, idx_fit, y_fit, prep_val, idx_val, y_val, cw):
        a = self.cfg["step2"]["adamw"]
        cent = self._centroids(prep, idx_fit, y_fit)
        W = torch.nn.Parameter(cent / cent.norm(dim=1, keepdim=True).clamp_min(1e-12))
        b = torch.nn.Parameter(torch.zeros(self.C, device=self.device))
        self.M, self.bias = W.data, b.data
        opt = torch.optim.AdamW([W, b], lr=a["lr"], weight_decay=a["weight_decay"])
        cwt = torch.as_tensor(cw, device=self.device) if cw is not None else None
        scale = 1.0 / math.sqrt(self.enc.D)

        def step(idx, y):
            for s, e in _batches(idx.size, a["batch"]):
                H = self._H(prep, idx[s:e])
                yb = torch.as_tensor(y[s:e], device=self.device)
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy((H * scale) @ W.T + b, yb, weight=cwt)
                loss.backward()
                opt.step()
            self.M, self.bias = W.data, b.data

        best = {}

        def restore():
            self.M, self.bias = best["W"], best["b"]

        self._train_epochs(prep, idx_fit, y_fit, prep_val, idx_val, y_val, step,
                           lambda: best.update(W=W.data.clone(), b=b.data.clone()), restore)

    def _fit_samme(self, prep, idx_fit, y_fit, cw):
        """SAMME with weighted-centroid weak learners (MT hdc.py retrain_boost_samme)."""
        T, cap = int(self.cfg["boosting"]["T"]), float(self.cfg["boosting"]["d_cap_mult"])
        N = idx_fit.size
        yt = torch.as_tensor(y_fit, device=self.device)
        Dt = torch.full((N,), 1.0 / N, device=self.device)
        if cw is not None:
            Dt = torch.as_tensor(cw, device=self.device)[yt]
            Dt = Dt / Dt.sum()
        protos, alphas, log = [], [], []
        for t in range(T):
            Ct = torch.zeros(self.C, self.enc.D, device=self.device)
            for s, e, H in self._iter_H(prep, idx_fit):
                Ct.index_add_(0, yt[s:e], H * Dt[s:e, None])
            wrong = torch.empty(N, device=self.device)
            for s, e, H in self._iter_H(prep, idx_fit):
                wrong[s:e] = (_cos(H, Ct).argmax(1) != yt[s:e]).float()
            eps = max(float((Dt * wrong).sum()), 1e-10)
            log.append({"t": t + 1, "eps": eps})
            if eps >= 1.0 - 1.0 / self.C:
                print(f"  [samme] round {t + 1}: eps={eps:.4f} >= 1-1/C, stop", flush=True)
                break
            alpha = math.log((1 - eps) / eps) + math.log(self.C - 1)
            protos.append(Ct)
            alphas.append(alpha)
            Dt = torch.clamp(Dt * torch.exp(alpha * wrong), max=cap / N)
            Dt = Dt / Dt.sum()
            print(f"  [samme] round {t + 1}: eps={eps:.4f} alpha={alpha:.3f}", flush=True)
        if not protos:                         # first learner already worse than chance
            protos, alphas = [Ct], [1.0]
        self.ensemble = (torch.stack(protos), np.asarray(alphas, np.float64))
        self.history["samme"] = log

    def fit_step2(self, prep_fit, y_fit, prep_val, y_val):
        cw = class_weights(y_fit, self.C, float(self.cfg["class_weight"]["p"]),
                           float(self.cfg["class_weight"]["cap"]))
        idx_fit = np.arange(y_fit.size)
        sel = stratified_subset(np.arange(y_val.size), y_val, max_rows=self.cfg["step2"]["val_select_max"],
                                seed=self.cfg["seed"], min_per_class=200)   # rare classes kept whole
        if int(self.cfg["boosting"]["T"]) > 0:
            self.readout = "samme"
            self._fit_samme(prep_fit, idx_fit, y_fit, cw)
        else:
            self.readout = self.cfg["step2"]["readout"]
            fit = {"mass": self._fit_mass, "adamw": self._fit_adamw}[self.readout]
            fit(prep_fit, idx_fit, y_fit, prep_val, sel, y_val[sel], cw)

    # ------------------------------------------------------------------ Step1
    def fit_step1(self, prep_fit, y_fit):
        s1 = self.cfg["step1"]
        D = self.enc.D
        normal = np.flatnonzero(y_fit == self.benign)
        K = self.K if normal.size > self.K else 1
        if K > 1:
            sample = np.sort(self.rng.choice(normal, size=min(s1["kmeans_fit_max"], normal.size), replace=False))
            Xs = torch.cat([H.cpu() for _, _, H in self._iter_H(prep_fit, sample)]).numpy()
            km = KMeans(n_clusters=K, n_init=10, random_state=self.cfg["seed"]).fit(Xs)
            centers = torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=self.device)
        self.N = torch.zeros(K, D, device=self.device)
        for s, e, H in self._iter_H(prep_fit, normal):
            assign = _cos(H, centers).argmax(1) if K > 1 else torch.zeros(e - s, dtype=torch.long,
                                                                         device=self.device)
            self.N.index_add_(0, assign, H)
        if s1["retrain"]["enabled"]:
            r = s1["retrain"]
            yt_all = torch.as_tensor(y_fit, device=self.device)
            for epoch in range(r["epochs"]):
                order = self.rng.permutation(y_fit.size)
                pulls = pushes = 0
                for s, e, Hc in self._iter_H(prep_fit, order):
                    yc = yt_all[order[s:e]]
                    for bs, be in _batches(e - s, r["batch"]):
                        H, yb = Hc[bs:be], yc[bs:be]
                        sims = _cos(H, self.N)
                        nm = yb == self.benign
                        if nm.any():                    # pull: closest prototype += lr (1 - sim) S
                            mx, best = sims[nm].max(1)
                            self.N.index_add_(0, best, H[nm] * (r["lr"] * (1 - mx))[:, None])
                            pulls += int(nm.sum())
                        am = ~nm
                        if am.any():                    # push: every prototype -= lr max(sim, 0) S
                            self.N.sub_(r["lr"] * (sims[am].clamp_min(0).T @ H[am]))
                            pushes += int(am.sum())
                print(f"  [step1 retrain] epoch {epoch + 1}: pull={pulls} push={pushes}", flush=True)
        self.history["step1_K"] = K

    def fit(self, prep_fit, y_fit, prep_val, y_val):
        t0 = time.time()
        self.fit_step2(prep_fit, y_fit, prep_val, y_val)
        self.history["step2_sec"] = round(time.time() - t0, 2)
        if self.cfg["step1"]["enabled"]:
            t1 = time.time()
            self.fit_step1(prep_fit, y_fit)
            self.history["step1_sec"] = round(time.time() - t1, 2)
        return self.history

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def scores(self, prep, n: int) -> dict:
        int8 = bool(self.cfg["eval"]["int8_proto"])
        idx = np.arange(n)
        s1, s2, s2cos = [], [], []
        N = (torch.sign(self.N) if int8 else self.N) if self.N is not None else None
        for _, _, H in self._iter_H(prep, idx):
            if N is not None:
                s1.append(_cos(H, N).max(1).values.cpu().numpy())
            s2.append(self._logits(H, int8).cpu().numpy())
            s2cos.append(self._class_cos(H, int8).max(1).values.cpu().numpy())
        return {"s1": np.concatenate(s1) if s1 else None, "s2": np.concatenate(s2),
                "s2cos": np.concatenate(s2cos)}

    @staticmethod
    def predict(sc: dict, theta1: float | None, theta2: float | None, benign: int = 0) -> np.ndarray:
        """Step1 gate (if theta1) -> argmax Step2 -> unknown (-1) if theta2 and max class cos < theta2."""
        pred = sc["s2"].argmax(1).astype(np.int64)
        gate = np.zeros(pred.size, bool)
        if theta1 is not None and sc["s1"] is not None:
            gate = sc["s1"] >= theta1
            pred[gate] = benign
        if theta2 is not None:
            pred[(~gate) & (sc["s2cos"] < theta2)] = -1
        return pred

    @staticmethod
    def calibrate_theta2(sc_val: dict, y_val: np.ndarray, rate: float, theta1: float | None,
                         benign: int = 0) -> float:
        """theta2 = `rate` quantile of max-class-cos over known ATTACK val rows that pass Step1."""
        mask = y_val != benign
        if theta1 is not None and sc_val["s1"] is not None:
            mask &= sc_val["s1"] < theta1
        vals = sc_val["s2cos"][mask]
        return float(np.quantile(vals, rate)) if vals.size else -1.0

    # ------------------------------------------------------------------ export
    def save(self, path, meta: dict) -> None:
        arrays = {f"enc_{k}": np.asarray(v) for k, v in self.enc.state_dict().items()}
        if self.N is not None:
            arrays["step1_N"] = self.N.cpu().numpy()
        if self.readout == "samme":
            arrays["samme_T"] = self.ensemble[0].cpu().numpy()
            arrays["samme_alpha"] = self.ensemble[1]
        else:
            arrays["step2_M"] = self.M.cpu().numpy()
            if self.bias is not None:
                arrays["step2_bias"] = self.bias.cpu().numpy()
        meta = {**meta, "readout": self.readout, "encoder": self.enc.kind, "D": self.enc.D}
        save_npz(path, compressed=True, meta=np.asarray(json.dumps(meta)), **arrays)
