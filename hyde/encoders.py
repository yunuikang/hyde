"""Hypervector encoders: x -> H in {-1,+1}^D (torch.int8).

Both encoders share one interface:
    enc.fit(X_num, X_cat, cat_widths, y)   # fit rows only
    prep = enc.prepare(X_num, X_cat)        # one-off per split (cheap, CPU)
    H = enc.encode(prep, idx)               # batch of rows -> (B, D) int8 on enc.device
    enc.state_dict()                        # arrays needed by infer.py

PairBindEncoder is ported from pairbind/pairbind.py (same random streams, same maths).
NonlinearEncoder is the attackhd / multi-tester baseline sign(cos(x W)).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier

from .data import stratified_subset


def _sign_int8(x: torch.Tensor) -> torch.Tensor:
    """+1 where x >= 0, else -1 (0 -> +1, as attackhd/multi-tester)."""
    out = torch.ones(x.shape, dtype=torch.int8, device=x.device)
    out[x < 0] = -1
    return out


# =========================================================================== Nonlinear
class NonlinearEncoder:
    """MinMax(numeric) ++ one-hot(categorical) -> sign(cos(x @ W)),  W ~ N(0,1)^{F x D}."""

    kind = "nonlinear"

    def __init__(self, D: int, seed: int, device: torch.device):
        self.D, self.seed, self.device = D, seed, device

    def fit(self, X_num, X_cat, cat_widths, y=None):
        with np.errstate(all="ignore"):
            lo = np.nanmin(X_num, axis=0) if X_num.shape[0] else np.zeros(X_num.shape[1])
            hi = np.nanmax(X_num, axis=0) if X_num.shape[0] else np.ones(X_num.shape[1])
        lo = np.where(np.isfinite(lo), lo, 0.0).astype(np.float32)
        hi = np.where(np.isfinite(hi), hi, 0.0).astype(np.float32)
        self.lo, self.span = lo, np.where(hi - lo > 0, hi - lo, 1.0).astype(np.float32)
        self.cat_widths = list(cat_widths)
        F = X_num.shape[1] + sum(self.cat_widths)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        self.W = torch.randn(F, self.D, generator=g).to(self.device)
        return self

    def prepare(self, X_num, X_cat):
        return (X_num, X_cat)

    def features(self, prep, idx) -> torch.Tensor:
        """Scaled numeric ++ one-hot categorical, (B, F) float32 on device."""
        X_num, X_cat = prep
        xn = torch.as_tensor(X_num[idx], device=self.device)
        xn = (xn - torch.as_tensor(self.lo, device=self.device)) / torch.as_tensor(self.span, device=self.device)
        xn = torch.nan_to_num(xn, nan=0.0).clamp_(0.0, 1.0)
        parts = [xn]
        if self.cat_widths:
            xc = torch.as_tensor(X_cat[idx], device=self.device, dtype=torch.long)
            for j, w in enumerate(self.cat_widths):
                parts.append(torch.nn.functional.one_hot(xc[:, j].clamp(0, w - 1), w).float())
        return torch.cat(parts, dim=1)

    def encode(self, prep, idx) -> torch.Tensor:
        h = self.features(prep, idx) @ self.W
        return _sign_int8(torch.cos(h))

    def state_dict(self) -> dict:
        return {"lo": self.lo, "span": self.span, "W": self.W.cpu().numpy(),
                "cat_widths": np.asarray(self.cat_widths, np.int64)}


# =========================================================================== PairBind
class _Rule:
    """One input feature: its local projection onto the coordinates it takes part in."""
    __slots__ = ("kind", "source", "width", "coords", "weights", "ties", "prefixes",
                 "edges", "categories", "alpha_col")


class PairBindPrepared:
    """Per-row feature states: ids (N, F) int16 and PLE fractions alpha (N, F_numerical)."""

    def __init__(self, ids: np.ndarray, alpha: np.ndarray):
        self.ids, self.alpha = ids, alpha


class PairBindEncoder:
    """Order-2 binding of local per-feature atoms (pairbind/pairbind.py `PairBindEncoder`).

    numerical feature  : target-aware PLE (univariate decision-tree bin edges) ->
                         atom = sign(r^T (2*PLE - 1)) via prefix sums
    low-cardinality num: value -> own random atom (+ unknown slot)
    categorical        : code  -> own random atom (+ unknown slot)
    coordinate d       : H_d = atom_a(x_a) * atom_b(x_b), {a, b} drawn uniformly per d
    """

    kind = "pairbind"
    MAX_STATES = np.iinfo(np.int16).max

    def __init__(self, D: int, seed: int, device: torch.device, bins: int = 16,
                 categorical_threshold: int = 32, tree_min_samples_leaf: int = 64,
                 tree_min_fraction_leaf: float = 0.001, tree_fit_max: int = 100_000, order: int = 2):
        self.D, self.seed, self.device = D, seed, device
        self.bins, self.categorical_threshold = bins, categorical_threshold
        self.tree_min_samples_leaf = tree_min_samples_leaf
        self.tree_min_fraction_leaf = tree_min_fraction_leaf
        self.tree_fit_max = tree_fit_max
        self.order = order          # features bound per coordinate: 2 = PairBind, 1 = PLE atoms only (no binding)
        self.rules: list[_Rule] = []
        self.pairs: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def _target_edges(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        valid = values[finite]
        if valid.size == 0:
            return np.asarray([0.0, 1.0], np.float32)
        lo, hi = float(valid.min()), float(valid.max())
        if lo == hi:
            return np.asarray([lo, hi], np.float32)
        rows = stratified_subset(np.flatnonzero(finite), y, max_rows=self.tree_fit_max,
                                 seed=self.seed, min_per_class=1)
        leaf = max(self.tree_min_samples_leaf, int(math.ceil(self.tree_min_fraction_leaf * rows.size)))
        tree = DecisionTreeClassifier(max_leaf_nodes=self.bins, min_samples_leaf=leaf,
                                      random_state=self.seed)
        tree.fit(values[rows, None], y[rows])
        th = tree.tree_.threshold
        th = th[th != -2.0]
        edges = np.unique(np.asarray([lo, *np.sort(th), hi], np.float32))
        return edges if edges.size >= 2 else np.asarray([lo, hi], np.float32)

    def fit(self, X_num, X_cat, cat_widths, y):
        n_num, n_cat = X_num.shape[1], X_cat.shape[1]
        F = n_num + n_cat
        if F < 2:
            raise ValueError("PairBind needs at least two features")
        pair_rng = np.random.default_rng(self.seed + 86028121)
        self.pairs = np.empty((self.D, self.order), np.int32)
        for d in range(self.D):
            self.pairs[d] = pair_rng.choice(F, size=self.order, replace=False)
        coord_lists = [np.flatnonzero(np.any(self.pairs == f, axis=1)) for f in range(F)]
        tie_gen = torch.Generator(device="cpu").manual_seed(self.seed + 49979687)
        all_ties = (torch.randint(0, 2, (F, self.D), generator=tie_gen, dtype=torch.int8) * 2 - 1)
        weight_rng = np.random.default_rng(self.seed)

        def projection(rule: _Rule, width: int, f: int):
            if width > self.MAX_STATES:
                raise ValueError(f"feature {f} needs {width} states (> int16)")
            full = weight_rng.standard_normal((width, self.D), dtype=np.float32) / math.sqrt(max(width, 1))
            sel = coord_lists[f]
            rule.width = width
            rule.coords = torch.as_tensor(sel, dtype=torch.long, device=self.device)
            rule.weights = torch.as_tensor(np.ascontiguousarray(full[:, sel]), device=self.device)
            rule.ties = all_ties[f, sel].to(self.device)

        self.rules, n_alpha = [], 0
        for j in range(n_num):
            rule, f = _Rule(), len(self.rules)
            rule.source, rule.edges, rule.categories, rule.prefixes, rule.alpha_col = j, None, None, None, -1
            col = X_num[:, j]
            uniq = np.unique(col[np.isfinite(col)])
            if uniq.size <= self.categorical_threshold:
                rule.kind = "numeric_categorical"
                rule.categories = uniq.astype(np.float32)
                projection(rule, uniq.size + 1, f)                     # +1 unknown / NaN
            else:
                rule.kind = "numerical"
                rule.edges = self._target_edges(col, y)
                projection(rule, max(rule.edges.size - 1, 1), f)
                rule.prefixes = torch.cat((torch.zeros(1, rule.weights.shape[1], device=self.device),
                                           torch.cumsum(rule.weights, dim=0)))
                rule.alpha_col, n_alpha = n_alpha, n_alpha + 1
            self.rules.append(rule)
        for j in range(n_cat):
            rule, f = _Rule(), len(self.rules)
            rule.kind, rule.source = "categorical", j
            rule.edges = rule.categories = rule.prefixes = None
            rule.alpha_col = -1
            projection(rule, int(cat_widths[j]), f)                    # vocab + unknown slot
            self.rules.append(rule)
        self.n_alpha = n_alpha
        return self

    # ------------------------------------------------------------------ prepare / encode
    def prepare(self, X_num, X_cat, chunk: int = 1_000_000) -> PairBindPrepared:
        N = X_num.shape[0]
        ids = np.empty((N, len(self.rules)), np.int16)
        alpha = np.zeros((N, self.n_alpha), np.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            for f, r in enumerate(self.rules):
                if r.kind == "numerical":
                    v = X_num[s:e, r.source]
                    ok = np.isfinite(v)
                    safe = np.where(ok, v, r.edges[0])
                    t = np.searchsorted(r.edges[1:-1], safe, side="right").clip(0, r.width - 1)
                    left, right = r.edges[t], r.edges[np.minimum(t + 1, r.edges.size - 1)]
                    span = right - left
                    a = np.divide(safe - left, span, out=np.zeros_like(safe), where=np.abs(span) > 1e-12)
                    a = np.clip(a, -1.0, 2.0)
                    t = t.astype(np.int16)
                    t[~ok] = r.width                       # NaN -> contribution 0 -> tie atom
                    a[~ok] = 0.0
                    ids[s:e, f] = t
                    alpha[s:e, r.alpha_col] = a
                elif r.kind == "numeric_categorical":
                    v = X_num[s:e, r.source]
                    out = np.full(v.size, r.width - 1, np.int16)
                    ok = np.isfinite(v)
                    pos = np.searchsorted(r.categories, v[ok])
                    hit = pos < r.categories.size
                    hit[hit] &= r.categories[pos[hit]] == v[ok][hit]
                    rows = np.flatnonzero(ok)
                    out[rows[hit]] = pos[hit]
                    ids[s:e, f] = out
                else:
                    ids[s:e, f] = np.clip(X_cat[s:e, r.source], 0, r.width - 1)
        return PairBindPrepared(ids, alpha)

    @torch.no_grad()
    def encode(self, prep: PairBindPrepared, idx) -> torch.Tensor:
        idx = np.asarray(idx)
        ids_b = torch.as_tensor(prep.ids[idx], device=self.device, dtype=torch.long)
        alpha_b = torch.as_tensor(prep.alpha[idx], device=self.device) if self.n_alpha else None
        H = torch.ones((idx.size, self.D), dtype=torch.int8, device=self.device)
        one = torch.ones((), dtype=torch.int8, device=self.device)
        for f, r in enumerate(self.rules):
            ids = ids_b[:, f]
            if r.kind == "numerical":
                safe = ids.clamp_max(r.width - 1)
                c = r.prefixes[safe] + r.weights[safe] * alpha_b[:, r.alpha_col, None]
                c = 2.0 * c - r.weights.sum(dim=0, keepdim=True)
                c = c * (ids < r.width)[:, None]
            else:
                c = r.weights[ids]
            atom = torch.where(c > 0, one, torch.where(c < 0, -one, r.ties))
            H[:, r.coords] = H[:, r.coords] * atom
        return H

    # ------------------------------------------------------------------ export
    def state_dict(self) -> dict:
        out = {"pairs": self.pairs, "n_rules": np.asarray(len(self.rules))}
        for f, r in enumerate(self.rules):
            p = f"r{f}_"
            out[p + "kind"] = np.asarray(["numerical", "numeric_categorical", "categorical"].index(r.kind))
            out[p + "source"] = np.asarray(r.source)
            out[p + "width"] = np.asarray(r.width)
            out[p + "coords"] = r.coords.cpu().numpy()
            out[p + "weights"] = r.weights.cpu().numpy()
            out[p + "ties"] = r.ties.cpu().numpy()
            if r.edges is not None:
                out[p + "edges"] = r.edges
            if r.categories is not None:
                out[p + "categories"] = r.categories
        return out


def build_encoder(cfg: dict, device: torch.device):
    e = cfg["encoder"]
    if e["type"] == "nonlinear":
        return NonlinearEncoder(e["D"], cfg["seed"], device)
    if e["type"] == "pairbind":
        return PairBindEncoder(e["D"], cfg["seed"], device, bins=e["bins"],
                               categorical_threshold=e["categorical_threshold"],
                               tree_min_samples_leaf=e["tree_min_samples_leaf"],
                               tree_min_fraction_leaf=e["tree_min_fraction_leaf"],
                               tree_fit_max=e["tree_fit_max"], order=int(e.get("order", 2)))
    raise ValueError(f"unknown encoder {e['type']!r}")
