#!/usr/bin/env python3
"""Stand-alone HyDE inference — needs only numpy and a model .npz (no torch, no sklearn).

Copy this file + models/<name>.npz to a Raspberry Pi and run:
    python3 infer.py --model mirai_s0.npz --csv test-flow.csv --bench
Outputs predictions (optional), metrics when the CSV carries labels, and throughput / latency /
memory figures for the encode+score path (CSV parsing is timed separately).

Library use (run.py does this to check agreement with the training code):
    pred, scores = Predictor("model.npz").predict(X_num, X_cat)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time

import numpy as np

UNKNOWN = -1


# ============================================================================ model
class Predictor:
    def __init__(self, path: str, int8_proto: bool | None = None):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.size_bytes = os.path.getsize(path)
        self.a = {k: z[k] for k in z.files if k != "meta"}
        self.int8 = self.meta["int8_proto"] if int8_proto is None else int8_proto
        self.kind, self.readout, self.D = self.meta["encoder"], self.meta["readout"], self.meta["D"]
        sgn = (lambda m: np.where(m >= 0, 1.0, -1.0).astype(np.float32)) if self.int8 else (lambda m: m)
        self.N = sgn(self.a["step1_N"]) if "step1_N" in self.a else None
        if self.readout == "samme":
            self.T, self.alpha = sgn(self.a["samme_T"]), self.a["samme_alpha"]
        else:
            self.M = sgn(self.a["step2_M"])
            self.bias = self.a.get("step2_bias")
        if self.kind == "pairbind":
            self._load_pairbind()

    # ---------------------------------------------------------------- PairBind
    def _load_pairbind(self):
        self.rules = []
        n_alpha = 0
        for f in range(int(self.a["enc_n_rules"])):
            g = lambda k: self.a.get(f"enc_r{f}_{k}")
            r = {"kind": ("numerical", "numeric_categorical", "categorical")[int(g("kind"))],
                 "source": int(g("source")), "width": int(g("width")), "coords": g("coords"),
                 "weights": g("weights").astype(np.float32), "ties": g("ties").astype(np.int8),
                 "edges": g("edges"), "categories": g("categories"), "alpha_col": -1}
            if r["kind"] == "numerical":
                w = r["weights"]
                r["prefixes"] = np.concatenate([np.zeros((1, w.shape[1]), np.float32), np.cumsum(w, 0)])
                r["wsum"] = w.sum(0, keepdims=True)
                r["alpha_col"], n_alpha = n_alpha, n_alpha + 1
            self.rules.append(r)
        self.n_alpha = n_alpha

    def _prepare(self, X_num, X_cat):
        N = X_num.shape[0]
        ids = np.empty((N, len(self.rules)), np.int64)
        alpha = np.zeros((N, self.n_alpha), np.float32)
        for f, r in enumerate(self.rules):
            if r["kind"] == "numerical":
                e = r["edges"]
                v = X_num[:, r["source"]]
                ok = np.isfinite(v)
                safe = np.where(ok, v, e[0])
                t = np.searchsorted(e[1:-1], safe, side="right").clip(0, r["width"] - 1)
                left, right = e[t], e[np.minimum(t + 1, e.size - 1)]
                span = right - left
                a = np.divide(safe - left, span, out=np.zeros_like(safe), where=np.abs(span) > 1e-12)
                a = np.clip(a, -1.0, 2.0)
                t[~ok], a[~ok] = r["width"], 0.0
                ids[:, f], alpha[:, r["alpha_col"]] = t, a
            elif r["kind"] == "numeric_categorical":
                v, cats = X_num[:, r["source"]], r["categories"]
                out = np.full(v.size, r["width"] - 1, np.int64)
                ok = np.isfinite(v)
                pos = np.searchsorted(cats, v[ok])
                hit = pos < cats.size
                hit[hit] &= cats[pos[hit]] == v[ok][hit]
                out[np.flatnonzero(ok)[hit]] = pos[hit]
                ids[:, f] = out
            else:
                ids[:, f] = np.clip(X_cat[:, r["source"]], 0, r["width"] - 1)
        return ids, alpha

    def _encode_pairbind(self, X_num, X_cat):
        ids, alpha = self._prepare(X_num, X_cat)
        H = np.ones((ids.shape[0], self.D), np.int8)
        for f, r in enumerate(self.rules):
            i = ids[:, f]
            if r["kind"] == "numerical":
                safe = np.minimum(i, r["width"] - 1)
                c = r["prefixes"][safe] + r["weights"][safe] * alpha[:, r["alpha_col"], None]
                c = 2.0 * c - r["wsum"]
                c *= (i < r["width"])[:, None]
            else:
                c = r["weights"][i]
            atom = np.where(c > 0, 1, np.where(c < 0, -1, r["ties"][None, :])).astype(np.int8)
            H[:, r["coords"]] *= atom
        return H

    # ---------------------------------------------------------------- Nonlinear
    def _encode_nonlinear(self, X_num, X_cat):
        x = (X_num - self.a["enc_lo"]) / self.a["enc_span"]
        x = np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)
        parts = [x.astype(np.float32)]
        for j, w in enumerate(self.a["enc_cat_widths"]):
            oh = np.zeros((X_cat.shape[0], int(w)), np.float32)
            oh[np.arange(X_cat.shape[0]), np.clip(X_cat[:, j], 0, w - 1)] = 1.0
            parts.append(oh)
        h = np.cos(np.concatenate(parts, 1) @ self.a["enc_W"])
        return np.where(h >= 0, 1, -1).astype(np.int8)

    def encode(self, X_num, X_cat) -> np.ndarray:
        enc = self._encode_pairbind if self.kind == "pairbind" else self._encode_nonlinear
        return enc(X_num, X_cat)

    # ---------------------------------------------------------------- scores / predict
    @staticmethod
    def _cos(H, P):
        return (H @ P.T) / (np.linalg.norm(H, axis=1, keepdims=True).clip(1e-12)
                            * np.linalg.norm(P, axis=1).clip(1e-12)[None])

    def scores(self, H: np.ndarray) -> dict:
        H = H.astype(np.float32)
        if self.readout == "samme":
            s2 = sum(float(a) * self._cos(H, T) for a, T in zip(self.alpha, self.T)) / float(self.alpha.sum())
            s2cos = s2.max(1)
        else:
            s2 = (H / math.sqrt(self.D)) @ self.M.T + self.bias if self.readout == "adamw" else H @ self.M.T
            s2cos = self._cos(H, self.M).max(1)
        if "step2_merge" in self.a:                            # benign sub-prototypes -> benign
            C = len(self.meta["classes"])
            extra = s2[:, C:]
            s2 = s2[:, :C].copy()
            s2[:, 0] = np.maximum(s2[:, 0], extra.max(1))
        s1 = self._cos(H, self.N).max(1) if self.N is not None else None
        return {"s1": s1, "s2": s2, "s2cos": s2cos}

    def predict(self, X_num, X_cat, batch: int = 4096):
        th1, th2 = self.meta.get("theta1"), self.meta.get("theta2")
        preds, all_sc = [], {"s1": [], "s2cos": []}
        for s in range(0, X_num.shape[0], batch):
            sc = self.scores(self.encode(X_num[s:s + batch], X_cat[s:s + batch]))
            p = sc["s2"].argmax(1).astype(np.int64)
            gate = np.zeros(p.size, bool)
            if th1 is not None and sc["s1"] is not None:
                gate = sc["s1"] >= th1
                p[gate] = 0                                           # benign
            if th2 is not None:
                p[(~gate) & (sc["s2cos"] < th2)] = UNKNOWN
            preds.append(p)
            all_sc["s2cos"].append(sc["s2cos"])
            if sc["s1"] is not None:
                all_sc["s1"].append(sc["s1"])
        return np.concatenate(preds), {k: (np.concatenate(v) if v else None) for k, v in all_sc.items()}


# ============================================================================ CSV -> arrays
def _norm(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def read_csv(path: str, meta: dict, limit: int | None = None):
    """Rebuild (X_num, X_cat, y_local) exactly as hyde/data.py + run.py did for training."""
    pre = meta["preprocess"]
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        seen, cols = {}, []
        for h in header:                                  # same duplicate renaming as data.py
            if h in seen:
                seen[h] += 1
                cols.append(f"{h}__dup{seen[h]}")
            else:
                seen[h] = 0
                cols.append(h)
        pos = {c: i for i, c in enumerate(cols)}
        num_i = [pos[c] for c in pre["num_names"]]
        cat_i = [pos[c] for c in pre["cat_names"]]
        lab_i = next((i for i, c in enumerate(cols) if _norm(c) == pre["label_column"]), None)
        vocab = [{v: k for k, v in enumerate(voc)} for voc in pre["vocabs"]]
        rows_num, rows_cat, labels = [], [], []
        for n, row in enumerate(reader):
            if limit and n >= limit:
                break
            vals = []
            for i in num_i:
                try:
                    v = float(row[i])
                    vals.append(v if math.isfinite(v) else math.nan)
                except ValueError:
                    vals.append(math.nan)
            rows_num.append(vals)
            rows_cat.append([vocab[j].get(row[i].strip() or "<NA>", len(vocab[j])) for j, i in enumerate(cat_i)])
            if lab_i is not None:
                labels.append(row[lab_i])
    X_num = np.asarray(rows_num, np.float32).reshape(-1, len(num_i))
    X_cat = np.asarray(rows_cat, np.int64).reshape(-1, len(cat_i))
    y = None
    if labels:
        benign = set(pre["benign_aliases"])
        local = {c: i for i, c in enumerate(meta["classes"])}
        y = []
        for raw in labels:
            lab = raw.strip()
            lab = "benign" if lab.lower() in benign else _canon(lab, pre.get("label_canon"))
            y.append(local.get(lab, UNKNOWN))
        y = np.asarray(y, np.int64)
    return X_num, X_cat, y


def _canon(label: str, mode):
    if mode != "cicids2017":
        return label
    c = label.lower()
    for t in ("�", "–", "—", "-", ":", "/"):
        c = c.replace(t, " ")
    c = _norm(c)
    m = {"brute_force": "web attack brute force", "web_attack_brute_force": "web attack brute force",
         "xss": "web attack xss", "web_attack_xss": "web attack xss",
         "sql_injection": "web attack sql injection", "web_attack_sql_injection": "web attack sql injection"}
    return m.get(c, label)


def macro_weighted_f1(y, p):
    labels = sorted(set(y.tolist()))
    f1s, sup = [], []
    for c in labels:
        tp = int(((p == c) & (y == c)).sum())
        fp, fn = int(((p == c) & (y != c)).sum()), int(((p != c) & (y == c)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        sup.append(int((y == c).sum()))
    f1s, sup = np.asarray(f1s), np.asarray(sup)
    return float(f1s.mean()), float((f1s * sup).sum() / sup.sum())


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--limit", type=int, default=None, help="read only the first N rows")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--int8-proto", action="store_true", help="score with sign(prototypes)")
    ap.add_argument("--bench", action="store_true", help="time encode+score per batch")
    ap.add_argument("--repeat-seconds", type=float, default=0.0,
                    help="loop inference for this long (for power measurement)")
    ap.add_argument("--out", default=None, help="write predicted class names to this file")
    args = ap.parse_args()

    model = Predictor(args.model, int8_proto=True if args.int8_proto else None)
    t0 = time.time()
    X_num, X_cat, y = read_csv(args.csv, model.meta, args.limit)
    parse_sec = time.time() - t0
    print(f"model {args.model}: {model.size_bytes / 1e6:.2f} MB, encoder={model.kind}, readout={model.readout}, "
          f"D={model.D}, int8={model.int8}")
    print(f"parsed {X_num.shape[0]:,} rows in {parse_sec:.2f}s")

    lat, t_start, n_done = [], time.time(), 0
    preds = None
    while True:
        out = []
        for s in range(0, X_num.shape[0], args.batch):
            tb = time.perf_counter()
            p, _ = model.predict(X_num[s:s + args.batch], X_cat[s:s + args.batch], batch=args.batch)
            lat.append(time.perf_counter() - tb)
            out.append(p)
            n_done += p.size
        preds = np.concatenate(out)
        if time.time() - t_start >= args.repeat_seconds:
            break
    total = time.time() - t_start
    names = model.meta["classes"]
    if y is not None:
        macro, weighted = macro_weighted_f1(y, preds)
        print(f"macro-F1 {macro * 100:.2f}  weighted-F1 {weighted * 100:.2f}  accuracy {(y == preds).mean() * 100:.2f}")
    if args.bench or args.repeat_seconds:
        lat_ms = np.asarray(lat) * 1e3
        print(f"throughput {n_done / total:,.0f} flows/s | batch {args.batch}: p50 {np.percentile(lat_ms, 50):.2f} ms, "
              f"p95 {np.percentile(lat_ms, 95):.2f} ms | peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB "
              f"| elapsed {total:.1f}s ({n_done:,} flows)")
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(("unknown" if p == UNKNOWN else names[p]) + "\n" for p in preds)


if __name__ == "__main__":
    main()
