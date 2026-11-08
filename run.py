#!/usr/bin/env python3
"""One training + evaluation run -> results/<tag>/<dataset>__<model>__s<seed>.json

Examples
    python run.py --data mirai                                   # HyDE, tuned config if present
    python run.py --data mirai --seed 1 --tag final --save-model
    python run.py --data mirai --set step1.enabled=false --tag ablate_no_step1
    python run.py --data mirai --holdout "udp flooding (action)" --set unknown.enabled=true --tag unk
    python run.py --data mirai --model random_forest --tag ml
"""

from __future__ import annotations

import os
import sys


def _early_threads() -> None:
    """Thread env vars must be set before numpy/torch are imported."""
    n = "6"
    if "--threads" in sys.argv:
        n = sys.argv[sys.argv.index("--threads") + 1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, n)


_early_threads()

import argparse                                                  # noqa: E402
import json                                                      # noqa: E402
import resource                                                  # noqa: E402
import subprocess                                                # noqa: E402
import time                                                      # noqa: E402
from dataclasses import dataclass                                # noqa: E402
from pathlib import Path                                         # noqa: E402

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402

from hyde.config import ROOT, build_config, datasets_registry    # noqa: E402
from hyde.data import load_dataset, split_fit_val, stratified_subset   # noqa: E402
from hyde.encoders import build_encoder                          # noqa: E402
from hyde.metrics import classification_report, unknown_report   # noqa: E402
from hyde.model import HyDE                                      # noqa: E402

RESULTS, MODELS = ROOT / "results", ROOT / "models"
UNKNOWN = -1


# =========================================================================== data for one run
@dataclass
class RunData:
    ds: object                 # hyde.data.Dataset
    fit: np.ndarray            # row indices into the train file
    val: np.ndarray
    y_fit: np.ndarray          # LOCAL labels 0..C-1 (benign = 0)
    y_val: np.ndarray
    y_test: np.ndarray         # local labels, -1 = unknown (test-only or held out)
    classes: list[str]         # local index -> name
    held_out: list[str]


def prepare_data(ds, cfg: dict, holdout: list[str] | None = None) -> RunData:
    """fit/val split (+ optional holdout and Tier-B subsample) and local class re-indexing."""
    fit, val = split_fit_val(ds.y, cfg["data"]["val_frac"], cfg["seed"])
    held = []
    if holdout:
        names = {c: i for i, c in enumerate(ds.classes)}
        missing = [h for h in holdout if h not in names]
        if missing:
            raise ValueError(f"holdout classes not in {ds.name}: {missing}")
        hid = np.asarray([names[h] for h in holdout])
        fit, val = fit[~np.isin(ds.y[fit], hid)], val[~np.isin(ds.y[val], hid)]
        held = list(holdout)
    if cfg["data"]["tune_subsample"]:
        fit = stratified_subset(fit, ds.y, frac=float(cfg["data"]["tune_subsample"]), seed=cfg["seed"])
    present = np.unique(ds.y[fit])
    g2l = np.full(len(ds.classes) + 1, UNKNOWN, np.int64)       # last slot maps global -1
    g2l[present] = np.arange(present.size)
    return RunData(ds, fit, val, g2l[ds.y[fit]], g2l[ds.y[val]], g2l[ds.y_test],
                   [ds.classes[i] for i in present], held)


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "none"
    except Exception:
        return "none"


def _device(cfg) -> torch.device:
    return torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")


def _preprocess_meta(ds) -> dict:
    reg = datasets_registry()
    return {"num_names": ds.num_names, "cat_names": ds.cat_names, "vocabs": ds.vocabs,
            "label_column": reg["label_column"], "benign_aliases": [a.lower() for a in reg["benign_aliases"]],
            "label_canon": ds.meta.get("label_canon")}


# =========================================================================== HyDE
class HydeSession:
    """Encoder fitted once; reused across Step2/Step1 candidates by tune.py."""

    def __init__(self, cfg: dict, rd: RunData):
        self.cfg, self.rd, self.device = cfg, rd, _device(cfg)
        ds = rd.ds
        t0 = time.time()
        self.enc = build_encoder(cfg, self.device)
        self.enc.fit(ds.X_num[rd.fit], ds.X_cat[rd.fit], ds.cat_widths, rd.y_fit)
        self.prep_fit = self.enc.prepare(ds.X_num[rd.fit], ds.X_cat[rd.fit])
        self.prep_val = self.enc.prepare(ds.X_num[rd.val], ds.X_cat[rd.val])
        self.encoder_sec = time.time() - t0

    def prep_test(self):
        return self.enc.prepare(self.rd.ds.X_num_test, self.rd.ds.X_cat_test)

    def new_model(self, cfg: dict | None = None) -> HyDE:
        cfg = cfg or self.cfg
        K = datasets_registry()["datasets"][self.rd.ds.name]["K"]
        return HyDE(cfg, self.enc, len(self.rd.classes), K, self.device)


def _thetas(cfg):
    return cfg["step1"]["theta1"] if cfg["step1"]["enabled"] else None


def evaluate_hyde(model: HyDE, sess: HydeSession, cfg: dict, prep_test=None, sweep_theta1=None) -> dict:
    rd = sess.rd
    th1 = _thetas(cfg)
    t0 = time.time()
    sc_val = model.scores(sess.prep_val, rd.y_val.size)
    prep_test = prep_test if prep_test is not None else sess.prep_test()
    t1 = time.time()
    sc_test = model.scores(prep_test, rd.y_test.size)
    infer_sec = time.time() - t1
    th2 = None
    if cfg["unknown"]["enabled"]:
        th2 = HyDE.calibrate_theta2(sc_val, rd.y_val, cfg["unknown"]["reject_rate"], th1)
    p_val, p_test = HyDE.predict(sc_val, th1, th2), HyDE.predict(sc_test, th1, th2)
    out = {"theta1": th1, "theta2": th2,
           "val": classification_report(rd.y_val, p_val, rd.classes),
           "test": classification_report(rd.y_test, p_test, rd.classes),
           "infer_ms_per_sample": 1e3 * infer_sec / max(rd.y_test.size, 1),
           "_pred_test": p_test, "_scores_test": sc_test}
    out["val"].pop("per_class")
    if cfg["unknown"]["enabled"]:
        out["unknown"] = unknown_report(rd.y_test, p_test, sc_test["s2cos"], rd.classes)
    if sweep_theta1:
        out["theta1_sweep"] = []
        for t in sweep_theta1:
            pv, pt = HyDE.predict(sc_val, t, th2), HyDE.predict(sc_test, t, th2)
            v, te = classification_report(rd.y_val, pv, rd.classes), classification_report(rd.y_test, pt, rd.classes)
            out["theta1_sweep"].append({"theta1": t, "val_macro_f1": v["macro_f1"], "val_weighted_f1": v["weighted_f1"],
                                        "test_macro_f1": te["macro_f1"], "test_weighted_f1": te["weighted_f1"],
                                        "test_fpr": te["binary"]["fpr"]})
    # deployment-size variant: same model scored with sign(prototypes)
    if not cfg["eval"]["int8_proto"]:
        cfg8 = {**cfg, "eval": {**cfg["eval"], "int8_proto": True}}
        model.cfg = cfg8
        sc8 = model.scores(prep_test, rd.y_test.size)
        model.cfg = cfg
        r8 = classification_report(rd.y_test, HyDE.predict(sc8, th1, th2), rd.classes)
        out["test_int8_proto"] = {k: r8[k] for k in ("macro_f1", "weighted_f1", "accuracy")}
    out["eval_sec"] = time.time() - t0
    return out


def run_hyde(cfg: dict, rd: RunData, save_path: Path | None) -> dict:
    t0 = time.time()
    sess = HydeSession(cfg, rd)
    model = sess.new_model()
    history = model.fit(sess.prep_fit, rd.y_fit, sess.prep_val, rd.y_val)
    train_sec = time.time() - t0
    ev = evaluate_hyde(model, sess, cfg)
    res = {"history": history, "train_sec": train_sec, "encoder_sec": sess.encoder_sec}
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path, {"classes": rd.classes, "theta1": ev["theta1"], "theta2": ev["theta2"],
                               "int8_proto": bool(cfg["eval"]["int8_proto"]), "config": cfg,
                               "dataset": rd.ds.name, "preprocess": _preprocess_meta(rd.ds)})
        res["model_path"] = str(save_path)
        res["model_bytes"] = save_path.stat().st_size
        n = min(int(cfg["eval"]["infer_check_rows"]), rd.y_test.size)
        if n:
            from infer import Predictor
            p_np, _ = Predictor(str(save_path)).predict(rd.ds.X_num_test[:n], rd.ds.X_cat_test[:n])
            res["infer_agreement"] = float((p_np == ev["_pred_test"][:n]).mean())
            print(f"  infer.py agreement on {n} test rows: {res['infer_agreement'] * 100:.3f}%", flush=True)
    res.update({k: v for k, v in ev.items() if not k.startswith("_")})
    return res


# =========================================================================== ML baselines
def run_ml(name: str, cfg: dict, rd: RunData, measure_size: bool = False) -> dict:
    from hyde.baselines import FeatureMatrix, fit_baseline, predict_proba
    ds, C = rd.ds, len(rd.classes)
    fm = FeatureMatrix().fit(ds.X_num[rd.fit], ds.cat_widths)
    X_fit, X_val = fm.transform(ds.X_num[rd.fit], ds.X_cat[rd.fit]), fm.transform(ds.X_num[rd.val], ds.X_cat[rd.val])
    model, info = fit_baseline(name, cfg, X_fit, rd.y_fit, X_val, rd.y_val, C, _device(cfg))
    del X_fit
    pv = predict_proba(model, X_val, C)
    X_test = fm.transform(ds.X_num_test, ds.X_cat_test)
    t0 = time.time()
    pt = predict_proba(model, X_test, C)
    infer_sec = time.time() - t0
    p_val, p_test = pv.argmax(1).astype(np.int64), pt.argmax(1).astype(np.int64)
    res = {"ml": info, "train_sec": info["train_sec"], "infer_ms_per_sample": 1e3 * infer_sec / max(pt.shape[0], 1)}
    if cfg["unknown"]["enabled"]:                                # max-probability open-set rule
        known_attack = rd.y_val != 0
        th = float(np.quantile(pv.max(1)[known_attack], cfg["unknown"]["reject_rate"])) if known_attack.any() else -1.0
        p_val[(p_val != 0) & (pv.max(1) < th)] = UNKNOWN
        p_test[(p_test != 0) & (pt.max(1) < th)] = UNKNOWN
        res["theta2"] = th
        res["unknown"] = unknown_report(rd.y_test, p_test, pt.max(1), rd.classes)
    res["val"] = classification_report(rd.y_val, p_val, rd.classes)
    res["val"].pop("per_class")
    res["test"] = classification_report(rd.y_test, p_test, rd.classes)
    if measure_size:
        import pickle
        res["model_bytes"] = len(pickle.dumps(model.net.state_dict() if name == "mlp" else model))
    return res


# =========================================================================== main
def result_path(tag: str, data: str, model: str, seed: int) -> Path:
    return RESULTS / tag / f"{data}__{model}__s{seed}.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="hyde", help="hyde | xgboost | lightgbm | random_forest | decision_tree | naive_bayes | mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--set", action="append", default=[], help="config override key=value (repeatable)")
    ap.add_argument("--config", action="append", default=[], help="extra yaml merged after tuned config")
    ap.add_argument("--no-tuned", action="store_true", help="ignore configs/tuned/<data>.yaml")
    ap.add_argument("--holdout", default="", help="comma-separated class names removed from training (unknown test)")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--measure-size", action="store_true", help="ML: pickle size of the fitted model")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-run even if the result JSON exists")
    args = ap.parse_args(argv)

    out = result_path(args.tag, args.data, args.model, args.seed)
    if out.exists() and not args.force:
        print(f"skip (exists): {out}")
        return
    cfg = build_config(args.data, args.set + [f"seed={args.seed}"], args.config, use_tuned=not args.no_tuned)
    if args.threads:
        cfg["threads"] = args.threads
    torch.set_num_threads(int(cfg["threads"]))
    print(f"=== {args.data} | {args.model} | seed {args.seed} | tag {args.tag}", flush=True)
    t0 = time.time()
    ds = load_dataset(args.data, threads=int(cfg["threads"]))
    holdout = [h.strip() for h in args.holdout.split(",") if h.strip()]
    rd = prepare_data(ds, cfg, holdout)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.model == "hyde":
        save = MODELS / args.tag / f"{args.data}__s{args.seed}.npz" if args.save_model else None
        res = run_hyde(cfg, rd, save)
    else:
        res = run_ml(args.model, cfg, rd, args.measure_size)

    res.update({
        "dataset": args.data, "model": args.model, "seed": args.seed, "tag": args.tag,
        "holdout": rd.held_out, "config": cfg, "git": _git_hash(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {"n_fit": int(rd.fit.size), "n_val": int(rd.val.size), "n_test": int(rd.y_test.size),
                 "n_numeric": len(ds.num_names), "n_categorical": len(ds.cat_names),
                 "classes": rd.classes, "dropped_columns": ds.meta["dropped"],
                 "test_unknown_rows": int((rd.y_test == UNKNOWN).sum())},
        "wall_sec": time.time() - t0,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    t = res["test"]
    print(f"=== test macro-F1 {t['macro_f1'] * 100:.2f} | weighted-F1 {t['weighted_f1'] * 100:.2f} "
          f"| acc {t['accuracy'] * 100:.2f} | {res['wall_sec']:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
