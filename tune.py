#!/usr/bin/env python3
"""Per-dataset sequential tuning on the VALIDATION split -> configs/tuned/<dataset>.yaml

Order (each axis fixes its best value before the next one is tried; ties keep the earlier,
simpler candidate, which is always listed first):
  A. Step2 only (Step1 off):  readout -> learning rate -> class-weight p -> hard-negative beta -> boosting T
  B. Step1 jointly:           K x retrain x theta1 (Step2 model trained once; theta1 swept post-hoc)
  C. PairBind bins            (full pipeline)
Criterion: cfg.step2.select_metric on val (default macro-F1). Test data is never touched.

    python tune.py --data mirai
    python tune.py --data lspr23            # Tier B: fit rows subsampled to 10 % automatically
"""

from __future__ import annotations

import run  # noqa: F401  (sets thread env vars before numpy/torch)

import argparse
import copy
import json
import time

import torch
import yaml

from hyde.config import CONFIGS, ROOT, build_config, datasets_registry
from hyde.data import load_dataset
from hyde.io import write_text
from hyde.metrics import classification_report
from hyde.model import HyDE
from run import HydeSession, prepare_data

GRID = {
    "readout": ["mass", "adamw"],
    "lr": {"mass": [0.035, 0.01, 0.1], "adamw": [0.01, 0.003, 0.03]},
    "cw_p": [0.0, 0.25, 0.5, 0.75, 1.0],
    "beta": [0.0, 0.5, 1.0, 2.0, 4.0],
    "boost_T": [0, 5, 10, 20],
    "benign_split": ["off", "K", "2K"],                 # multi-prototype normal traffic (x elbow K)
    "retrain": [(False, None), (True, 0.01), (True, 0.05)],
    "theta1": [0.70, 0.80, 0.85, 0.90, 0.95, 0.97],
    "bins": [16, 8, 32],
}


class Tuner:
    def __init__(self, data: str, seed: int, subsample: float | None, threads: int | None = None,
                 from_tuned: bool = False):
        self.data = data
        cfg = build_config(data, [f"seed={seed}"], use_tuned=from_tuned)
        cfg["step1"]["enabled"] = False
        if subsample:
            cfg["data"]["tune_subsample"] = subsample
        if threads:
            cfg["threads"] = threads
        torch.set_num_threads(int(cfg["threads"]))
        self.cfg = cfg
        self.key = "macro_f1" if cfg["step2"]["select_metric"] == "macro_f1" else "weighted_f1"
        self.ds = load_dataset(data, threads=int(cfg["threads"]))
        self.rd = prepare_data(self.ds, cfg)
        self.sessions: dict = {}
        self.log: list = []
        self.out_dir = ROOT / "results" / "tune" / data
        self.out_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.out_dir / "log.json"            # resume: candidates finished before a crash
        if log_path.exists():
            self.log = json.loads(log_path.read_text())
        self.done = {r["key"]: r for r in self.log if r.get("key")}

    def session(self, cfg) -> HydeSession:
        key = json.dumps(cfg["encoder"], sort_keys=True)
        if key not in self.sessions:
            self.sessions = {key: HydeSession(cfg, self.rd)}          # keep one encoder in memory
        return self.sessions[key]

    def val_metrics(self, model: HyDE, sess: HydeSession, theta1=None) -> dict:
        sc = model.scores(sess.prep_val, self.rd.y_val.size)
        rep = classification_report(self.rd.y_val, HyDE.predict(sc, theta1, None), self.rd.classes)
        return {"macro_f1": rep["macro_f1"], "weighted_f1": rep["weighted_f1"]}, sc

    def train(self, cfg) -> tuple[HyDE, HydeSession, dict]:
        sess = self.session(cfg)
        model = sess.new_model(cfg)
        model.fit(sess.prep_fit, self.rd.y_fit, sess.prep_val, self.rd.y_val)
        th1 = cfg["step1"]["theta1"] if cfg["step1"]["enabled"] else None
        m, _ = self.val_metrics(model, sess, th1)
        return model, sess, m

    def record(self, axis, value, metrics, sec, key=None):
        row = {"axis": axis, "value": value, **metrics, "sec": round(sec, 1), "key": key}
        self.log.append(row)
        print(f"[tune {self.data}] {axis}={value}: val macro={metrics['macro_f1']:.4f} "
              f"weighted={metrics['weighted_f1']:.4f} ({sec:.0f}s)", flush=True)
        write_text(self.out_dir / "log.json", json.dumps(self.log, indent=1))

    def axis(self, name, candidates, apply):
        """Try each candidate on a copy of self.cfg; keep the best (first wins ties)."""
        best = None
        for value in candidates:
            cfg = copy.deepcopy(self.cfg)
            apply(cfg, value)
            key = json.dumps(cfg, sort_keys=True, default=str)
            if key in self.done:                          # finished before a restart
                r = self.done[key]
                m = {"macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"]}
                print(f"[tune {self.data}] {name}={value}: cached val macro={m['macro_f1']:.4f}", flush=True)
            else:
                t0 = time.time()
                _, _, m = self.train(cfg)
                self.record(name, value, m, time.time() - t0, key)
            if best is None or m[self.key] > best[0] + 1e-9:
                best = (m[self.key], value, cfg)
        self.cfg = best[2]
        print(f"[tune {self.data}] -> {name} = {best[1]}", flush=True)

    # ------------------------------------------------------------------ stages
    def stage_a(self):
        self.axis("readout", GRID["readout"], lambda c, v: c["step2"].__setitem__("readout", v))
        r = self.cfg["step2"]["readout"]
        self.axis("lr", GRID["lr"][r], lambda c, v: c["step2"][r].__setitem__("lr", v))
        self.axis("class_weight.p", GRID["cw_p"], lambda c, v: c["class_weight"].__setitem__("p", v))
        self.axis("hard_negative.beta", GRID["beta"], lambda c, v: c["hard_negative"].__setitem__("beta", v))
        self.axis("boosting.T", GRID["boost_T"], lambda c, v: c["boosting"].__setitem__("T", v))
        self.axis_benign_split()

    def axis_benign_split(self):
        K0 = datasets_registry()["datasets"][self.data]["K"]
        values = {"off": 0, "K": K0, "2K": 2 * K0}
        self.axis("step2.benign_split", [values[v] for v in GRID["benign_split"]],
                  lambda c, v: c["step2"].__setitem__("benign_split", v))

    def stage_b(self):
        """Step1: train Step2 once, then K x retrain candidates with a post-hoc theta1 sweep."""
        base = copy.deepcopy(self.cfg)
        base["step1"]["enabled"] = False
        model, sess, m_off = self.train(base)
        best = (m_off[self.key], {"enabled": False})
        self.record("step1", "off", m_off, 0)
        K0 = datasets_registry()["datasets"][self.data]["K"]
        for K in sorted({1, K0, 2 * K0}):
            for enabled, lr in GRID["retrain"]:
                cfg = copy.deepcopy(base)
                cfg["step1"].update(enabled=True, K=K)
                cfg["step1"]["retrain"]["enabled"] = enabled
                if lr is not None:
                    cfg["step1"]["retrain"]["lr"] = lr
                t0 = time.time()
                model.cfg, model.K = cfg, K
                model.fit_step1(sess.prep_fit, self.rd.y_fit)
                _, sc = self.val_metrics(model, sess, None)
                for th in GRID["theta1"]:
                    rep = classification_report(self.rd.y_val, HyDE.predict(sc, th, None), self.rd.classes)
                    m = {"macro_f1": rep["macro_f1"], "weighted_f1": rep["weighted_f1"]}
                    self.record("step1", {"K": K, "retrain": enabled, "lr": lr, "theta1": th}, m,
                                time.time() - t0)
                    if m[self.key] > best[0] + 1e-9:
                        best = (m[self.key], {"enabled": True, "K": K, "retrain": enabled, "lr": lr, "theta1": th})
        choice = best[1]
        self.cfg["step1"]["enabled"] = choice["enabled"]
        if choice["enabled"]:
            self.cfg["step1"].update(K=choice["K"], theta1=choice["theta1"])
            self.cfg["step1"]["retrain"]["enabled"] = choice["retrain"]
            if choice["lr"] is not None:
                self.cfg["step1"]["retrain"]["lr"] = choice["lr"]
        print(f"[tune {self.data}] -> step1 = {choice}", flush=True)

    def stage_c(self):
        if self.cfg["encoder"]["type"] == "pairbind":
            self.axis("encoder.bins", GRID["bins"], lambda c, v: c["encoder"].__setitem__("bins", v))

    def save(self):
        c = self.cfg
        tuned = {
            "encoder": {"bins": c["encoder"]["bins"]},
            "step1": {"enabled": c["step1"]["enabled"], "K": c["step1"]["K"], "theta1": c["step1"]["theta1"],
                      "retrain": {"enabled": c["step1"]["retrain"]["enabled"], "lr": c["step1"]["retrain"]["lr"]}},
            "step2": {"readout": c["step2"]["readout"], "benign_split": c["step2"]["benign_split"],
                      c["step2"]["readout"]: {"lr": c["step2"][c["step2"]["readout"]]["lr"]}},
            "class_weight": {"p": c["class_weight"]["p"]},
            "hard_negative": {"beta": c["hard_negative"]["beta"]},
            "boosting": {"T": c["boosting"]["T"]},
        }
        path = CONFIGS / "tuned" / f"{self.data}.yaml"
        header = (f"# tuned by tune.py on {time.strftime('%Y-%m-%d %H:%M')} (seed {c['seed']}, "
                  f"criterion val {self.key}, subsample {c['data']['tune_subsample']})\n")
        write_text(path, header + yaml.safe_dump(tuned, sort_keys=False))
        print(f"[tune {self.data}] wrote {path}\n{yaml.safe_dump(tuned, sort_keys=False)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subsample", type=float, default=None,
                    help="fraction of fit rows used for tuning (default: 0.1 for Tier B, else all)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    tier = datasets_registry()["datasets"][args.data].get("tier", "A")
    sub = args.subsample if args.subsample is not None else (0.1 if tier == "B" else None)
    path = ROOT / "configs" / "tuned" / f"{args.data}.yaml"
    if path.exists():
        if "benign_split" in (yaml.safe_load(open(path)).get("step2") or {}):
            print(f"skip (exists): configs/tuned/{args.data}.yaml  — delete it to re-tune")
            return
        # tuned before the benign_split axis existed: keep every earlier choice, add only that axis
        # (Step1 is re-checked because Step2 changed).
        print(f"[tune {args.data}] extending existing tuning with step2.benign_split", flush=True)
        t = Tuner(args.data, args.seed, sub, args.threads, from_tuned=True)
        t.cfg["step1"]["enabled"] = False
        t.axis_benign_split()
        t.stage_b()
        t.save()
        return
    t = Tuner(args.data, args.seed, sub, args.threads)
    t0 = time.time()
    t.stage_a()
    t.stage_b()
    t.stage_c()
    t.save()
    print(f"[tune {args.data}] done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
