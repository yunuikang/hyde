#!/usr/bin/env python3
"""Experiment stages E0-E5 as job lists; run them sequentially, optionally split across sessions.

    python jobs.py E2 --dry                 # list the commands
    python jobs.py E2                       # run all (finished results are skipped)
    python jobs.py E0 --shard 0/3           # this session runs every 3rd job starting at 0
    python jobs.py E3 --datasets mirai nsl-kdd

Stages (EXPERIMENT_PLAN.md):
  cache  parse every CSV once -> cache/<ds>.npz   (run this BEFORE E0/E1 run in parallel)
  E0  ML baselines            6 models x 11 datasets x 3 seeds              -> results/ml/
  E1  tuning                  tune.py per dataset                            -> configs/tuned/
  E2  final HyDE              tuned config x 3 seeds, model saved            -> results/final/, models/final/
  E3  ablation                one component changed per arm x 3 seeds        -> results/ablation/<arm>/
  E4  unknown                 largest/median/smallest attack held out        -> results/unknown/<variant>/<scenario>/
  E5  dimension D             D in {1000, 2000, 5000} (seed 0, Tier A)       -> results/e5/D<d>/
E2-E4 need configs/tuned/<dataset>.yaml (E1) to exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REG = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))
DATASETS = [d for d, s in REG["datasets"].items() if s["tier"] == "A"] + \
           [d for d, s in REG["datasets"].items() if s["tier"] == "B"]
TIER_A = [d for d in DATASETS if REG["datasets"][d]["tier"] == "A"]
SEEDS = [0, 1, 2]
ML_MODELS = ["xgboost", "lightgbm", "random_forest", "decision_tree", "naive_bayes", "mlp"]
SCENARIO_FILE = ROOT / "results" / "unknown" / "scenarios.csv"


def tuned(ds: str) -> dict:
    path = ROOT / "configs" / "tuned" / f"{ds}.yaml"
    if not path.exists():
        raise SystemExit(f"missing {path} — run `python jobs.py E1` first")
    return yaml.safe_load(open(path))


def ready(datasets) -> list[str]:
    """Datasets whose tuning (E1) is finished; the others are skipped with a note."""
    ok = [d for d in datasets if (ROOT / "configs" / "tuned" / f"{d}.yaml").exists()]
    if len(ok) < len(datasets):
        print(f"(waiting for E1 tuning: {[d for d in datasets if d not in ok]})", flush=True)
    return ok


def R(*args) -> list[str]:
    return [sys.executable, str(ROOT / "run.py"), *map(str, args)]


# --------------------------------------------------------------------------- stages
def cache(datasets):
    code = "import sys; from hyde.data import load_dataset; load_dataset(sys.argv[1])"
    return [[sys.executable, "-c", code, d] for d in datasets if not (ROOT / "cache" / f"{d}.npz").exists()]


def e0(datasets):
    return [R("--data", d, "--model", m, "--seed", s, "--tag", "ml", "--measure-size")
            for d in datasets for m in ML_MODELS for s in SEEDS]


def e1(datasets):
    return [[sys.executable, str(ROOT / "tune.py"), "--data", d] for d in datasets]


def e2(datasets):
    jobs = []
    for d in ready(datasets):
        tuned(d)
        jobs += [R("--data", d, "--seed", s, "--tag", "final", "--save-model") for s in SEEDS]
    return jobs


def ablation_arms(ds: str) -> dict[str, list[str]]:
    """Each arm flips exactly one component of the tuned config."""
    t = tuned(ds)
    cw_on = t["class_weight"]["p"] > 0
    hn_on = t["hard_negative"]["beta"] > 0
    bt_on = t["boosting"]["T"] > 0
    arms = {
        "encoder_nonlinear": ["encoder.type=nonlinear"],
        "class_weight_" + ("off" if cw_on else "on"): [f"class_weight.p={0.0 if cw_on else 0.5}"],
        "hard_negative_" + ("off" if hn_on else "on"): [f"hard_negative.beta={0.0 if hn_on else 1.0}"],
        "boosting_" + ("off" if bt_on else "on"): [f"boosting.T={0 if bt_on else 10}"],
    }
    if t["step1"]["enabled"]:
        arms["step1_off"] = ["step1.enabled=false"]
        arms["step1_K1"] = ["step1.K=1"]
        arms["step1_retrain_" + ("off" if t["step1"]["retrain"]["enabled"] else "on")] = \
            [f"step1.retrain.enabled={str(not t['step1']['retrain']['enabled']).lower()}"]
    else:                                           # tuned without Step1: show what Step1 would do
        arms["step1_on"] = ["step1.enabled=true", "step1.theta1=0.95"]
    return arms


def e3(datasets):
    jobs = []
    for d in ready(datasets):
        for arm, sets in ablation_arms(d).items():
            extra = [x for s in sets for x in ("--set", s)]
            jobs += [R("--data", d, "--seed", s, "--tag", f"ablation/{arm}", *extra) for s in SEEDS]
    return jobs


def scenarios() -> list[dict]:
    """Largest / median / smallest attack class (test >= 30 rows) per dataset, fixed once on disk."""
    if SCENARIO_FILE.exists():
        return list(csv.DictReader(open(SCENARIO_FILE)))
    rows = []
    for d in DATASETS:
        z = np.load(ROOT / "cache" / f"{d}.npz")
        meta = json.loads(str(z["meta"]))
        y, yt = z["y"], z["y_test"]
        cand = [(int((y == i).sum()), c) for i, c in enumerate(meta["classes"])
                if c != "benign" and int((yt == i).sum()) >= 30]
        cand.sort(reverse=True)
        pick = cand if len(cand) <= 3 else [cand[0], cand[len(cand) // 2], cand[-1]]
        for (n, c), sid in zip(pick, ["large", "median", "small"]):
            rows.append({"dataset": d, "scenario": sid, "holdout": c, "train_rows": n})
        if meta["test_only_labels"]:
            rows.append({"dataset": d, "scenario": "natural", "holdout": "", "train_rows": 0})
    SCENARIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCENARIO_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "scenario", "holdout", "train_rows"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {SCENARIO_FILE}")
    return rows


def e4(datasets):
    jobs = []
    ok = ready(datasets)
    for sc in scenarios():
        d = sc["dataset"]
        if d not in ok:
            continue
        tuned(d)
        base = ["--data", d, "--set", "unknown.enabled=true"] + (["--holdout", sc["holdout"]] if sc["holdout"] else [])
        for s in SEEDS:
            jobs.append(R(*base, "--seed", s, "--tag", f"unknown/2step/{sc['scenario']}",
                          "--set", "step1.enabled=true"))
            jobs.append(R(*base, "--seed", s, "--tag", f"unknown/1step/{sc['scenario']}",
                          "--set", "step1.enabled=false"))
            for m in ("random_forest", "xgboost"):
                jobs.append(R(*base, "--seed", s, "--model", m, "--tag", f"unknown/ml/{sc['scenario']}"))
    return jobs


def e5(datasets):
    jobs = []
    for d in ready(datasets):
        if d not in TIER_A:
            continue
        tuned(d)
        jobs += [R("--data", d, "--seed", 0, "--tag", f"e5/D{D}", "--set", f"encoder.D={D}", "--save-model")
                 for D in (1000, 2000, 5000)]
    return jobs


STAGES = {"cache": cache, "E0": e0, "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--shard", default="0/1", help="i/n: run jobs i, i+n, i+2n, ...")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads per job (passed to run.py / tune.py)")
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    jobs = STAGES[args.stage](args.datasets)
    if args.threads and args.stage != "cache":
        jobs = [cmd + ["--threads", str(args.threads)] for cmd in jobs]
    mine = jobs[i::n]
    print(f"{args.stage}: {len(jobs)} jobs, this shard ({args.shard}) runs {len(mine)}", flush=True)
    failed = []
    for k, cmd in enumerate(mine, 1):
        print(f"\n##### [{k}/{len(mine)}] {' '.join(cmd[1:])}", flush=True)
        if args.dry:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            failed.append(" ".join(cmd[1:]))
            print(f"##### FAILED (exit {rc})", flush=True)
        else:
            print(f"##### done in {time.time() - t0:.0f}s", flush=True)
    if failed:
        print("\nFAILED JOBS:\n" + "\n".join(failed))


if __name__ == "__main__":
    main()
