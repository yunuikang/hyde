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
  E5  dimension D             D in {1000, 2000, 5000} x 3 seeds              -> results/e5/D<d>/
  E6  cost benchmark          RUN ALONE at the very end, seed 0              -> results/cost/
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


def outdated(datasets) -> list[str]:
    """Tuned before the step2.benign_split axis existed -> need `tune.py` (extend mode) first."""
    out = []
    for d in datasets:
        p = ROOT / "configs" / "tuned" / f"{d}.yaml"
        if p.exists() and "benign_split" not in (yaml.safe_load(open(p)).get("step2") or {}):
            out.append(d)
    return out


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
    jobs = [[sys.executable, str(ROOT / "tune.py"), "--data", d] for d in outdated(ready(datasets))]
    for d in ready(datasets):
        tuned(d)
        jobs += [R("--data", d, "--seed", s, "--tag", "final", "--save-model") for s in SEEDS]
    return jobs


def tune_log(ds: str) -> list[dict]:
    p = ROOT / "results" / "tune" / ds / "log.json"
    return json.loads(p.read_text()) if p.exists() else []


def best_on(ds: str, axis: str, default):
    """Val-best NON-off value of an axis from the tuning log (used when tuning chose 'off').
    Keeps E3/E4 free of hand-picked values."""
    rows = [r for r in tune_log(ds) if r["axis"] == axis]
    if axis == "step1":
        # the proposed Step1 = elbow-K clustering + retrain; only theta1 (and retrain lr) are chosen on val.
        # Ties (gate never fires) -> the lowest theta1, i.e. the most active gate that does not hurt val.
        K0 = REG["datasets"][ds]["K"]
        rows = [r for r in rows if isinstance(r["value"], dict) and r["value"]["K"] == K0 and r["value"]["retrain"]]
        return max(rows, key=lambda r: (round(r["macro_f1"], 6), -r["value"]["theta1"]))["value"] if rows else default
    rows = [r for r in rows if r["value"] not in (0, 0.0, None, "off")]
    return max(rows, key=lambda r: r["macro_f1"])["value"] if rows else default


def step1_sets(v: dict) -> list[str]:
    s = ["step1.enabled=true", f"step1.K={v['K']}", f"step1.theta1={v['theta1']}",
         f"step1.retrain.enabled={str(bool(v['retrain'])).lower()}"]
    return s + ([f"step1.retrain.lr={v['lr']}"] if v.get("lr") is not None else [])


def step1_on(ds: str) -> list[str]:
    """Step1 settings for 'with Step1' runs: tuned ones if tuning kept Step1, else the val-best enabled combo."""
    t = tuned(ds)
    if t["step1"]["enabled"]:
        return []
    K0 = REG["datasets"][ds]["K"]
    return step1_sets(best_on(ds, "step1", {"K": K0, "retrain": True, "lr": 0.05, "theta1": 0.95}))


def ablation_arms(ds: str) -> dict[str, list[str]]:
    """Each arm flips exactly one component of the tuned config. When tuning switched a component
    off, the 'on' arm uses that component's val-best non-off value from the tuning log."""
    t = tuned(ds)
    K0 = REG["datasets"][ds]["K"]
    cw, hn, bt = t["class_weight"]["p"], t["hard_negative"]["beta"], t["boosting"]["T"]
    bs = int((t["step2"] or {}).get("benign_split", 0) or 0)
    arms = {
        "encoder_nonlinear": ["encoder.type=nonlinear"],
        "class_weight_" + ("off" if cw > 0 else "on"):
            [f"class_weight.p={0.0 if cw > 0 else best_on(ds, 'class_weight.p', 0.5)}"],
        "hard_negative_" + ("off" if hn > 0 else "on"):
            [f"hard_negative.beta={0.0 if hn > 0 else best_on(ds, 'hard_negative.beta', 1.0)}"],
        "boosting_" + ("off" if bt > 0 else "on"):
            [f"boosting.T={0 if bt > 0 else best_on(ds, 'boosting.T', 10)}"],
        "benign_split_" + ("off" if bs > 1 else "on"):
            [f"step2.benign_split={0 if bs > 1 else best_on(ds, 'step2.benign_split', K0)}"],
    }
    if t["step1"]["enabled"]:
        arms["step1_off"] = ["step1.enabled=false"]
        arms["step1_K1"] = ["step1.K=1"]
        arms["step1_retrain_" + ("off" if t["step1"]["retrain"]["enabled"] else "on")] = \
            [f"step1.retrain.enabled={str(not t['step1']['retrain']['enabled']).lower()}"]
    else:                     # tuning dropped Step1: show its effect with the val-best enabled setting
        on = step1_on(ds)
        retrain = "step1.retrain.enabled=true" in on
        arms["step1_on"] = on
        arms["step1_on_K1"] = on + ["step1.K=1"]
        arms["step1_on_retrain_" + ("off" if retrain else "on")] = \
            on + [f"step1.retrain.enabled={str(not retrain).lower()}"]
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
            s1 = [x for v in (step1_on(d) or ["step1.enabled=true"]) for x in ("--set", v)]
            jobs.append(R(*base, "--seed", s, "--tag", f"unknown/2step/{sc['scenario']}", *s1))
            jobs.append(R(*base, "--seed", s, "--tag", f"unknown/1step/{sc['scenario']}",
                          "--set", "step1.enabled=false"))
            for m in ("random_forest", "xgboost"):
                jobs.append(R(*base, "--seed", s, "--model", m, "--tag", f"unknown/ml/{sc['scenario']}"))
    return jobs


def e5(datasets):
    jobs = []
    for d in ready(datasets):
        jobs += [R("--data", d, "--seed", s, "--tag", f"e5/D{D}", "--set", f"encoder.D={D}", "--save-model")
                 for D in (1000, 2000, 5000) for s in SEEDS]
    return jobs


def e6(datasets):
    """Cost benchmark. Run ALONE (no other job on the machine), seed 0, same threads for every model:
    training time, GPU inference, numpy CPU inference (float / int8 prototypes), model bytes."""
    jobs = []
    for d in ready(datasets):
        jobs.append(R("--data", d, "--seed", 0, "--tag", "cost/hyde", "--save-model", "--bench-cpu", 100000))
        jobs += [R("--data", d, "--seed", 0, "--model", m, "--tag", "cost/ml", "--measure-size") for m in ML_MODELS]
    return jobs


STAGES = {"cache": cache, "E0": e0, "E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def wait_for_disk(min_gb: float = 5.0, poll_sec: int = 300) -> None:
    """The volume is shared with other users; when it is (nearly) full, wait instead of failing jobs."""
    import shutil
    warned = False
    while shutil.disk_usage(ROOT).free / 2**30 < min_gb:
        if not warned:
            print(f"##### disk free < {min_gb} GB — waiting (checks every {poll_sec}s)", flush=True)
            warned = True
        time.sleep(poll_sec)
    if warned:
        print("##### disk space available again — resuming", flush=True)


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
        wait_for_disk()
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
