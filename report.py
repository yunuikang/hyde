#!/usr/bin/env python3
"""Summarise result JSONs into markdown/CSV tables (mean ± std over seeds).

    python report.py final ml                      # Table A: HyDE vs ML, macro & weighted F1
    python report.py "ablation/*" final            # Table B: ablation arms next to the full model
    python report.py "unknown/*/*" --block unknown --metrics unknown_recall unknown_precision auroc known_macro_f1
    python report.py e5/D1000 e5/D2000 e5/D5000 final --metrics macro_f1 weighted_f1 --extra model_bytes infer_ms_per_sample

Every column is one (tag, model) pair; rows are datasets. Output: stdout + results/reports/<name>.{md,csv}.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ORDER = ["mirai", "nsl-kdd", "unsw-nb15", "ton-iot", "cicids2017", "xiiotid", "wustl-iiot-2021",
         "edge-iiot", "cicids2018", "lspr23", "nf-ton-iot-v3"]


def load(tag_patterns: list[str]) -> list[dict]:
    rows = []
    for pat in tag_patterns:
        for path in sorted(glob.glob(str(RESULTS / pat / "*.json"))):
            r = json.loads(Path(path).read_text())
            r["_tag"] = str(Path(path).parent.relative_to(RESULTS))
            rows.append(r)
    return rows


def value(r: dict, block: str, metric: str):
    src = r.get(block, {}) if block else r
    v = src.get(metric) if isinstance(src, dict) else None
    if v is None and block == "test" and metric.startswith("binary_"):
        v = r["test"]["binary"].get(metric[len("binary_"):])
    return v


def fmt(vals, scale):
    vals = [v * scale for v in vals if v is not None]
    if not vals:
        return "—"
    m = statistics.mean(vals)
    return f"{m:.2f}" if len(vals) == 1 else f"{m:.2f} ± {statistics.stdev(vals):.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="result tags (glob patterns allowed)")
    ap.add_argument("--block", default="test", help="test | val | unknown | '' (top level)")
    ap.add_argument("--metrics", nargs="+", default=["macro_f1", "weighted_f1"])
    ap.add_argument("--extra", nargs="*", default=[], help="top-level numeric fields (train_sec, model_bytes, ...)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    rows = load(args.tags)
    if not rows:
        raise SystemExit("no results found")
    groups = defaultdict(list)                       # (tag, model) -> dataset -> [result]
    for r in rows:
        groups[(r["_tag"], r["model"])].append(r)
    cols = sorted(groups)
    datasets = [d for d in ORDER if any(r["dataset"] == d for r in rows)]
    specs = [(args.block, m, 100.0) for m in args.metrics] + [("", e, 1.0) for e in args.extra]

    header = ["dataset"] + [f"{t}:{m} {met}" for (t, m) in cols for (_, met, _) in specs]
    table = []
    for d in datasets:
        line = [d]
        for col in cols:
            rs = [r for r in groups[col] if r["dataset"] == d]
            for block, met, scale in specs:
                line.append(fmt([value(r, block, met) for r in rs], scale) if rs else "")
        table.append(line)

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(l) + " |" for l in table]
    print("\n".join(md))
    name = args.name or "_".join(t.replace("/", "-").replace("*", "x") for t in args.tags)
    out = RESULTS / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text("\n".join(md) + "\n")
    with open(out / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    print(f"\n-> {out / name}.md / .csv")


if __name__ == "__main__":
    main()
