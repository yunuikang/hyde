#!/usr/bin/env python3
"""Write docs/DATASETS.md from the caches (sizes, kept/dropped columns, classes, imbalance).

    python dataset_info.py            # all datasets that have a cache
Numbers come straight from cache/<ds>.npz, i.e. exactly what every experiment sees.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REG = yaml.safe_load(open(ROOT / "configs" / "datasets.yaml"))
REASON = {"leaky": "누수(포트·날짜·난이도)", "metadata": "메타데이터(IP·시각·ID)", "constant": "상수(train 고유값 1개)",
          "label": "다른 라벨 컬럼"}


def info(name: str) -> dict | None:
    path = ROOT / "cache" / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    y, yt = z["y"], z["y_test"]
    classes = meta["classes"]
    tr = np.bincount(y, minlength=len(classes))
    te = np.bincount(yt[yt >= 0], minlength=len(classes))
    return {"meta": meta, "classes": classes, "train": tr, "test": te, "test_unknown": int((yt < 0).sum()),
            "n_train": int(y.size), "n_test": int(yt.size)}


def main():
    rows, sections = [], []
    for name, spec in REG["datasets"].items():
        d = info(name)
        if d is None:
            rows.append(f"| {name} | {spec['tier']} | (캐시 없음) |||||||||")
            continue
        m, tr = d["meta"], d["train"]
        attack = tr[1:]
        rare = int((tr < 100).sum())
        imb = int(tr.max() / max(tr[tr > 0].min(), 1))
        widths = [len(v) for v in m["vocabs"]]
        dropped = defaultdict(list)
        for col, why in m["dropped"].items():
            dropped[why.split("(")[0]].append(col)
        rows.append(f"| {name} | {spec['tier']} | {d['n_train']:,} | {d['n_test']:,} | {len(m['num_names'])} | "
                    f"{len(m['cat_names'])} ({sum(widths)}) | {len(d['classes'])} | {100 * tr[0] / tr.sum():.1f} | "
                    f"{imb:,} | {rare} | {len(m['dropped'])} | {spec['K']} |")

        s = [f"## {name}", ""]
        s.append(f"- 위치: `{REG['root']}/{spec['dir']}/{{training,test}}-flow.csv` · Tier {spec['tier']} · "
                 f"Step1 K(elbow) = {spec['K']}" + (f" · 라벨 통일: {m['label_canon']}" if m.get("label_canon") else "")
                 + (f" · train 캡: {m['train_cap']:,}" if m.get("train_cap") else ""))
        s.append(f"- 행: train {d['n_train']:,} / test {d['n_test']:,}"
                 + (f" (test 중 unknown {d['test_unknown']:,}행 = test에만 있는 라벨 {len(m['test_only_labels'])}종: "
                    f"{', '.join(m['test_only_labels'])})" if d["test_unknown"] else ""))
        s.append(f"- 사용 피처: 숫자 {len(m['num_names'])}개 + 범주 {len(m['cat_names'])}개"
                 + (" (" + ", ".join(f"`{c}` {w}종" for c, w in zip(m["cat_names"], widths)) + ")" if widths else ""))
        for why in ("leaky", "metadata", "constant", "high_cardinality", "label"):
            if dropped.get(why):
                s.append(f"- 드롭 — {REASON.get(why, why)}: " + ", ".join(f"`{c}`" for c in dropped[why]))
        s.append(f"- 숫자 피처 목록: " + ", ".join(f"`{c}`" for c in m["num_names"]))
        s += ["", "| # | 클래스 | train | train % | test |", "|---:|---|---:|---:|---:|"]
        for i, c in enumerate(d["classes"]):
            s.append(f"| {i} | {c} | {tr[i]:,} | {100 * tr[i] / tr.sum():.3f} | {d['test'][i]:,} |")
        if d["test_unknown"]:
            s.append(f"| −1 | unknown (test에만 있는 라벨) | 0 | 0 | {d['test_unknown']:,} |")
        sections.append("\n".join(s) + "\n")

    doc = [
        "# 데이터셋 정보 (자동 생성)", "",
        "`python dataset_info.py`가 `cache/<ds>.npz`에서 만든다. 모든 실험이 실제로 쓰는 데이터와 같다.",
        "컬럼 규칙: `configs/datasets.yaml`, 처리 방식: ARCHITECTURE.md §3.", "",
        "| 데이터셋 | Tier | train | test | 숫자 피처 | 범주 피처 (원-핫 폭) | 클래스 | benign % | 불균형 (최대/최소) | 100행 미만 클래스 | 드롭 컬럼 | K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", *rows, "",
        "- 클래스 수는 benign 포함. 불균형 = train에서 가장 큰 클래스 ÷ 가장 작은 클래스.",
        "- 범주 피처의 원-핫 폭은 train에서 본 값의 수(+ 처음 보는 값 슬롯 1개는 제외).", "",
        *sections,
    ]
    out = ROOT / "docs" / "DATASETS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(doc))
    print("\n".join(doc[:len(rows) + 8]))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
