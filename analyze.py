#!/usr/bin/env python3
"""Collect every finished result into docs/RESULTS.md (re-run any time; incomplete cells are marked).

    python analyze.py
Sections: coverage · tuned configs · A HyDE vs ML · A' binary (folded) · B ablation · C unknown ·
D dimension / int8 · per-class files. All numbers are test-set means ± std over seeds.
Cost numbers are NOT reported here (only E6 measures them cleanly).
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
DATASETS = ["mirai", "nsl-kdd", "unsw-nb15", "ton-iot", "cicids2017", "xiiotid", "wustl-iiot-2021",
            "edge-iiot", "cicids2018", "lspr23", "nf-ton-iot-v3"]
ML = ["xgboost", "lightgbm", "random_forest", "decision_tree", "naive_bayes", "mlp"]
SHORT = {"xgboost": "XGB", "lightgbm": "LGBM", "random_forest": "RF", "decision_tree": "DT",
         "naive_bayes": "NB", "mlp": "MLP"}
SEEDS = 3


def load(pattern: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(RES / pattern)))]


def by_seed(rows):
    return {r["seed"]: r for r in rows}


def m(rows, f):
    vals = [f(r) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return st.mean(vals) * 100, (st.stdev(vals) * 100 if len(vals) > 1 else 0.0), len(vals)


def fmt(mean, sd, n, bold=False, show_n=True):
    if mean is None:
        return "—"
    s = f"{mean:.2f}" + (f" ± {sd:.2f}" if n > 1 else "")
    s = f"**{s}**" if bold else s
    return s + (f" ({n})" if show_n and n < SEEDS else "")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def macro(r):
    return r["test"]["macro_f1"]


def weighted(r):
    return r["test"]["weighted_f1"]


def main():
    doc = ["# HyDE 결과 정리 (자동 생성)", "",
           "`python analyze.py`로 생성한다. 숫자는 test 3시드 평균 ± 표준편차. 괄호 안 숫자는 아직 덜 끝난 시드 수다.",
           "튜닝·선택은 val로만 했고 test는 여기서 처음 본다. **시간·처리량·모델 크기 비교는 E6(비용 측정)에서만** 가져온다.", ""]

    # ------------------------------------------------------------------ coverage
    final = {d: load(f"final/{d}__hyde__s*.json") for d in DATASETS}
    ml = {(d, a): load(f"ml/{d}__{a}__s*.json") for d in DATASETS for a in ML}
    rows = []
    for d in DATASETS:
        abl = len(glob.glob(str(RES / "ablation" / "*" / f"{d}__hyde__s*.json")))
        unk = len([p for p in glob.glob(str(RES / "unknown" / "*" / "*" / f"{d}__*.json")) if "xgboost" not in p])
        e5 = len(glob.glob(str(RES / "e5" / "*" / f"{d}__hyde__s*.json")))
        rows.append([d, len(final[d]), sum(len(ml[(d, a)]) for a in ML), abl, unk, e5])
    doc += ["## 0. 완료 현황", "", table(["데이터셋", "E2 최종 (3)", "E0 ML (18)", "E3 ablation", "E4 unknown", "E5 D (9)"], rows), ""]

    # ------------------------------------------------------------------ tuned configs
    rows = []
    for d in DATASETS:
        p = ROOT / "configs" / "tuned" / f"{d}.yaml"
        if not p.exists():
            rows.append([d] + ["—"] * 8); continue
        t = yaml.safe_load(open(p))
        ro = t["step2"]["readout"]
        s1 = t["step1"]
        rows.append([d, f"{ro} (lr {t['step2'][ro]['lr']})", t["class_weight"]["p"], t["hard_negative"]["beta"],
                     t["boosting"]["T"], t["step2"].get("benign_split", "—"),
                     (f"K={s1['K'] or 'elbow'}, θ1={s1['theta1']}, retrain={'on' if s1['retrain']['enabled'] else 'off'}"
                      if s1["enabled"] else "off"), t["encoder"]["bins"]])
    doc += ["## 1. 튜닝 결과 (val macro-F1 기준)", "",
            table(["데이터셋", "readout", "CW p", "HN β", "boost T", "정상 분할 k", "Step1", "bins"], rows), ""]

    # ------------------------------------------------------------------ A: HyDE vs ML
    for title, f in (("macro-F1", macro), ("weighted-F1", weighted)):
        rows = []
        wins = 0
        for d in DATASETS:
            hy = m(final[d], f)
            nl = m(load(f"ablation/encoder_nonlinear/{d}__hyde__s*.json"), f)
            mls = {a: m(ml[(d, a)], f) for a in ML}
            best_ml = max((v[0] for v in mls.values() if v[0] is not None), default=None)
            best = max([x for x in [hy[0], best_ml] if x is not None], default=None)
            gap = (f"{hy[0] - best_ml:+.2f}" if hy[0] is not None and best_ml is not None else "—")
            wins += hy[0] is not None and best_ml is not None and hy[0] >= best_ml
            rows.append([d, fmt(*hy, bold=hy[0] == best), fmt(*nl)] +
                        [fmt(*mls[a], bold=mls[a][0] == best) for a in ML] + [gap])
        doc += [f"## A. HyDE vs ML — {title}", "",
                table(["데이터셋", "**HyDE**", "기존 HDC (nonlinear)"] + [SHORT[a] for a in ML] + ["HyDE − ML 최선"], rows),
                "", f"HyDE가 ML 최선 이상인 데이터셋: **{wins}개**. 기존 HDC = 같은 튜닝값에서 인코더만 nonlinear(E3 `encoder_nonlinear`).", ""]

    # ------------------------------------------------------------------ A': binary (folded)
    rows = []
    for d in DATASETS:
        hy_b = m(final[d], lambda r: r["test"]["binary"]["macro_f1"])
        hy_f = m(final[d], lambda r: r["test"]["binary"]["fpr"])
        best = None
        for a in ML:
            v = m(ml[(d, a)], lambda r: r["test"]["binary"]["macro_f1"])
            if v[0] is not None and (best is None or v[0] > best[1][0]):
                best = (a, v, m(ml[(d, a)], lambda r: r["test"]["binary"]["fpr"]))
        rows.append([d, fmt(*hy_b), fmt(*hy_f)] + ([f"{fmt(*best[1])} ({SHORT[best[0]]})", fmt(*best[2])] if best else ["—", "—"]))
    doc += ["## A'. 이진 탐지 (다중 예측을 benign/공격으로 접은 값)", "",
            table(["데이터셋", "HyDE 이진 macro-F1", "HyDE FPR", "ML 최선 이진 macro-F1", "그 모델 FPR"], rows), ""]

    # ------------------------------------------------------------------ B: ablation
    arms = sorted({Path(p).name for p in glob.glob(str(RES / "ablation" / "*"))})
    for title, f in (("macro-F1", macro), ("weighted-F1", weighted)):
        rows = []
        for a in arms:
            line = [a]
            for d in DATASETS:
                fin, ab = by_seed(final[d]), by_seed(load(f"ablation/{a}/{d}__hyde__s*.json"))
                common = sorted(set(fin) & set(ab))
                if not common:
                    line.append("."); continue
                deltas = [(f(ab[s]) - f(fin[s])) * 100 for s in common]
                mu = st.mean(deltas)
                sd = st.stdev(deltas) if len(deltas) > 1 else 0.0
                line.append(f"{mu:+.2f}" + (f"±{sd:.2f}" if len(deltas) > 1 else "") + ("" if len(deltas) == SEEDS else f"({len(deltas)})"))
            rows.append(line)
        doc += [f"## B. Ablation — 최종 대비 {title} 변화 (같은 시드끼리 뺀 값)", "",
                table(["팔"] + [d for d in DATASETS], rows), ""]
    doc += ["- 팔 이름의 `_on`/`_off`: 튜닝 결과에서 그 기법을 켜거나 끈 것. 켤 때 값은 튜닝 기록의 val 최고값.",
            "- `.` = 그 데이터셋에는 해당 팔이 없음(예: 튜닝에서 이미 켜진 기법은 `_on` 팔이 없다).", ""]

    # ------------------------------------------------------------------ C: unknown
    variants = [("2step", "HyDE 2-step"), ("1step", "HyDE 1-step"), ("ml", "RF")]
    rate = "0.05"
    rows = []
    for d in DATASETS:
        line = [d]
        for v, _ in variants:
            rs = [r for r in load(f"unknown/{v}/*/{d}__*.json") if r["model"] in ("hyde", "random_forest")]
            if not rs:
                line += ["—"] * 4; continue
            u = [r["unknown_by_rate"][rate] for r in rs]
            line += [f"{st.mean(x['unknown_recall'] for x in u) * 100:.1f}",
                     f"{st.mean(x['unknown_precision'] for x in u) * 100:.1f}",
                     f"{st.mean((x['auroc'] or 0) for x in u) * 100:.1f}",
                     f"{st.mean(x['known_macro_f1'] for x in u) * 100:.1f}"]
        rows.append(line)
    hdr = ["데이터셋"] + [f"{n} {k}" for _, n in variants for k in ("recall", "prec", "AUROC", "known-F1")]
    doc += ["## C. Unknown 탐지 (reject rate 5%, 시나리오 × 시드 평균)", "",
            "시나리오: 데이터셋마다 가장 큰/중간/작은 공격 클래스를 학습에서 뺌(+ nsl-kdd는 test에만 있는 신종 공격). "
            "recall/prec = unknown 탐지율/정밀도, known-F1 = 알려진 클래스만의 macro-F1.", "", table(hdr, rows), ""]
    rows = []
    for sc in ("large", "median", "small", "natural"):
        for v, n in variants:
            rs = [r for r in load(f"unknown/{v}/{sc}/*.json") if r["model"] in ("hyde", "random_forest")]
            if rs:
                u = [r["unknown_by_rate"][rate] for r in rs]
                rows.append([sc, n, len({r["dataset"] for r in rs}), f"{st.mean(x['unknown_recall'] for x in u) * 100:.1f}",
                             f"{st.mean((x['auroc'] or 0) for x in u) * 100:.1f}"])
    doc += ["### C-2. 시나리오별 (데이터셋 평균)", "", table(["시나리오", "방법", "데이터셋 수", "unknown recall", "AUROC"], rows), ""]
    rows = []
    for v, n in variants:
        rs = [r for r in load(f"unknown/{v}/*/*.json") if r["model"] in ("hyde", "random_forest")]
        if rs:
            rows.append([n] + [f"{st.mean(r['unknown_by_rate'][k]['unknown_recall'] for r in rs) * 100:.1f} / "
                               f"{st.mean(r['unknown_by_rate'][k]['known_macro_f1'] for r in rs) * 100:.1f}"
                               for k in ("0.01", "0.05", "0.1")])
    doc += ["### C-3. reject rate별 (전체 평균: unknown recall / known macro-F1)", "",
            table(["방법", "1%", "5%", "10%"], rows), ""]

    # ------------------------------------------------------------------ D: dimension / int8
    rows = []
    for d in DATASETS:
        line = [d]
        for D in (1000, 2000, 5000):
            line.append(fmt(*m(load(f"e5/D{D}/{d}__hyde__s*.json"), macro)))
        line.append(fmt(*m(final[d], macro)))
        line.append(fmt(*m(final[d], lambda r: r.get("test_int8_proto", {}).get("macro_f1"))))
        rows.append(line)
    doc += ["## D. 차원 D와 int8 프로토타입 — macro-F1", "",
            table(["데이터셋", "D=1000", "D=2000", "D=5000", "D=10000 (최종)", "D=10000 int8"], rows),
            "", "모델 파일 크기와 추론 속도는 E6에서 측정한다.", ""]

    doc += ["## 원자료", "",
            "- 실행별 JSON: `results/<tag>/<dataset>__<model>__s<seed>.json` (per-class 지표는 `test.per_class`)",
            "- 튜닝 기록: `results/tune/<dataset>/log.json`, unknown 시나리오: `results/unknown/scenarios.csv`",
            "- 그 밖의 표: `python report.py ...` (README)", ""]
    out = ROOT / "docs" / "RESULTS.md"
    out.write_text("\n".join(doc))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
