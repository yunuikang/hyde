#!/usr/bin/env python3
"""Progress of every stage: finished / planned, running processes, last log lines.

    python status.py
"""

from __future__ import annotations

import contextlib
import io
import subprocess
from pathlib import Path

import jobs

ROOT = Path(__file__).resolve().parent


def outputs(cmd: list[str]) -> Path | None:
    """Expected result file of a run.py / tune.py / cache job."""
    a = cmd[2:] if cmd[1].endswith((".py",)) else cmd[1:]
    if cmd[1].endswith("tune.py"):
        return ROOT / "configs" / "tuned" / f"{a[a.index('--data') + 1]}.yaml"
    if cmd[1] == "-c":
        return ROOT / "cache" / f"{cmd[-1]}.npz"
    get = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    return ROOT / "results" / get("--tag", "dev") / f"{get('--data')}__{get('--model', 'hyde')}__s{get('--seed', '0')}.json"


def main():
    print(f"{'stage':6} {'done':>6} / {'planned':<8}")
    for name, fn in jobs.STAGES.items():
        err = None
        with contextlib.redirect_stdout(io.StringIO()) as note:
            try:
                js = fn(jobs.DATASETS)
            except Exception as e:                         # e.g. caches missing for E4 scenarios
                err = e
        if err is not None:
            print(f"{name:6} (not ready: {type(err).__name__}: {err})"[:160])
            continue
        if name == "cache":
            done = sum((ROOT / "cache" / f"{d}.npz").exists() for d in jobs.DATASETS)
            print(f"{name:6} {done:>6} / {len(jobs.DATASETS):<8}")
            continue
        if name != "E1":                                   # E2 may prepend tune.py (extend) jobs
            js = [c for c in js if not c[1].endswith("tune.py")]
        done = sum(outputs(c).exists() for c in js)
        extra = " ".join(dict.fromkeys(note.getvalue().split("\n"))).strip()
        print(f"{name:6} {done:>6} / {len(js):<8} {extra}")
    ps = subprocess.run("ps -eo pid,etime,args | grep -E 'run.py|tune.py|load_dataset|run_all.sh' | grep -v grep",
                        shell=True, capture_output=True, text=True).stdout.strip()
    print("\nrunning:\n" + (ps or "  (nothing)"))
    for log in sorted((ROOT / "logs").glob("*.log")):
        lines = [l for l in log.read_text(errors="ignore").splitlines() if l.strip()]
        if lines:
            print(f"\n[{log.name}] {lines[-1][:160]}")


if __name__ == "__main__":
    main()
