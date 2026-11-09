#!/usr/bin/env python3
"""Move any lspr23 result produced with the leaky Label_*/Segment_* features to results/_invalid/ (2026-11-09 fix)."""
import glob, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
dest = ROOT / "results" / "_invalid" / "lspr23_label_leak"
moved = 0
for p in glob.glob(str(ROOT / "results" / "**" / "lspr23__*.json"), recursive=True):
    if "/_invalid/" in p:
        continue
    r = json.loads(Path(p).read_text())
    if "Label_dst" not in r["data"]["dropped_columns"]:
        rel = Path(p).relative_to(ROOT / "results")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dest / rel)
        moved += 1
print(f"check_lspr23: moved {moved} stale result(s)")
