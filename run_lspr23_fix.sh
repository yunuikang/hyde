#!/bin/bash
# lspr23 re-run after dropping Label_src/Label_dst/Segment_src/Segment_dst (2026-11-09).
# Launch: nohup bash run_lspr23_fix.sh >> logs/lspr23_fix.log 2>&1 &     (do NOT edit while it runs)
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/work/yunuikang/miniconda3/envs/final-env/bin/python3}
echo "[lspr23-fix] start $(date)"
$PY -c "from hyde.data import load_dataset; load_dataset('lspr23', rebuild=True)"
$PY check_lspr23.py
$PY tune.py --data lspr23 --threads 6
for st in E2 E3 E3b E4 E5; do
    $PY check_lspr23.py
    $PY jobs.py "$st" --datasets lspr23 --threads 6
    echo "[lspr23-fix] $st done $(date)"
done
$PY check_lspr23.py
echo "[lspr23-fix] finished $(date)"
