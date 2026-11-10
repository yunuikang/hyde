#!/bin/bash
# Session 1 (re-opened 2026-11-10 with a full GPU): two parallel lanes, resumable.
#   lane L : lspr23 re-run stages (E3, E3b, E4, E5)            forward
#   lane H : helper for Tier B (E3, E4, E5) then ML (E0)        reverse
# Launch: nohup bash run_session1.sh >> logs/session1.log 2>&1 &     (do NOT edit while it runs)
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/work/yunuikang/miniconda3/envs/final-env/bin/python3}
B="cicids2018 lspr23 nf-ton-iot-v3"
ALL="mirai nsl-kdd unsw-nb15 ton-iot cicids2017 xiiotid wustl-iiot-2021 edge-iiot $B"
echo "[session1] start $(date)"
$PY check_lspr23.py
(
  for st in E3 E3b E4 E5; do
      $PY jobs.py "$st" --datasets lspr23 --threads 3 >> logs/L_${st}.log 2>&1
      $PY check_lspr23.py
      echo "[session1:L] $st done $(date)"
  done
) &
(
  for st in E3 E4 E5; do
      $PY jobs.py "$st" --datasets $B --threads 3 --reverse >> logs/H2_${st}.log 2>&1
      echo "[session1:H] $st done $(date)"
  done
  $PY jobs.py E0 --datasets $ALL --threads 3 --reverse >> logs/H2_E0.log 2>&1
  echo "[session1:H] E0 done $(date)"
) &
wait
echo "[session1] finished $(date)"
