#!/bin/bash
# Session 1: finish lane A, then help lane B (same stages, back-to-front) and finally lane ML.
# Launch: nohup bash run_after_A.sh >> logs/after_A.log 2>&1 &     (do NOT edit while it runs)
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/work/yunuikang/miniconda3/envs/final-env/bin/python3}
B="cicids2018 lspr23 nf-ton-iot-v3"
ALL="mirai nsl-kdd unsw-nb15 ton-iot cicids2017 xiiotid wustl-iiot-2021 edge-iiot $B"
bash run_lane.sh A 6 >> logs/lane_A.log 2>&1
for st in E3 E4 E5; do
    echo "[helper] $st (lane B datasets, reverse) start $(date)"
    $PY jobs.py "$st" --datasets $B --threads 6 --reverse >> "logs/H_${st}.log" 2>&1
done
echo "[helper] E0 (reverse) start $(date)"
$PY jobs.py E0 --datasets $ALL --threads 6 --reverse >> logs/H_E0.log 2>&1
echo "[helper] finished $(date)"
