#!/bin/bash
# CPU sweeps run directly on ada, not on mira/vega (D-007).
#   ssh ada 'nohup bash /ada1/u/harsh/assist-adapt/scripts/ada_sweep.sh p1 16 > /ada1/u/harsh/assist-adapt/logs/p1.log 2>&1 &'
# Safe to re-run: a lock stops a second copy, and a finished shard is skipped.
set -euo pipefail
REPO=/ada1/u/harsh/assist-adapt
PY=/ada1/u/harsh/miniconda3/envs/sote/bin/python
NAME=$1
W=${2:-16}
cd "$REPO"
mkdir -p logs results
exec 9>"$REPO/logs/$NAME.lock"
flock -n 9 || { echo "$NAME already running"; exit 0; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hostname; date
case "$NAME" in
  p0)  # replication of Fazelpour et al. Fig 1 and Fig 3 (K=9, H=10, p_social=0.1)
    $PY -u -m src.run --name p0 --seeds 0:1000 --rhos 0.6,0.7,0.8,0.9,1.0 \
      --assistants np,tdr,sbp --uses 0.125,0.25,0.375,0.5,0.625,0.75,0.875 --workers "$W" ;;
  p1)  # help now, adapt later: withdrawal and shift probes
    $PY -u -m src.run --name p1 --seeds 0:1000 --rhos 0.6,0.8,1.0 \
      --assistants np,tdr,sbp,rand3,div3,randall,divall --uses 0.25,0.5,0.75 \
      --probe-times 10,30,60,100 --shifts 0,0.2 --post-rounds 30 --workers "$W" ;;
  p1b)  # copy-based rules where P1 found the flip (D-010)
    $PY -u -m src.run --name p1b --seeds 0:1000 --rhos 0.8,0.9,1.0 \
      --assistants np,tdr,tdr25,tdrinv,pcc,sbp --uses 0.25,0.5,0.75 \
      --probe-times 10,30,60,100 --shifts 0,0.2 --post-rounds 30 --workers "$W" ;;
  *) echo "unknown sweep $NAME"; exit 1 ;;
esac
date
