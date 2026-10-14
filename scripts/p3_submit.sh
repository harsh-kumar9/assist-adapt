#!/bin/bash
# Submit a P3 run against a frozen copy of src/, so later edits cannot change a queued job.
#   bash scripts/p3_submit.sh <run name> <args for src.llm.online...>
set -euo pipefail
REPO=/ada1/u/harsh/assist-adapt
NAME=$1
shift
OUT=$REPO/results/p3/$NAME
mkdir -p "$OUT/code"
rsync -a --delete --exclude __pycache__ "$REPO/src/" "$OUT/code/src/"
cd "$REPO"
sbatch --job-name="p3_$NAME" --export=ALL,CODE="$OUT/code" scripts/p3.sbatch "$OUT" "$@"
