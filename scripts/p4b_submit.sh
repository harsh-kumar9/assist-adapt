#!/bin/bash
# Submit a P4b run against a frozen copy of src/, so later edits cannot change a queued job.
#   bash scripts/p3_submit.sh <run name> <args for src.llm.online...>
#   P3DIR=p3_r08 bash scripts/p3_submit.sh ...   (results/<P3DIR>/<run name>, default p3)
set -euo pipefail
REPO=/ada1/u/harsh/assist-adapt
NAME=$1
shift
OUT=$REPO/results/${P3DIR:-p4b}/$NAME
mkdir -p "$OUT/code"
rsync -a --delete --exclude __pycache__ "$REPO/src/" "$OUT/code/src/"
cd "$REPO"
sbatch --job-name="p4b_$NAME" --export=ALL,CODE="$OUT/code" scripts/p4b.sbatch "$OUT" "$@"
