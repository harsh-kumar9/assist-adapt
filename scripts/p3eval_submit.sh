#!/bin/bash
# Submit an evaluation job against a frozen copy of src/.
#   bash scripts/p3eval_submit.sh <tag> <args for src.llm.evaluate...>
set -euo pipefail
REPO=/ada1/u/harsh/assist-adapt
TAG=$1
shift
CODE=$REPO/results/p3eval/code_$TAG
mkdir -p "$CODE"
rsync -a --delete --exclude __pycache__ "$REPO/src/" "$CODE/src/"
cd "$REPO"
sbatch --job-name="p3eval_$TAG" --export=ALL,CODE="$CODE" scripts/p3eval.sbatch "$@"
