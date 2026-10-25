#!/bin/bash
# P4b: LLM assistant retrained once per epoch in a culture with turnover. One GPU (mira, vega or grace).
#   sbatch scripts/p3.sbatch <out dir> <args for src.llm.online...>
#SBATCH --job-name=p4b
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

cleanup() {
  rc=$?
  pkill -9 -P $$ 2>/dev/null || true
  if [ "$rc" -ne 0 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # only processes on this job's own GPU
    for p in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$p" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT TERM INT

export HF_HOME=/ada1/u/harsh/.cache/huggingface
export HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache
export XDG_CONFIG_HOME=/ada1/u/harsh/.config
export TMPDIR=/ada1/u/harsh/tmp
export TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor
export CUDA_CACHE_PATH=/ada1/u/harsh/.cache/nv/ComputeCache
export FLASHINFER_WORKSPACE_BASE=/ada1/u/harsh/.cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" /ada1/u/harsh/.config logs

pwd; hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }
done

set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u

cd "${CODE:-/ada1/u/harsh/assist-adapt}"  # frozen copy of src/ when submitted with p3_submit.sh
OUT=$1
shift
[ -f "$OUT/DONE" ] && { echo "already done: $OUT"; exit 0; }
python -u -m src.llm.culture_llm --out "$OUT" "$@"
date
