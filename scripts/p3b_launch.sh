#!/bin/bash
# D-014: (b) runs, context 3, feedback = scalar / thumbs / pairwise, 6 seeds x rho {1, 0.8}.
set -euo pipefail
cd /ada1/u/harsh/assist-adapt
for s in 0 1 2 3 4 5; do
  for fb in now thumbs pairwise; do
    P3DIR=p3b bash scripts/p3_submit.sh ${fb}_s$s --assistant llm --reward $fb --context 3 --rho 1.0 --seed $s
    P3DIR=p3b_r08 bash scripts/p3_submit.sh ${fb}_s$s --assistant llm --reward $fb --context 3 --rho 0.8 --seed $s
  done
done
