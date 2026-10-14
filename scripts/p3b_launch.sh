#!/bin/bash
# D-014: (b) runs, context 3. Feedback = scalar / thumbs / pairwise, plus KL (beta 0.05) for thumbs and
# pairwise (RLHF is KL-anchored in practice). 6 seeds x rho {1, 0.8}. Same seeds across arms.
set -euo pipefail
cd /ada1/u/harsh/assist-adapt
for s in 0 1 2 3 4 5; do
  for rho in 1.0 0.8; do
    D=p3b; [ "$rho" = 0.8 ] && D=p3b_r08
    for fb in now thumbs pairwise; do
      P3DIR=$D bash scripts/p3_submit.sh ${fb}_s$s --assistant llm --reward $fb --context 3 --rho $rho --seed $s
    done
    for fb in thumbs pairwise; do
      P3DIR=$D bash scripts/p3_submit.sh ${fb}kl_s$s --assistant llm --reward $fb --kl 0.05 --context 3 --rho $rho --seed $s
    done
  done
done
