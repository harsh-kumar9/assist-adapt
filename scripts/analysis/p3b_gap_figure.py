"""Gap to the oracle nudge over continual training, (b) frozen evaluation. Paper-style figure."""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save  # noqa: E402
from src.llm.analyze_p3 import ARM_COLORS  # noqa: E402

D = "results/p3eval/b"
res = {}
for f in glob.glob(os.path.join(D, "rho*_s*_p*_*.json")):
    m = re.match(r"rho([\d.]+)_s(\d+)_p(\d+)_", os.path.basename(f))
    res[(float(m.group(1)), int(m.group(2)), int(m.group(3)))] = json.load(open(f))
ARMS = [("base", "LLM, untrained (with context)", "#b8b6ad"), ("thumbs", "LLM, thumbs up/down", ARM_COLORS["thumbs"]),
        ("pairwisefix", "LLM, pick one of two", ARM_COLORS["pairwisefix"])]


def gap(rho, k, arm, key):
    seeds = sorted(s for (r, s, kk) in res if r == rho and kk == k and arm in res[(r, s, kk)])
    base = np.mean([np.mean(res[(rho, s, k)]["none"][key]) for s in seeds])
    d = np.array([100 * (np.mean(res[(rho, s, k)][arm][key]) - np.mean(res[(rho, s, k)]["sbp"][key])) / base for s in seeds])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return d.mean(), t * se, len(d), d


# direct thumbs - pick one of two, paired by seed
for rho in (1.0, 0.8):
    for key in ("during", "next_off"):
        per_ck = []
        for k in range(4):
            _, _, _, dt = gap(rho, k, "thumbs", key)
            _, _, _, dp = gap(rho, k, "pairwisefix", key)
            per_ck.append(dt - dp)
        pooled = np.mean(per_ck, 0)
        print(f"rho {rho:g} {key}: thumbs - pick one of two, per checkpoint " +
              " ".join(f"{np.mean(x):+.1f}(p={stats.ttest_1samp(x, 0).pvalue:.2f})" for x in per_ck) +
              f" | pooled {pooled.mean():+.1f} p={stats.ttest_1samp(pooled, 0).pvalue:.3f} n={len(pooled)}")

fig, axes = plt.subplots(2, 2, figsize=(7, 4.2), constrained_layout=True, sharex=True)
for col, rho in enumerate((1.0, 0.8)):
    for row, (key, lab) in enumerate((("during", "While assisted, 30 rounds"),
                                      ("next_off", "Next problem, AI removed"))):
        ax = axes[row, col]
        for j, (arm, name, c) in enumerate(ARMS):
            ks = np.arange(1, 5) + (j - 1) * 0.08
            vals = [gap(rho, k, arm, key) for k in range(4)]
            ax.errorbar(ks, [v[0] for v in vals], yerr=[v[1] for v in vals], fmt="-o", ms=3, lw=1, color=c,
                        elinewidth=0.6, capsize=0, label=f"{name} (n = {vals[0][2]})" if row == 0 else None)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.set_title(f"ρ = {rho:g}: {lab}", loc="left", color=INK)
        ax.set_xticks(range(1, 5))
        if row == 1:
            ax.set_xlabel("Checkpoint (after problem)")
        if col == 0:
            ax.set_ylabel("Difference from oracle nudge\n(points, % of no-AI fitness)")
    axes[0, col].legend(frameon=False, fontsize=5.5, loc="lower left" if col == 0 else "upper right")
mark_panels(axes.ravel())
save(fig, "p3b_gap_over_training")
