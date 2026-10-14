"""G-3 (a): frozen checkpoints on fresh communities.

  python -m src.llm.analyze_eval results/p3eval/a

For each training seed (one landscape) every assistant served the same 200 fresh communities.
We take the mean over communities per seed, express it as % vs no AI on the same communities,
and report the mean over seeds with a t(5) 95% CI (the training seed is the unit, since each
seed is one trained model on one landscape).
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

from src.analyze import GRID, INK, INK2, MUTED, mark_panels, plt, save, write_rows
from src.llm.analyze_p3 import ARM_COLORS, ARM_NAMES

ORDER = ["sbp", "pcc", "gated", "base", "now", "kl", "stepback"]
COLORS = dict(ARM_COLORS, base="#b8b6ad", pcc="#eda100")
NAMES = dict(ARM_NAMES, base="LLM, untrained", now="LLM, R_now (collapsed)", kl="LLM, R_now + KL",
             stepback="LLM, R_now + step back")


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "rho*_s*_p*_*.json")):
        m = re.match(r"rho([\d.]+)_s(\d+)_p(\d+)_", os.path.basename(f))
        out[(float(m.group(1)), int(m.group(2)))] = json.load(open(f))
    return out


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.mean(), np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se, stats.ttest_1samp(x, 0).pvalue


def main(d="results/p3eval/a"):
    res = load(d)
    rhos = sorted({k[0] for k in res})
    rows = []
    for rho in rhos:
        seeds = sorted(s for r, s in res if r == rho and "none" in res[(r, s)])
        for a in ORDER:
            seeds_a = [s for s in seeds if a in res[(rho, s)]]
            if not seeds_a:
                continue
            row = dict(rho=rho, assistant=a, n_seeds=len(seeds_a))
            for key in ("during", "same_on", "next_off"):
                pct = [100 * (np.mean(res[(rho, s)][a][key]) - np.mean(res[(rho, s)]["none"][key]))
                       / np.mean(res[(rho, s)]["none"][key]) for s in seeds_a]
                m, lo, hi, p = ci(pct)
                row.update({key: m, key + "_lo": lo, key + "_hi": hi, key + "_p": p})
            row["div_T"] = float(np.mean([np.mean(res[(rho, s)][a]["div_T"]) for s in seeds_a]))
            row["div_T_none"] = float(np.mean([np.mean(res[(rho, s)]["none"]["div_T"]) for s in seeds_a]))
            for key in ("top_share", "distinct_answers", "step_back"):
                v = [res[(rho, s)][a].get(key) for s in seeds_a if res[(rho, s)][a].get(key) is not None]
                row[key] = float(np.mean(v)) if v else None
            rows.append(row)
    tag = os.path.basename(os.path.normpath(d))
    write_rows(rows, f"p3eval_{tag}")
    for r in rows:
        print(f"rho {r['rho']:g} {r['assistant']:9s} n={r['n_seeds']} during {r['during']:+6.1f} [{r['during_lo']:+.1f},{r['during_hi']:+.1f}] "
              f"| same_on {r['same_on']:+6.1f} | next_off {r['next_off']:+6.1f} [{r['next_off_lo']:+.1f},{r['next_off_hi']:+.1f}] "
              f"| div_T {r['div_T']:.2f} (none {r['div_T_none']:.2f}) top_share {r['top_share']} distinct {r['distinct_answers']} back {r['step_back']}")

    # paired LLM comparisons: KL vs R_now, R_now vs base
    print()
    for rho in rhos:
        seeds = sorted(s for r, s in res if r == rho)
        for a, b in (("kl", "now"), ("now", "base"), ("kl", "base"), ("stepback", "now")):
            for key in ("during", "next_off"):
                dd = [100 * (np.mean(res[(rho, s)][a][key]) - np.mean(res[(rho, s)][b][key]))
                      / np.mean(res[(rho, s)]["none"][key]) for s in seeds if a in res[(rho, s)] and b in res[(rho, s)]]
                if len(dd) > 1:
                    m, lo, hi, p = ci(dd)
                    print(f"rho {rho:g} {a} - {b} {key}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}] p = {p:.3f}")

    fig, axes = plt.subplots(1, len(rhos), figsize=(7.2, 2.6), constrained_layout=True, squeeze=False)
    for ax, rho in zip(axes[0], rhos):
        for r in [r for r in rows if r["rho"] == rho]:
            a = r["assistant"]
            ax.errorbar(r["during"], r["next_off"], xerr=[[r["during"] - r["during_lo"]], [r["during_hi"] - r["during"]]],
                        yerr=[[r["next_off"] - r["next_off_lo"]], [r["next_off_hi"] - r["next_off"]]], fmt="o", ms=3.5,
                        color=COLORS[a], elinewidth=0.6, capsize=0, mec="#fcfcfb", mew=0.5,
                        label=NAMES[a] if rho == rhos[0] else None)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"Modularity ρ = {rho:g}", loc="left", color=INK)
        ax.set_xlabel("Fitness while assisted, 30 rounds (% vs no AI)")
    axes[0][0].set_ylabel("Next problem, AI removed\n(% vs no AI)")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside right center", frameon=False, fontsize=6)
    mark_panels(axes[0])
    save(fig, f"p3eval_{tag}")


if __name__ == "__main__":
    main(*sys.argv[1:])
