"""P4a figures and summary: culture over time with turnover, oracle assistants.

  python -m src.analyze_culture results/p4a
"""
import glob
import os
import re
import sys

import numpy as np
from scipy import stats

from src.analyze import GRID, INK, mark_panels, plt, save, write_rows
from src.llm.analyze_p3 import ARM_COLORS, ARM_NAMES


def load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "rho*_mu*_*.npz")):
        m = re.match(r"rho([\d.]+)_mu([\d.]+)_(\w+)\.npz", os.path.basename(f))
        out[(float(m.group(1)), float(m.group(2)), m.group(3))] = dict(np.load(f))
    return out


def per_epoch(x, rounds, lo=None, hi=None):
    """(seeds, epochs*rounds) -> (seeds, epochs), mean over rounds [lo, hi) of each epoch."""
    s, T = x.shape
    e = x.reshape(s, T // rounds, rounds)[:, :, lo:hi]
    return e.mean(-1)


def ci(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se


def main(d="results/p4a", rounds=60, burn=5):
    res = load(d)
    rhos = sorted({k[0] for k in res}, reverse=True)
    mus = sorted({k[1] for k in res})
    arms = [a for a in ("none", "sbp", "pcc", "gated") if any(k[2] == a for k in res)]
    rows = []
    for rho in rhos:
        for mu in mus:
            if (rho, mu, "none") not in res:
                continue
            base = res[(rho, mu, "none")]
            b_div = per_epoch(base["div"], rounds, -20)[:, burn:].mean(1)
            b_fit = per_epoch(base["fit"], rounds)[:, burn:].mean(1)
            b_adapt = per_epoch(base["fit"], rounds, 0, 10)[:, burn:].mean(1)
            for a in arms:
                r = res[(rho, mu, a)]
                div_e = per_epoch(r["div"], rounds, -20)
                div = div_e[:, burn:].mean(1)
                fit = per_epoch(r["fit"], rounds)[:, burn:].mean(1)
                adapt = per_epoch(r["fit"], rounds, 0, 10)[:, burn:].mean(1)
                # ratchet: slope of end-of-epoch diversity over epochs, minus the no-AI slope, per seed
                ep = np.arange(div_e.shape[1])[burn:]
                b_div_e = per_epoch(base["div"], rounds, -20)
                slope = np.array([np.polyfit(ep, div_e[i, burn:] - b_div_e[i, burn:], 1)[0] for i in range(len(div_e))])
                row = dict(rho=rho, mu=mu, assistant=a, div=float(div.mean()), div_none=float(b_div.mean()),
                           fit_pct=ci(100 * (fit - b_fit) / b_fit.mean())[0],
                           adapt_pct=ci(100 * (adapt - b_adapt) / b_adapt.mean())[0],
                           adapt_lo=ci(100 * (adapt - b_adapt) / b_adapt.mean())[1],
                           adapt_hi=ci(100 * (adapt - b_adapt) / b_adapt.mean())[2],
                           slope=float(slope.mean()), slope_p=float(stats.ttest_1samp(slope, 0).pvalue) if a != "none" else 1.0)
                rows.append(row)
                print(f"rho {rho:g} mu {mu:g} {a:6s} div {row['div']:.2f} (none {row['div_none']:.2f}) "
                      f"fit {row['fit_pct']:+6.1f}% adapt(first 10 rounds of an epoch) {row['adapt_pct']:+6.1f}% "
                      f"[{row['adapt_lo']:+.1f},{row['adapt_hi']:+.1f}] ratchet slope {row['slope']:+.3f}/epoch p={row['slope_p']:.2f}")
    write_rows(rows, "p4a_culture")

    rho = rhos[0]
    mu_main = 0.01 if 0.01 in mus else mus[len(mus) // 2]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    ax = axes[0]
    for a in arms:
        x = res[(rho, mu_main, a)]["div"].mean(0)
        ax.plot(np.arange(len(x)) / rounds, x, lw=0.8, color=ARM_COLORS[a], label=ARM_NAMES[a])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean pairwise Hamming distance")
    ax.set_title(f"ρ = {rho:g}, turnover {mu_main:g}", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    for ax, key, lab in ((axes[1], "div", "Diversity at steady state"),
                         (axes[2], "fit_pct", "Fitness while assisted\n(% vs no AI)"),
                         (axes[3], "adapt_pct", "Fitness in first 10 rounds\nafter a change (% vs no AI)")):
        for a in arms:
            ys = [next(r[key] for r in rows if r["rho"] == rho and r["mu"] == m and r["assistant"] == a) for m in mus]
            if key != "div" and a == "none":
                continue
            ax.plot(mus, ys, "-o", ms=3, lw=1, color=ARM_COLORS[a])
        if key != "div":
            ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("Turnover per round")
        ax.set_ylabel(lab)
    mark_panels(axes)
    save(fig, "p4a_culture")


if __name__ == "__main__":
    main(*sys.argv[1:])
