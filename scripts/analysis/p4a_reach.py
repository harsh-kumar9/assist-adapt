"""P4a reach sweep: how far may the assistant move a team's approach? Culture diversity, fitness while
assisted, and the next problem with the assistant removed, against how many bits of the module the
assistant may change (rad1 = SBP ... rad10 = best module anywhere), with the ideal learners of the P4b
rewards and PCC (best approach already in the culture) for reference. Paired by seed with no AI.

  python scripts/analysis/p4a_reach.py results/p4a_reach
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save, write_rows  # noqa: E402
from src.llm.analyze_p3 import ARM_COLORS  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "results/p4a_reach"
ROUNDS, BURN = 60, 5
REACH = [("sbp", 1), ("rad2", 2), ("rad3", 3), ("rad5", 5), ("rad10", 10)]
OTHER = [("pcc", "PCC: best approach already in the culture", ARM_COLORS["pcc"], "s"),
         ("rhelp", "Any helpful module (ideal thumbs)", "#4a3aa7", "o"),
         ("maxp", "Farthest helpful from the asker (ideal P-creativity)", "#eda100", "^"),
         ("maxh", "Farthest helpful from the culture (ideal H-creativity)", "#1baf7a", "D")]

z = {a: np.load(os.path.join(D, f"rho1_mu0.01_{a}.npz")) for a in ["none"] + [a for a, _ in REACH] + [a for a, *_ in OTHER]}


def per_epoch(x, lo=None, hi=None):
    s, T = x.shape
    return x.reshape(s, T // ROUNDS, ROUNDS)[:, :, lo:hi].mean(-1)[:, BURN:].mean(1)


def summary(a):
    r, b = z[a], z["none"]
    out = dict(assistant=a, div=float(per_epoch(r["div"], -20).mean()))
    for key, x, y in (("fit", per_epoch(r["fit"]), per_epoch(b["fit"])),
                      ("next_off", r["next_off"][:, BURN:].mean(1), b["next_off"][:, BURN:].mean(1))):
        d = 100 * (x - y) / y.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.update({key: float(d.mean()), key + "_ci": float(stats.t.ppf(0.975, len(d) - 1) * se)})
    return out


rows = [summary(a) for a in z]
for r in rows:
    print(f"{r['assistant']:6s} div {r['div']:.2f}  fit {r['fit']:+6.1f}% ±{r['fit_ci']:.1f}  "
          f"next problem, AI removed {r['next_off']:+6.1f}% ±{r['next_off_ci']:.1f}")
write_rows(rows, "p4a_reach")
by = {r["assistant"]: r for r in rows}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
for ax, key, lab in ((axes[0], "div", "Culture diversity\n(mean pairwise Hamming)"),
                     (axes[1], "fit", "Fitness while assisted\n(% vs no AI)"),
                     (axes[2], "next_off", "Next problem, AI removed\n(% vs no AI)")):
    xs = [k for _, k in REACH]
    ys = [by[a][key] for a, _ in REACH]
    ax.plot(xs, ys, "-o", ms=3, lw=1, color=INK, label="Best module within reach")
    if key != "div":
        ax.errorbar(xs, ys, yerr=[by[a][key + "_ci"] for a, _ in REACH], fmt="none", color=INK, lw=0.6)
    for a, name, c, m in OTHER:
        ax.plot([10.6 + 0.25 * [o[0] for o in OTHER].index(a)], [by[a][key]], m, ms=3.5, color=c, label=name)
    ax.axhline(by["none"]["div"] if key == "div" else 0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xlabel("Reach (most bits the assistant may change)")
    ax.set_ylabel(lab)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=5.5)
mark_panels(axes)
save(fig, "p4a_reach")
