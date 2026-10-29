"""P4a memory screen: can an assistant that knows only which of its answers helped beat random edits without
narrowing the culture? (Memory assistants in src/culture.py; rho 1, turnover 0.01, shift 0.5, 200 seeds.)

Per design, epochs 3-20, paired by seed with no AI: culture diversity (last 20 rounds of each epoch), fitness
while assisted, fitness on the next problem with the assistant removed (mean of 30 rounds), share of answers
that help (all, from memory, random fallback), share of answers taken from memory. Contrasts with random edits
(rnd3) and with each variant's base design. Blue: memory used on a share of requests, else a random edit.

  python scripts/analysis/p4a_mem.py [results/p4a_mem/shift0.5]
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save, write_rows  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "results/p4a_mem/shift0.5"
ROUNDS, BURN = 60, 2
MEMS = [("mem_now_recent", "This epoch, always", "#0d366b", None),
        ("mem_now_recent_p75", "This epoch, 75% of answers", "#1c5cab", None),
        ("mem_now_recent_p50", "This epoch, 50%", "#2a78d6", None),
        ("mem_now_recent_p25", "This epoch, 25%", "#5598e7", None),
        ("mem_now_recent_p10", "This epoch, 10%", "#86b6ef", None),
        ("mem_now_recent_p50_q3", "50%, 3 adoptions each", "#1baf7a", "mem_now_recent_p50"),
        ("mem_now_recent_p50_i4", "50%, 4 separate memories", "#1baf7a", "mem_now_recent_p50"),
        ("mem_now_recent_p25_q3", "25%, 3 adoptions each", "#1baf7a", "mem_now_recent_p25"),
        ("mem_now_rare", "Always, rarest in culture", "#eb6834", "mem_now_recent"),
        ("mem_now_recent_q3", "Always, 3 adoptions each", "#eb6834", "mem_now_recent"),
        ("mem_now_recent_i4", "Always, 4 separate memories", "#eb6834", "mem_now_recent"),
        ("mem_last_any", "Last epoch, always", "#6250d6", None),
        ("mem_last_any_p25", "Last epoch, 25%", "#6250d6", "mem_last_any")]
REFS = [("rnd3", "Random edit (1-3 flips)"), ("rad3", "Best edit (≤3)"), ("pcc", "Best answer in culture")]

z = {a: np.load(os.path.join(D, f"rho1_mu0.01_{a}.npz")) for a, *_ in MEMS + REFS + [("none",)]
     if os.path.exists(os.path.join(D, f"rho1_mu0.01_{a}.npz"))}
mems = [x for x in MEMS if x[0] in z]
refs = [x for x in REFS if x[0] in z]


def per_seed(a, k):
    x = z[a][k]
    if k in ("div", "fit"):
        x = x.reshape(len(x), -1, ROUNDS)
        x = x[:, :, -20:] if k == "div" else x
        return x[:, BURN:].mean((1, 2))
    return np.nanmean(x[:, BURN:], 1)


def diff(a, b, k):
    """Paired difference a - b per seed: Hamming units for diversity, % of no-AI fitness otherwise."""
    d = per_seed(a, k) - per_seed(b, k)
    return d if k == "div" else 100 * d / per_seed("none", k).mean()


def ci(d):
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


rows = []
for a, name, *_ in mems + refs:
    row = dict(design=a, name=name, n=len(z[a]["seeds"]))
    for k in ("div", "fit", "next_off"):
        row[k + "_vs_none"], row[k + "_ci"] = (float(v) for v in ci(diff(a, "none", k)))
    row["helps"] = float(per_seed(a, "helps").mean())
    for k in ("mem_use", "mem_help", "rnd_help"):
        row[k] = float(per_seed(a, k).mean()) if k in z[a] else float("nan")
    rows.append(row)
    print(f"{name:32s} div {row['div_vs_none']:+.2f}±{row['div_ci']:.2f}  fit {row['fit_vs_none']:+6.1f}±{row['fit_ci']:.1f}%"
          f"  next {row['next_off_vs_none']:+5.1f}±{row['next_off_ci']:.1f}%  helps {row['helps']:.3f}"
          f"  (memory {row['mem_help']:.3f}, random {row['rnd_help']:.3f})  from memory {row['mem_use']:.2f}")
write_rows(rows, "p4a_mem")

print("\npaired contrasts (diversity in Hamming units, fitness in % of no-AI fitness)")
for a, _, _, base in mems:
    for b in ("rnd3", base):
        if b is None or b not in z:
            continue
        out = []
        for k in ("div", "fit", "next_off"):
            d = diff(a, b, k)
            out.append(f"{k} {d.mean():+.2f} [{d.mean() - ci(d)[1]:+.2f}, {d.mean() + ci(d)[1]:+.2f}]")
        print(f"  {a} - {b}: " + "; ".join(out))

labels = [name for _, name, *_ in mems] + [name for _, name in refs]
y = np.r_[np.arange(len(mems)), len(mems) + 0.6 + np.arange(len(refs))]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.6 + 0.2 * len(labels)), sharey=True, constrained_layout=True)
for ax, k, lab in ((axes[0], "div", "Culture diversity\n(Hamming, vs no AI)"),
                   (axes[1], "fit", "Fitness while assisted\n(% vs no AI)"),
                   (axes[2], "next_off", "Next problem, AI removed\n(% vs no AI)")):
    for j, (a, name, c, _) in enumerate(mems):
        mu_, h = ci(diff(a, "none", k))
        ax.errorbar(mu_, y[j], xerr=h, fmt="o", ms=3.5, color=c, lw=1, capsize=0)
    for j, (a, name) in enumerate(refs, len(mems)):
        mu_, h = ci(diff(a, "none", k))
        ax.errorbar(mu_, y[j], xerr=h, fmt="s", ms=3, mfc="white", color=INK, lw=0.8, capsize=0)
    if "rnd3" in z:
        ax.axvline(ci(diff("rnd3", "none", k))[0], color=INK, lw=0.6, ls=":", zorder=0)
    ax.axvline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_xlabel(lab)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
mark_panels(axes)
save(fig, "p4a_mem")
