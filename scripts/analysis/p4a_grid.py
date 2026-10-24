"""P4a design-space grid: oracle assistants by answer source x preference x reach (x islands).

For each design: culture diversity, fitness while assisted, and the next problem with the assistant
removed (mean over 30 rounds, and the last round), all paired by seed with no AI, epochs 6-20.

  python scripts/analysis/p4a_grid.py results/p4a_grid/<setting> [rho] [mu] [tag]
"""
import glob
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save, write_rows  # noqa: E402

D = sys.argv[1]
RHO = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
MU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
TAG = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(D.rstrip("/"))
ROUNDS, BURN = 60, 5
PREF = {"gain": ("largest gain", "#4a3aa7"), "rand": ("any helpful", "#1baf7a"), "rare": ("rarest", "#eda100")}
SRC = {"any": ("-", "o", "any module"), "cul": (":", "s", "only modules in the culture")}

z = {}
for f in glob.glob(os.path.join(D, f"rho{RHO:g}_mu{MU:g}_*.npz")):
    z[re.match(rf"rho{RHO:g}_mu{MU:g}_(\w+)\.npz", os.path.basename(f)).group(1)] = np.load(f)


def per_epoch(x, lo=None, hi=None):
    s, T = x.shape
    return x.reshape(s, T // ROUNDS, ROUNDS)[:, :, lo:hi].mean(-1)[:, BURN:].mean(1)


def summary(a):
    r, b = z[a], z["none"]
    out = dict(design=a, div=float(per_epoch(r["div"], -20).mean()))
    pairs = [("fit", per_epoch(r["fit"]), per_epoch(b["fit"])),
             ("next_off", r["next_off"][:, BURN:].mean(1), b["next_off"][:, BURN:].mean(1))]
    if "next_end" in r:
        pairs.append(("next_end", r["next_end"][:, BURN:].mean(1), b["next_end"][:, BURN:].mean(1)))
    for key, x, y in pairs:
        d = 100 * (x - y) / y.mean()
        se = d.std(ddof=1) / np.sqrt(len(d))
        out.update({key: float(d.mean()), key + "_ci": float(stats.t.ppf(0.975, len(d) - 1) * se)})
    return out


rows = [summary(a) for a in sorted(z)]
by = {r["design"]: r for r in rows}
for r in rows:
    print(f"{r['design']:20s} div {r['div']:.2f}  fit {r['fit']:+6.1f}%  next(AI off) {r['next_off']:+6.1f}%"
          + (f"  next, round 30 {r['next_end']:+6.1f}% ±{r['next_end_ci']:.1f}" if "next_end" in r else ""))
write_rows(rows, f"p4a_grid_{TAG}")

reaches = sorted({int(a.split("_")[3]) for a in z if a.startswith("x_")})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), constrained_layout=True)
for ax, key, lab in ((axes[0], "div", "Culture diversity"), (axes[1], "fit", "Fitness while assisted\n(% vs no AI)"),
                     (axes[2], "next_end" if "next_end" in by["none"] else "next_off",
                      "Next problem, AI removed,\nround 30 (% vs no AI)")):
    for src, (ls, mk, _) in SRC.items():
        for pref, (_, c) in PREF.items():
            names = [f"x_{src}_{pref}_{k}_q1" for k in reaches]
            ks = [k for k, n in zip(reaches, names) if n in by]
            if ks:
                ax.plot(ks, [by[f"x_{src}_{pref}_{k}_q1"][key] for k in ks], ls, marker=mk, ms=2.5, lw=0.9, color=c)
    ax.axhline(by["none"][key] if key == "div" else 0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_xticks(reaches)
    ax.set_xlabel("Reach (bits)")
    ax.set_ylabel(lab)
ax = axes[3]
for a, r in by.items():
    if not a.startswith("x_"):
        continue
    _, src, pref, k, q = a.split("_")
    ax.plot(r["div"], r["fit"], SRC[src][1], ms=2 + 0.35 * int(k), mfc="none" if q != "q1" else PREF[pref][1],
            color=PREF[pref][1], lw=0)
ax.axvline(by["none"]["div"], color=GRID, lw=0.7, ls="--", zorder=0)
ax.set_xlabel("Culture diversity")
ax.set_ylabel("Fitness while assisted\n(% vs no AI)")
handles = [plt.Line2D([], [], color=c, lw=1, label=f"Prefers {n}") for n, c in PREF.values()]
handles += [plt.Line2D([], [], color=INK, ls=ls, marker=mk, ms=2.5, lw=0.9, label=f"Answers from {n}")
            for ls, mk, n in SRC.values()]
fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=5.5)
mark_panels(axes)
save(fig, f"p4a_grid_{TAG}")
