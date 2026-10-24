"""Does homogenisation cost anything once teams turn over? P4a oracles across problem shift, rho and
turnover. For each assistant: culture diversity, and fitness on the next problem with the assistant
removed (mean of 30 rounds, and round 30), as % vs no AI on the same seeds, epochs 6-20.

  python scripts/analysis/p4a_cost.py results/p4a_cost
"""
import glob
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save, write_rows  # noqa: E402
from src.llm.analyze_p3 import ARM_COLORS  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "results/p4a_cost"
ROUNDS, BURN = 60, 5
REACH = [("sbp", 1), ("rad2", 2), ("rad3", 3), ("rad5", 5), ("rad10", 10)]
EXTRA = ["pcc", "rhelp", "maxh"]


def per_epoch(x, lo=None, hi=None):
    s, T = x.shape
    return x.reshape(s, T // ROUNDS, ROUNDS)[:, :, lo:hi].mean(-1)[:, BURN:].mean(1)


def pct(x, y):
    d = 100 * (x - y) / y.mean()
    return float(d.mean()), float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


rows = []
for sd in sorted(glob.glob(os.path.join(D, "shift*"))):
    shift = float(os.path.basename(sd)[5:])
    files = glob.glob(os.path.join(sd, "rho*_mu*_*.npz"))
    keys = {re.match(r"rho([\d.]+)_mu([\d.]+)_(\w+)\.npz", os.path.basename(f)).groups() for f in files}
    for rho, mu in sorted({(k[0], k[1]) for k in keys}):
        z = {a: np.load(os.path.join(sd, f"rho{rho}_mu{mu}_{a}.npz")) for (r, m, a) in keys if (r, m) == (rho, mu)}
        if "none" not in z:
            continue
        b = z["none"]
        for a, r in z.items():
            row = dict(shift=shift, rho=float(rho), mu=float(mu), assistant=a, div=float(per_epoch(r["div"], -20).mean()),
                       div_none=float(per_epoch(b["div"], -20).mean()))
            row["fit"], row["fit_ci"] = pct(per_epoch(r["fit"]), per_epoch(b["fit"]))
            row["next_off"], row["next_off_ci"] = pct(r["next_off"][:, BURN:].mean(1), b["next_off"][:, BURN:].mean(1))
            row["next_end"], row["next_end_ci"] = pct(r["next_end"][:, BURN:].mean(1), b["next_end"][:, BURN:].mean(1))
            rows.append(row)
write_rows(rows, "p4a_cost")
order = [a for a, _ in REACH] + EXTRA
for (shift, rho, mu) in sorted({(r["shift"], r["rho"], r["mu"]) for r in rows}):
    print(f"\nshift {shift:g}  rho {rho:g}  turnover {mu:g}   (no AI diversity "
          f"{next(r['div_none'] for r in rows if (r['shift'], r['rho'], r['mu']) == (shift, rho, mu)):.2f})")
    for a in order:
        r = next((r for r in rows if (r["shift"], r["rho"], r["mu"], r["assistant"]) == (shift, rho, mu, a)), None)
        if r:
            print(f"  {a:6s} div {r['div']:.2f}  assisted {r['fit']:+6.1f}%  next problem AI off: mean {r['next_off']:+6.1f}%"
                  f"  round 30 {r['next_end']:+6.1f}% ±{r['next_end_ci']:.1f}")

settings = sorted({(r["rho"], r["mu"]) for r in rows}, key=lambda x: (-x[0], x[1]))
shifts = sorted({r["shift"] for r in rows})
fig, axes = plt.subplots(2, len(settings), figsize=(7.2, 3.8), constrained_layout=True, squeeze=False)
shades = {s: c for s, c in zip(shifts, ["#8d82d9", "#4a3aa7", "#1f1660"])}
for j, (rho, mu) in enumerate(settings):
    for s in shifts:
        get = lambda a, key: next((r[key] for r in rows if (r["shift"], r["rho"], r["mu"], r["assistant"]) == (s, rho, mu, a)),
                                  np.nan)
        xs = [k for _, k in REACH]
        axes[0, j].plot(xs, [get(a, "div") for a, _ in REACH], "-o", ms=2.5, lw=0.9, color=shades[s], label=f"shift {s:g}")
        axes[1, j].plot(xs, [get(a, "next_end") for a, _ in REACH], "-o", ms=2.5, lw=0.9, color=shades[s])
        axes[1, j].plot([12], [get("pcc", "next_end")], "s", ms=3, color=shades[s])
        axes[0, j].plot([12], [get("pcc", "div")], "s", ms=3, color=shades[s])
    axes[0, j].set_title(f"ρ = {rho:g}, turnover {mu:g}", loc="left", color=INK)
    axes[1, j].axhline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    for i in range(2):
        axes[i, j].set_xticks([1, 2, 3, 5, 10, 12], ["1", "2", "3", "5", "10", "PCC"])
    axes[1, j].set_xlabel("Reach (bits)")
axes[0, 0].set_ylabel("Culture diversity")
axes[1, 0].set_ylabel("Next problem, AI removed,\nround 30 (% vs no AI)")
axes[0, 0].legend(frameon=False, fontsize=5.5)
mark_panels(axes.ravel())
save(fig, "p4a_cost")
