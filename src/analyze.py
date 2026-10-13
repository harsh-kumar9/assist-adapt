"""Figures and summary tables for P0 (replication) and P1 (help now, adapt later).

  python -m src.analyze p0 results/p0
  python -m src.analyze p1 results/p1

Every comparison is paired by seed against the no-AI community on the same landscape,
network and starting points. We report percent difference of means and a 95% CI from the
paired per-seed differences, plus a paired t-test p-value (the paper's convention).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("rb", [RED, NEUTRAL, BLUE])
NAMES = {"none": "No AI", "np": "NP", "tdr": "TDR", "sbp": "SBP",
         "rand3": "SBP-rand3", "div3": "SBP-div3", "randall": "SBP-rand-all", "divall": "SBP-div-all"}
PAPER = ("np", "tdr", "sbp")

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})


def load(d):
    shards = sorted(glob.glob(os.path.join(d, "shard*.npz")))
    parts = [np.load(s) for s in shards]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    cfg = json.load(open(os.path.join(d, "config.json")))
    cfg["cells"] = [tuple(c) for c in cfg["cells"]]
    return data, cfg


def paired(x, base):
    """x, base: per-seed values. Returns % diff, CI low, CI high, p."""
    d = x - base
    m0 = base.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_rel(x, base).pvalue if np.any(d != 0) else 1.0
    return 100 * d.mean() / m0, 100 * (d.mean() - 1.96 * se) / m0, 100 * (d.mean() + 1.96 * se) / m0, p


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


def mark_panels(axes):
    for i, ax in enumerate(axes):
        ax.text(-0.18, 1.06, chr(97 + i), transform=ax.transAxes, fontsize=8, fontweight="bold", color=INK)


# ---------------------------------------------------------------- P0

def p0(d):
    data, cfg = load(d)
    rhos = sorted(set(data["rho"].tolist()))
    uses = sorted({u for a, u in cfg["cells"] if a != "none"})
    fit, div = data["fit"], data["div"]
    early_n = max(1, cfg["rounds"] // 10)
    measures = {"final fitness": fit[:, :, -1],
                "early fitness (rounds 1-10)": fit[:, :, 1:early_n + 1].mean(-1),
                "diversity AUC": div.sum(-1)}
    rows = []
    grids = {}
    for mname, vals in measures.items():
        for a in PAPER:
            grid = np.full((len(rhos), len(uses)), np.nan)
            sig = np.zeros_like(grid, bool)
            for i, rho in enumerate(rhos):
                sel = data["rho"] == rho
                base = vals[sel, cfg["cells"].index(("none", 0.0))]
                for j, u in enumerate(uses):
                    pct, lo, hi, p = paired(vals[sel, cfg["cells"].index((a, u))], base)
                    grid[i, j], sig[i, j] = pct, p < 0.05
                    rows.append(dict(measure=mname, assistant=a, rho=rho, use=u, pct=pct, lo=lo, hi=hi, p=p))
            grids[mname, a] = grid, sig
    fig, axes = plt.subplots(len(measures), 3, figsize=(7, 2.0 * len(measures)), constrained_layout=True)
    for r, mname in enumerate(measures):
        lim = max(np.nanmax(np.abs(grids[mname, a][0])) for a in PAPER)
        for c, a in enumerate(PAPER):
            grid, sig = grids[mname, a]
            ax = axes[r, c]
            shown = np.where(sig, grid, np.nan)
            ax.imshow(np.full_like(grid, 0), cmap=LinearSegmentedColormap.from_list("g", [NEUTRAL, NEUTRAL]),
                      origin="lower", aspect="auto")
            ax.imshow(shown, cmap=DIVERGING, vmin=-lim, vmax=lim, origin="lower", aspect="auto")
            for i in range(len(rhos)):
                for j in range(len(uses)):
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                            color=INK if sig[i, j] else MUTED)
            ax.set_xticks(range(len(uses)), [f"{u:g}" for u in uses])
            ax.set_yticks(range(len(rhos)), [f"{x:g}" for x in rhos])
            ax.set_title(f"{NAMES[a]}: {mname}", loc="left", color=INK)
            if r == len(measures) - 1:
                ax.set_xlabel("AI use rate")
            if c == 0:
                ax.set_ylabel("Modularity ρ")
            for s in ax.spines.values():
                s.set_visible(False)
    mark_panels(axes.ravel())
    save(fig, "p0_replication")
    write_rows(rows, "p0_replication")


# ---------------------------------------------------------------- P1

def p1_values(data, cfg):
    """Per run: mean fitness while assisted (rounds 1..T) and the probe outcomes."""
    T = cfg["probe_times"] = [int(t) for t in cfg["probe_times"].split(",")]
    fit, probe = data["fit"], data["probe"]
    during = np.stack([fit[:, :, 1:t + 1].mean(-1) for t in T], -1)        # (run, cell, T)
    post = probe[..., 1:].mean(-1)                                           # (run, cell, T, land, on)
    return T, during, post


def p1(d, rho_main=0.8, use_main=0.5):
    data, cfg = load(d)
    T, during, post = p1_values(data, cfg)
    shifts = [float(s) for s in cfg["shifts"].split(",")]
    li_same, li_shift = shifts.index(0.0), shifts.index(0.2)
    rhos = sorted(set(data["rho"].tolist()))
    uses = sorted({u for a, u in cfg["cells"] if a != "none"})
    assistants = [a for a in cfg["assistants"].split(",")]
    none = cfg["cells"].index(("none", 0.0))

    rows = []
    for rho in rhos:
        sel = data["rho"] == rho
        for a in assistants:
            for u in uses:
                c = cfg["cells"].index((a, u))
                for ti, t in enumerate(T):
                    out = dict(assistant=a, rho=rho, use=u, T=t)
                    for key, x, b in [
                        ("during", during[sel, c, ti], during[sel, none, ti]),
                        ("same_on", post[sel, c, ti, li_same, 1], post[sel, none, ti, li_same, 0]),
                        ("same_off", post[sel, c, ti, li_same, 0], post[sel, none, ti, li_same, 0]),
                        ("shift_on", post[sel, c, ti, li_shift, 1], post[sel, none, ti, li_shift, 0]),
                        ("shift_off", post[sel, c, ti, li_shift, 0], post[sel, none, ti, li_shift, 0]),
                        ("div_at_T", data["div"][sel, c, t], data["div"][sel, none, t]),
                    ]:
                        pct, lo, hi, p = paired(x, b)
                        out.update({key: pct, key + "_lo": lo, key + "_hi": hi, key + "_p": p})
                    out["div_abs"] = float(data["div"][sel, c, t].mean() - data["div"][sel, none, t].mean())
                    out["shift_penalty"] = out["shift_off"] - out["same_off"]
                    rows.append(out)
    write_rows(rows, "p1_probes")

    def get(a, rho, u, t, key):
        r = next(r for r in rows if (r["assistant"], r["rho"], r["use"], r["T"]) == (a, rho, u, t))
        return r[key], r.get(key + "_lo", r[key]), r.get(key + "_hi", r[key])

    def point(ax, x, y, col, ms=3.5, label=None, marker="o"):
        ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                    fmt=marker, ms=ms, color=col, ecolor=col, elinewidth=0.6, capsize=0,
                    mec="#fcfcfb", mew=0.5, label=label)

    variants = [a for a in assistants if a not in PAPER]

    # Figure 1: during vs after (shift, AI off), one panel per exposure length T
    fig, axes = plt.subplots(1, len(T), figsize=(7, 2.1), constrained_layout=True)
    for ax, t in zip(axes, T):
        for a in assistants:
            x = get(a, rho_main, use_main, t, "during")
            y = get(a, rho_main, use_main, t, "shift_off")
            if a in PAPER:
                point(ax, x, y, BLUE)
                ax.annotate(NAMES[a], (x[0], y[0]), xytext=(3, 3), textcoords="offset points",
                            fontsize=6, color=INK2)
            else:
                point(ax, x, y, ORANGE, ms=2.5, label="SBP variants" if a == variants[0] else None)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"{t} assisted rounds", loc="left", color=INK)
        ax.set_xlabel("Fitness while assisted\n(% vs no AI)")
    axes[0].set_ylabel("Fitness after shift, AI removed\n(% vs no AI)")
    axes[-1].legend(frameon=False, loc="lower right", handletextpad=0.2)
    mark_panels(axes)
    save(fig, "p1_during_vs_after")

    # Figure 2: mechanism
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), constrained_layout=True)
    sel = data["rho"] == rho_main
    show = ["none", "np", "tdr", "sbp"]
    cols = {"none": MUTED, "np": BLUE, "tdr": AQUA, "sbp": ORANGE}
    ax = axes[0]
    for a in show:
        c = none if a == "none" else cfg["cells"].index((a, use_main))
        m = data["div"][sel, c].mean(0)
        ax.plot(np.arange(len(m)), m, lw=1.2, color=cols[a], label=NAMES[a])
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean pairwise Hamming distance")
    ax.set_title("Diversity", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax = axes[1]
    keys = [("same_on", "same\nAI on"), ("same_off", "same\nAI off"),
            ("shift_on", "shift\nAI on"), ("shift_off", "shift\nAI off")]
    t_main = 30 if 30 in T else T[len(T) // 2]
    for k, (key, lab) in enumerate(keys):
        for j, a in enumerate(show[1:]):
            y, lo, hi = get(a, rho_main, use_main, t_main, key)
            ax.errorbar(k + (j - 1) * 0.2, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=3, color=cols[a],
                        elinewidth=0.6, capsize=0)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(4), [l for _, l in keys])
    ax.set_ylabel(f"Fitness, next 30 rounds\n(% vs no AI)")
    ax.set_title(f"Probes after {t_main} rounds", loc="left", color=INK)
    ax = axes[2]
    for a in show[1:]:
        ys = [get(a, rho_main, use_main, t, "shift_off") for t in T]
        ax.errorbar(T, [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
                    fmt="-o", ms=3, lw=1, color=cols[a], elinewidth=0.6, capsize=0)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Assisted rounds before shift")
    ax.set_ylabel("Fitness after shift, AI removed\n(% vs no AI)")
    ax.set_title("Exposure length", loc="left", color=INK)
    ax = axes[3]
    fam = {a: (cols.get(a, RED)) for a in assistants}
    for r in rows:
        if r["T"] == 100:
            continue
        ax.plot(r["div_abs"], r["shift_penalty"], "o", ms=2, color=fam[r["assistant"]], alpha=0.7, mew=0)
    xs = np.array([r["div_abs"] for r in rows if r["T"] != 100])
    ys = np.array([r["shift_penalty"] for r in rows if r["T"] != 100])
    b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=INK2, lw=0.8)
    ax.text(0.03, 0.95, f"r = {np.corrcoef(xs, ys)[0, 1]:.2f}", transform=ax.transAxes, va="top",
            fontsize=6, color=INK2)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Diversity at shift\n(Hamming, vs no AI)")
    ax.set_ylabel("Shift penalty\n(shift minus same, AI off, pp)")
    ax.set_title("All cells, T < 100", loc="left", color=INK)
    mark_panels(axes)
    save(fig, "p1_mechanism")

    # Figure 3: the SBP family up close, every rho x use at T = 30
    fig, axes = plt.subplots(len(rhos), len(uses), figsize=(7, 1.9 * len(rhos)), constrained_layout=True)
    markers = dict(zip(["sbp"] + variants, ["o", "s", "D", "^", "v", "P", "X"]))
    for i, rho in enumerate(rhos):
        for j, u in enumerate(uses):
            ax = axes[i, j]
            for a in ["sbp"] + variants:
                point(ax, get(a, rho, u, t_main, "during"), get(a, rho, u, t_main, "shift_off"),
                      BLUE if a == "sbp" else ORANGE, ms=3, marker=markers[a],
                      label=NAMES[a] if (i, j) == (0, 0) else None)
            ax.set_title(f"ρ = {rho:g}, use = {u:g}", loc="left", color=INK)
            if i == len(rhos) - 1:
                ax.set_xlabel("Fitness while assisted (% vs no AI)")
            if j == 0:
                ax.set_ylabel("After shift, AI off\n(% vs no AI)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False)
    mark_panels(axes.ravel())
    save(fig, "p1_sbp_family")


def write_rows(rows, name):
    os.makedirs("results/summary", exist_ok=True)
    keys = list(rows[0].keys())
    with open(f"results/summary/{name}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"wrote results/summary/{name}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    {"p0": p0, "p1": p1}[sys.argv[1]](sys.argv[2])
