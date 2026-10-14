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
NAMES = {"none": "No AI", "np": "NP", "tdr": "TDR", "sbp": "SBP", "tdr25": "TDR-25%",
         "tdrinv": "TDR-inverse", "pcc": "PCC",
         "rand3": "SBP-rand3", "div3": "SBP-div3", "randall": "SBP-rand-all", "divall": "SBP-div-all"}
PAPER = ("np", "tdr", "sbp")
# one colour per assistant in every figure (categorical slots of the dataviz reference palette)
COLORS = {"np": BLUE, "tdr": AQUA, "sbp": ORANGE, "pcc": "#eda100", "tdr25": "#4a3aa7",
          "tdrinv": "#e87ba4", "rand3": MUTED, "div3": MUTED, "randall": MUTED, "divall": MUTED}

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


def p1(d, use_main=0.5, t_main=30):
    tag = os.path.basename(os.path.normpath(d))
    data, cfg = load(d)
    T, during, post = p1_values(data, cfg)
    shifts = [float(s) for s in cfg["shifts"].split(",")]
    li_same, li_shift = shifts.index(0.0), next(i for i, x in enumerate(shifts) if x > 0)
    rhos = sorted(set(data["rho"].tolist()))
    rho_main = max(rhos)
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
    write_rows(rows, f"{tag}_probes")
    p1_reversals(tag, data, cfg, T, during, post, li_same, li_shift, rhos, uses, assistants)

    def get(a, rho, u, t, key):
        r = next(r for r in rows if (r["assistant"], r["rho"], r["use"], r["T"]) == (a, rho, u, t))
        return r[key], r.get(key + "_lo", r[key]), r.get(key + "_hi", r[key])

    def point(ax, x, y, col, ms=3.5, label=None, marker="o"):
        ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                    fmt=marker, ms=ms, color=col, ecolor=col, elinewidth=0.6, capsize=0,
                    mec="#fcfcfb", mew=0.5, label=label)

    sbp_variants = [a for a in assistants if a.startswith(("rand", "div"))]
    labelled = [a for a in assistants if a not in sbp_variants]
    col = {a: COLORS[a] for a in labelled}
    direct = len(labelled) <= 4

    # Figure 1: while assisted vs after a shift with the AI off, one panel per modularity
    fig, axes = plt.subplots(1, len(rhos), figsize=(7, 2.3), constrained_layout=True)
    for ax, rho in zip(axes, rhos):
        for a in sbp_variants:
            point(ax, get(a, rho, use_main, t_main, "during"), get(a, rho, use_main, t_main, "shift_off"),
                  MUTED, ms=2.5, label="SBP variants" if a == sbp_variants[0] else None)
        for a in labelled:
            x, y = get(a, rho, use_main, t_main, "during"), get(a, rho, use_main, t_main, "shift_off")
            point(ax, x, y, col[a], label=NAMES[a])
            if direct:
                ax.annotate(NAMES[a], (x[0], y[0]), xytext=(4, 3), textcoords="offset points",
                            fontsize=6, color=INK2)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.axvline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f"Modularity ρ = {rho:g}", loc="left", color=INK)
        ax.set_xlabel(f"Fitness while assisted, rounds 1-{t_main}\n(% vs no AI)")
    axes[0].set_ylabel("Fitness after shift, AI removed\n(% vs no AI)")
    axes[0].legend(frameon=False, loc="upper left", handletextpad=0.2, fontsize=6)
    mark_panels(axes)
    save(fig, f"{tag}_during_vs_after")

    # Figure 2: mechanism at the highest modularity
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    sel = data["rho"] == rho_main
    show = ["none"] + labelled
    ax = axes[0]
    for a in show:
        c = none if a == "none" else cfg["cells"].index((a, use_main))
        m = data["div"][sel, c].mean(0)
        ax.plot(np.arange(len(m)), m, lw=1.1, color=MUTED if a == "none" else col[a], label=NAMES[a])
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean pairwise Hamming distance")
    ax.set_title(f"Diversity, ρ = {rho_main:g}", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right", handlelength=1.0)
    ax = axes[1]
    keys = [("same_on", "same\nAI on"), ("same_off", "same\nAI off"),
            ("shift_on", "shift\nAI on"), ("shift_off", "shift\nAI off")]
    k_off = (len(labelled) - 1) / 2
    for k, (key, lab) in enumerate(keys):
        for j, a in enumerate(labelled):
            y, lo, hi = get(a, rho_main, use_main, t_main, key)
            ax.errorbar(k + (j - k_off) * 0.13, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=2.5, color=col[a],
                        elinewidth=0.6, capsize=0)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xticks(range(4), [l for _, l in keys])
    ax.set_ylabel("Fitness, next 30 rounds\n(% vs no AI)")
    ax.set_title(f"Probes after {t_main} rounds", loc="left", color=INK)
    ax = axes[2]
    for a in labelled:
        ys = [get(a, rho_main, use_main, t, "shift_off") for t in T]
        ax.errorbar(T, [y[0] for y in ys], yerr=[[y[0] - y[1] for y in ys], [y[2] - y[0] for y in ys]],
                    fmt="-o", ms=2.5, lw=1, color=col[a], elinewidth=0.6, capsize=0)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Assisted rounds before shift")
    ax.set_ylabel("Fitness after shift, AI removed\n(% vs no AI)")
    ax.set_title("Exposure length", loc="left", color=INK)
    ax = axes[3]
    pts = [r for r in rows if r["T"] != 100]
    for r in pts:
        ax.plot(r["div_abs"], r["shift_penalty"], "o", ms=2, mew=0, alpha=0.7,
                color=col.get(r["assistant"], MUTED))
    xs = np.array([r["div_abs"] for r in pts])
    ys = np.array([r["shift_penalty"] for r in pts])
    b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 10)
    ax.plot(xx, np.polyval(b, xx), color=INK2, lw=0.8)
    ax.text(0.03, 0.95, f"r = {np.corrcoef(xs, ys)[0, 1]:.2f}", transform=ax.transAxes, va="top",
            fontsize=6, color=INK2)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=GRID, lw=0.6, zorder=0)
    ax.set_xlabel("Diversity at shift\n(Hamming, vs no AI)")
    ax.set_ylabel("Shift penalty\n(shift minus same, AI off, pp)")
    ax.set_title("All settings, T < 100", loc="left", color=INK)
    mark_panels(axes)
    save(fig, f"{tag}_mechanism")


# ---------------------------------------------------------------- P2

P2_LEARNED = {"now": BLUE, "long": AQUA, "adapt": ORANGE}


def p2(d="results/p2"):
    """Learned help-mode policies against fixed mixes of full answers and nudges (D-011)."""
    import collections
    rows = json.load(open(os.path.join(d, "eval.json")))
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["rho"], r["shift"], r["use"], r["seed"])][r["policy"]] = r
    policies = sorted({r["policy"] for r in rows} - {"none"})
    settings = sorted({(r["rho"], r["shift"], r["use"]) for r in rows})
    out = []
    for st in settings:
        keys = [k for k in by if k[:3] == st]
        base = {m: np.array([by[k]["none"][m] for k in keys]) for m in ("now", "shift_off", "same_on")}
        for pol in policies:
            row = dict(rho=st[0], shift=st[1], use=st[2], policy=pol)
            for m in ("now", "shift_off", "same_on"):
                pct, lo, hi, pv = paired(np.array([by[k][pol][m] for k in keys]), base[m])
                row.update({m: pct, m + "_lo": lo, m + "_hi": hi, m + "_p": pv})
            tot = sum(np.array([by[k][pol][a] for k in keys]).sum() for a in ("full", "nudge", "back"))
            for a in ("full", "nudge", "back"):
                row["share_" + a] = float(np.array([by[k][pol][a] for k in keys]).sum() / max(tot, 1))
            out.append(row)
    write_rows(out, "p2_eval")

    def get(pol, st, m):
        r = next(r for r in out if r["policy"] == pol and (r["rho"], r["shift"], r["use"]) == st)
        return r[m], r[m + "_lo"], r[m + "_hi"]

    learned = [p for p in policies if p.split("_")[0] in P2_LEARNED]
    mixes = ["always_nudge", "mix0.25", "mix0.5", "mix0.75", "always_full"]
    scheds = [p for p in policies if p.startswith("full_first")]
    main = [st for st in settings if st[1] == 0.2 and st[2] == 0.5]
    held = [st for st in settings if st not in main]
    for name, sts in (("p2_frontier", main), ("p2_heldout", held)):
        if not sts:
            continue
        nrow = (len(sts) + 2) // 3
        fig, axs = plt.subplots(nrow, 3, figsize=(7.2, 2.3 * nrow), constrained_layout=True, squeeze=False)
        axs = list(axs.ravel())
        for ax, st in zip(axs, sts):
            xs = [get(p, st, "now")[0] for p in mixes]
            ys = [get(p, st, "shift_off")[0] for p in mixes]
            ax.plot(xs, ys, "-", color=MUTED, lw=0.8, zorder=1)
            for p, x, y in zip(mixes, xs, ys):
                ax.plot(x, y, "o", ms=3, color=MUTED, zorder=2)
            ax.annotate("full", (xs[-1], ys[-1]), xytext=(3, -8), textcoords="offset points", fontsize=5.5, color=MUTED)
            ax.annotate("nudge", (xs[0], ys[0]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=MUTED)
            for p in scheds:
                ax.plot(get(p, st, "now")[0], get(p, st, "shift_off")[0], "s", ms=3, color=MUTED, mfc="none", mew=0.6)
            gated = [p for p in ["gated_nudge", "gated_mix0.25", "gated_mix0.5", "gated_mix0.75", "gated_full"]
                     if p in policies]
            if gated:
                gx = [get(p, st, "now")[0] for p in gated]
                gy = [get(p, st, "shift_off")[0] for p in gated]
                ax.plot(gx, gy, "-D", ms=2.5, lw=0.8, color=INK2, zorder=2, label="gated mixes")
                if "gated_best" in policies:
                    ax.plot(get("gated_best", st, "now")[0], get("gated_best", st, "shift_off")[0], "*", ms=6,
                            color=INK2, zorder=3, label="gated, best gain")
            for p in learned:
                x, y = get(p, st, "now"), get(p, st, "shift_off")
                c = P2_LEARNED[p.split("_")[0]]
                ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                            fmt="o", ms=3.5, color=c, elinewidth=0.6, capsize=0, mec="#fcfcfb", mew=0.5,
                            label="learned, R_" + p.split("_")[0] if p.endswith("s0") else None, zorder=4)
            ax.set_title(f"ρ = {st[0]:g}, shift {st[1]:g}, use {st[2]:g}", loc="left", color=INK)
            ax.set_xlabel("Fitness while assisted (% vs no AI)")
            ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        axs[0].set_ylabel("Fitness after shift, AI removed\n(% vs no AI)")
        axs[0].legend(frameon=False, loc="upper left", fontsize=6, handletextpad=0.2)
        mark_panels(axs)
        save(fig, name)

    # action mix over the assisted rounds, learned policies, highest rho, main setting
    st = main[-1]
    fig, axes = plt.subplots(1, len(learned), figsize=(7.2, 1.9), constrained_layout=True, squeeze=False)
    for ax, p in zip(axes[0], learned):
        acts = np.array([r["acts"] for r in rows if r["policy"] == p and (r["rho"], r["shift"], r["use"]) == st])
        share = acts.sum(0) / np.maximum(acts.sum(0).sum(1, keepdims=True), 1)
        ax.stackplot(np.arange(1, share.shape[0] + 1), share.T, colors=["#eda100", ORANGE, "#c3c2b7"],
                     labels=["full answer", "nudge", "step back"], edgecolor="#fcfcfb", linewidth=0.3)
        ax.set_title(p.replace("_s", ", ES seed "), loc="left", color=INK)
        ax.set_xlabel("Assisted round")
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("Share of queries")
    axes[0][-1].legend(frameon=False, loc="lower right", fontsize=5.5)
    mark_panels(axes[0])
    save(fig, "p2_actions")


def robust(tags=("p1b", "rb_edge10", "rb_shift50", "rb_k5"), rho=1.0, use=0.5, T=30):
    """One panel per robustness condition: while assisted vs after shift with AI off."""
    import csv
    labels = {"p1b": "Main", "rb_edge10": "Denser network (p_edge 0.1)", "rb_shift50": "Bigger shift (50%)",
              "rb_k5": "Smoother landscape (K = 5)"}
    fig, axes = plt.subplots(1, len(tags), figsize=(7.2, 2.2), constrained_layout=True)
    for ax, t in zip(axes, tags):
        rows = {r["assistant"]: r for r in csv.DictReader(open(f"results/summary/{t}_probes.csv"))
                if float(r["rho"]) == rho and float(r["use"]) == use and int(float(r["T"])) == T}
        for a, r in rows.items():
            x, y = float(r["during"]), float(r["shift_off"])
            ax.errorbar(x, y, xerr=[[x - float(r["during_lo"])], [float(r["during_hi"]) - x]],
                        yerr=[[y - float(r["shift_off_lo"])], [float(r["shift_off_hi"]) - y]], fmt="o", ms=3,
                        color=COLORS[a], elinewidth=0.6, capsize=0, mec="#fcfcfb", mew=0.5,
                        label=NAMES[a] if t == tags[0] else None)
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(labels.get(t, t), loc="left", color=INK)
        ax.set_xlabel("While assisted (% vs no AI)")
    axes[0].set_ylabel("After shift, AI removed\n(% vs no AI)")
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right", handletextpad=0.2)
    mark_panels(axes)
    save(fig, "robustness")


def p1_reversals(tag, data, cfg, T, during, post, li_same, li_shift, rhos, uses, assistants):
    """Ranking flips between assistants: A beats B while assisted (paired p < .05) but B beats A
    after a shift with the AI off (paired p < .05). Same rho, use and T for both."""
    rows = []
    for rho in rhos:
        sel = data["rho"] == rho
        for u in uses:
            for ti, t in enumerate(T[:-1] if T[-1] == 100 else T):
                for a in assistants:
                    for b in assistants:
                        if a == b:
                            continue
                        ca, cb = cfg["cells"].index((a, u)), cfg["cells"].index((b, u))
                        d_now = during[sel, ca, ti] - during[sel, cb, ti]
                        d_after = post[sel, cb, ti, li_shift, 0] - post[sel, ca, ti, li_shift, 0]
                        p_now = stats.ttest_1samp(d_now, 0).pvalue
                        p_after = stats.ttest_1samp(d_after, 0).pvalue
                        if d_now.mean() > 0 and p_now < .05 and d_after.mean() > 0 and p_after < .05:
                            rows.append(dict(better_now=a, better_after=b, rho=rho, use=u, T=t,
                                             now_gap=float(d_now.mean()), after_gap=float(d_after.mean()),
                                             p_now=float(p_now), p_after=float(p_after)))
    if rows:
        write_rows(rows, f"{tag}_reversals")
    else:
        print("no ranking flips")
    return rows


def p1_gate(tag="p1"):
    """Gate numbers for P1 -> P2, written to results/summary/p1_gate.md.

    Dissociation: fitness while assisted is significantly above no AI (p < .05) while fitness
    after a shift with the AI off is significantly below no AI. Counted over rho x use x T<100.
    """
    import csv
    rows = list(csv.DictReader(open(f"results/summary/{tag}_probes.csv")))
    for r in rows:
        for k, v in r.items():
            if k != "assistant":
                r[k] = float(v)
    out = ["| assistant | settings | helpful while assisted | worse after shift, AI off | both (dissociation) |",
           "|---|---|---|---|---|"]
    assistants = list(dict.fromkeys(r["assistant"] for r in rows))
    for a in assistants:
        rs = [r for r in rows if r["assistant"] == a and r["T"] < 100]
        up = [r for r in rs if r["during"] > 0 and r["during_p"] < .05]
        down = [r for r in rs if r["shift_off"] < 0 and r["shift_off_p"] < .05]
        both = [r for r in up if r in down]
        out.append(f"| {NAMES[a]} | {len(rs)} | {len(up)} | {len(down)} | {len(both)} |")
    flips_path = f"results/summary/{tag}_reversals.csv"
    if os.path.exists(flips_path):
        flips = list(csv.DictReader(open(flips_path)))
        n_tests = len(assistants) * (len(assistants) - 1) * len({(r["rho"], r["use"], r["T"]) for r in rows
                                                                 if r["T"] < 100})
        strict = [f for f in flips if max(float(f["p_now"]), float(f["p_after"])) < .05 / n_tests]
        out += ["", f"Ranking flips (A beats B while assisted, B beats A after shift with AI off): "
                    f"{len(flips)} of {n_tests} ordered pairs x settings at p < .05 on both; "
                    f"{len(strict)} at Bonferroni p < {.05 / n_tests:.1e} on both.", ""]
        out += ["| better while assisted | better after shift | rho | use | T | strict |", "|---|---|---|---|---|---|"]
        for f in flips:
            ok = max(float(f["p_now"]), float(f["p_after"])) < .05 / n_tests
            out.append(f"| {NAMES[f['better_now']]} | {NAMES[f['better_after']]} | {f['rho']} | {f['use']} | "
                       f"{f['T']} | {'yes' if ok else ''} |")
    out += ["", "T = 30, after shift with AI off vs while assisted (% vs no AI):", "",
            "| assistant | rho | use | while assisted | after shift, AI off | same problem, AI off | diversity at T (Hamming diff) |",
            "|---|---|---|---|---|---|---|"]
    for a in assistants:
        for r in rows:
            if r["assistant"] == a and r["T"] == 30:
                f = lambda k: f"{r[k]:+.1f} [{r[k + '_lo']:+.1f}, {r[k + '_hi']:+.1f}]"
                out.append(f"| {NAMES[a]} | {r['rho']:g} | {r['use']:g} | {f('during')} | {f('shift_off')} | "
                           f"{f('same_off')} | {r['div_abs']:+.2f} |")
    open(f"results/summary/{tag}_gate.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out[:len(assistants) + 2]))


def write_rows(rows, name):
    os.makedirs("results/summary", exist_ok=True)
    keys = list(rows[0].keys())
    with open(f"results/summary/{name}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"wrote results/summary/{name}.csv ({len(rows)} rows)")


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        p1_gate(sys.argv[2] if len(sys.argv) > 2 else "p1")
    elif sys.argv[1] == "robust":
        robust()
    elif sys.argv[1] == "p2":
        p2(*sys.argv[2:])
    else:
        {"p0": p0, "p1": p1}[sys.argv[1]](sys.argv[2])
