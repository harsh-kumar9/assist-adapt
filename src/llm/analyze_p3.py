"""Figures for P3: LLM assistants learning online across a sequence of problems.

  python -m src.llm.analyze_p3 results/p3

Each run directory is named <arm>_s<seed> (for example now_s0, kl_s1, pcc_s0).
"""
import collections
import glob
import json
import os
import sys

import numpy as np

from src.analyze import AQUA, BLUE, GRID, INK, INK2, MUTED, ORANGE, RED, mark_panels, plt, save, write_rows

ARM_COLORS = {"none": MUTED, "sbp": ORANGE, "pcc": "#eda100", "gated": INK2, "now": BLUE, "kl": AQUA,
              "stepback": "#e87ba4", "edit": "#4a3aa7", "cf": RED}
ARM_NAMES = {"none": "No AI", "sbp": "SBP (oracle nudge)", "pcc": "PCC (oracle full answer)",
             "gated": "Gated oracle (P2 rule)", "now": "LLM, R_now", "kl": "LLM, R_now + KL",
             "stepback": "LLM, R_now + step back", "edit": "LLM, nudge prior", "cf": "LLM, R_adapt"}


def load(root):
    runs = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        if not os.path.exists(os.path.join(d, "rounds.jsonl")):
            continue
        arm = os.path.basename(d).rsplit("_s", 1)[0]
        rounds = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
        probes = [json.loads(l) for l in open(os.path.join(d, "probes.jsonl"))] if os.path.exists(
            os.path.join(d, "probes.jsonl")) else []
        runs[arm].append(dict(dir=d, rounds=rounds, probes=probes, cfg=json.load(open(os.path.join(d, "config.json")))))
    return runs


def series(run, key):
    R = run["cfg"]["rounds"]
    x = np.array([r["problem"] * R + r["round"] for r in run["rounds"]])
    y = np.array([np.nan if r.get(key) is None else r[key] for r in run["rounds"]], float)
    return x, y


def mean_series(rs, key):
    xs = [series(r, key) for r in rs]
    n = min(len(x) for x, _ in xs)
    return xs[0][0][:n], np.nanmean(np.stack([y[:n] for _, y in xs]), 0)


def main(root="results/p3"):
    runs = load(root)
    arms = [a for a in ARM_COLORS if a in runs]
    if not arms:
        print("no runs")
        return
    R = runs[arms[0]][0]["cfg"]["rounds"]
    llm_arms = [a for a in arms if a not in ("none", "sbp", "pcc", "gated")]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), constrained_layout=True)
    axes = axes.ravel()
    panels = [("fitness", "Community fitness", arms), ("diversity", "Mean pairwise Hamming distance", arms),
              ("modules", "Distinct computational modules", arms),
              ("top_share", "Share of askers given the\nmost common suggestion", llm_arms),
              ("group_distinct", "Distinct answers in a group of 8", llm_arms)]
    for ax, (key, label, show) in zip(axes, panels):
        for a in show:
            x, y = mean_series(runs[a], key)
            ax.plot(x, y, lw=1.0, color=ARM_COLORS[a], label=ARM_NAMES[a])
        n_prob = runs[arms[0]][0]["cfg"]["problems"]
        for k in range(1, n_prob):
            ax.axvline(k * R, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("Round (shift every %d)" % R)
        ax.set_ylabel(label)
    ax = axes[5]
    rows = []
    for a in arms:
        by_k = collections.defaultdict(list)
        for r in runs[a]:
            for p in r["probes"]:
                if p["landscape"] == "next":
                    by_k[p["problem"]].append(p["mean"])
        ks = sorted(by_k)
        if ks:
            ax.plot(ks, [np.mean(by_k[k]) for k in ks], "-o", ms=3, lw=1, color=ARM_COLORS[a])
            rows += [dict(arm=a, problem=k, next_ai_off=float(np.mean(by_k[k])), n=len(by_k[k])) for k in ks]
    ax.set_xlabel("Problem")
    ax.set_ylabel("Fitness on next problem,\nAI removed (30 rounds)")
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right")
    mark_panels(axes)
    save(fig, "p3_continual")
    if rows:
        write_rows(rows, "p3_probes")




def per_problem(runs, arm, base="none"):
    """Per problem: mean fitness while assisted and next-problem AI-off fitness, % vs no AI,
    paired by seed, with 95% CIs."""
    seeds = sorted({int(r["dir"].rsplit("_s", 1)[1]) for r in runs[arm]} &
                   {int(r["dir"].rsplit("_s", 1)[1]) for r in runs[base]})
    by = {a: {int(r["dir"].rsplit("_s", 1)[1]): r for r in runs[a]} for a in (arm, base)}
    out = []
    n_prob = runs[arm][0]["cfg"]["problems"]
    for k in range(n_prob):
        vals = {}
        for a in (arm, base):
            vals[a] = (np.array([np.mean([x["fitness"] for x in by[a][s]["rounds"] if x["problem"] == k]) for s in seeds]),
                       np.array([next(p["mean"] for p in by[a][s]["probes"] if p["problem"] == k and p["landscape"] == "next")
                                 for s in seeds]))
        row = dict(arm=arm, problem=k, n=len(seeds))
        for i, key in enumerate(("during", "next_off")):
            d = vals[arm][i] - vals[base][i]
            m0 = vals[base][i].mean()
            se = d.std(ddof=1) / np.sqrt(len(d))
            row.update({key: 100 * d.mean() / m0, key + "_lo": 100 * (d.mean() - 1.96 * se) / m0,
                        key + "_hi": 100 * (d.mean() + 1.96 * se) / m0})
        out.append(row)
    return out


def oracle_fig(root="results/p3_oracle"):
    runs = load(root)
    arms = [a for a in ("none", "sbp", "pcc", "gated") if a in runs]
    R = runs[arms[0]][0]["cfg"]["rounds"]
    n_prob = runs[arms[0]][0]["cfg"]["problems"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
    for ax, key, label in ((axes[0], "fitness", "Community fitness"),
                           (axes[1], "diversity", "Mean pairwise Hamming distance")):
        for a in arms:
            x, y = mean_series(runs[a], key)
            ax.plot(x, y, lw=1.0, color=ARM_COLORS[a], label=ARM_NAMES[a])
        for k in range(1, n_prob):
            ax.axvline(k * R, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel(f"Round (new problem every {R})")
        ax.set_ylabel(label)
    rows = []
    for a in arms[1:]:
        pp = per_problem(runs, a)
        rows += pp
        for ax, key in ((axes[2], "during"), (axes[3], "next_off")):
            ks = np.array([r["problem"] for r in pp]) + 1 + (arms.index(a) - 2) * 0.12
            ax.errorbar(ks, [r[key] for r in pp], yerr=[[r[key] - r[key + "_lo"] for r in pp],
                                                       [r[key + "_hi"] - r[key] for r in pp]],
                        fmt="-o", ms=3, lw=1, color=ARM_COLORS[a], elinewidth=0.6, capsize=0)
    for ax, lab in ((axes[2], "Fitness while assisted\n(% vs no AI)"),
                    (axes[3], "Next problem, AI removed\n(% vs no AI)")):
        ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_xticks(range(1, n_prob + 1))
        ax.set_xlabel("Problem")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right")
    mark_panels(axes)
    save(fig, "p3_oracle_continual")
    write_rows(rows, "p3_oracle_per_problem")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "oracle":
        oracle_fig(*sys.argv[2:])
    else:
        main(*sys.argv[1:])
