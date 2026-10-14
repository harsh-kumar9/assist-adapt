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


if __name__ == "__main__":
    main(*sys.argv[1:])
