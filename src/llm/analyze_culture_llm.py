"""P4b analysis: LLM assistants retrained once per epoch in a culture with turnover.

  python -m src.llm.analyze_culture_llm results/p4b results/p4a [arms,comma,separated] [output tag]

Per run and epoch: the culture's diversity (mean over the last 20 rounds of the epoch), the model's
answer diversity while deployed, fitness while assisted, fitness in the first 10 rounds after each change,
and the next-problem fitness with the assistant removed. Comparisons are paired by seed with the frozen
model (same problems, network and starting points) and with no AI and the oracles from P4a.
"""
import collections
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

from src.analyze import GRID, INK, mark_panels, plt, save, write_rows
from src.llm.analyze_p3 import ARM_COLORS

ARMS = ["frozen", "thumbs", "thumbskl", "pcreative", "hcreative", "thumbsT2", "pcreativeT2", "hcreativeT2", "localT2",
        "frozenE3", "thumbsE3T2"]
NAMES = {"frozen": "LLM, never updated", "thumbs": "thumbs", "thumbskl": "thumbs + KL",
         "pcreative": "P-creativity", "hcreative": "H-creativity", "thumbsT2": "thumbs, explore", "pcreativeT2": "P-creativity, explore",
         "hcreativeT2": "H-creativity, explore", "localT2": "edit within 3 bits, explore",
         "frozenE3": "edit interface (3), never updated", "thumbsE3T2": "edit interface (3), thumbs", "none": "No AI", "pcc": "PCC (oracle full answer)",
         "sbp": "SBP (oracle nudge)"}
COLORS = {"frozen": "#b8b6ad", "thumbs": ARM_COLORS["thumbs"], "thumbskl": ARM_COLORS["thumbskl"],
          "pcreative": "#eda100", "hcreative": "#1baf7a", "thumbsT2": ARM_COLORS["thumbs"], "pcreativeT2": "#eda100",
          "hcreativeT2": "#1baf7a", "localT2": "#e87ba4", "frozenE3": "#7fd1b0",
          "thumbsE3T2": "#0f7a52", "none": "#898781", "pcc": "#c98500", "sbp": ARM_COLORS["sbp"]}


def load(root, rounds=60):
    runs = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        arm, s = os.path.basename(d).rsplit("_s", 1)
        f = os.path.join(d, "rounds.jsonl")
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f)]
        ep = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"))] if os.path.exists(
            os.path.join(d, "epochs.jsonl")) else []
        n_ep = len(rows) // rounds
        if n_ep == 0:
            continue
        rows = rows[:n_ep * rounds]
        g = lambda key: np.array([np.nan if r.get(key) is None else r[key] for r in rows], float).reshape(n_ep, rounds)
        runs[arm][int(s)] = dict(
            div=g("diversity")[:, -20:].mean(1), mods=g("modules")[:, -20:].mean(1), fit=g("fitness").mean(1),
            adapt=g("fitness")[:, :10].mean(1), top=np.nanmean(g("top_share"), 1),
            uniq=np.nanmean(g("shown_unique"), 1), improve=np.nanmean(g("improve"), 1),
            next_off=np.array([x["next_off"] for x in ep[:n_ep]]),
            grp=np.array([x.get("group_distinct", np.nan) for x in ep[:n_ep]]), n_ep=n_ep)
    return runs


def oracle(p4a, name, seeds, rounds=60, mu=0.01):
    f = os.path.join(p4a, f"rho1_mu{mu:g}_{name}.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    idx = [int(np.where(z["seeds"] == s)[0][0]) for s in seeds]
    div = z["div"][idx]
    fit = z["fit"][idx]
    s, T = div.shape
    return dict(div=div.reshape(s, T // rounds, rounds)[:, :, -20:].mean(-1),
                fit=fit.reshape(s, T // rounds, rounds).mean(-1),
                adapt=fit.reshape(s, T // rounds, rounds)[:, :, :10].mean(-1))


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se, stats.ttest_1samp(x, 0).pvalue


def main(root="results/p4b", p4a="results/p4a", arms=None, tag="", burn=2):
    runs = load(root)
    arms = [a for a in (arms.split(",") if arms else ARMS) if a in runs]
    full = max(runs[a][s]["n_ep"] for a in arms for s in runs[a])
    seeds = sorted(s for s in set.intersection(*[set(runs[a]) for a in arms])
                   if all(runs[a][s]["n_ep"] == full for a in arms))  # finished runs only
    n_ep = full
    none = oracle(p4a, "none", seeds)
    orc = {k: oracle(p4a, k, seeds) for k in ("pcc", "sbp")}
    print(f"{len(seeds)} seeds with all arms, {n_ep} epochs complete")
    rows = []
    for a in arms:
        R = runs[a]
        mean_over = lambda key: np.array([R[s][key][burn:n_ep].mean() for s in seeds])
        row = dict(arm=a, n=len(seeds), epochs=n_ep)
        for key in ("div", "mods", "top", "uniq", "improve", "grp"):
            row[key] = float(np.nanmean(mean_over(key)))
        if none is not None:
            for key in ("fit", "adapt"):
                b = none[key][:, burn:n_ep].mean(1)
                m, lo, hi, p = ci(100 * (mean_over(key) - b) / b.mean())
                row.update({key + "_vs_none": m, key + "_lo": lo, key + "_hi": hi})
            row["div_none"] = float(none["div"][:, burn:n_ep].mean())
        if a != "frozen":
            for key in ("div", "next_off"):
                d = np.array([R[s][key][burn:n_ep].mean() - runs["frozen"][s][key][burn:n_ep].mean() for s in seeds])
                m, lo, hi, p = ci(d)
                row.update({key + "_vs_frozen": m, key + "_vs_frozen_p": p})
            # ratchet: slope of (diversity - frozen diversity) over epochs
            sl = [np.polyfit(np.arange(burn, n_ep), R[s]["div"][burn:n_ep] - runs["frozen"][s]["div"][burn:n_ep], 1)[0]
                  for s in seeds]
            m, lo, hi, p = ci(sl)
            row.update(ratchet=m, ratchet_lo=lo, ratchet_hi=hi, ratchet_p=p)
        rows.append(row)
        print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    if rows:
        write_rows(rows, "p4b_culture_llm" + tag)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), constrained_layout=True)
    ep = np.arange(1, n_ep + 1)
    series = [("div", axes[0], "Culture diversity\n(mean pairwise Hamming)"),
              ("uniq", axes[1], "Distinct answers / answers given\n(per round, while deployed)"),
              ("fit", axes[2], "Fitness while assisted"),
              ("next_off", axes[3], "Next problem, AI removed")]
    for key, ax, lab in series:
        for a in arms:
            y = np.nanmean([runs[a][s][key][:n_ep] for s in seeds], 0)
            ax.plot(ep, y, "--" if a.endswith("T2") and "E" not in a else "-", lw=1.1, color=COLORS[a], label=NAMES[a])
        if key in ("div", "fit") and none is not None:
            ax.plot(ep, none[key][:, :n_ep].mean(0), "--", lw=0.9, color=COLORS["none"], label=NAMES["none"])
            if orc["pcc"] is not None:
                ax.plot(ep, orc["pcc"][key][:, :n_ep].mean(0), ":", lw=0.9, color=COLORS["pcc"], label=NAMES["pcc"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=5.2, loc="upper right")
    mark_panels(axes)
    save(fig, "p4b_culture_llm" + tag)


if __name__ == "__main__":
    main(*sys.argv[1:])
