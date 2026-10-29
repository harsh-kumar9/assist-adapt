"""P4b step 2: does an LLM retrained on thumbs learn to copy what helped others, and narrow the culture?

Arms (results/p4b_s05): edits of at most 3 choices with up to 3 remembered helpful modules in the prompt
(`--memory 3`), retrained on thumbs or with the duplicate penalty; the same arms without memory. Per epoch:
share of answers that copy a listed edit (among answers to prompts with a listing), the same if the answer were a
random edit of 1-3 flips (chance, from the rebuilt listings), help rate of copies and of other answers, share of
the 8 update samples that copy, and diversity and fitness while assisted vs no AI on the same seeds (P4a).
Summaries over epochs 3-20, paired by seed. Sim memory designs (P4a, 200 seeds) for reference.

  python scripts/analysis/p4b_memory.py [results/p4b_s05]
"""
import json
import os
import sys
from math import comb

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, MUTED, mark_panels, plt, save, write_rows  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/p4b_s05"
ROUNDS, BURN, EPOCHS = 60, 2, 20
ARMS = [("memthumbsE3T2", "Memory, thumbs", "#2a78d6", "-"), ("memdupE3T2", "Memory, duplicate penalty", "#eb6834", "-"),
        ("thumbsE3T2", "No memory, thumbs", "#2a78d6", "--"), ("dupE3T2", "No memory, duplicate penalty", "#eb6834", "--")]
POP = lambda x: bin(int(x)).count("1")


def chance(d):
    """Per epoch, the share of answers that would copy a listed edit if each were a random edit of 1-3 flips
    (count uniform), from listings rebuilt out of requests.jsonl as the driver builds them."""
    out = []
    for line in open(os.path.join(d, "requests.jsonl")):
        r, mem, p = json.loads(line), {}, []
        rows = list(zip(r["t"], r["code"], r["answer"], r["helped"]))
        for t in sorted(set(r["t"])):
            cur = [x for x in rows if x[0] == t]
            for _, code, ans, _ in cur:
                m = code & 1023
                near = sorted(((rr, x) for x, rr in mem.items() if 0 < POP(x ^ m) <= 3), key=lambda q: -q[0])[:3]
                if near and ans >= 0:
                    p.append(sum(1 / 3 / comb(10, POP(x ^ m)) for _, x in near))
            for _, _, ans, h in cur:
                if h and ans >= 0:
                    mem[ans] = t
        out.append(np.mean(p) if p else np.nan)
    return np.array(out)


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
    ep = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"))]
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows], float).reshape(-1, ROUNDS)
    out = dict(div=g("diversity")[:, -20:].mean(1), fit=g("fitness").mean(1), help=np.nanmean(g("improve"), 1),
               next_off=np.array([e["next_off"] for e in ep]))
    if "listed" in rows[0] or any("listed" in r for r in rows):
        out.update(listed=np.nanmean(g("listed"), 1), copied=np.nanmean(g("copied"), 1),
                   copy_help=np.nanmean(g("copy_help"), 1), other_help=np.nanmean(g("other_help"), 1),
                   update_copy=np.array([e.get("update_copy", np.nan) for e in ep], float), chance=chance(d))
    return out


runs = {}
for a, *_ in ARMS:
    for s in range(6):
        d = os.path.join(ROOT, f"{a}_s{s}")
        if os.path.exists(os.path.join(d, "DONE")):
            runs.setdefault(a, {})[s] = load(d)
arms = [x for x in ARMS if x[0] in runs]
seeds = sorted(set.intersection(*[set(runs[a]) for a, *_ in arms]))
print(f"seeds {seeds} with all of: {', '.join(a for a, *_ in arms)}")
z = np.load("results/p4a_cost/shift0.5/rho1_mu0.01_none.npz")
ix = [int(np.where(z["seeds"] == s)[0][0]) for s in seeds]
none = dict(div=z["div"][ix].reshape(len(seeds), -1, ROUNDS)[:, :EPOCHS, -20:].mean(-1),
            fit=z["fit"][ix].reshape(len(seeds), -1, ROUNDS)[:, :EPOCHS].mean(-1), next_off=z["next_off"][ix][:, :EPOCHS])
scale = {k: 1 if k == "div" else 100 / none[k][:, BURN:].mean() for k in none}


def per_epoch(a, k):
    x = np.array([runs[a][s][k][:EPOCHS] for s in seeds])
    return scale[k] * (x - none[k]) if k in none else x


def ci(d):
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


rows = []
for a, name, *_ in arms:
    row = dict(arm=a, name=name, n=len(seeds))
    for k in ("div", "fit", "next_off", "help", "listed", "copied", "chance", "copy_help", "other_help", "update_copy"):
        if k in runs[a][seeds[0]]:
            row[k], row[k + "_ci"] = (float(v) for v in ci(np.nanmean(per_epoch(a, k)[:, BURN:], 1)))
    rows.append(row)
    print(f"{name:30s} " + "  ".join(f"{k} {v:+.3g}" for k, v in row.items() if isinstance(v, float) and not k.endswith("_ci")))
    if "copied" in row:
        print(f"{'':30s} copied by epoch: " + " ".join(f"{v:.2f}" for v in np.nanmean(per_epoch(a, "copied"), 0)))
keys = list(dict.fromkeys(k for r in rows for k in r))
write_rows([{k: r.get(k, float("nan")) for k in keys} for r in rows], "p4b_memory")

print("\npaired contrasts, epochs 3-20 (diversity in Hamming units, fitness in % of no-AI fitness)")
for a, b in [("memthumbsE3T2", "thumbsE3T2"), ("memdupE3T2", "dupE3T2"), ("memdupE3T2", "memthumbsE3T2")]:
    if a not in runs or b not in runs:
        continue
    out = []
    for k in ("div", "fit", "next_off", "copied"):
        if k not in runs[a][seeds[0]] or k not in runs[b][seeds[0]]:
            continue
        d = np.nanmean(per_epoch(a, k)[:, BURN:], 1) - np.nanmean(per_epoch(b, k)[:, BURN:], 1)
        out.append(f"{k} {d.mean():+.3f} (p={stats.ttest_1samp(d, 0).pvalue:.3f})")
    print(f"  {a} - {b}: " + "; ".join(out))
if os.path.isdir("results/p4a_mem/shift0.5"):
    print("\nsim references (P4a, 200 seeds, from results/summary/p4a_mem.csv): memory always fit +10.8%, div -0.16; "
          "50% fit +24.0%, div -0.09; random edit fit +15.2%, div +0.14")

e = np.arange(1, EPOCHS + 1)
mem_arms = [x for x in arms if "copied" in runs[x[0]][seeds[0]]]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), layout="constrained")
for a, name, c, ls in mem_arms:
    for k, lw in (("copied", 1.4), ("update_copy", 0.8)):
        mu_ = np.nanmean(per_epoch(a, k), 0)
        h = stats.t.ppf(0.975, len(seeds) - 1) * np.nanstd(per_epoch(a, k), 0, ddof=1) / np.sqrt(len(seeds))
        axes[0].plot(e, mu_, color=c, lw=lw, ls="-" if k == "copied" else ":")
        if k == "copied":
            axes[0].fill_between(e, mu_ - h, mu_ + h, color=c, alpha=0.15, lw=0)
if mem_arms:
    axes[0].plot(e, np.nanmean([np.nanmean(per_epoch(a, "chance"), 0) for a, *_ in mem_arms], 0), color=MUTED, lw=0.8, ls="--")
axes[0].set_ylabel("Answers copying a listed edit")
axes[0].set_ylim(0, 1)
for ax, k, lab in ((axes[1], "div", "Culture diversity (Hamming, vs no AI)"), (axes[2], "fit", "Fitness while assisted (% vs no AI)")):
    for a, name, c, ls in arms:
        mu_ = per_epoch(a, k).mean(0)
        ax.plot(e, mu_, color=c, lw=1.2, ls=ls, label=name)
    ax.axhline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_ylabel(lab)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.axvspan(0.5, BURN + 0.5, color=GRID, alpha=0.4, lw=0, zorder=0)
fig.legend(*axes[2].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7, frameon=False)
mark_panels(axes)
save(fig, "p4b_memory")
