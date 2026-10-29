"""P4b steps 2-3: does an LLM retrained on thumbs learn to copy what helped others, and narrow the culture? And
does it beat random edits when the provider sets the mix (memory on a share of requests, thumbs there, dup elsewhere)?

Arms (results/p4b_s05): edits of at most 3 choices with up to 3 remembered helpful modules in the prompt
(`--memory 3`), retrained on thumbs or with the duplicate penalty; the same arms without memory; step 3, memory
shown on 50% or 25% of requests (`--memory-share`), thumbs on prompts with a listing, dup on the rest. Per epoch:
share of answers that copy a listed edit (among answers to prompts with a listing), the same if the answer were a
random edit of 1-3 flips (chance, from the rebuilt listings), help rate of copies and of other answers (pooled
over the epoch's requests), helpful modules in memory, share of the 8 update samples that copy, distinct texts of 8
update samples on prompts with and without a listing, and diversity and fitness while assisted vs no AI on the same
seeds (P4a); fitness also vs random edits (rnd3).
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
        ("thumbsE3T2", "No memory, thumbs", "#2a78d6", "--"), ("dupE3T2", "No memory, duplicate penalty", "#eb6834", "--"),
        ("memmix50E3T2", "Memory on 50%, thumbs there, dup elsewhere", "#1baf7a", "-"),
        ("memmix25E3T2", "Memory on 25%, thumbs there, dup elsewhere", "#eda100", "-")]
POP = lambda x: bin(int(x)).count("1")


def from_requests(d):
    """Per epoch, pooled over requests: share with a listing; share of answers (to prompts with a listing) that copy
    a listed edit; help rate of copies and of other answers to listed prompts; chance, the share that would copy if
    each answer were a random edit of 1-3 flips (count uniform); helpful modules in memory at the epoch's end.
    Listings are rebuilt out of requests.jsonl as the driver builds them."""
    out = {k: [] for k in ("listed", "copied", "copy_help", "other_help", "chance", "mem_size",
                           "n_copy", "n_copy_help", "n_other", "n_other_help")}
    for line in open(os.path.join(d, "requests.jsonl")):
        r, mem, p = json.loads(line), {}, []
        rows = list(zip(r["t"], r["code"], r["answer"], r["helped"], r["n_listed"]))
        for t in sorted(set(r["t"])):
            cur = [x for x in rows if x[0] == t]
            for _, code, ans, _, nl in cur:
                m = code & 1023
                near = sorted(((rr, x) for x, rr in mem.items() if 0 < POP(x ^ m) <= 3), key=lambda q: -q[0])[:3]
                if nl and ans >= 0:  # listing shown (with --memory-share some are not)
                    p.append(sum(1 / 3 / comb(10, POP(x ^ m)) for _, x in near))
            for _, _, ans, h, _ in cur:
                if h and ans >= 0:
                    mem[ans] = t
        a, h = np.array(r["answer"]), np.array(r["helped"], bool)
        L, cp = (np.array(r["n_listed"]) > 0) & (a >= 0), np.array(r["copied"], bool)
        nan_mean = lambda x: x.mean() if len(x) else np.nan
        out["listed"].append((np.array(r["n_listed"]) > 0).mean())
        out["copied"].append(nan_mean(cp[L]))
        out["copy_help"].append(nan_mean(h[cp]))
        out["other_help"].append(nan_mean(h[L & ~cp]))
        out["chance"].append(np.mean(p) if p else np.nan)
        out["mem_size"].append(len(mem))
        for k, x in (("n_copy", cp), ("n_copy_help", cp & h), ("n_other", L & ~cp), ("n_other_help", L & ~cp & h)):
            out[k].append(x.sum())
    return {k: np.array(v, float) for k, v in out.items()}


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
    ep = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"))]
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows], float).reshape(-1, ROUNDS)
    out = dict(div=g("diversity")[:, -20:].mean(1), fit=g("fitness").mean(1), help=np.nanmean(g("improve"), 1),
               next_off=np.array([e["next_off"] for e in ep]))
    if any("listed" in r for r in rows):
        out.update(from_requests(d), update_copy=np.array([e.get("update_copy", np.nan) for e in ep], float))
    for k in ("group_distinct", "group_distinct_listed", "group_distinct_unlisted"):
        if any(e.get(k) is not None for e in ep):
            out[k] = np.array([np.nan if e.get(k) is None else e[k] for e in ep], float)
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


def sim(name):
    z = np.load(f"results/p4a_cost/shift0.5/rho1_mu0.01_{name}.npz")
    ix = [int(np.where(z["seeds"] == s)[0][0]) for s in seeds]
    return dict(div=z["div"][ix].reshape(len(seeds), -1, ROUNDS)[:, :EPOCHS, -20:].mean(-1),
                fit=z["fit"][ix].reshape(len(seeds), -1, ROUNDS)[:, :EPOCHS].mean(-1), next_off=z["next_off"][ix][:, :EPOCHS])


none, rnd3 = sim("none"), sim("rnd3")
scale = {k: 1 if k == "div" else 100 / none[k][:, BURN:].mean() for k in none}


def per_epoch(a, k):
    x = np.array([runs[a][s][k][:EPOCHS] for s in seeds])
    return scale[k] * (x - none[k]) if k in none else x


def ci(d):
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


rows = []
for a, name, *_ in arms:
    row = dict(arm=a, name=name, n=len(seeds))
    for k in ("div", "fit", "next_off", "help", "listed", "copied", "chance", "copy_help", "other_help", "update_copy",
              "mem_size", "group_distinct", "group_distinct_listed", "group_distinct_unlisted"):
        if k not in runs[a][seeds[0]]:
            continue
        if k in ("copy_help", "other_help"):  # pooled over the requests of epochs 3-20, per seed
            n = "n_" + k.split("_")[0]
            x = np.array([runs[a][s][n + "_help"][BURN:].sum() / runs[a][s][n][BURN:].sum() for s in seeds])
        else:
            x = np.nanmean(per_epoch(a, k)[:, BURN:], 1)
        row[k], row[k + "_ci"] = (float(v) for v in ci(x))
    rows.append(row)
    print(f"{name:30s} " + "  ".join(f"{k} {v:+.3g}" for k, v in row.items() if isinstance(v, float) and not k.endswith("_ci")))
    if "copied" in row:
        print(f"{'':30s} copied by epoch: " + " ".join(f"{v:.2f}" for v in np.nanmean(per_epoch(a, "copied"), 0)))
keys = list(dict.fromkeys(k for r in rows for k in r))
write_rows([{k: r.get(k, float("nan")) for k in keys} for r in rows], "p4b_memory")

print("\npaired contrasts, epochs 3-20 (diversity in Hamming units, fitness in % of no-AI fitness)")
mix = ["memmix50E3T2", "memmix25E3T2"]
for a, b in [("memthumbsE3T2", "thumbsE3T2"), ("memdupE3T2", "dupE3T2"), ("memdupE3T2", "memthumbsE3T2"),
             ("memthumbsE3T2", "rnd3"), ("memdupE3T2", "rnd3")] + [(x, y) for x in mix for y in (
                 "rnd3", "memdupE3T2", "memthumbsE3T2", "dupE3T2")] + [tuple(mix)]:
    if a not in runs or (b not in runs and b != "rnd3"):
        continue
    out = []
    for k in ("div", "fit", "next_off", "copied"):
        if k not in runs[a][seeds[0]] or (b != "rnd3" and k not in runs[b][seeds[0]]) or (b == "rnd3" and k == "copied"):
            continue
        vb = scale[k] * (rnd3[k] - none[k]) if b == "rnd3" else per_epoch(b, k)
        d = np.nanmean(per_epoch(a, k)[:, BURN:], 1) - np.nanmean(vb[:, BURN:], 1)
        out.append(f"{k} {d.mean():+.3f} (p={stats.ttest_1samp(d, 0).pvalue:.3f})")
    print(f"  {a} - {b}: " + "; ".join(out))
if os.path.isdir("results/p4a_mem/shift0.5"):
    print("\nsim references (P4a, 200 seeds, from results/summary/p4a_mem.csv): memory always fit +10.8%, div -0.16; "
          "50% fit +24.0%, div -0.09; random edit fit +15.2%, div +0.14")

e = np.arange(1, EPOCHS + 1)
mem_arms = [x for x in arms if x[0] in ("memthumbsE3T2", "memdupE3T2")]
base = {"memthumbsE3T2": "thumbsE3T2", "memdupE3T2": "dupE3T2"}
tq = stats.t.ppf(0.975, len(seeds) - 1)


def band(ax, x, c, lw=1.4, ls="-", fill=True):
    mu_, h = np.nanmean(x, 0), tq * np.nanstd(x, 0, ddof=1) / np.sqrt(len(seeds))
    ax.plot(e, mu_, color=c, lw=lw, ls=ls)
    if fill:
        ax.fill_between(e, mu_ - h, mu_ + h, color=c, alpha=0.15, lw=0)


fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), layout="constrained")
axes = axes.ravel()
for a, name, c, ls in mem_arms:
    band(axes[0], per_epoch(a, "copied"), c)
    band(axes[0], per_epoch(a, "update_copy"), c, lw=0.8, ls=":", fill=False)
    band(axes[1], per_epoch(a, "mem_size"), c)
    for ax, k in ((axes[2], "div"), (axes[3], "fit")):
        if base[a] in runs:
            band(ax, per_epoch(a, k) - per_epoch(base[a], k), c)
if mem_arms:
    axes[0].plot(e, np.nanmean([np.nanmean(per_epoch(a, "chance"), 0) for a, *_ in mem_arms], 0), color=MUTED, lw=0.8, ls="--")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Answers copying a listed edit")
axes[1].set_ylabel("Helpful modules in memory")
axes[2].set_ylabel("Culture diversity, memory − none\n(Hamming)")
axes[3].set_ylabel("Fitness while assisted, memory − none\n(% of no-AI fitness)")
for ax in axes[2:]:
    ax.axhline(0, color=GRID, lw=0.7, ls="--", zorder=0)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.axvspan(0.5, BURN + 0.5, color=GRID, alpha=0.4, lw=0, zorder=0)
fig.legend([plt.Line2D([], [], color=c, lw=1.4) for _, _, c, _ in mem_arms], ["Thumbs", "Duplicate penalty"][:len(mem_arms)],
           loc="outside lower center", ncol=2, fontsize=7, frameon=False)
mark_panels(axes)
save(fig, "p4b_memory")

# Step 3: the provider sets the mix. (a) copying on prompts with a listing, (b) distinct texts of 8 update samples
# on prompts without one, (c) fitness while assisted vs random edits, (d) diversity vs no AI. Step-2 arms dashed.
mix_arms = [x for x in arms if x[0] in mix]
if mix_arms:
    ref = [x for x in arms if x[0] in ("memthumbsE3T2", "memdupE3T2")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), layout="constrained")
    axes = axes.ravel()
    for (a, name, c, _), ls in [(x, "-") for x in mix_arms] + [(x, "--") for x in ref]:
        lw = 1.4 if ls == "-" else 0.9
        band(axes[0], per_epoch(a, "copied"), c, lw, ls, ls == "-")
        k = "group_distinct_unlisted" if a in mix else "group_distinct"
        band(axes[2], per_epoch(a, "fit") - scale["fit"] * (rnd3["fit"] - none["fit"]), c, lw, ls, ls == "-")
        band(axes[3], per_epoch(a, "div"), c, lw, ls, ls == "-")
    for a, name, c, _ in mix_arms:
        band(axes[1], per_epoch(a, "group_distinct_unlisted"), c)
    for a, c in (("dupE3T2", "#eb6834"), ("thumbsE3T2", "#2a78d6")):
        if a in runs and "group_distinct" in runs[a][seeds[0]]:
            band(axes[1], per_epoch(a, "group_distinct"), c, 0.9, "--", False)
    axes[0].plot(e, np.nanmean([np.nanmean(per_epoch(a, "chance"), 0) for a, *_ in mix_arms], 0), color=MUTED, lw=0.8, ls="--")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(1, 8)
    axes[0].set_ylabel("Answers copying a listed edit\n(prompts with a listing)")
    axes[1].set_ylabel("Distinct answers of 8\n(prompts without a listing)")
    axes[2].set_ylabel("Fitness while assisted, vs random edit\n(% of no-AI fitness)")
    axes[3].set_ylabel("Culture diversity, vs no AI\n(Hamming)")
    for ax in axes[2:]:
        ax.axhline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.axvspan(0.5, BURN + 0.5, color=GRID, alpha=0.4, lw=0, zorder=0)
    lab = {"memmix50E3T2": "Memory on 50%", "memmix25E3T2": "Memory on 25%", "memthumbsE3T2": "Memory always, thumbs",
           "memdupE3T2": "Memory always, duplicate penalty"}
    fig.legend([plt.Line2D([], [], color=c, lw=1.4 if a in mix else 0.9, ls="-" if a in mix else "--")
                for a, _, c, _ in mix_arms + ref], [lab[a] for a, *_ in mix_arms + ref],
               loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    mark_panels(axes)
    save(fig, "p4b_mix")
