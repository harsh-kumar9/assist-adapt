"""P4b at 50% shift: does giving the LLM reach through the interface keep the culture able to adapt?

Arms (results/p4b_s05): full answers vs edits of at most 3 choices vs tweak or leap, each never updated
and retrained on thumbs (update samples at temperature 2); a menu of 3; and rewards beyond thumbs (spread,
duplicate penalty, crowding penalty, signed rating); edits with a memory of what helped in the prompt
(details in p4b_memory.py). Per run, epochs 3-20: culture diversity (last
20 rounds of each epoch), fitness while assisted, fitness on the next problem with the assistant removed
(mean of 30 rounds), share of answers that help. Paired by seed; no AI and oracles from P4a, same seeds.

  python scripts/analysis/p4b_interface.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, INK, mark_panels, plt, save, write_rows  # noqa: E402

ROOT, ROUNDS, BURN = "results/p4b_s05", 60, 2
ARMS = [("frozen", "Full answer, never updated", "#b8b6ad", "--"), ("thumbsT2", "Full answer, thumbs", "#4a3aa7", "--"),
        ("frozenE3", "Edit (≤3), never updated", "#7fd1b0", "-"), ("thumbsE3T2", "Edit (≤3), thumbs", "#0f7a52", "-"),
        ("tlE3", "Tweak or leap, never updated", "#f2c46b", "-"), ("tlE3T2", "Tweak or leap, thumbs", "#c98500", "-"),
        ("frozenE3M3", "Edit, best of 3, never updated", "#e66b8f", "-"), ("spreadE3T2", "Edit, spread reward", "#3b8fd6", "-"),
        ("dupE3T2", "Edit, duplicate penalty", "#2aa3a3", "-"), ("rateE3T2", "Edit, signed rating", "#8a5a44", "-"),
        ("crowdT2", "Full answer, crowding penalty", "#c03b2b", "--"), ("dupT2", "Full answer, duplicate penalty", "#7a4fb8", "--"),
        ("memthumbsE3T2", "Edit + memory, thumbs", "#1f2f8a", "-"), ("memdupE3T2", "Edit + memory, duplicate penalty", "#b0127a", "-"),
        ("memmix50E3T2", "Edit + memory on 50%", "#1baf7a", "-"),
        ("memmix25E3T2", "Edit + memory on 25%", "#eda100", "-")]
ORACLES = [("results/p4a_cost/shift0.5", "none"), ("results/p4a_cost/shift0.5", "pcc"),
           ("results/p4a_cost/shift0.5", "rad3"), ("results/p4a_grid/shift0.5", "leap_3_10"),
           ("results/p4a_cost/shift0.5", "rnd1"), ("results/p4a_cost/shift0.5", "rnd3"),
           ("results/p4a_cost/shift0.5", "rndfull"), ("results/p4a_cost/shift0.5", "rnd3x3")]
REFS = [("rnd3", "Random edit (1-3 flips)"), ("rnd3x3", "Random edit, best of 3"), ("rndfull", "Random full answer"), ("pcc", "Best answer in culture"),
        ("rad3", "Best edit (≤3)"), ("leap_3_10", "Best tweak or leap")]


def load_run(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
    ep = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"))]
    n = len(rows) // ROUNDS
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows[:n * ROUNDS]], float).reshape(n, ROUNDS)
    return dict(n=n, div=g("diversity")[:, -20:].mean(1), fit=g("fitness").mean(1), improve=np.nanmean(g("improve"), 1),
                uniq=np.nanmean(g("shown_unique"), 1), next_off=np.array([e["next_off"] for e in ep[:n]]),
                leap_good=np.nanmean(g("leap_good"), 1) if any("leap_good" in r for r in rows) else np.full(n, np.nan),
                edit_size=np.nanmean(g("edit_size"), 1) if any("edit_size" in r for r in rows) else np.full(n, np.nan),
                group_any=np.array([e.get("group_any", np.nan) for e in ep[:n]], float),
                group_multi=np.array([e.get("group_multi", np.nan) for e in ep[:n]], float),
                repeat=repeats(d, n))


def repeats(d, n):
    """Per epoch, share of requests that got the same answer as the same team's previous request with the same
    approach (a repeat of an answer that did not help, since a helped team has moved). NaN if not logged."""
    f = os.path.join(d, "requests.jsonl")
    if not os.path.exists(f):
        return np.full(n, np.nan)
    out = []
    for line in list(open(f))[:n]:
        r, last, rep, tot = json.loads(line), {}, 0, 0
        for slot, code, ans in zip(r.get("slot", []), r.get("code", []), r.get("answer", [])):
            if ans >= 0 and last.get(slot, (None, None))[0] == code:
                tot += 1
                rep += last[slot][1] == ans
            last[slot] = (code, ans)
        out.append(rep / tot if tot else np.nan)
    return np.array(out)


runs = {}
for a, *_ in ARMS:
    for d in sorted(glob.glob(os.path.join(ROOT, f"{a}_s*"))):
        if os.path.exists(os.path.join(d, "DONE")):
            runs.setdefault(a, {})[int(d.rsplit("_s", 1)[1])] = load_run(d)
arms = [x for x in ARMS if x[0] in runs and len(runs[x[0]]) >= 6]
seeds = sorted(set.intersection(*[set(runs[a]) for a, *_ in arms]))
print(f"{len(seeds)} seeds with all of: {', '.join(a for a, *_ in arms)}")
orc, full = {}, {}
for d, name in ORACLES:
    if not os.path.exists(os.path.join(d, f"rho1_mu0.01_{name}.npz")):
        continue
    z = np.load(os.path.join(d, f"rho1_mu0.01_{name}.npz"))
    idx = [int(np.where(z["seeds"] == s)[0][0]) for s in seeds]
    e = lambda x: x[idx].reshape(len(seeds), -1, ROUNDS)
    orc[name] = dict(div=e(z["div"])[:, :, -20:].mean(-1), fit=e(z["fit"]).mean(-1), next_off=z["next_off"][idx])
    S = len(z["seeds"])  # all P4a seeds, for the reference rows of the figure
    full[name] = dict(div=z["div"].reshape(S, -1, ROUNDS)[:, BURN:, -20:].mean((1, 2)),
                      fit=z["fit"].reshape(S, -1, ROUNDS)[:, BURN:].mean((1, 2)), next_off=z["next_off"][:, BURN:].mean(1))
base = orc["none"]
m = lambda x: x[:, BURN:].mean(1)


def pct(x):
    return 100 * x / m(base["fit"]).mean()


rows = []
for a, name, *_ in arms:
    R = runs[a]
    get = lambda k: np.array([np.nanmean(R[s][k][BURN:]) if not np.isnan(R[s][k][BURN:]).all() else np.nan for s in seeds])
    row = dict(arm=a, name=name, n=len(seeds), div=float(get("div").mean()), improve=float(get("improve").mean()),
               uniq=float(get("uniq").mean()), fit_vs_none=float(pct(get("fit") - m(base["fit"])).mean()),
               next_vs_none=float((100 * (get("next_off") - m(base["next_off"])) / m(base["next_off"]).mean()).mean()))
    for k in ("leap_good", "edit_size", "group_any", "group_multi", "repeat"):
        if not np.isnan(get(k)).all():
            row[k] = float(np.nanmean(get(k)))
    rows.append(row)
for name, o in orc.items():
    rows.append(dict(arm="oracle_" + name, name=name, n=len(seeds), div=float(m(o["div"]).mean()),
                     fit_vs_none=float(pct(m(o["fit"]) - m(base["fit"])).mean()),
                     next_vs_none=float((100 * (m(o["next_off"]) - m(base["next_off"])) / m(base["next_off"]).mean()).mean())))
for r in rows:
    print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
keys = list(dict.fromkeys(k for r in rows for k in r))
write_rows([{k: r.get(k, float("nan")) for k in keys} for r in rows], "p4b_interface")

print("\npaired contrasts (seeds paired; diversity in Hamming units, fitness in % of no-AI fitness)")
for a, b in [("frozenE3", "frozen"), ("thumbsE3T2", "thumbsT2"), ("thumbsE3T2", "frozenE3"), ("thumbsT2", "frozen"),
             ("tlE3", "frozenE3"), ("tlE3T2", "thumbsE3T2"), ("tlE3T2", "tlE3"),
             ("frozen", "rndfull"), ("thumbsT2", "rndfull"), ("frozenE3", "rnd3"), ("thumbsE3T2", "rnd3"),
             ("tlE3", "rnd3"), ("tlE3T2", "rnd3"), ("frozenE3M3", "frozenE3"), ("frozenE3M3", "rnd3x3"),
             ("spreadE3T2", "thumbsE3T2"), ("spreadE3T2", "frozenE3"), ("spreadE3T2", "rnd3"),
             ("dupE3T2", "thumbsE3T2"), ("dupE3T2", "frozenE3"), ("dupE3T2", "rnd3"),
             ("rateE3T2", "thumbsE3T2"), ("rateE3T2", "rnd3"),
             ("crowdT2", "thumbsT2"), ("crowdT2", "frozen"), ("crowdT2", "rndfull"),
             ("dupT2", "thumbsT2"), ("dupT2", "rndfull"), ("dupT2", "crowdT2"),
             ("memthumbsE3T2", "thumbsE3T2"), ("memthumbsE3T2", "rnd3"), ("memdupE3T2", "dupE3T2"),
             ("memdupE3T2", "rnd3"), ("memdupE3T2", "memthumbsE3T2"),
             ("memmix50E3T2", "rnd3"), ("memmix50E3T2", "memdupE3T2"), ("memmix50E3T2", "memthumbsE3T2"),
             ("memmix25E3T2", "rnd3"), ("memmix25E3T2", "memdupE3T2"), ("memmix25E3T2", "memthumbsE3T2"),
             ("memmix50E3T2", "memmix25E3T2")]:
    done = [x[0] for x in arms]
    if a not in done or not (b in done or b in orc):
        continue
    out = []
    for k in ("div", "fit", "next_off"):
        vb = m(orc[b][k]) if b in orc else np.array([runs[b][s][k][BURN:].mean() for s in seeds])
        d = np.array([runs[a][s][k][BURN:].mean() for s in seeds]) - vb
        if k != "div":
            d = 100 * d / m(base[k]).mean()
        out.append(f"{k} {d.mean():+.2f} (p={stats.ttest_1samp(d, 0).pvalue:.3f})")
    print(f"  {a} - {b}: " + "; ".join(out))



def ci(d):
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


refs = [x for x in REFS if x[0] in full]
labels = [name for _, name, *_ in arms] + [name for _, name in refs]
y = np.r_[np.arange(len(arms)), len(arms) + 0.6 + np.arange(len(refs))]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.6 + 0.2 * len(labels)), sharey=True, constrained_layout=True)
for ax, k, lab in ((axes[0], "div", "Culture diversity\n(Hamming, vs no AI)"),
                   (axes[1], "fit", "Fitness while assisted\n(% vs no AI)"),
                   (axes[2], "next_off", "Next problem, AI removed\n(% vs no AI)")):
    scale = 1 if k == "div" else 100 / m(base[k]).mean()
    for j, (a, name, c, ls) in enumerate(arms):
        d = scale * (np.array([runs[a][s][k][BURN:].mean() for s in seeds]) - m(base[k]))
        ax.plot(d, np.full(len(d), y[j]), "o", ms=2, color=c, alpha=0.45, mec="none")
        mu_, h = ci(d)
        ax.errorbar(mu_, y[j], xerr=h, fmt="o", ms=3.5, color=c, lw=1, capsize=0)
    sc = 1 if k == "div" else 100 / full["none"][k].mean()
    for j, (a, name) in enumerate(refs, len(arms)):
        mu_, h = ci(sc * (full[a][k] - full["none"][k]))
        ax.errorbar(mu_, y[j], xerr=h, fmt="s", ms=3, mfc="white", color=INK, lw=0.8, capsize=0)
    ax.axvline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_xlabel(lab)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
mark_panels(axes)
save(fig, "p4b_interface")
