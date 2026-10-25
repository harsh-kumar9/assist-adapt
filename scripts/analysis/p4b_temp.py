"""P4b at 50% shift: does answering at a higher temperature make the never-updated LLM as useful as
random proposals of the same form? Full answers and edits of at most 3 choices at deployment temperature
1.0 (the interface runs), 1.5, 2.0 and 2.5. Per run, epochs 3-20, paired by seed with no AI; random
proposals (rndfull, rnd3) from P4a on all 200 seeds as references.

  python scripts/analysis/p4b_temp.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.analyze import GRID, mark_panels, plt, save, write_rows  # noqa: E402

ROOT, P4A, ROUNDS, BURN = "results/p4b_s05", "results/p4a_cost/shift0.5", 60, 2
TEMPS = [(1.0, ""), (1.5, "D15"), (2.0, "D20"), (2.5, "D25")]
IFACES = [("frozen", "Full answer", "rndfull", "#4a3aa7"), ("frozenE3", "Edit (≤3)", "rnd3", "#0f7a52")]
KEYS = [("div", "Culture diversity\n(Hamming, vs no AI)"), ("fit", "Fitness while assisted\n(% vs no AI)"),
        ("next_off", "Next problem, AI removed\n(% vs no AI)"), ("uniq", "Distinct answers\nper round")]


def load_run(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rounds.jsonl"))]
    ep = [json.loads(l) for l in open(os.path.join(d, "epochs.jsonl"))]
    n = len(rows) // ROUNDS
    g = lambda k: np.array([np.nan if r.get(k) is None else r[k] for r in rows[:n * ROUNDS]], float).reshape(n, ROUNDS)
    v = dict(div=g("diversity")[:, -20:].mean(1), fit=g("fitness").mean(1), improve=np.nanmean(g("improve"), 1),
             uniq=np.nanmean(g("shown_unique"), 1), valid=np.nanmean(g("valid"), 1),
             next_off=np.array([e["next_off"] for e in ep[:n]]))
    return {k: x[BURN:].mean() for k, x in v.items()}


runs = {}
for a, *_ in IFACES:
    for T, tag in TEMPS:
        for d in glob.glob(os.path.join(ROOT, f"{a}{tag}_s*")):
            if os.path.exists(os.path.join(d, "DONE")):
                runs.setdefault((a, T), {})[int(d.rsplit("_s", 1)[1])] = load_run(d)
seeds = sorted(set.intersection(*[set(r) for r in runs.values()]))
print(f"{len(seeds)} seeds; cells: {', '.join(f'{a}@{T:g}' for a, T in sorted(runs))}")

z = {n: np.load(os.path.join(P4A, f"rho1_mu0.01_{n}.npz")) for n in ("none", "rnd3", "rndfull")}
S = len(z["none"]["seeds"])
ep = lambda x: x.reshape(S, -1, ROUNDS)
full = {n: dict(div=ep(v["div"])[:, BURN:, -20:].mean((1, 2)), fit=ep(v["fit"])[:, BURN:].mean((1, 2)),
                next_off=v["next_off"][:, BURN:].mean(1)) for n, v in z.items()}
idx = [int(np.where(z["none"]["seeds"] == s)[0][0]) for s in seeds]
base = {k: v[idx] for k, v in full["none"].items()}


def diff(a, T, k):
    x = np.array([runs[(a, T)][s][k] for s in seeds])
    if k == "uniq":
        return x
    d = x - base[k]
    return d if k == "div" else 100 * d / base[k].mean()


def ci(d):
    return d.mean(), stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


rows = []
for a, name, rnd, _ in IFACES:
    print(f"\n{name}  (random of the same form, 200 seeds: div {full[rnd]['div'].mean() - full['none']['div'].mean():+.2f},"
          f" fit {100 * (full[rnd]['fit'] - full['none']['fit']).mean() / full['none']['fit'].mean():+.1f}%,"
          f" next {100 * (full[rnd]['next_off'] - full['none']['next_off']).mean() / full['none']['next_off'].mean():+.1f}%)")
    for T, _ in TEMPS:
        if (a, T) not in runs:
            continue
        r = dict(interface=a, temperature=T, n=len(seeds),
                 valid=float(np.mean([runs[(a, T)][s]["valid"] for s in seeds])),
                 improve=float(np.mean([runs[(a, T)][s]["improve"] for s in seeds])))
        for k, _ in KEYS:
            r[k] = float(diff(a, T, k).mean())
        # gap to random of the same form, paired by seed
        for k in ("fit", "next_off"):
            g = diff(a, T, k) - 100 * (full[rnd][k][idx] - base[k]) / base[k].mean()
            r[f"{k}_vs_random"], r[f"{k}_vs_random_p"] = float(g.mean()), float(stats.ttest_1samp(g, 0).pvalue)
        rows.append(r)
        print(f"  T {T:g}: valid {r['valid']:.2f} helps {r['improve']:.3f} distinct {r['uniq']:.2f} div {r['div']:+.2f}"
              f" fit {r['fit']:+.1f}% next {r['next_off']:+.1f}%  vs random: fit {r['fit_vs_random']:+.1f}"
              f" (p={r['fit_vs_random_p']:.3f}) next {r['next_off_vs_random']:+.1f} (p={r['next_off_vs_random_p']:.3f})")
write_rows(rows, "p4b_temp")

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), constrained_layout=True)
for ax, (k, lab) in zip(axes, KEYS):
    for j, (a, name, rnd, c) in enumerate(IFACES):
        Ts = [T for T, _ in TEMPS if (a, T) in runs]
        st = [ci(diff(a, T, k)) for T in Ts]
        x = np.array(Ts) + (j - 0.5) * 0.04
        ax.errorbar(x, [m for m, _ in st], yerr=[h for _, h in st], fmt="-o", ms=3, lw=1, color=c, capsize=0,
                    label=f"LLM, {name.lower()}")
        if k == "uniq":
            continue
        ref = full[rnd][k] - full["none"][k]
        ref = ref.mean() if k == "div" else 100 * ref.mean() / full["none"][k].mean()
        ax.axhline(ref, color=c, lw=0.8, ls=":", label=f"Random {name.lower()}")
    if k != "uniq":
        ax.axhline(0, color=GRID, lw=0.7, ls="--", zorder=0)
    ax.set_xticks([T for T, _ in TEMPS])
    ax.set_xlabel("Temperature when answering")
    ax.set_ylabel(lab)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False, fontsize=5.5)
mark_panels(axes)
save(fig, "p4b_temp")
