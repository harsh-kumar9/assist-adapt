"""(b) training runs: collapse and community measures per feedback type, 6 seeds, paired by seed."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.analyze_p3 import load  # noqa: E402

ARMS = ["now", "thumbs", "thumbskl", "pairwise", "pairwisekl"]
seed = lambda x: int(x["dir"].rsplit("_s", 1)[1])


def late(run, k, key):
    v = [x[key] for x in run["rounds"] if x["problem"] == k and x["round"] >= 30 and x.get(key) is not None]
    return np.mean(v) if v else np.nan


def early(run, k, key, lo=0, hi=10):
    v = [x[key] for x in run["rounds"] if x["problem"] == k and lo <= x["round"] < hi and x.get(key) is not None]
    return np.mean(v) if v else np.nan


def fit(run, k):
    return np.mean([x["fitness"] for x in run["rounds"] if x["problem"] == k])


def ci(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 2.571 * se, d.mean() + 2.571 * se, stats.ttest_1samp(d, 0).pvalue


for rho, root, oroot in (("1", "results/p3b", "results/p3_oracle"), ("0.8", "results/p3b_r08", "results/p3_oracle_r08")):
    r = load(root)
    o = load(oroot)
    R = {a: {seed(x): x for x in r[a]} for a in ARMS if a in r}
    for a in ("none", "sbp", "pcc", "gated"):
        R[a] = {seed(x): x for x in o[a] if seed(x) < 6}
    S = range(6)
    print(f"\n===== rho {rho}: (b) with context, 6 seeds =====")
    print("collapse late in each problem: share of askers on the top answer | distinct of 8")
    for a in ARMS:
        print(f"  {a:11s} " + "  ".join(f"p{k+1} {np.mean([late(R[a][s], k, 'top_share') for s in S]):.2f} | "
                                       f"{np.mean([late(R[a][s], k, 'group_distinct') for s in S]):.1f}" for k in range(4)))
    print("early in problem 1 (rounds 0-9): answers that improve the asker | top share | community modules at round 10")
    for a in ARMS:
        mods = np.mean([[x for x in R[a][s]["rounds"] if x["problem"] == 0][10]["modules"] for s in S])
        print(f"  {a:11s} improve {np.mean([early(R[a][s], 0, 'improve_shown') for s in S]):.2f}  "
              f"top {np.mean([early(R[a][s], 0, 'top_share') for s in S]):.2f}  modules@10 {mods:.1f}")
    for a in ("none", "sbp", "pcc", "gated"):
        mods = np.mean([[x for x in R[a][s]["rounds"] if x["problem"] == 0][10]["modules"] for s in S])
        print(f"  {a:11s} modules@10 {mods:.1f}")
    print("distinct of 8, mean over problems, paired by seed:")
    for a, b in (("thumbs", "pairwise"), ("thumbskl", "thumbs"), ("pairwisekl", "pairwise"), ("thumbs", "now"), ("pairwise", "now")):
        d = [np.mean([late(R[a][s], k, "group_distinct") - late(R[b][s], k, "group_distinct") for k in range(4)]) for s in S]
        m, lo, hi, p = ci(d)
        print(f"  {a} - {b}: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] p = {p:.3f}")
    print("fitness while assisted, % of no-AI mean, paired by seed: problem 1 | problems 2-4")
    base1 = np.mean([fit(R["none"][s], 0) for s in S])
    base24 = np.mean([np.mean([fit(R["none"][s], k) for k in (1, 2, 3)]) for s in S])
    for a in ARMS + ["sbp", "pcc", "gated"]:
        d1 = [100 * (fit(R[a][s], 0) - fit(R["none"][s], 0)) / base1 for s in S]
        d24 = [100 * np.mean([fit(R[a][s], k) - fit(R["none"][s], k) for k in (1, 2, 3)]) / base24 for s in S]
        m1, m2 = ci(d1), ci(d24)
        print(f"  {a:11s} p1 {m1[0]:+6.1f} [{m1[1]:+.1f}, {m1[2]:+.1f}]   p2-4 {m2[0]:+6.1f} [{m2[1]:+.1f}, {m2[2]:+.1f}]")
