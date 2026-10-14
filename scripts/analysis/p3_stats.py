import json, numpy as np, sys
from scipy import stats
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from src.llm.analyze_p3 import load

def runs_for(root, oracle_root):
    r = load(root); o = load(oracle_root)
    seed = lambda x: int(x["dir"].rsplit("_s", 1)[1])
    out = {a: {seed(x): x for x in r[a]} for a in ("now", "kl", "stepback")}
    for a in ("none", "sbp", "pcc", "gated"):
        out[a] = {seed(x): x for x in o[a] if seed(x) < 6}
    return out

def fit(run, k):
    return np.mean([x["fitness"] for x in run["rounds"] if x["problem"] == k])

def probe(run, k):
    return next(p["mean"] for p in run["probes"] if p["problem"] == k and p["landscape"] == "next")

def late(run, k, key):
    v = [x[key] for x in run["rounds"] if x["problem"] == k and x["round"] >= 30 and x.get(key) is not None]
    return np.mean(v)

def ci(d):
    d = np.asarray(d); se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 2.571 * se, d.mean() + 2.571 * se, stats.ttest_1samp(d, 0).pvalue  # t(5)

for rho, root, oroot in (("1", "results/p3", "results/p3_oracle"), ("0.8", "results/p3_r08", "results/p3_oracle_r08")):
    R = runs_for(root, oroot); S = range(6)
    print(f"\n===== rho {rho}, 6 seeds, t(5) 95% CIs =====")
    print("collapse, late in each problem (mean over seeds [min, max]): share of askers on the top answer | distinct of 8")
    for a in ("now", "kl", "stepback"):
        ts = [[late(R[a][s], k, "top_share") for s in S] for k in range(4)]
        gd = [[late(R[a][s], k, "group_distinct") for s in S] for k in range(4)]
        print(f"  {a:9s} " + "  ".join(f"p{k+1} {np.mean(ts[k]):.2f} | {np.mean(gd[k]):.1f}" for k in range(4)))
    # paired arm differences in group_distinct pooled over problems
    for a, b in (("kl", "now"), ("kl", "stepback"), ("stepback", "now")):
        d = [np.mean([late(R[a][s], k, "group_distinct") - late(R[b][s], k, "group_distinct") for k in range(4)]) for s in S]
        m, lo, hi, p = ci(d); print(f"  distinct-of-8, {a} - {b}, mean over problems: {m:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3f}")
    print("fitness while assisted, % of no-AI mean (paired by seed): problem 1 | problems 2-4 pooled")
    for a in ("now", "kl", "stepback", "sbp", "gated"):
        base1 = np.mean([fit(R["none"][s], 0) for s in S]); base24 = np.mean([np.mean([fit(R["none"][s], k) for k in (1,2,3)]) for s in S])
        d1 = [100 * (fit(R[a][s], 0) - fit(R["none"][s], 0)) / base1 for s in S]
        d24 = [100 * np.mean([fit(R[a][s], k) - fit(R["none"][s], k) for k in (1,2,3)]) / base24 for s in S]
        m1 = ci(d1); m2 = ci(d24)
        print(f"  {a:9s} p1 {m1[0]:+6.1f} [{m1[1]:+.1f}, {m1[2]:+.1f}]   p2-4 {m2[0]:+6.1f} [{m2[1]:+.1f}, {m2[2]:+.1f}]")
    print("next problem with AI removed, % of no-AI mean, pooled over the 4 probes")
    for a in ("now", "kl", "stepback", "sbp", "gated"):
        base = np.mean([np.mean([probe(R["none"][s], k) for k in range(4)]) for s in S])
        d = [100 * np.mean([probe(R[a][s], k) - probe(R["none"][s], k) for k in range(4)]) / base for s in S]
        m = ci(d); print(f"  {a:9s} {m[0]:+6.1f} [{m[1]:+.1f}, {m[2]:+.1f}]")
    print("community diversity at end of problem 1 (Hamming) and distinct modules: ")
    for a in ("none", "now", "kl", "stepback", "sbp", "gated"):
        dv = [ [x for x in R[a][s]["rounds"] if x["problem"] == 0][-1]["diversity"] for s in S]
        md = [ [x for x in R[a][s]["rounds"] if x["problem"] == 0][29]["diversity"] for s in S]
        print(f"  {a:9s} round 30: {np.mean(md):.2f}   round 60: {np.mean(dv):.2f}")
