import numpy as np, sys
from scipy import stats
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from src.llm.analyze_p3 import load
seed = lambda x: int(x["dir"].rsplit("_s", 1)[1])
for rho, root in (("1", "results/p3"), ("0.8", "results/p3_r08")):
    R = load(root); R = {a: {seed(x): x for x in R[a]} for a in ("now", "kl", "stepback")}
    print(f"\nrho {rho}: share of the oracle nudge's available gain the model's shown answers capture, rounds 0-14 of each problem")
    vals = {}
    for a in ("now", "kl", "stepback"):
        per_seed = {}
        for window, ks in (("problem 1", [0]), ("after shifts (2-4)", [1, 2, 3])):
            v = []
            for s in range(6):
                rows = [x for x in R[a][s]["rounds"] if x["problem"] in ks and x["round"] < 15]
                g = sum(x.get("gain_shown") or 0 for x in rows); o = sum(x.get("sbp_gain") or 0 for x in rows)
                v.append(g / o if o > 0 else np.nan)
            per_seed[window] = np.array(v)
        vals[a] = per_seed
        print(f"  {a:9s} " + "   ".join(f"{w}: {np.nanmean(v):.2f} (seeds {' '.join(f'{x:.2f}' for x in v)})" for w, v in per_seed.items()))
    for a, b in (("kl", "now"), ("kl", "stepback")):
        d = vals[a]["after shifts (2-4)"] - vals[b]["after shifts (2-4)"]
        d = d[~np.isnan(d)]
        print(f"  {a} - {b}, after shifts: {d.mean():+.2f}, p = {stats.ttest_1samp(d, 0).pvalue:.3f} (n={len(d)}); wilcoxon p = {stats.wilcoxon(d).pvalue:.3f}")
