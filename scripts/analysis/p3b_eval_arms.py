"""(b) frozen evaluation per arm and checkpoint: arm - oracle nudge (SBP) and arm - no AI, % of the
no-AI mean, paired by training seed (every seed that has the arm), t CIs over seeds."""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

D = sys.argv[1] if len(sys.argv) > 1 else "results/p3eval/b"
ARMS = ["base", "now", "thumbs", "thumbskl", "pairwise", "pairwisekl", "pairwisefix"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return "n/a"
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return f"{x.mean():+6.1f} [{x.mean() - t * se:+.1f}, {x.mean() + t * se:+.1f}] p={stats.ttest_1samp(x, 0).pvalue:.3f}"


res = {}
for f in glob.glob(os.path.join(D, "rho*_s*_p*_*.json")):
    m = re.match(r"rho([\d.]+)_s(\d+)_p(\d+)_", os.path.basename(f))
    res[(float(m.group(1)), int(m.group(2)), int(m.group(3)))] = json.load(open(f))
rows = []
for rho in sorted({k[0] for k in res}, reverse=True):
    for k in sorted({k[2] for k in res}):
        print(f"\nrho {rho:g}, checkpoint after problem {k + 1}")
        for a in ARMS:
            seeds = sorted(s for (r, s, kk) in res if r == rho and kk == k and a in res[(r, s, kk)])
            if len(seeds) < 2:
                continue
            g = lambda s, x, key: np.mean(res[(rho, s, k)][x][key])
            out = {}
            for key in ("during", "next_off"):
                base = np.mean([g(s, "none", key) for s in seeds])
                out[key + "_vs_sbp"] = [100 * (g(s, a, key) - g(s, "sbp", key)) / base for s in seeds]
                out[key + "_vs_none"] = [100 * (g(s, a, key) - g(s, "none", key)) / base for s in seeds]
            div = np.mean([g(s, a, "div_T") for s in seeds])
            print(f"  {a:12s} n={len(seeds):2d}  vs nudge: now {ci(out['during_vs_sbp'])} | next {ci(out['next_off_vs_sbp'])}"
                  f"   vs no AI next {ci(out['next_off_vs_none'])}  div {div:.2f}")
            rows.append(dict(rho=rho, checkpoint=k + 1, arm=a, n=len(seeds),
                             **{key: float(np.mean(v)) for key, v in out.items()}, div_T=float(div)))
os.makedirs("results/summary", exist_ok=True)
with open("results/summary/p3b_eval_arms.csv", "w") as f:
    keys = list(rows[0].keys())
    f.write(",".join(keys) + "\n")
    for r in rows:
        f.write(",".join(f"{r[x]:.4g}" if isinstance(r[x], float) else str(r[x]) for x in keys) + "\n")
