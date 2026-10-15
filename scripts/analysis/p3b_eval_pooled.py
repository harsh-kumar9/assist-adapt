"""(b) frozen evaluation: trained LLM arms pooled (mean over the 5 arms within each seed), vs no AI
and vs the oracle nudge (SBP), on the same communities. t(5) over training seeds."""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.llm.analyze_eval import load  # noqa: E402

ARMS = ["now", "thumbs", "thumbskl", "pairwise", "pairwisekl"]


def ci(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return f"{x.mean():+6.1f} [{x.mean() - t * se:+.1f}, {x.mean() + t * se:+.1f}] p={stats.ttest_1samp(x, 0).pvalue:.3f}"


for ck in [c for c in ("b_p0", "b_p1", "b_p2", "b_p3") if os.path.isdir(f"results/p3eval/{c}")]:
    res = load(f"results/p3eval/{ck}")
    for rho in sorted({k[0] for k in res}):
        seeds = sorted(s for r, s in res if r == rho)
        print(f"\n{ck} rho {rho:g} (n={len(seeds)} seeds; checkpoint after problem {int(ck[-1]) + 1})")
        for key, lab in (("during", "while assisted"), ("next_off", "next problem, AI removed")):
            m = lambda s, a: np.mean(res[(rho, s)][a][key])
            none = [m(s, "none") for s in seeds]
            llm = [np.mean([m(s, a) for a in ARMS if a in res[(rho, s)]]) for s in seeds]
            sbp = [m(s, "sbp") for s in seeds]
            base = [m(s, "base") for s in seeds]
            pct = lambda xs: [100 * (x - n) / np.mean(none) for x, n in zip(xs, none)]
            print(f"  {lab:26s} trained LLM vs no AI {ci(pct(llm))} | untrained vs no AI {ci(pct(base))} | "
                  f"trained LLM - SBP {ci([100 * (l - b) / np.mean(none) for l, b in zip(llm, sbp)])}")
        dv = lambda a: np.mean([np.mean(res[(rho, s)][a]["div_T"]) for s in seeds])
        print(f"  diversity after 30 rounds: none {dv('none'):.2f} sbp {dv('sbp'):.2f} pcc {dv('pcc'):.2f} "
              f"base {dv('base'):.2f} trained LLM {np.mean([dv(a) for a in ARMS if all(a in res[(rho, s)] for s in seeds)]):.2f}")
