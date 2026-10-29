"""P4b: what could let an assistant beat random edits? Structure of the module landscape, and the help rate
of proposals built from each information source, on the requests one run actually served.

Structure (per seed, epochs 3-19): fitness correlation between a module and its 1- and 2-flip neighbours (fixed
rest), between the same module under two different rests, and between the same module before and after a shift.

Help rate on the run's own requests (open loop: the culture is the one the run produced), each proposal kept
only if it helps, as in the sim:
  random edit          1-3 flips, count uniform (rnd3)
  last epoch's log     a random module that helped some team last epoch, if within 3 flips; else a random edit
  perfect last epoch   the best edit of at most 3 flips on last epoch's problem
  this epoch's log     the most recent module that helped some team earlier this epoch, if within 3 flips; else
                       a random edit (also: a random one of them)
  shared edit          the most recent edit that helped some team this epoch, applied to this team's module
  ceiling              some edit of at most 3 flips helps

  python scripts/analysis/p4b_memory_sources.py [run, default dupE3T2]
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.nk import Landscape  # noqa: E402

ROOT, BURN, SEEDS = "results/p4b_s05", 2, range(6)
ARM = sys.argv[1] if len(sys.argv) > 1 else "dupE3T2"
N, K, M = 20, 9, 10
MC = (1 << M) - 1
ALL = np.arange(1 << M)
POP = np.array([bin(x).count("1") for x in range(1 << M)])
FLIPS = {k: np.array([sum(1 << b for b in c) for c in itertools.combinations(range(M), k)]) for k in (1, 2, 3)}
R3 = np.concatenate(list(FLIPS.values()))


def landscapes(cfg):
    lands = [Landscape(N, K, M, cfg["rho"], np.random.default_rng([cfg["seed"], int(round(cfg["rho"] * 1000)), 0]))]
    srng = np.random.default_rng([cfg["seed"], int(round(cfg["rho"] * 1000)), 3])
    for _ in range(cfg["epochs"]):
        lands.append(lands[-1].shifted(cfg["shift"], srng))
    return lands


corr = {k: [] for k in ("1 flip", "2 flips", "other rest", "next epoch")}
rate = {k: [] for k in ("random edit", "last epoch's log", "perfect last epoch", "this epoch's log, most recent",
                        "this epoch's log, any", "shared edit", "ceiling")}
covered = []
for s in SEEDS:
    d = os.path.join(ROOT, f"{ARM}_s{s}")
    lands = landscapes(json.load(open(os.path.join(d, "config.json"))))
    rng = np.random.default_rng(s)
    logs = [json.loads(line) for line in open(os.path.join(d, "requests.jsonl"))]
    for r, prev in zip(logs[BURN + 1:], logs[BURN:]):
        e = r["epoch"]
        F, Fp = lands[e].F, lands[e - 1].F
        rests = rng.integers(0, 1 << (N - M), 2) << M
        a, b = F[rests[0] | ALL], F[rests[1] | ALL]
        corr["1 flip"].append(np.mean([np.corrcoef(a, a[ALL ^ m])[0, 1] for m in FLIPS[1]]))
        corr["2 flips"].append(np.mean([np.corrcoef(a, a[ALL ^ m])[0, 1] for m in FLIPS[2]]))
        corr["other rest"].append(np.corrcoef(a, b)[0, 1])
        corr["next epoch"].append(np.corrcoef(a, Fp[rests[0] | ALL])[0, 1])

        code = np.array(r["code"], dtype=np.int64)
        fa = F[code]
        cand = code[:, None] ^ R3[None, :]
        rate["random edit"].append(np.mean(sum((F[code[:, None] ^ FLIPS[k][None, :]] > fa[:, None]).mean(1)
                                               for k in FLIPS) / 3))
        rate["ceiling"].append((F[cand] > fa[:, None]).any(1).mean())
        rate["perfect last epoch"].append((F[cand[np.arange(len(code)), np.argmax(Fp[cand], 1)]] > fa).mean())
        last = [int(x) for x, h in zip(prev["answer"], prev["helped"]) if h and x >= 0]
        seen, edits = [], []
        hits = {k: [] for k in ("last epoch's log", "this epoch's log, most recent", "this epoch's log, any", "shared edit")}
        for j in range(len(code)):
            m, rest = int(code[j]) & MC, int(code[j]) & ~MC
            rnd = m ^ int(R3[rng.integers(len(R3))])
            near = [x for x in seen if 0 < POP[x ^ m] <= 3]
            near_last = [x for x in last if 0 < POP[x ^ m] <= 3]
            props = {"last epoch's log": near_last[rng.integers(len(near_last))] if near_last else rnd,
                     "this epoch's log, most recent": near[-1] if near else rnd,
                     "this epoch's log, any": near[rng.integers(len(near))] if near else rnd,
                     "shared edit": m ^ edits[-1] if edits else rnd}
            for k, p in props.items():
                hits[k].append(F[rest | p] > fa[j])
            covered.append(bool(near))
            if r["helped"][j] and r["answer"][j] >= 0:
                seen.append(int(r["answer"][j]))
                edits.append(int(r["answer"][j]) ^ m)
        for k, v in hits.items():
            rate[k].append(np.mean(v))

print(f"{ARM}, seeds {SEEDS.start}-{SEEDS.stop - 1}, epochs {BURN + 1}-{len(logs) - 1}")
print("fitness correlation of a module with: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in corr.items()))
for k, v in rate.items():
    print(f"help rate, {k:30s} {np.mean(v):.4f}")
print(f"requests with a module that helped someone earlier this epoch within 3 flips: {np.mean(covered):.2f}")
