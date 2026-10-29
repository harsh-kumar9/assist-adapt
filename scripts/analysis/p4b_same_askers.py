"""P4b: does the trained LLM beat random proposals on the very requests it answered?

For each logged request (requests.jsonl, epochs 3-20) the share of the LLM's shown answers that helped, against
the exact chance that a random proposal helps the same asker on that epoch's landscape: rnd3 (flip 1-3 choices,
count uniform), exactly 3 flips, and a uniformly random module (rndfull). Rebuilds each seed's landscapes from
its config. Only runs that log requests (dup, crowd, rate and later).

  python scripts/analysis/p4b_same_askers.py dupE3T2 crowdT2
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.nk import Landscape  # noqa: E402

ROOT, BURN, SEEDS = "results/p4b_s05", 2, range(6)
N, K, M = 20, 9, 10
MC = (1 << M) - 1
FLIPS = {k: np.array([sum(1 << b for b in c) for c in itertools.combinations(range(M), k)]) for k in (1, 2, 3)}


def landscapes(cfg):
    lands = [Landscape(N, K, M, cfg["rho"], np.random.default_rng([cfg["seed"], int(round(cfg["rho"] * 1000)), 0]))]
    srng = np.random.default_rng([cfg["seed"], int(round(cfg["rho"] * 1000)), 3])
    for _ in range(cfg["epochs"]):
        lands.append(lands[-1].shifted(cfg["shift"], srng))
    return lands


def help_rates(d):
    """Mean over epochs of: LLM, rnd3, exactly 3 flips, rndfull help rate on the run's own requests."""
    cfg = json.load(open(os.path.join(d, "config.json")))
    lands, out = landscapes(cfg), []
    for line in open(os.path.join(d, "requests.jsonl")):
        r = json.loads(line)
        if r["epoch"] <= BURN or not r.get("code"):
            continue
        F = lands[r["epoch"]].F
        code, ans = np.array(r["code"], dtype=np.int64), np.array(r["answer"], dtype=np.int64)
        fa = F[code]
        llm = np.where(ans >= 0, F[(code & ~MC) | np.maximum(ans, 0)] > fa, False).mean()
        flip = {k: (F[code[:, None] ^ FLIPS[k][None, :]] > fa[:, None]).mean(1) for k in FLIPS}
        full = (F[(code[:, None] & ~MC) | np.arange(1 << M)[None, :]] > fa[:, None]).mean(1)
        out.append((llm, np.mean(sum(flip.values()) / 3), flip[3].mean(), full.mean()))
    return np.mean(out, 0)


for arm in sys.argv[1:]:
    a = np.array([help_rates(os.path.join(ROOT, f"{arm}_s{s}")) for s in SEEDS])
    print(f"{arm}: help rate on its own requests, LLM {a[:, 0].mean():.4f} | rnd3 {a[:, 1].mean():.4f} | "
          f"exactly 3 flips {a[:, 2].mean():.4f} | rndfull {a[:, 3].mean():.4f}")
    for j, name in ((1, "rnd3"), (3, "rndfull")):
        diff = a[:, 0] - a[:, j]
        print(f"  LLM - {name} per seed {np.round(diff, 4)}; mean {diff.mean():+.4f}")
