"""Sweep runner. One work unit = one (rho, seed): build the landscape once, then run every
assistant x use-rate cell on it. Writes one .npz per shard.

  python -m src.run --name p0 --seeds 0:200 --rhos 0.6,0.7,0.8,0.9,1.0 \
      --assistants np,tdr,sbp --uses 0.125,0.25,0.375,0.5,0.625,0.75,0.875

Probes (P1): --probe-times 10,30,60,100 --shifts 0,0.2 --post-rounds 30
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from src.nk import Landscape
from src.sim import Community, run


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", default="0:200", help="start:stop")
    p.add_argument("--rhos", default="0.6,0.8,1.0")
    p.add_argument("--assistants", default="np,tdr,sbp")
    p.add_argument("--uses", default="0.25,0.5,0.75")
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--K", type=int, default=9)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--H", type=int, default=10)
    p.add_argument("--agents", type=int, default=100)
    p.add_argument("--p-edge", type=float, default=0.05)
    p.add_argument("--p-social", type=float, default=0.1)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--probe-times", default="")
    p.add_argument("--shifts", default="0,0.2")
    p.add_argument("--post-rounds", type=int, default=30)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    return p.parse_args(argv)


def floats(s):
    return [float(x) for x in s.split(",") if x != ""]


def cells(args):
    """('none', 0) once, then every assistant x use rate."""
    out = [("none", 0.0)]
    for a in args.assistants.split(","):
        for u in floats(args.uses):
            out.append((a, u))
    return out


def unit(job):
    args, rho, seed = job
    land = Landscape(args.N, args.K, args.M, rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))
    comm = Community(args.agents, args.N, args.H, args.p_edge, np.random.default_rng([seed, 2]))
    probe_times = [int(t) for t in args.probe_times.split(",") if t]
    probe_lands = []
    if probe_times:
        srng = np.random.default_rng([seed, int(round(rho * 1000)), 3])
        probe_lands = [land.shifted(s, srng) for s in floats(args.shifts)]
    res = [run(land, comm, a, u, args.p_social, args.rounds, seed,
               probe_times, probe_lands, args.post_rounds) for a, u in cells(args)]
    return {k: np.stack([r[k] for r in res]) for k in res[0]}


def main(argv=None):
    args = parse_args(argv)
    lo, hi = (int(x) for x in args.seeds.split(":"))
    rhos = floats(args.rhos)
    jobs = [(args, rho, seed) for rho in rhos for seed in range(lo, hi)]
    jobs = jobs[args.shard::args.nshards]
    os.makedirs(os.path.join(args.out, args.name), exist_ok=True)
    path = os.path.join(args.out, args.name, f"shard{args.shard:03d}of{args.nshards:03d}.npz")
    if os.path.exists(path):
        print("exists, skipping", path)
        return
    t0 = time.time()
    outs = []
    with Pool(args.workers) as pool:
        for o in pool.imap(unit, jobs, chunksize=max(1, len(jobs) // (20 * args.workers))):
            outs.append(o)
            if len(outs) % max(1, len(jobs) // 20) == 0:
                print(f"{len(outs)}/{len(jobs)} units, {time.time() - t0:.0f}s", flush=True)
    keys = outs[0].keys()
    np.savez_compressed(
        path + ".tmp.npz",
        rho=np.array([j[1] for j in jobs]), seed=np.array([j[2] for j in jobs]),
        **{k: np.stack([o[k] for o in outs]) for k in keys})
    os.replace(path + ".tmp.npz", path)
    meta = dict(vars(args), cells=cells(args), n_units=len(jobs), seconds=round(time.time() - t0, 1))
    with open(os.path.join(args.out, args.name, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {path}: {len(jobs)} units in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
