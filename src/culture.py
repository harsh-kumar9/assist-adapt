"""P4a: culture over time with oracle assistants.

The paper's model (Fazelpour, O'Brien & Rubin) with two additions:
  turnover: each round each team leaves with probability mu and is replaced by a newcomer with a random
            approach and a random specialization, in the same network position;
  epochs:   every `rounds` rounds the problem changes (20% of payoff tables redrawn), as in P3.
Oracle assistants do not learn, so this is the reference for the learning assistants of P4b.

  python -m src.culture --out results/p4a --rhos 1.0,0.8 --mus 0,0.005,0.01,0.02 --seeds 0:200
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, propose, step

N, K, M, H = 20, 9, 10, 10
MC = np.int64((1 << M) - 1)


def advise_gated(codes, fit, F, idx, u, explore):
    """Simulation 2's rule: help with whichever of full answer / nudge gains more, else step back."""
    pcc = propose("pcc", codes, fit, F, M, idx, u)
    sbp = propose("sbp", codes, fit, F, M, idx, u)
    gp, gs = F[pcc] - fit[idx], F[sbp] - fit[idx]
    return np.where(np.maximum(gp, gs) > 0, np.where(gp >= gs, pcc, sbp), explore[idx])


def round_with(codes, F, comm, assistant, use, p_social, rng):
    if assistant != "gated":
        return step(codes, F, comm, assistant, use, p_social, M, rng)
    n = len(codes)
    u_social, u_query, u_ai = rng.random(n), rng.random(n), rng.random(n)
    col = rng.integers(0, comm.H, n)
    fit = F[codes]
    social = u_social < p_social
    query = ~social & (u_query < use)
    prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])
    fit_pad = np.append(fit, -np.inf)
    best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
    prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
    idx = np.flatnonzero(query)
    if len(idx):
        prop[idx] = advise_gated(codes, fit, F, idx, u_ai[idx], prop)
    return np.where(F[prop] > fit, prop, codes)


def run(job):
    rho, seed, assistant, mu, epochs, rounds, shift, use = job
    lands = [Landscape(N, K, M, rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))]
    srng = np.random.default_rng([seed, int(round(rho * 1000)), 3])
    for _ in range(epochs - 1):
        lands.append(lands[-1].shifted(shift, srng))
    comm = Community(100, N, H, 0.05, np.random.default_rng([seed, 2]))
    comm.spec = comm.spec.copy()
    rng = np.random.default_rng([seed, 1])
    trng = np.random.default_rng([seed, 11])  # turnover draws, separate so other draws stay paired
    codes = comm.codes0.copy()
    T = epochs * rounds
    fit, div, mods = np.zeros(T, np.float32), np.zeros(T, np.float32), np.zeros(T, np.int16)
    for e in range(epochs):
        F = lands[e].F
        for t in range(rounds):
            codes = round_with(codes, F, comm, assistant if assistant != "none" else "none", use, 0.1, rng)
            leave = np.flatnonzero(trng.random(len(codes)) < mu)
            if len(leave):
                codes[leave] = trng.integers(0, 2 ** N, len(leave), dtype=np.int64)
                for i in leave:
                    comm.spec[i] = trng.choice(N, H, replace=False)
            k = e * rounds + t
            fit[k], div[k] = F[codes].mean(), mean_pairwise_hamming(codes, N)
            mods[k] = len(np.unique(codes & MC))
    return dict(fit=fit, div=div, mods=mods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rhos", default="1.0,0.8")
    ap.add_argument("--mus", default="0,0.005,0.01,0.02")
    ap.add_argument("--assistants", default="none,sbp,pcc,gated")
    ap.add_argument("--seeds", default="0:200")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--shift", type=float, default=0.2)
    ap.add_argument("--use", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split(":"))
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for rho in [float(x) for x in args.rhos.split(",")]:
            for mu in [float(x) for x in args.mus.split(",")]:
                for a in args.assistants.split(","):
                    path = os.path.join(args.out, f"rho{rho:g}_mu{mu:g}_{a}.npz")
                    if os.path.exists(path):
                        continue
                    jobs = [(rho, s, a, mu, args.epochs, args.rounds, args.shift, args.use) for s in range(lo, hi)]
                    res = pool.map(run, jobs, chunksize=4)
                    np.savez_compressed(path, seeds=np.arange(lo, hi),
                                        **{k: np.stack([r[k] for r in res]) for k in res[0]})
                    print(f"{path} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
