"""P2 (D-011): learn when to give a full answer, a nudge, or nothing.

Each time an agent asks, the assistant picks one action:
  full  = PCC, the peer computational module that is best in this agent's context
  nudge = SBP, the best single flip in the computational module
  back  = no suggestion; the agent explores on its own this round
The policy is a softmax over a linear score of a few features ("back" is the zero logit).
We fit it with an evolution strategy on one of three episode rewards:
  now   = mean fitness over the T assisted rounds
  long  = now + mean fitness over the next 30 rounds, same problem, assistant still on
  adapt = now + mean fitness over the next 30 rounds after a shift, assistant gone

Random draws match src/sim.py exactly, so "always nudge" reproduces the SBP runs and
"always full" the PCC runs from P1b on the same seeds.

  python -m src.p2 train --reward adapt --out results/p2
  python -m src.p2 eval --out results/p2
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from src.nk import Landscape
from src.sim import Community, mean_pairwise_hamming, propose_pcc, propose_single_flip

FEATURES = ("bias", "round", "diversity", "rank", "gain_full", "gain_nudge", "popularity_full")
N_FEAT = len(FEATURES)
BIG = 50.0


# ---------------------------------------------------------------- policies

class Linear:
    """theta (2, N_FEAT): logits for full and nudge; back has logit 0."""

    def __init__(self, theta, name="learned"):
        self.theta = np.asarray(theta, float).reshape(2, N_FEAT)
        self.name = name

    def act(self, X, u, t_frac):
        logits = np.c_[X @ self.theta.T, np.zeros(len(X))]
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        return (u[:, None] > np.cumsum(p, 1)).sum(1).clip(0, 2)


class Mix:
    """Full answer with probability p, else nudge. p = 1 is PCC, p = 0 is SBP."""

    def __init__(self, p, name=None):
        self.p, self.name = p, name or f"mix{p:g}"

    def act(self, X, u, t_frac):
        return np.where(u < self.p, 0, 1)


class Schedule:
    """Full answer for the first `frac` of the assisted rounds, then nudges."""

    def __init__(self, frac, name=None):
        self.frac, self.name = frac, name or f"full_first{frac:g}"

    def act(self, X, u, t_frac):
        return np.full(len(X), 0 if t_frac < self.frac else 1)


def rules():
    return [Mix(1.0, "always_full"), Mix(0.0, "always_nudge"), Mix(0.75), Mix(0.5), Mix(0.25),
            Schedule(1 / 6), Schedule(1 / 3), Schedule(2 / 3)]


# ---------------------------------------------------------------- dynamics

def features(codes, fit, F, M, idx, full, nudge, t_frac, div):
    mc = np.int64((1 << M) - 1)
    rank = (fit[idx][:, None] > fit[None, :]).mean(1)
    pop = ((codes & mc)[None, :] == (full & mc)[:, None]).mean(1)
    n = len(idx)
    return np.stack([np.ones(n), np.full(n, t_frac), np.full(n, div / 10.0), rank,
                     10 * (F[full] - fit[idx]), 10 * (F[nudge] - fit[idx]), pop], 1)


def step(codes, F, comm, policy, use_rate, p_social, M, rng, t_frac, div):
    """Same draws and order as src.sim.step. Returns new codes and action counts."""
    n = len(codes)
    u_social, u_query, u_ai = rng.random(n), rng.random(n), rng.random(n)
    col = rng.integers(0, comm.H, n)
    fit = F[codes]
    social = u_social < p_social
    query = ~social & (u_query < use_rate) if policy is not None else np.zeros(n, bool)
    prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])
    fit_pad = np.append(fit, -np.inf)
    best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
    prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
    counts = np.zeros(3, np.int64)
    if query.any():
        idx = np.flatnonzero(query)
        mc = np.int64((1 << M) - 1)
        full = propose_pcc(codes, fit, F, mc, idx)
        nudge = propose_single_flip(codes, fit, F, M, idx, u_ai[idx], "sbp", 1)
        a = policy.act(features(codes, fit, F, M, idx, full, nudge, t_frac, div), u_ai[idx], t_frac)
        prop[idx] = np.where(a == 0, full, np.where(a == 1, nudge, prop[idx]))
        counts = np.bincount(a, minlength=3)
    return np.where(F[prop] > fit, prop, codes), counts


def episode(land, shifted, comm, policy, use, p_social, T, post, seed):
    M, N = land.M, land.N
    rng = np.random.default_rng([seed, 1])
    codes = comm.codes0.copy()
    div = mean_pairwise_hamming(codes, N)
    during, acts = [], np.zeros((T, 3))
    for t in range(1, T + 1):
        codes, acts[t - 1] = step(codes, land.F, comm, policy, use, p_social, M, rng, (t - 1) / T, div)
        during.append(land.F[codes].mean())
        div = mean_pairwise_hamming(codes, N)
    out = dict(now=float(np.mean(during)), div_T=div, acts=acts)
    for key, pl, li, on in (("same_on", land, 0, 1), ("shift_off", shifted, 1, 0)):
        prng = np.random.default_rng([seed, 7, T, li, on if policy is not None else 0])  # as in sim.run
        pc, d, vals = codes.copy(), div, []
        for _ in range(post):
            pc, _ = step(pc, pl.F, comm, policy if on else None, use, p_social, M, prng, 1.0, d)
            vals.append(pl.F[pc].mean())
            d = mean_pairwise_hamming(pc, N)
        out[key] = float(np.mean(vals))
    return out


def reward(ep, kind):
    return ep["now"] + {"now": 0.0, "long": ep["same_on"], "adapt": ep["shift_off"]}[kind]


# ---------------------------------------------------------------- world

CFG = dict(N=20, K=9, M=10, H=10, agents=100, p_edge=0.05, p_social=0.1, T=30, post=30)


def world(seed, rho, shift, K=None):
    c = CFG
    K = K or c["K"]
    land = Landscape(c["N"], K, c["M"], rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))
    shifted = land.shifted(shift, np.random.default_rng([seed, int(round(rho * 1000)), 3]))
    comm = Community(c["agents"], c["N"], c["H"], c["p_edge"], np.random.default_rng([seed, 2]))
    return land, shifted, comm


def eval_unit(job):
    """All policies on one seed. Returns {policy name: episode dict}."""
    seed, rho, shift, use, policies = job
    land, shifted, comm = world(seed, rho, shift)
    out = {}
    for pol in [None] + list(policies):
        ep = episode(land, shifted, comm, pol, use, CFG["p_social"], CFG["T"], CFG["post"], seed)
        out["none" if pol is None else pol.name] = ep
    return out


def es_unit(job):
    seed, rho, use, thetas, kind = job
    land, shifted, comm = world(seed, rho, 0.2)
    return np.array([reward(episode(land, shifted, comm, Linear(th), use, CFG["p_social"],
                                    CFG["T"], CFG["post"], seed), kind) for th in thetas])


# ---------------------------------------------------------------- training

def train(args):
    rng = np.random.default_rng(args.es_seed)
    theta = np.zeros(2 * N_FEAT)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    rhos = [float(r) for r in args.rhos.split(",")]
    log = []
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"es_{args.reward}_s{args.es_seed}.json")
    t0 = time.time()
    with Pool(args.workers) as pool:
        for g in range(args.gens):
            eps = rng.standard_normal((args.pairs, theta.size))
            thetas = np.concatenate([theta + args.sigma * eps, theta - args.sigma * eps])
            seeds = rng.choice(args.train_seeds, args.seeds_per_gen, replace=False)
            jobs = [(int(s), float(rng.choice(rhos)), args.use, thetas, args.reward) for s in seeds]
            R = np.mean(pool.map(es_unit, jobs), 0)
            ranks = np.empty(len(R))
            ranks[np.argsort(R)] = np.arange(len(R))
            shaped = ranks / (len(R) - 1) - 0.5
            grad = (shaped[:args.pairs] - shaped[args.pairs:]) @ eps / (args.pairs * args.sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            theta = theta + args.lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            log.append(dict(gen=g, mean_R=float(R.mean()), max_R=float(R.max()), seconds=round(time.time() - t0)))
            if g % 10 == 0 or g == args.gens - 1:
                print(f"gen {g} mean R {R.mean():.4f} max {R.max():.4f} {time.time() - t0:.0f}s", flush=True)
                json.dump(dict(config=vars(args), theta=theta.tolist(), features=FEATURES, log=log),
                          open(path + ".tmp", "w"), indent=1)
                os.replace(path + ".tmp", path)
    print("wrote", path)


# ---------------------------------------------------------------- evaluation

def evaluate(args):
    pols = rules()
    for f in sorted(os.listdir(args.out)):
        if f.startswith("es_") and f.endswith(".json"):
            d = json.load(open(os.path.join(args.out, f)))
            pols.append(Linear(d["theta"], name=f[3:-5]))
    rhos = [float(r) for r in args.rhos.split(",")]
    settings = [(rho, 0.2, args.use) for rho in rhos] + [(rho, 0.5, args.use) for rho in rhos] + \
               [(rho, 0.2, 0.75) for rho in rhos]
    lo, hi = (int(x) for x in args.test_seeds.split(":"))
    jobs = [(s, rho, shift, use, pols) for rho, shift, use in settings for s in range(lo, hi)]
    with Pool(args.workers) as pool:
        res = pool.map(eval_unit, jobs, chunksize=4)
    rows = []
    for (s, rho, shift, use, _), r in zip(jobs, res):
        for name, ep in r.items():
            rows.append(dict(seed=s, rho=rho, shift=shift, use=use, policy=name, now=ep["now"],
                             same_on=ep["same_on"], shift_off=ep["shift_off"], div_T=ep["div_T"],
                             full=float(ep["acts"][:, 0].sum()), nudge=float(ep["acts"][:, 1].sum()),
                             back=float(ep["acts"][:, 2].sum()),
                             acts=ep["acts"].tolist()))
    json.dump(rows, open(os.path.join(args.out, "eval.json"), "w"))
    print(f"wrote {args.out}/eval.json: {len(rows)} rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--reward", default="adapt", choices=["now", "long", "adapt"])
    p.add_argument("--out", default="results/p2")
    p.add_argument("--rhos", default="0.8,0.9,1.0")
    p.add_argument("--use", type=float, default=0.5)
    p.add_argument("--gens", type=int, default=150)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seeds-per-gen", type=int, default=24)
    p.add_argument("--train-seeds", type=int, default=800)
    p.add_argument("--test-seeds", default="800:1000")
    p.add_argument("--es-seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    args = p.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
