"""P4a: culture over time with oracle assistants.

The paper's model (Fazelpour, O'Brien & Rubin) with two additions:
  turnover: each round each team leaves with probability mu and is replaced by a newcomer with a random
            approach and a random specialization, in the same network position;
  epochs:   every `rounds` rounds the problem changes (20% of payoff tables redrawn), as in P3.
Oracle assistants do not learn, so this is the reference for the learning assistants of P4b. The
`rhelp`, `maxp`, `maxh` and `minp` assistants are what a perfect learner of each P4b reward would do.

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


POP = np.array([bin(i).count("1") for i in range(1 << M)])
ALL = np.arange(1 << M, dtype=np.int64)


def advise_ideal(codes, fit, F, idx, u, explore, kind):
    """What a perfect learner of each P4b reward would answer: among the modules that help the asker,
    rhelp: any one at random (thumbs is indifferent); maxp: the farthest from the asker's own module
    (P-creativity); maxh: the farthest from every module in the culture (H-creativity); minp: the closest
    to the asker's own module (a nudge); rad<r>: the best within r bits of the asker's own module (the
    reach of help; rad1 is SBP, rad10 the best module anywhere). Ties go to the larger gain; with nothing
    helpful, step back."""
    cand = (codes[idx, None] & ~MC) | ALL[None, :]
    gain = F[cand] - fit[idx, None]
    ok = gain > 0
    if kind.startswith("rad"):
        ok &= POP[ALL[None, :] ^ (codes[idx, None] & MC)] <= int(kind[3:])
        pick = np.argmax(np.where(ok, gain, -np.inf), 1)
    elif kind == "rhelp":
        k = np.floor(u * ok.sum(1)).astype(int)
        pick = np.argmax(np.cumsum(ok, 1) > k[:, None], 1)
    else:
        if kind == "maxh":
            cult = np.unique(codes & MC)
            score = np.broadcast_to(POP[ALL[:, None] ^ cult[None, :]].min(1), gain.shape)
        else:
            score = POP[ALL[None, :] ^ (codes[idx, None] & MC)]
            score = -score if kind == "minp" else score
        pick = np.argmax(np.where(ok, score + 0.5 * gain, -np.inf), 1)
    return np.where(ok.any(1), cand[np.arange(len(idx)), pick], explore[idx])


def advise_grid(codes, fit, F, idx, u, explore, name):
    """Design-space oracles, name = x_<source>_<preference>_<reach>_q<islands>.

    source      any: any of the 2^M modules; cul: only modules some team in the asker's island holds
    preference  among helpful candidates within reach: gain (largest gain), rand (any, uniformly),
                rare (fewest teams in the island hold it, i.e. fitness sharing / negative
                frequency-dependent selection; new modules count 0), near (the fewest bits changed:
                the least help that works), elite (the new approach beats every team whose module is
                within 2 bits of it: local competition); ties go to the larger gain
    reach       at most this many bits of the asker's module change
    islands     q assistants, each serving and seeing one fixed group of teams (team i in group i mod q)
    Nothing helpful within reach: step back (the team explores on its own)."""
    _, src, pref, k, q = name.split("_")
    k, q = int(k), int(q[1:])
    n = len(codes)
    island = np.arange(n) % q
    out = explore[idx].copy()
    for g in np.unique(island[idx]):
        sel = idx[island[idx] == g]
        pos = np.flatnonzero(island[idx] == g)
        held = codes[island == g] & MC
        mods = ALL if src == "any" else np.unique(held)
        cand = (codes[sel, None] & ~MC) | mods[None, :]
        gain = F[cand] - fit[sel, None]
        ok = (gain > 0) & (POP[mods[None, :] ^ (codes[sel, None] & MC)] <= k)
        if pref == "gain":
            key = gain
        elif pref == "near":
            key = -POP[mods[None, :] ^ (codes[sel, None] & MC)] + 0.5 * gain
        elif pref == "elite":
            fit_g = fit[island == g]
            close = POP[mods[:, None] ^ held[None, :]] <= 2
            best_near = np.where(close, fit_g[None, :], -np.inf).max(1)
            key = (F[cand] > best_near[None, :]) + 0.5 * gain
        elif pref == "rand":
            key = np.random.default_rng(int(u[pos[0]] * 2 ** 31)).random(gain.shape)
        else:
            count = np.bincount(held, minlength=1 << M)[mods]
            key = -count[None, :] + 0.5 * gain
        pick = np.argmax(np.where(ok, key, -np.inf), 1)
        out[pos] = np.where(ok.any(1), cand[np.arange(len(sel)), pick], explore[sel])
    return out


def advise_leap(codes, fit, F, idx, u, explore, name):
    """leap_<r>_<q>: tweak or leap (Miu et al. 2018). With probability q/100 the best helpful module that is
    more than r bits from the asker's and held by no team; otherwise the best helpful module within r bits."""
    _, r, q = name.split("_")
    r, q = int(r), int(q) / 100
    cand = (codes[idx, None] & ~MC) | ALL[None, :]
    gain = F[cand] - fit[idx, None]
    d = POP[ALL[None, :] ^ (codes[idx, None] & MC)]
    held = np.bincount(codes & MC, minlength=1 << M) > 0
    tweak = (gain > 0) & (d <= r)
    leap = (gain > 0) & (d > r) & ~held[None, :]
    go = (np.random.default_rng(int(u[0] * 2 ** 31)).random(len(idx)) < q) & leap.any(1)
    ok = np.where(go[:, None], leap, tweak)
    pick = np.argmax(np.where(ok, gain, -np.inf), 1)
    return np.where(ok.any(1), cand[np.arange(len(idx)), pick], explore[idx])


def advise_recombine(codes, fit, F, idx, u, explore, name):
    """rec_<b>: recombination (Uzzi et al. 2013; Fleming 2001). Draw a donor module held by some team, at
    least 3 bits from the asker's, with probability proportional to 1 / (teams holding it); answer the best
    helpful change of at most b bits, each toward the donor."""
    b = int(name.split("_")[1])
    held, count = np.unique(codes & MC, return_counts=True)
    rng = np.random.default_rng(int(u[0] * 2 ** 31))
    out = explore[idx].copy()
    for j, i in enumerate(idx):
        m = codes[i] & MC
        far = POP[held ^ m] >= 3
        if not far.any():
            continue
        w = far / count
        c = rng.choice(held, p=w / w.sum())
        diff = c ^ m
        sub = ALL[((ALL & ~diff) == 0) & (POP[ALL] <= b) & (ALL > 0)]  # flips toward the donor
        cand = (codes[i] & ~MC) | (m ^ sub)
        gain = F[cand] - fit[i]
        if (gain > 0).any():
            out[j] = cand[np.argmax(gain)]
    return out


def round_with(codes, F, comm, assistant, use, p_social, rng):
    if assistant not in ("gated", "rhelp", "maxp", "maxh", "minp") and not assistant.startswith(("rad", "x_", "leap_", "rec_")):
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
        if assistant == "gated":
            prop[idx] = advise_gated(codes, fit, F, idx, u_ai[idx], prop)
        elif assistant.startswith("x_"):
            prop[idx] = advise_grid(codes, fit, F, idx, u_ai, prop, assistant)
        elif assistant.startswith("leap_"):
            prop[idx] = advise_leap(codes, fit, F, idx, u_ai[idx], prop, assistant)
        elif assistant.startswith("rec_"):
            prop[idx] = advise_recombine(codes, fit, F, idx, u_ai[idx], prop, assistant)
        else:
            prop[idx] = advise_ideal(codes, fit, F, idx, u_ai[idx], prop, assistant)
    return np.where(F[prop] > fit, prop, codes)


def run(job):
    rho, seed, assistant, mu, epochs, rounds, shift, use = job[:8]
    post = job[8] if len(job) > 8 else 30
    lands = [Landscape(N, K, M, rho, np.random.default_rng([seed, int(round(rho * 1000)), 0]))]
    srng = np.random.default_rng([seed, int(round(rho * 1000)), 3])
    for _ in range(epochs):  # one extra: the next problem after the last epoch
        lands.append(lands[-1].shifted(shift, srng))
    comm = Community(100, N, H, 0.05, np.random.default_rng([seed, 2]))
    comm.spec = comm.spec.copy()
    rng = np.random.default_rng([seed, 1])
    trng = np.random.default_rng([seed, 11])  # turnover draws, separate so other draws stay paired
    codes = comm.codes0.copy()
    T = epochs * rounds
    fit, div, mods = np.zeros(T, np.float32), np.zeros(T, np.float32), np.zeros(T, np.int16)
    next_off, next_end = np.zeros(epochs, np.float32), np.zeros(epochs, np.float32)
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
        # next problem with the assistant removed, from this epoch's end state (own draws; as in P4b)
        prng = np.random.default_rng([seed, 7, e, 1, 0])
        pc, vals = codes.copy(), []
        for _ in range(post):
            pc = step(pc, lands[e + 1].F, comm, "none", 0.0, 0.1, M, prng)
            vals.append(lands[e + 1].F[pc].mean())
        next_off[e], next_end[e] = np.mean(vals), vals[-1]
    return dict(fit=fit, div=div, mods=mods, next_off=next_off, next_end=next_end)


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
