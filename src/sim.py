"""Community search on an NK landscape with optional AI assistance.

Follows Fazelpour, O'Brien & Rubin (arXiv:2608.19390), section 3.5. Each round every agent
takes one action:
  social learning (prob p_social): copy the best neighbour's full vector if it is better
  query the AI (prob use_rate of the remaining rounds): adopt the AI's proposal if it is better
  explore (otherwise): flip one random decision from its specialisation set, keep it if better
Updates are synchronous: every agent acts on the state at the start of the round (D-003).

Random draws are made the same way in every condition, so runs with the same seed share the
landscape, network, starting points and per-round coin flips (common random numbers).
"""
import numpy as np

TOP_DECILE = 0.1


# ---------------------------------------------------------------- community

def er_connected(n, p, rng):
    """Erdos-Renyi G(n, p), resampled until connected."""
    while True:
        upper = np.triu(rng.random((n, n)) < p, 1)
        adj = upper | upper.T
        seen = np.zeros(n, bool)
        seen[0] = True
        frontier = [0]
        while frontier:
            nxt = np.flatnonzero(adj[frontier].any(0) & ~seen)
            seen[nxt] = True
            frontier = list(nxt)
        if seen.all():
            return adj


class Community:
    """Network, specialisation sets and starting configurations. Fixed within a seed."""

    def __init__(self, n_agents, N, H, p_edge, rng):
        self.n, self.N, self.H = n_agents, N, H
        adj = er_connected(n_agents, p_edge, rng)
        deg = adj.sum(1)
        self.nbr = np.full((n_agents, deg.max()), n_agents, dtype=np.int64)  # pad = n_agents
        for a in range(n_agents):
            self.nbr[a, :deg[a]] = np.flatnonzero(adj[a])
        self.spec = np.stack([rng.choice(N, H, replace=False) for _ in range(n_agents)])
        self.codes0 = rng.integers(0, 2 ** N, n_agents, dtype=np.int64)


# ---------------------------------------------------------------- metrics

def bit_counts(codes, N):
    """ones[j] = number of agents with decision j set."""
    return ((codes[:, None] >> np.arange(N)) & 1).sum(0)


def mean_pairwise_hamming(codes, N):
    n = len(codes)
    ones = bit_counts(codes, N)
    return float((ones * (n - ones)).sum() / (n * (n - 1) / 2))


# ---------------------------------------------------------------- assistants
# Each returns a proposed configuration for the querying agents `idx`.
# The agent adopts it only if it is strictly better (same rule as the paper).

def parse_assistant(name, M):
    """'none' | 'np' | 'tdr' | 'sbp' | 'rand<q>' | 'div<q>' (q an int or 'all')."""
    if name in ("none", "np", "tdr", "sbp"):
        return name, 1
    for mode in ("rand", "div"):
        if name.startswith(mode):
            q = name[len(mode):]
            return mode, M if q == "all" else int(q)
    raise ValueError(name)


def propose_np(codes, fit, mc, idx, u):
    """Non-personalised: computational module of the current best agent (paper section 4.1.1)."""
    best = codes[np.argmax(fit)]
    return (codes[idx] & ~mc) | (best & mc)


def propose_tdr(codes, fit, mc, idx, u):
    """Top-decile randomised: computational module of a random top-10% agent (section 5.1.1)."""
    n_top = int(np.ceil(TOP_DECILE * len(codes)))
    top = np.argpartition(-fit, n_top - 1)[:n_top]
    src = codes[top[(u * n_top).astype(np.int64)]]
    return (codes[idx] & ~mc) | (src & mc)


def propose_single_flip(codes, fit, F, M, idx, u, mode, q):
    """Single-bit personalised family.

    sbp:     the best single flip in m_c for this agent (paper section 5.1.2).
    rand<q>: a uniformly random flip among the q best improving flips.
    div<q>:  among the q best improving flips, the one that moves the agent furthest from the
             rest of the population (largest increase in summed Hamming distance), ties broken
             by fitness. Flipping bit j raises that distance by 2*agree_j - (n-1), where
             agree_j is the number of other agents that share the agent's current value on j.
    With q = 1 all three modes are the same rule.
    """
    c = codes[idx]
    flips = c[:, None] ^ (np.int64(1) << np.arange(M))[None, :]
    cf = F[flips]
    if mode == "sbp" or q == 1:
        return flips[np.arange(len(idx)), np.argmax(cf, 1)]
    order = np.argsort(-cf, axis=1, kind="stable")
    n_ok = np.minimum((cf > fit[idx][:, None]).sum(1), q)  # improving flips sort first
    if mode == "rand":
        pos = np.minimum((u * n_ok).astype(np.int64), np.maximum(n_ok - 1, 0))
    else:
        n = len(codes)
        ones = bit_counts(codes, M)
        mine = (c[:, None] >> np.arange(M)) & 1
        agree = np.where(mine == 1, ones - 1, (n - ones) - 1)
        score = np.take_along_axis(agree, order, 1).astype(float)
        score[np.arange(M)[None, :] >= n_ok[:, None]] = -np.inf
        pos = np.argmax(score, 1)
    out = flips[np.arange(len(idx)), order[np.arange(len(idx)), pos]]
    return np.where(n_ok > 0, out, c)


def propose(assistant, codes, fit, F, M, idx, u):
    mode, q = parse_assistant(assistant, M)
    mc = np.int64((1 << M) - 1)
    if mode == "np":
        return propose_np(codes, fit, mc, idx, u)
    if mode == "tdr":
        return propose_tdr(codes, fit, mc, idx, u)
    return propose_single_flip(codes, fit, F, M, idx, u, mode, q)


# ---------------------------------------------------------------- dynamics

def step(codes, F, comm, assistant, use_rate, p_social, M, rng):
    n = len(codes)
    u_social, u_query, u_ai = rng.random(n), rng.random(n), rng.random(n)
    col = rng.integers(0, comm.H, n)
    fit = F[codes]

    social = u_social < p_social
    query = ~social & (u_query < use_rate) if assistant != "none" else np.zeros(n, bool)

    prop = codes ^ (np.int64(1) << comm.spec[np.arange(n), col])  # explore
    fit_pad = np.append(fit, -np.inf)
    best_nb = comm.nbr[np.arange(n), np.argmax(fit_pad[comm.nbr], 1)]
    prop = np.where(social, codes[np.minimum(best_nb, n - 1)], prop)
    if query.any():
        idx = np.flatnonzero(query)
        prop[idx] = propose(assistant, codes, fit, F, M, idx, u_ai[idx])
    return np.where(F[prop] > fit, prop, codes)


def run(land, comm, assistant, use_rate, p_social, rounds, seed,
        probe_times=(), probe_lands=(), post_rounds=30):
    """One community trajectory, plus withdrawal/shift probes branched from it.

    At each probe time T we copy the community and run `post_rounds` more rounds on each
    probe landscape (same problem, shifted problem, ...) with the assistant on and off.
    Probe randomness depends on (seed, T, landscape, on/off) only, so it is shared across
    conditions.
    """
    M, N = land.M, land.N
    rng = np.random.default_rng([seed, 1])
    codes = comm.codes0.copy()
    fit = np.zeros(rounds + 1, np.float32)
    div = np.zeros(rounds + 1, np.float32)
    fit[0], div[0] = land.F[codes].mean(), mean_pairwise_hamming(codes, N)
    probe = np.zeros((len(probe_times), len(probe_lands), 2, post_rounds + 1), np.float32)
    probe_div = np.zeros((len(probe_times), len(probe_lands), 2), np.float32)
    for t in range(1, rounds + 1):
        codes = step(codes, land.F, comm, assistant, use_rate, p_social, M, rng)
        fit[t], div[t] = land.F[codes].mean(), mean_pairwise_hamming(codes, N)
        if t in probe_times:
            ti = probe_times.index(t)
            for li, pl in enumerate(probe_lands):
                for on in (0, 1):
                    if on and assistant == "none":
                        probe[ti, li, 1] = probe[ti, li, 0]
                        probe_div[ti, li, 1] = probe_div[ti, li, 0]
                        continue
                    prng = np.random.default_rng([seed, 7, t, li, on])
                    pc = codes.copy()
                    probe[ti, li, on, 0] = pl.F[pc].mean()
                    for s in range(1, post_rounds + 1):
                        pc = step(pc, pl.F, comm, assistant if on else "none",
                                  use_rate, p_social, M, prng)
                        probe[ti, li, on, s] = pl.F[pc].mean()
                    probe_div[ti, li, on] = mean_pairwise_hamming(pc, N)
    return dict(fit=fit, div=div, probe=probe, probe_div=probe_div)
