"""NK landscapes with two modules, following Fazelpour, O'Brien & Rubin (arXiv:2608.19390).

Decision j is bit j of an integer code. The computational module m_c is decisions [0, M);
the non-computational module is [M, N). Because N = 20, we precompute the fitness of every
one of the 2^N configurations once per landscape. The simulator then only does integer ops
and table lookups.
"""
import numpy as np

FITNESS_EXPONENT = 8  # Lazer & Friedman (2007) monotone transform; see D-003


def make_deps(N, K, M, rho, rng):
    """Draw the K dependencies S(d_i) of each decision so both modules hit modularity rho.

    rho(m) = sum k_in / sum (k_in + k_out), with k_in(i) = |S(d_i) in own module| + 1.
    For a module of size m that means sum |S(d_i) in own module| = rho*m*(K+1) - m.
    We spread that total as evenly as possible over the module's decisions.
    Returns deps (N, K) int array and the realised rho of each module.
    """
    comp, noncomp = np.arange(0, M), np.arange(M, N)
    deps = np.zeros((N, K), dtype=np.int64)
    for own, other in ((comp, noncomp), (noncomp, comp)):
        m = len(own)
        n_in_total = int(round(rho * m * (K + 1))) - m
        base, extra = divmod(n_in_total, m)
        n_in = np.full(m, base)
        n_in[rng.choice(m, size=extra, replace=False)] += 1
        for pos, i in enumerate(own):
            k_in = int(n_in[pos])
            k_out = K - k_in
            same = own[own != i]
            if k_in > len(same) or k_out > len(other) or k_out < 0:
                raise ValueError(f"rho={rho} infeasible for N={N} K={K} M={M}")
            deps[i] = np.concatenate([rng.choice(same, k_in, replace=False),
                                      rng.choice(other, k_out, replace=False)])
    return deps, realised_rho(deps, M)


def realised_rho(deps, M):
    N, K = deps.shape
    out = []
    for lo, hi in ((0, M), (M, N)):
        k_in = sum(int(np.sum((deps[i] >= lo) & (deps[i] < hi))) + 1 for i in range(lo, hi))
        out.append(k_in / ((hi - lo) * (K + 1)))
    return tuple(out)


def raw_fitness_table(deps, tables):
    """Mean contribution over decisions, for every code in [0, 2^N)."""
    N, K = deps.shape
    codes = np.arange(2 ** N, dtype=np.int64)
    total = np.zeros(2 ** N, dtype=np.float64)
    for i in range(N):
        idx = (codes >> i) & 1
        for c in deps[i]:
            idx = (idx << 1) | ((codes >> c) & 1)
        total += tables[i][idx]
    return total / N


def transform(raw):
    """Normalise so the global max is 1, then apply the monotone power transform.
    Every update rule in the model compares fitness values, so the dynamics are unchanged
    by this transform. It only changes the scale we report on."""
    return (raw / raw.max()) ** FITNESS_EXPONENT


class Landscape:
    """One problem: dependency structure, payoff tables, and the full fitness table F."""

    def __init__(self, N, K, M, rho, rng):
        self.N, self.K, self.M = N, K, M
        self.deps, self.rho = make_deps(N, K, M, rho, rng)
        self.tables = rng.random((N, 2 ** (K + 1)))
        self.F = transform(raw_fitness_table(self.deps, self.tables))

    def shifted(self, frac, rng):
        """A related problem: redraw the payoff tables of round(frac*N) decisions, chosen at
        random from both modules. The dependency structure stays the same (D-004)."""
        new = object.__new__(Landscape)
        new.N, new.K, new.M, new.deps, new.rho = self.N, self.K, self.M, self.deps, self.rho
        new.tables = self.tables.copy()
        n_change = int(round(frac * self.N))
        if n_change:
            changed = rng.choice(self.N, size=n_change, replace=False)
            new.tables[changed] = rng.random((n_change, self.tables.shape[1]))
        new.F = self.F if n_change == 0 else transform(raw_fitness_table(new.deps, new.tables))
        return new
