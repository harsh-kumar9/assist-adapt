"""Sanity checks. Run with: python tests/test_sim.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nk import Landscape, make_deps, raw_fitness_table  # noqa: E402
from src.sim import Community, mean_pairwise_hamming, propose, run, step  # noqa: E402


class TestNK(unittest.TestCase):
    def test_rho_targets(self):
        for K in (5, 9):
            for rho in (0.6, 0.7, 0.8, 0.9, 1.0):
                deps, real = make_deps(20, K, 10, rho, np.random.default_rng(0))
                for r in real:
                    self.assertAlmostEqual(r, rho, delta=1 / (10 * (K + 1)) + 1e-9)
                for i in range(20):
                    self.assertNotIn(i, deps[i])
                    self.assertEqual(len(set(deps[i])), K)

    def test_fitness_matches_direct(self):
        rng = np.random.default_rng(1)
        deps, _ = make_deps(20, 9, 10, 0.8, rng)
        tables = rng.random((20, 2 ** 10))
        raw = raw_fitness_table(deps, tables)
        for code in rng.integers(0, 2 ** 20, 50):
            bits = [(int(code) >> j) & 1 for j in range(20)]
            direct = 0.0
            for i in range(20):
                idx = bits[i]
                for c in deps[i]:
                    idx = (idx << 1) | bits[c]
                direct += tables[i][idx]
            self.assertAlmostEqual(raw[code], direct / 20, places=12)

    def test_transform_and_shift(self):
        land = Landscape(20, 9, 10, 0.8, np.random.default_rng(2))
        self.assertAlmostEqual(land.F.max(), 1.0)
        same = land.shifted(0.0, np.random.default_rng(3))
        self.assertIs(same.F, land.F)
        sh = land.shifted(0.2, np.random.default_rng(3))
        self.assertEqual(int((sh.tables != land.tables).any(1).sum()), 4)
        self.assertAlmostEqual(sh.F.max(), 1.0)


class TestSim(unittest.TestCase):
    def setUp(self):
        self.land = Landscape(20, 9, 10, 0.8, np.random.default_rng(4))
        self.comm = Community(100, 20, 10, 0.05, np.random.default_rng(5))

    def test_hamming(self):
        codes = np.array([0b0, 0b1, 0b11], dtype=np.int64)
        self.assertAlmostEqual(mean_pairwise_hamming(codes, 20), (1 + 2 + 1) / 3)

    def test_monotone_and_mc_only(self):
        F, mc = self.land.F, (1 << 10) - 1
        codes = self.comm.codes0.copy()
        fit = F[codes]
        idx = np.arange(100)
        u = np.random.default_rng(0).random(100)
        for a in ("np", "tdr", "tdr25", "tdrinv", "pcc", "sbp", "rand3", "div3", "randall", "divall"):
            prop = propose(a, codes, fit, F, 10, idx, u)
            self.assertTrue(np.all((prop & ~mc) == (codes & ~mc)), a)  # only m_c changes
            if a not in ("np", "tdr", "tdr25", "tdrinv", "pcc"):
                n_diff = np.array([bin(int(x)).count("1") for x in prop ^ codes])
                self.assertTrue(np.all(n_diff <= 1), a)
        new = step(codes, F, self.comm, "sbp", 0.5, 0.1, 10, np.random.default_rng(0))
        self.assertTrue(np.all(F[new] >= fit))

    def test_pcc_and_tdrinv(self):
        F, mc = self.land.F, (1 << 10) - 1
        codes = self.comm.codes0.copy()
        fit = F[codes]
        prop = propose("pcc", codes, fit, F, 10, np.array([3]), np.array([0.5]))
        best = max(F[(codes[3] & ~mc) | (c & mc)] for c in codes)
        self.assertEqual(F[prop[0]], best)
        # inverse popularity: a module held by 90 agents is drawn far less often than one held by 1
        c2 = np.zeros(100, dtype=np.int64)
        c2[:10] = np.arange(10)          # ten distinct modules, all in the "top decile" by fitness below
        c2[10:] = 0                      # module 0 is also held by the other 90 agents
        f2 = np.r_[np.ones(10), np.zeros(90)]
        u = np.random.default_rng(1).random(20000)
        picks = propose("tdrinv", c2, f2, F, 10, np.zeros(20000, dtype=np.int64), u) & mc
        share0 = np.mean(picks == 0)
        self.assertLess(share0, 0.02)    # weight 1/91 vs 1 for each of the other nine

    def test_div_moves_away_from_majority(self):
        F = self.land.F
        codes = np.zeros(100, dtype=np.int64)  # everyone identical
        codes[0] = 0  # the querier
        fit = F[codes]
        prop_div = propose("divall", codes, fit, F, 10, np.array([0]), np.array([0.5]))
        prop_sbp = propose("sbp", codes, fit, F, 10, np.array([0]), np.array([0.5]))
        # with everyone identical every flip moves equally far, so div ties -> best fitness
        if F[prop_sbp[0]] > fit[0]:
            self.assertEqual(prop_div[0], prop_sbp[0])

    def test_run_shapes_and_none_probe(self):
        sh = self.land.shifted(0.2, np.random.default_rng(6))
        out = run(self.land, self.comm, "none", 0.0, 0.1, 40, 0, [10, 40], [self.land, sh], 5)
        self.assertEqual(out["fit"].shape, (41,))
        self.assertEqual(out["probe"].shape, (2, 2, 2, 6))
        np.testing.assert_array_equal(out["probe"][..., 0, :], out["probe"][..., 1, :])
        # probe on the same landscape starts where the main run was at T
        self.assertAlmostEqual(out["probe"][0, 0, 0, 0], out["fit"][10], places=5)

    def test_common_random_numbers(self):
        a = run(self.land, self.comm, "none", 0.0, 0.1, 20, 3)
        b = run(self.land, self.comm, "none", 0.0, 0.1, 20, 3)
        np.testing.assert_array_equal(a["fit"], b["fit"])


if __name__ == "__main__":
    unittest.main()
