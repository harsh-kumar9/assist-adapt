"""P2 runner must reproduce src.sim on the same seed. Run with: python tests/test_p2.py"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import p2  # noqa: E402
from src.sim import run  # noqa: E402


class TestP2(unittest.TestCase):
    def test_matches_sim(self):
        seed, rho = 5, 1.0
        land, shifted, comm = p2.world(seed, rho, 0.2)
        for rule, name in ((p2.Mix(0.0), "sbp"), (p2.Mix(1.0), "pcc"), (None, "none")):
            ep = p2.episode(land, shifted, comm, rule, 0.5, 0.1, 30, 30, seed)
            ref = run(land, comm, name, 0.5, 0.1, 30, seed, [30], [land, shifted], 30)
            self.assertAlmostEqual(ep["now"], float(ref["fit"][1:31].mean()), places=5)
            self.assertAlmostEqual(ep["shift_off"], float(ref["probe"][0, 1, 0, 1:].mean()), places=5)
            on = 0 if name == "none" else 1
            self.assertAlmostEqual(ep["same_on"], float(ref["probe"][0, 0, on, 1:].mean()), places=5)

    def test_linear_policy_probs(self):
        th = np.zeros((2, p2.N_FEAT))
        th[0, 0] = p2.BIG  # always full
        a = p2.Linear(th).act(np.ones((1000, p2.N_FEAT)), np.random.default_rng(0).random(1000), 0.0)
        self.assertTrue(np.all(a == 0))
        a = p2.Linear(np.zeros((2, p2.N_FEAT))).act(np.ones((30000, p2.N_FEAT)),
                                                    np.random.default_rng(0).random(30000), 0.0)
        np.testing.assert_allclose(np.bincount(a, minlength=3) / 30000, [1 / 3] * 3, atol=0.01)


if __name__ == "__main__":
    unittest.main()
