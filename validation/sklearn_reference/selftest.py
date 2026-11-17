"""Self-tests for the T16 reference tooling (stdlib unittest, no pytest).

Covers the mandatory checks:
  * independent objectives vs hand-computed values;
  * KL zero conventions (0/0, 0/positive, positive/0);
  * regularization mapping hand-verification: L1 only, L2 only, L1+L2, off,
    on a NON-SQUARE input (alpha_W != alpha_H);
  * mapped sklearn objective == NMFlib objective on a real sklearn fit;
  * dataset reproducibility (bit-for-bit), legality, rank deficiency of C;
  * sklearn fit reproducibility under a fixed seed;
  * results table schema completeness (when results.csv exists).

Usage:  python selftest.py
"""

from __future__ import annotations

import csv
import math
import unittest
from pathlib import Path

import numpy as np

import generate_data
import nmf_reference
import objective
from run_experiments import RESULT_FIELDS

HERE = Path(__file__).parent


class TestObjectives(unittest.TestCase):
    def test_frobenius_hand_computed(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([[1.0], [1.0]])
        h = np.array([[1.0, 1.0]])
        # WH = [[1,1],[1,1]]; diffs 0,1,2,3 -> 0.5*(0+1+4+9) = 7
        self.assertAlmostEqual(objective.frobenius_data_loss(v, w @ h), 7.0, places=12)
        self.assertAlmostEqual(objective.reconstruction_error(v, w @ h), math.sqrt(14.0), places=12)

    def test_kl_hand_computed(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        wh = np.ones((2, 2))
        expected = (2 * math.log(2) - 1) + (3 * math.log(3) - 2) + (4 * math.log(4) - 3)
        self.assertAlmostEqual(objective.kl_divergence(v, wh), expected, places=12)

    def test_kl_zero_conventions(self):
        self.assertEqual(objective.kl_divergence(np.array([[0.0]]), np.array([[0.0]])), 0.0)
        self.assertEqual(objective.kl_divergence(np.array([[0.0]]), np.array([[2.5]])), 2.5)
        with self.assertRaises(objective.KLInfiniteError):
            objective.kl_divergence(np.array([[1.0]]), np.array([[0.0]]))

    def test_kl_rejects_negative_and_nonfinite(self):
        with self.assertRaises(ValueError):
            objective.kl_divergence(np.array([[-1.0]]), np.array([[1.0]]))
        with self.assertRaises(ValueError):
            objective.kl_divergence(np.array([[np.nan]]), np.array([[1.0]]))

    def test_penalty_hand_computed(self):
        w = np.array([[1.0, 2.0], [3.0, 4.0]])
        h = np.array([[0.5, 1.5, 2.5]])
        # sum W = 10, sum H = 4.5; sum W^2 = 30, sum H^2 = 8.75
        self.assertAlmostEqual(objective.l1_penalty(w, h), 14.5, places=12)
        self.assertAlmostEqual(objective.l2_penalty(w, h), 0.5 * 38.75, places=12)
        self.assertAlmostEqual(
            objective.regularization_penalty(w, h, 0.3, 0.7),
            0.3 * 14.5 + 0.7 * 0.5 * 38.75, places=12)


class TestRegularizationMapping(unittest.TestCase):
    """Hand-verify the mapping on a NON-SQUARE problem (m=4, n=6, k=2)."""

    def setUp(self):
        self.m, self.n = 4, 6
        self.w = np.arange(1.0, 9.0).reshape(4, 2) * 0.1
        self.h = np.arange(1.0, 13.0).reshape(2, 6) * 0.2

    def _check(self, l1, l2):
        reg = nmf_reference.map_regularization(l1, l2, self.m, self.n)
        nmf_penalty = objective.regularization_penalty(self.w, self.h, l1, l2)
        sk_penalty = nmf_reference.sklearn_penalty_from_params(
            self.w, self.h, reg["alpha_W"], reg["alpha_H"], reg["l1_ratio"],
            self.m, self.n)
        self.assertAlmostEqual(nmf_penalty, sk_penalty, places=12)
        return reg

    def test_l1_only(self):
        reg = self._check(0.5, 0.0)
        self.assertAlmostEqual(reg["l1_ratio"], 1.0, places=12)
        self.assertAlmostEqual(reg["alpha_W"], 0.5 / 6, places=12)
        self.assertAlmostEqual(reg["alpha_H"], 0.5 / 4, places=12)

    def test_l2_only(self):
        reg = self._check(0.0, 0.7)
        self.assertAlmostEqual(reg["l1_ratio"], 0.0, places=12)
        self.assertAlmostEqual(reg["alpha_W"], 0.7 / 6, places=12)
        self.assertAlmostEqual(reg["alpha_H"], 0.7 / 4, places=12)

    def test_l1_plus_l2(self):
        reg = self._check(0.3, 0.7)
        self.assertAlmostEqual(reg["l1_ratio"], 0.3, places=12)
        self.assertAlmostEqual(reg["alpha_W"], 1.0 / 6, places=12)
        self.assertAlmostEqual(reg["alpha_H"], 1.0 / 4, places=12)

    def test_off(self):
        reg = self._check(0.0, 0.0)
        self.assertEqual(reg, {"alpha_W": 0.0, "alpha_H": 0.0, "l1_ratio": 0.0})

    def test_alpha_h_never_same_and_differs_on_non_square(self):
        reg = nmf_reference.map_regularization(0.3, 0.7, self.m, self.n)
        self.assertIsInstance(reg["alpha_H"], float)
        self.assertNotAlmostEqual(reg["alpha_W"], reg["alpha_H"], places=6)

    def test_mapped_sklearn_fit_matches_nmflib_objective(self):
        # End-to-end: fit the real sklearn with mapped params, then check the
        # documented sklearn objective equals NMFlib's Omega-form objective
        # on the SAME factors (catches version drift in the scaling).
        rng = np.random.default_rng(7)
        v = rng.uniform(0.1, 1.0, size=(4, 6))
        for l1, l2 in [(0.5, 0.0), (0.0, 0.7), (0.3, 0.7), (0.0, 0.0)]:
            r = nmf_reference.fit_sklearn(
                v, 2, "MU", "frobenius", "Random", 11, l1, l2, 200, 1e-6)
            reg = r["sklearn_params"]
            wh = r["w"] @ r["h"]
            sk_obj = objective.frobenius_data_loss(v, wh) + (
                nmf_reference.sklearn_penalty_from_params(
                    r["w"], r["h"], reg["alpha_W"], reg["alpha_H"],
                    reg["l1_ratio"], 4, 6))
            nmf_obj = objective.full_objective(v, r["w"], r["h"], l1, l2, "frobenius")
            self.assertAlmostEqual(sk_obj, nmf_obj, places=10)


class TestDatasets(unittest.TestCase):
    def test_reproducible_bit_for_bit(self):
        d1 = generate_data.build_datasets()
        d2 = generate_data.build_datasets()
        for name in d1:
            np.testing.assert_array_equal(d1[name]["v"], d2[name]["v"])

    def test_legality_and_shapes(self):
        ds = generate_data.build_datasets()
        self.assertEqual(ds["A1"]["v"].shape, (12, 9))
        self.assertEqual(ds["A2"]["v"].shape, (40, 25))
        self.assertEqual(ds["B"]["v"].shape, (30, 20))
        self.assertEqual(ds["C"]["v"].shape, (25, 18))
        for name, d in ds.items():
            v = d["v"]
            self.assertTrue(np.all(np.isfinite(v)), name)
            self.assertTrue(np.all(v >= 0), name)
            self.assertNotEqual(v.shape[0], v.shape[1], name)  # non-square

    def test_c_rank_deficient_with_exact_zeros(self):
        c = generate_data.build_datasets()["C"]
        self.assertLess(c["numeric_rank"], c["k"])  # actual rank < configured k
        self.assertEqual(c["numeric_rank"], 2)
        self.assertGreater(c["zero_fraction"], 0.1)
        v = c["v"]
        self.assertTrue(np.all(v.sum(axis=1) > 0))
        self.assertTrue(np.all(v.sum(axis=0) > 0))

    def test_two_different_ranks(self):
        ds = generate_data.build_datasets()
        self.assertGreaterEqual(len({d["k"] for d in ds.values()}), 2)


class TestSklearnReproducibility(unittest.TestCase):
    def test_same_seed_same_factors(self):
        rng = np.random.default_rng(3)
        v = rng.uniform(0.0, 1.0, size=(9, 7))
        a = nmf_reference.fit_sklearn(v, 3, "CD", "frobenius", "Random", 23, 0.0, 0.0, 200, 1e-6)
        b = nmf_reference.fit_sklearn(v, 3, "CD", "frobenius", "Random", 23, 0.0, 0.0, 200, 1e-6)
        np.testing.assert_array_equal(a["w"], b["w"])
        np.testing.assert_array_equal(a["h"], b["h"])
        self.assertEqual(a["n_iter"], b["n_iter"])


class TestResultsSchema(unittest.TestCase):
    def test_results_fields_complete(self):
        path = HERE / "results" / "results.csv"
        if not path.exists():
            self.skipTest("results.csv not generated yet (run run_experiments.py)")
        with path.open(encoding="utf-8") as f:
            header = next(csv.reader(f))
        for field in RESULT_FIELDS:
            self.assertIn(field, header)


if __name__ == "__main__":
    unittest.main(verbosity=2)