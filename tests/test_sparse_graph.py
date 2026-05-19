"""Tests for neurodecomp.sparse_graph + .block_finder on toy hand-built nets."""
from __future__ import annotations

import math
import random
import unittest

import torch

from neurodecomp import sparse_graph
from tests import toy_circuits


class SSAExtractTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def _check(self, model, inputs):
        prog = sparse_graph.extract_ssa(model)
        for x in inputs:
            xt = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                expected = model(xt).tolist()
            if not isinstance(expected, list):
                expected = [expected]
            got = sparse_graph.evaluate(prog, list(x))
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                self.assertTrue(math.isclose(g, e, abs_tol=1e-5),
                                f"input {x}: ssa={g}, torch={e}")

    def test_identity(self):
        self._check(toy_circuits.net_identity(),
                    [(-1, 2), (3, 4), (0, 0), (-2, -3)])

    def test_xor_truthtable(self):
        self._check(toy_circuits.net_xor_bool(),
                    [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_kron_delta_integer(self):
        model = toy_circuits.net_kron_delta()
        prog = sparse_graph.extract_ssa(model)
        for x in [-5, -1, 0, 1, 5]:
            (got,) = sparse_graph.evaluate(prog, [x])
            self.assertAlmostEqual(got, 1.0 if x == 0 else 0.0, places=5)

    def test_byte_equality(self):
        model = toy_circuits.net_byte_equality(target=42)
        prog = sparse_graph.extract_ssa(model)
        for x in [40, 41, 42, 43, 44, 0, 255]:
            (got,) = sparse_graph.evaluate(prog, [x])
            self.assertAlmostEqual(got, 1.0 if x == 42 else 0.0, places=5)

    def test_and_of_byte_eq(self):
        targets = [199, 239, 101, 35]      # like 4 bytes of the MD5 digest
        model = toy_circuits.net_and_of_byte_eq(targets)
        prog = sparse_graph.extract_ssa(model)
        # f equals 1 only on the exact target vector.
        for off in [-1, 0, 1]:
            x = [t + off for t in targets]
            (got,) = sparse_graph.evaluate(prog, x)
            expected = 1.0 if off == 0 else 0.0
            self.assertAlmostEqual(got, expected, places=5,
                                   msg=f"x={x}: got={got}, expected={expected}")

    def test_zero_edges_dropped(self):
        import torch.nn as nn
        model = nn.Sequential(
            toy_circuits._linear([[1.0, 0.0, 0.0, -1.0]], [0.0]),
            nn.ReLU(),
        )
        prog = sparse_graph.extract_ssa(model)
        affines = [s for s in prog.statements if s.kind == "affine"]
        self.assertEqual(len(affines), 1)
        self.assertEqual(affines[0].affine.nonzero_count(), 2)

    def test_weight_alphabet(self):
        prog = sparse_graph.extract_ssa(toy_circuits.net_and_of_byte_eq([1, 2, 3]))
        alpha = sparse_graph.weight_alphabet(prog)
        # All values should be small integers; alphabet must be small.
        self.assertLess(len(alpha), 12)
        for v in alpha:
            self.assertEqual(v, int(v), f"non-integer weight {v}")
            self.assertLess(abs(v), 10, f"weight {v} outside expected range")


if __name__ == "__main__":
    unittest.main()
