"""Tests for neurodecomp.domains + neurodecomp.interp on toy ReLU circuits."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from neurodecomp import interp
from neurodecomp.domains import Affine, affine_input_only
from tests.toy_circuits import _linear


class AffineAlgebraTests(unittest.TestCase):
    def test_const_and_predicates(self):
        a = Affine.const(7)
        self.assertTrue(a.is_const)
        self.assertEqual(a.const_value, 7)

    def test_addition_simplifies_zero_coefs(self):
        x0 = Affine.of(("in", 0), 1)
        x0_neg = Affine.of(("in", 0), -1)
        s = x0 + x0_neg
        self.assertTrue(s.is_const)
        self.assertEqual(s.const_value, 0)

    def test_scalar_multiplication_and_negation(self):
        a = Affine.of(("in", 3), 2, 5)
        b = a * 3
        self.assertEqual(b.coefs, ((("in", 3), 6),))
        self.assertEqual(b.bias, 15)
        self.assertEqual((-a).bias, -5)

    def test_subtraction(self):
        a = Affine.of(("in", 0), 1, 4)
        b = Affine.of(("in", 0), 1, 1)
        d = a - b
        self.assertTrue(d.is_const)
        self.assertEqual(d.const_value, 3)

    def test_range_byte_domain(self):
        # x0 + 2 * x1 - 1, x0, x1 in [0, 255]
        a = Affine.of(("in", 0), 1, 0) + Affine.of(("in", 1), 2, 0) + Affine.const(-1)
        ranges = {("in", 0): (0, 255), ("in", 1): (0, 255)}
        lo, hi = a.range(ranges)
        self.assertEqual(lo, -1)
        self.assertEqual(hi, 255 + 510 - 1)

    def test_range_missing_var_raises(self):
        a = Affine.of(("in", 0), 1)
        with self.assertRaises(KeyError):
            a.range({})


class InterpTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_identity_passes(self):
        # f(x) = ReLU(x) with x in [0, 10] -> non-saturating, identity.
        net = nn.Sequential(_linear([[1.0]], [0.0]), nn.ReLU())
        res = interp.run(net, [(0, 10)])
        (a,) = res.values
        self.assertEqual(a.coefs, ((("in", 0), 1),))
        self.assertEqual(a.bias, 0)
        self.assertEqual(res.num_opaque, 0)

    def test_dead_relu(self):
        # f(x) = ReLU(-x - 1) with x in [0, 10] -> always negative -> dead.
        net = nn.Sequential(_linear([[-1.0]], [-1.0]), nn.ReLU())
        res = interp.run(net, [(0, 10)])
        (a,) = res.values
        self.assertTrue(a.is_const)
        self.assertEqual(a.const_value, 0)

    def test_saturating_relu(self):
        # f(x) = ReLU(x - 5) with x in [0, 10] -> saturating.
        net = nn.Sequential(_linear([[1.0]], [-5.0]), nn.ReLU())
        res = interp.run(net, [(0, 10)])
        (a,) = res.values
        self.assertEqual(res.num_opaque, 1)
        # The single neuron should refer to the opaque ReLU symbol.
        self.assertEqual(len(a.coefs), 1)
        self.assertEqual(a.coefs[0][0][0], "relu")
        # And its range should be [0, 10-5] = [0, 5].
        self.assertEqual(res.var_ranges[("relu", 0)], (0, 5))

    def test_byte_passthrough_with_positive_bias(self):
        # The first head layer pattern: ReLU(byte_i + bias_i) for bias_i in {1..55}.
        # On input domain [0, 255], with positive bias, this is non-saturating.
        net = nn.Sequential(
            _linear([[1.0]], [3.0]),    # mimics "byte_0 + 3"
            nn.ReLU(),
        )
        res = interp.run(net, [(0, 255)])
        (a,) = res.values
        self.assertEqual(a.coefs, ((("in", 0), 1),))
        self.assertEqual(a.bias, 3)
        self.assertEqual(res.num_opaque, 0)

    def test_kron_delta_introduces_two_opaque(self):
        # Kronecker delta on x in [-5, 5]: ReLU(x+1) + ReLU(x-1) - 2 ReLU(x).
        # Two of the three pre-ReLU values straddle zero; one (ReLU(x)) also straddles.
        net = nn.Sequential(
            _linear([[1.0], [1.0], [1.0]], [1.0, -1.0, 0.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0, -2.0]], [0.0]),
        )
        res = interp.run(net, [(-5, 5)])
        # 3 ReLUs all saturate.
        self.assertEqual(res.num_opaque, 3)
        # The final output is one affine over the three relu symbols.
        (out,) = res.values
        rel_vars = [v for v, _ in out.coefs if v[0] == "relu"]
        self.assertEqual(set(rel_vars), {("relu", 0), ("relu", 1), ("relu", 2)})

    def test_byte_equality_passes_recovery(self):
        # The byte-equality net we tested in MVP-2 has data-dependent ReLUs.
        # The point of this test isn't full decompilation; just that interp runs
        # to completion without crashing and produces a single output Affine.
        from tests.test_tail import build_byte_eq_network
        net = build_byte_eq_network([0x42, 0x00, 0xFF])
        # Inputs are bits in {0, 1}.
        res = interp.run(net, [(0, 1)] * (8 * 3))
        self.assertEqual(len(res.values), 1)


if __name__ == "__main__":
    unittest.main()
