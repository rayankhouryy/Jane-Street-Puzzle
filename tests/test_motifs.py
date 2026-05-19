"""Tests for neurodecomp.motifs.

Three layers of testing:

1.  Z3 self-verification: every motif in CATALOG proves its identity over
    its declared input domain.
2.  Detection round-trip: build a synthetic network that implements a known
    motif, scan it, verify the motif is recognised at the expected layer.
3.  False-positive guards: confirm that subtly-different networks are not
    falsely matched.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from neurodecomp import motifs
from tests.toy_circuits import _linear


class MotifZ3Tests(unittest.TestCase):
    """Each motif in CATALOG must agree with its reference on its domain."""

    def test_z3_available(self):
        self.assertTrue(motifs._HAS_Z3)

    def test_all_motifs_verified(self):
        results = motifs.verify_all()
        names_failed = [r for r in results if not r.proved]
        self.assertEqual(names_failed, [], f"some motifs failed Z3 proof: {names_failed}")


class MotifDetectionTests(unittest.TestCase):

    def test_detects_kronecker_delta(self):
        # 1[x = 5] for integer x in {0..10}.
        net = nn.Sequential(
            _linear([[1.0], [1.0], [1.0]], [-4.0, -6.0, -5.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0, -2.0]], [0.0]),
        )
        hits = motifs.scan_model(net)
        delta_hits = [h for h in hits if h.motif_name == "delta"]
        self.assertEqual(len(delta_hits), 1)
        # Output neuron 0 of the second Linear should be the delta.
        self.assertEqual(delta_hits[0].layer_idx, 1)
        self.assertEqual(delta_hits[0].output_neurons, (0,))

    def test_detects_bool_xor(self):
        # XOR(a, b) for a, b in {0, 1}.
        net = nn.Sequential(
            _linear([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, -1.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0, -2.0]], [0.0]),
        )
        hits = motifs.scan_model(net)
        xor_hits = [h for h in hits if h.motif_name == "bool_xor"]
        self.assertEqual(len(xor_hits), 1)
        self.assertEqual(xor_hits[0].layer_idx, 1)
        self.assertEqual(xor_hits[0].input_neurons, (0, 1))

    def test_detects_bool_and(self):
        # AND(a, b) = ReLU(a + b - 1).
        net = nn.Sequential(
            _linear([[1.0, 1.0]], [-1.0]),
            nn.ReLU(),
        )
        hits = motifs.scan_model(net)
        and_hits = [h for h in hits if h.motif_name == "bool_and"]
        self.assertEqual(len(and_hits), 1)
        self.assertEqual(and_hits[0].layer_idx, 0)
        self.assertEqual(and_hits[0].input_neurons, (0, 1))

    def test_no_false_positive_for_wrong_bias(self):
        # AND-shaped row but bias is 0 instead of -1 -> should NOT match.
        net = nn.Sequential(
            _linear([[1.0, 1.0]], [0.0]),
            nn.ReLU(),
        )
        hits = motifs.scan_model(net)
        self.assertEqual([h for h in hits if h.motif_name == "bool_and"], [])

    def test_xor_truth_table_matches(self):
        net = nn.Sequential(
            _linear([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.0, -1.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0, -2.0]], [0.0]),
        )
        with torch.no_grad():
            for a in (0, 1):
                for b in (0, 1):
                    out = net(torch.tensor([float(a), float(b)])).item()
                    self.assertAlmostEqual(out, a ^ b, places=6,
                                           msg=f"({a}, {b}) -> {out}")


if __name__ == "__main__":
    unittest.main()
