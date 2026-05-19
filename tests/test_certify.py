"""Tests for neurodecomp.certify.

We verify the certifier:

* CONFIRMS bool_and motifs when their inputs are provably booleans;
* REJECTS bool_and-shaped rows whose inputs are wider integers (e.g. bytes);
* propagates abstract domains correctly through stitching layers.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from neurodecomp import certify
from tests.toy_circuits import _linear


class CertifyTests(unittest.TestCase):

    def test_confirms_bool_and_with_bool_inputs(self):
        # Inputs are direct passthroughs of two bool inputs.
        # Linear 0: identity on (a, b)
        # Linear 1: ReLU(a + b - 1) -> AND(a, b)
        net = nn.Sequential(
            _linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0]], [-1.0]),
            nn.ReLU(),
        )
        rep = certify.certify_motifs(net, input_ranges=[(0, 1), (0, 1)])
        self.assertEqual(rep.confirmed_counts["bool_and"], 1)
        self.assertEqual(rep.candidate_counts["bool_and"], 1)

    def test_rejects_bool_and_with_integer_inputs(self):
        # Same network but inputs can be bytes (0..255). The AND pattern row
        # is still found but should NOT be confirmed as a boolean AND.
        net = nn.Sequential(
            _linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0]], [-1.0]),
            nn.ReLU(),
        )
        rep = certify.certify_motifs(net, input_ranges=[(0, 255), (0, 255)])
        self.assertEqual(rep.candidate_counts["bool_and"], 1)
        self.assertEqual(rep.confirmed_counts["bool_and"], 0)

    def test_certifier_propagates_through_threshold_step(self):
        """Deeper certification through opaque ReLU symbols is *not* expected
        to succeed with our current Affine-only abstract domain.  This test
        documents the limitation: when the abstract interpreter introduces
        opaque symbols, downstream candidate ANDs over those symbols are
        correctly marked as *not* confirmed (because the symbol ranges are
        too wide to certify booleanness).

        Closing this gap requires a richer abstract domain (e.g. tracking
        that ``opaque[k] <= original_var[k]`` relations) -- noted as a
        future improvement in the README's MVP-7.5.
        """
        net = nn.Sequential(
            _linear(
                [
                    [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                    [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
                ],
                [1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
            ),
            nn.ReLU(),
            _linear(
                [
                    [1.0, 1.0, -2.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0, -2.0],
                ],
                [0.0, 0.0],
            ),
            nn.ReLU(),
            _linear([[1.0, 1.0]], [-1.0]),
            nn.ReLU(),
        )
        rep = certify.certify_motifs(net, input_ranges=[(0, 255), (0, 255)])
        # The scanner still finds the candidate; the certifier sets it to
        # unconfirmed because the upstream ReLUs saturate and the abstract
        # bounds widen beyond {0, 1}.
        self.assertGreaterEqual(rep.candidate_counts["bool_and"], 1)
        self.assertEqual(rep.confirmed_counts["bool_and"], 0)


if __name__ == "__main__":
    unittest.main()
