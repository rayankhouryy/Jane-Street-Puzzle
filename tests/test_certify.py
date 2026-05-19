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

    def test_certifier_confirms_bool_input_to_relu(self):
        """MVP-7.5 — when a ReLU is fed an affine with integer hi ≤ 1, the
        post-ReLU output is provably in {0, 1} even if the affine's lo is
        negative.

        Example: ReLU(a + b - 1) where a, b are bytes in [0, 1].
        Input affine has range [-1, 1] → straddles zero → opaque.
        Source-expression check (hi = 1 ≤ 1) certifies booleanness.

        With the old range-only check this opaque was unconfirmable; now it
        is.
        """
        # Network: two boolean inputs → one explicit ReLU(a + b - 1) →
        # another bool_and downstream that consumes the result.
        # The downstream Linear computes ReLU((a AND b) + (a AND b) - 1)
        # which the scanner recognises as bool_and; certifier should
        # confirm because the input is a relational-Boolean opaque.
        net = nn.Sequential(
            _linear(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                [-1.0, -1.0],
            ),
            nn.ReLU(),
            _linear([[1.0, 1.0]], [-1.0]),
            nn.ReLU(),
        )
        rep = certify.certify_motifs(net, input_ranges=[(0, 1), (0, 1)])
        # Two candidate AND patterns (one in each Linear); both should be
        # confirmed: the first directly from the input bool domain, the
        # second via the relational source-expression check.
        self.assertEqual(rep.candidate_counts["bool_and"], 3)
        self.assertGreaterEqual(rep.confirmed_counts["bool_and"], 2)

    def test_certifier_does_not_overclaim_kron_delta_chain(self):
        """Documented limitation: Boolean output of a Kronecker-delta triple
        (which sums three ReLUs to encode `1[x == c]`) is not yet certified.

        The triple's *combined* output is in {0, 1}, but each individual
        ReLU's source affine has range like [-254, 255], so the simple
        ``hi ≤ 1 on source`` check doesn't fire.  Certifying this case
        requires recognising the delta motif algebraically -- left for
        MVP-7.6 or motif-aware certification.
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
        # The scanner finds the final bool_and candidate; the current
        # certifier conservatively leaves it unconfirmed.
        self.assertGreaterEqual(rep.candidate_counts["bool_and"], 1)
        # NOTE: this expectation will tighten when motif-aware certification
        # (MVP-7.6) is added.
        self.assertEqual(rep.confirmed_counts["bool_and"], 0)


if __name__ == "__main__":
    unittest.main()
