"""Tests for MVP-9c register-bit probing on synthetic networks."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from neurodecomp import round_decode
from tests.test_round_decode import _md5_f_layer


class RegisterBitProbingTests(unittest.TestCase):
    def test_identify_round_via_register_bit_on_synthetic_f(self):
        """On a synthetic single-bit MD5-F network, register-bit probing
        recovers the round identity even without zero-input pre-conditions."""
        f_net = _md5_f_layer()
        result = round_decode.identify_iteration_round(
            f_net, max_outputs_to_try=1, output_offset=0,
        )
        # The output of f_net is exactly F(B, C, D); identify should find it.
        self.assertIn("MD5 round 1", result["round_name"])
        self.assertGreaterEqual(result["evidence_count"], 1)


if __name__ == "__main__":
    unittest.main()
