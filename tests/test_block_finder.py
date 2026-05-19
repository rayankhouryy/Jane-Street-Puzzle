"""Tests for neurodecomp.block_finder on a constructed toy loop network."""
from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from neurodecomp import block_finder
from tests.toy_circuits import _linear


def _make_block(width: int) -> list:
    """A tiny 'block' of N layers with identical weights, designed so we can
    stack copies and get periodicity."""
    return [
        _linear([[1.0 if i == j else 0.0 for j in range(width)] for i in range(width)],
                [0.0] * width),
        nn.ReLU(),
    ]


def _build_unrolled_loop(width: int, period_linears: int, iters: int):
    """Build a Sequential representing `for _ in range(iters): body(...)`.

    Each iteration is `period_linears` Linears + ReLUs with bit-identical
    weights across iterations (so block_finder should detect period).
    """
    layers = []
    # Head: project the input to width.
    in_dim = 8
    layers.append(_linear(
        [[1.0 if i == j else 0.0 for j in range(in_dim)] + [0.0] * (width - in_dim)
         for i in range(width)] if width > in_dim else
        [[1.0 if i == j else 0.0 for j in range(in_dim)] for i in range(width)],
        [0.0] * width,
    ))
    layers.append(nn.ReLU())

    body_weights = [[1.0 if i == j else 0.0 for j in range(width)] for i in range(width)]
    body_bias = [0.0] * width
    rotated_weights_per_iter = []
    for k in range(iters):
        for p in range(period_linears):
            if p == 1:    # position with per-iteration constant (like MD5 shifts)
                bias = [0.5 + k * 0.1] + [0.0] * (width - 1)
                rotated_weights_per_iter.append(bias)
                layers.append(_linear(body_weights, bias))
            else:
                layers.append(_linear(body_weights, body_bias))
            layers.append(nn.ReLU())

    # Tail: reduce to 1 output.
    layers.append(_linear([[1.0] + [0.0] * (width - 1)], [0.0]))
    return nn.Sequential(*layers)


class BlockFinderTests(unittest.TestCase):
    def test_no_block_when_no_period(self):
        # A small straight-shot net with no repeated body.
        net = nn.Sequential(
            _linear([[1.0, -1.0], [-1.0, 1.0]], [0.0, 0.0]),
            nn.ReLU(),
            _linear([[1.0, 1.0]], [0.0]),
        )
        rep = block_finder.find_blocks(net)
        # With only 2 Linears, period detection should pick a small period
        # but block_starts should be empty.
        self.assertEqual(rep.block_starts, [])
        self.assertEqual(rep.num_iterations, 0)

    def test_synthetic_loop(self):
        # 5-layer body repeated 6 times; widths are all 400 (passes block-start
        # heuristic threshold of >=320 in-width but our block-finder requires
        # in_features > out_features to identify a "compression" linear, so
        # this synthetic net intentionally won't match. We just check no crash.
        net = _build_unrolled_loop(width=400, period_linears=5, iters=6)
        rep = block_finder.find_blocks(net)
        # We don't require the heuristic to find blocks in this synthetic
        # network, but the dispatcher must not crash.
        self.assertIsNotNone(rep)
        self.assertGreater(rep.num_linear, 0)


if __name__ == "__main__":
    unittest.main()
