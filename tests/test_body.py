"""Tests for neurodecomp.body on a synthetic per-iteration rotate gadget."""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn as nn

from neurodecomp import body, block_finder
from tests.toy_circuits import _linear


def _identity_linear(n: int) -> nn.Linear:
    return _linear([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
                   [0.0] * n)


def _compress_linear(n: int) -> nn.Linear:
    """A wider->narrower Linear that the block finder will recognise as a
    block-start: 2n -> n with the first n outputs = first n inputs."""
    return _linear(
        [[1.0 if i == j else 0.0 for j in range(2 * n)] for i in range(n)],
        [0.0] * n,
    )


def _expand_linear(n: int) -> nn.Linear:
    """The reverse: n -> 2n that pads with zeros so the cycle resumes."""
    out_dim = 2 * n
    rows = []
    for i in range(out_dim):
        rows.append([1.0 if i == j else 0.0 for j in range(n)])
    return _linear(rows, [0.0] * out_dim)


def _rotate_linear(n_lanes: int, n_registers: int, shift: int, total_in: int) -> nn.Linear:
    """Build a Linear(total_in -> total_in) whose first `n_lanes * n_registers`
    output rows implement ROTL_shift on `n_registers` consecutive 32-bit lanes
    located at the *last* `n_lanes` input cols."""
    out_dim = total_in
    rows = []
    row_base = total_in - n_lanes * n_registers          # output row block start
    col_base = total_in - n_lanes                          # input col block start
    # First, identity for the leading rows.
    for o in range(total_in):
        if o < row_base or o >= row_base + n_lanes * n_registers:
            # identity passthrough
            rows.append([1.0 if o == j else 0.0 for j in range(total_in)])
        else:
            # rotate row
            local = o - row_base
            reg = local // n_lanes
            b = local % n_lanes
            src_col = col_base + ((b - shift) % n_lanes)
            rows.append([1.0 if j == src_col else 0.0 for j in range(total_in)])
    return _linear(rows, [0.0] * out_dim)


def build_synthetic_body(shifts):
    """Build a Sequential whose body has identical layers at all positions
    except position p (which implements a rotate-by-s_t gadget), with
    `len(shifts)` iterations.

    Block layout (per iteration): one compress (200 -> 100), then identity
    layers, then a rotate at position p, then identity layers, then expand
    (100 -> 200) so the next iter can start.  Period = 4 (compress, identity,
    rotate, expand)."""
    n_lanes = 32
    n_registers = 3
    total_in = 100
    layers = []

    # Initial layer to bring an external input dim up to 200 (so the first
    # compress is a "real" block-start).
    initial = _linear(
        [[1.0 if i == j else 0.0 for j in range(8)] + [0.0] * (200 - 8)
         for i in range(200)],
        [0.0] * 200,
    )
    layers.append(initial)
    layers.append(nn.ReLU())

    period = 4
    for t, s_t in enumerate(shifts):
        layers.append(_compress_linear(total_in))        # position 0: 200 -> 100
        layers.append(nn.ReLU())
        layers.append(_identity_linear(total_in))         # position 1: 100 -> 100
        layers.append(nn.ReLU())
        layers.append(_rotate_linear(n_lanes, n_registers, s_t, total_in))  # position 2: 100 -> 100
        layers.append(nn.ReLU())
        layers.append(_expand_linear(total_in))            # position 3: 100 -> 200
        layers.append(nn.ReLU())

    # Final reduction to a scalar (so the network is well-formed).
    final = _linear([[1.0] + [0.0] * 199], [0.0])
    layers.append(final)
    layers.append(nn.ReLU())
    return nn.Sequential(*layers), period


class BodyDecompilerTests(unittest.TestCase):

    def test_recovers_synthetic_rotate_schedule(self):
        shifts = [3, 7, 11, 15, 19, 23, 27, 31]
        net, period = build_synthetic_body(shifts)
        layout = block_finder.find_blocks(net, min_iterations=3)
        self.assertEqual(layout.period, period)
        # Block finder may not pick up every iteration depending on heuristics;
        # we only require that it finds a contiguous tail.
        self.assertGreaterEqual(layout.num_iterations, len(shifts) - 1)
        rep = body.decompile_body(net)
        self.assertTrue(rep.matches, rep.reason)
        # Exactly one position should vary (the rotate position).
        self.assertEqual(len(rep.positions), 1)
        pr = rep.positions[0]
        self.assertIsNotNone(pr.rotate_gadget)
        rg = pr.rotate_gadget
        self.assertEqual(rg["num_lanes"], 32)
        self.assertEqual(rg["num_registers"], 3)
        # The recovered shifts should be a contiguous suffix of the input list.
        n = len(rg["shifts_per_iter"])
        self.assertEqual(rg["shifts_per_iter"], shifts[-n:],
                         f"recovered {rg['shifts_per_iter']} should be the last "
                         f"{n} of {shifts}")


if __name__ == "__main__":
    unittest.main()
