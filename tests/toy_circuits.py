"""Toy hand-built ReLU circuits used as test fixtures.

These networks have known semantics on integer inputs; downstream stages
(SSA extraction, abstract interpretation, motif matching) should reproduce
them exactly.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _linear(weights: List[List[float]], bias: List[float]) -> nn.Linear:
    out_dim, in_dim = len(weights), len(weights[0])
    layer = nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))
    return layer


def net_identity() -> nn.Sequential:
    """f(x0, x1) = (ReLU(x0), ReLU(x1))"""
    return nn.Sequential(
        _linear([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0]),
        nn.ReLU(),
    )


def net_xor_bool() -> nn.Sequential:
    """XOR(a, b) for a, b in {0, 1}: a + b - 2 * ReLU(a + b - 1)"""
    return nn.Sequential(
        _linear([[1.0, 1.0], [1.0, 1.0]], [0.0, -1.0]),
        nn.ReLU(),
        _linear([[1.0, -2.0]], [0.0]),
    )


def net_kron_delta() -> nn.Sequential:
    """1[x = 0] for integer x: ReLU(x+1) + ReLU(x-1) - 2 ReLU(x)"""
    return nn.Sequential(
        _linear([[1.0], [1.0], [1.0]], [1.0, -1.0, 0.0]),
        nn.ReLU(),
        _linear([[1.0, 1.0, -2.0]], [0.0]),
    )


def net_byte_equality(target: int) -> nn.Sequential:
    """1[x = target] for integer x using Kronecker-delta construction."""
    return nn.Sequential(
        _linear([[1.0], [1.0], [1.0]], [1.0 - target, -1.0 - target, -float(target)]),
        nn.ReLU(),
        _linear([[1.0, 1.0, -2.0]], [0.0]),
    )


def net_and_of_byte_eq(targets: List[int]) -> nn.Sequential:
    """AND_i 1[x_i = targets[i]] using the same template as the puzzle's tail.

    Each Kronecker delta uses 3 ReLUs; for n targets the network has 3n hidden
    units. Final layer applies (+1, +1, …, -2, …, +1, …) – bias (n-1).
    """
    n = len(targets)

    # Layer 1: x_i - target_i, x_i - target_i - 1, x_i - target_i + 1 ... but
    # remember each Kronecker delta needs (x+1, x-1, x) shifted by target.
    # We use n inputs, output 3*n.
    W1 = []
    b1 = []
    for i, t in enumerate(targets):
        row_x_plus_1 = [0.0] * n
        row_x_plus_1[i] = 1.0
        row_x_minus_1 = [0.0] * n
        row_x_minus_1[i] = 1.0
        row_x = [0.0] * n
        row_x[i] = 1.0
        W1.append(row_x_plus_1)
        b1.append(1.0 - t)
        W1.append(row_x_minus_1)
        b1.append(-1.0 - t)
        W1.append(row_x)
        b1.append(-float(t))

    # Layer 2: sum_i (relu(x+1) + relu(x-1) - 2 relu(x))   (then bias -(n-1))
    W2 = [[0.0] * (3 * n)]
    for i in range(n):
        W2[0][3 * i + 0] = 1.0
        W2[0][3 * i + 1] = 1.0
        W2[0][3 * i + 2] = -2.0
    b2 = [-(n - 1)]

    return nn.Sequential(
        _linear(W1, b1),
        nn.ReLU(),
        _linear(W2, b2),
        nn.ReLU(),
    )
