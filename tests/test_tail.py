"""Tests for neurodecomp.tail.

We build synthetic AND-of-byte-equality networks of arbitrary size with
known targets, run the tail decompiler, and check that it recovers
targets and bit addresses exactly.
"""
from __future__ import annotations

import random
import unittest
from typing import List

import torch
import torch.nn as nn

from neurodecomp import tail
from tests.toy_circuits import _linear


def build_byte_eq_network(targets: List[int]) -> nn.Sequential:
    """Build a Sequential that computes 1[byte_i == targets[i] for all i].

    The input is 8N bits (each in {0, 1}), packed so that
    bits [8*i, 8*(i+1)) hold the binary representation of byte i (LSB first).

    The architecture is the canonical AND-of-Kronecker-deltas template.
    """
    N = len(targets)
    n_bits = 8 * N

    # Penultimate layer: (3N, 8N) computing z_i +/- {1, 0, -1} = byte_i - t + delta.
    # Bit weights 2^0, 2^1, ..., 2^7 for byte i.
    W_p = [[0.0] * n_bits for _ in range(3 * N)]
    b_p: List[float] = []
    # First N rows: z_i + 1 = byte_i - t_i + 1
    for i, t in enumerate(targets):
        for k in range(8):
            W_p[i][8 * i + k] = float(1 << k)
        b_p.append(-float(t) + 1.0)
    # Middle N rows: z_i = byte_i - t_i
    for i, t in enumerate(targets):
        for k in range(8):
            W_p[N + i][8 * i + k] = float(1 << k)
        b_p.append(-float(t))
    # Last N rows: z_i - 1 = byte_i - t_i - 1
    for i, t in enumerate(targets):
        for k in range(8):
            W_p[2 * N + i][8 * i + k] = float(1 << k)
        b_p.append(-float(t) - 1.0)

    # Final layer: (+1)*N, (-2)*N, (+1)*N, bias = -(N-1).
    W_f = [[0.0] * (3 * N)]
    for i in range(N):
        W_f[0][i] = 1.0
        W_f[0][N + i] = -2.0
        W_f[0][2 * N + i] = 1.0
    b_f = [-(N - 1)]

    return nn.Sequential(
        _linear(W_p, b_p),
        nn.ReLU(),
        _linear(W_f, b_f),
        nn.ReLU(),
    )


def bytes_to_bits(byte_values: List[int]) -> List[int]:
    bits: List[int] = []
    for v in byte_values:
        for k in range(8):
            bits.append((v >> k) & 1)
    return bits


class TailDecompilerTests(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)

    def test_recovers_targets_n_4(self):
        targets = [0xC7, 0xEF, 0x65, 0x23]
        net = build_byte_eq_network(targets)
        rep = tail.decompile_tail(net)
        self.assertTrue(rep.matches, rep.reason)
        self.assertEqual(rep.N, 4)
        self.assertEqual([int(t) for t in rep.targets], targets)
        self.assertTrue(rep.targets_are_bytes)
        self.assertEqual(rep.targets_hex, "c7ef6523")

    def test_recovers_targets_n_16(self):
        targets = [random.randint(0, 255) for _ in range(16)]
        net = build_byte_eq_network(targets)
        rep = tail.decompile_tail(net)
        self.assertTrue(rep.matches, rep.reason)
        self.assertEqual(rep.N, 16)
        self.assertEqual([int(t) for t in rep.targets], targets)
        self.assertTrue(rep.targets_are_bytes)
        self.assertEqual(rep.targets_hex, "".join(f"{t:02x}" for t in targets))

    def test_each_predicate_recovers_bit_addresses(self):
        targets = [0x42, 0x00, 0xFF]
        net = build_byte_eq_network(targets)
        rep = tail.decompile_tail(net)
        for p in rep.predicates:
            self.assertEqual(p.bit_kind, "powers_of_two")
            self.assertEqual(p.num_terms, 8)
            # Each predicate i should reference bits 8i..8i+7 with +1, +2, +4, ..., +128.
            expected = [(8 * p.i + k, float(1 << k)) for k in range(8)]
            actual = sorted(p.bit_addresses)
            self.assertEqual(actual, expected, f"predicate {p.i} bit addresses wrong")

    def test_network_behaves_correctly_on_targets(self):
        """Sanity: the synthetic network outputs 1 exactly on the target byte string."""
        targets = [0xC7, 0xEF, 0x65, 0x23, 0x3C, 0x40, 0xAA, 0x32]
        net = build_byte_eq_network(targets)
        # Hit -> output 1.
        x_hit = torch.tensor(bytes_to_bits(targets), dtype=torch.float32)
        self.assertEqual(float(net(x_hit).item()), 1.0)
        # Miss -> output 0.
        wrong = list(targets)
        wrong[3] ^= 0x01
        x_miss = torch.tensor(bytes_to_bits(wrong), dtype=torch.float32)
        self.assertEqual(float(net(x_miss).item()), 0.0)

    def test_rejects_non_template_last_layer(self):
        # Slightly off bias -> shouldn't match.
        net = build_byte_eq_network([1, 2, 3])
        # Mess with the last layer's bias.
        last = list(net.children())[-2]    # Linear before final ReLU
        with torch.no_grad():
            last.bias.add_(0.5)
        rep = tail.decompile_tail(net)
        self.assertFalse(rep.matches)

    def test_rejects_when_penu_not_triple_replicated(self):
        net = build_byte_eq_network([1, 2, 3])
        penu = list(net.children())[0]
        with torch.no_grad():
            penu.weight[0, 0] += 1.0    # break triple replication on first row
        rep = tail.decompile_tail(net)
        self.assertFalse(rep.matches)


if __name__ == "__main__":
    unittest.main()
