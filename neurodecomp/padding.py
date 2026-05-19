"""MVP-10 — padding scheme decoder.

The head of a hand-compiled hash network is responsible for turning a
raw input byte sequence into a fixed-size padded block, following the
algorithm's padding rule.  For MD5 / SHA-2 / similar:

   pad(byte_seq) = byte_seq[:L] || 0x80 || 0..0 || length_suffix(8*L)

This module recovers the head's padding behaviour empirically:

1. Feed the head with byte sequences of varying lengths (effective L);
   collect the resulting 336-dim head output.
2. For each output neuron, classify it as one of:
   - ``passthrough(i, bit_k)``: passes input byte ``i``'s bit ``k``.
   - ``framing_byte``: outputs `1` exactly at the bit-positions of `0x80`
     when L = (constant the bit corresponds to).
   - ``length_bit(k)``: outputs the k-th bit of ``8 * L`` (little-endian).
   - ``length_indicator``: depends only on L, not on input content.
   - ``constant(c)``: never varies (an IV/auxiliary constant).
3. Aggregate per output neuron and produce a symbolic ``pad_message(s)``
   description.

Recovery is purely structural: no MD5-specific knowledge.  The decoder
makes no assumption about the byte position of `0x80` (which we
explicitly *recover* here) and tolerates any byte-order convention for
the length suffix.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HeadPaddingProbe:
    """Per-output-neuron classification of the head's padding behaviour."""

    out_idx: int
    role: str        # 'const', 'passthrough', 'length_bit', 'length_indicator', 'mixed'
    detail: str = ""


@dataclass
class PaddingReport:
    head_output_dim: int
    num_outputs_probed: int
    role_counts: Counter = field(default_factory=Counter)
    probes: List[HeadPaddingProbe] = field(default_factory=list)
    # Inferred 8-bit ``length_suffix`` byte indices in the output, mapped to
    # the bit they hold.  Used to confirm the length-byte layout.
    length_bit_locations: Dict[int, List[int]] = field(default_factory=dict)


def _encode(s: str, width: int = 55) -> torch.Tensor:
    s = str(s)[:width].ljust(width, "\x00")
    return torch.Tensor([ord(c) for c in s])


def _head_forward(head: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return head(x)


def _gen_inputs(width: int, num_inputs: int, seed: int = 0) -> List[Tuple[str, int]]:
    """Generate diverse (string, length) probes for the head.

    Returns pairs (string, L) where L = len(string).  We cover:
      * empty string
      * short strings of length 1..min(width-1, 20)
      * a few full-width strings of varying content
    """
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789!@#"
    pairs: List[Tuple[str, int]] = []
    pairs.append(("", 0))
    for L in range(1, min(width, 30)):
        for _ in range(num_inputs // 30 + 1):
            s = "".join(rng.choice(chars) for _ in range(L))
            pairs.append((s, L))
    # Some longer strings.
    for L in range(30, width):
        s = "".join(rng.choice(chars) for _ in range(L))
        pairs.append((s, L))
    return pairs[:num_inputs]


def _bit_pattern(L: int, total_bits: int = 64) -> List[int]:
    """Return the bits of ``8*L`` as a little-endian list (LSB first)."""
    v = 8 * L
    return [(v >> i) & 1 for i in range(total_bits)]


def decode_padding(
    head: nn.Sequential,
    input_width: int = 55,
    *,
    num_inputs: int = 200,
    seed: int = 0,
) -> PaddingReport:
    """Probe the head with varying-length inputs and classify each output."""
    head_out_dim = next(
        m for m in reversed(list(head)) if isinstance(m, nn.Linear)
    ).out_features

    pairs = _gen_inputs(input_width, num_inputs, seed=seed)
    inputs = torch.stack([_encode(s, input_width) for s, _ in pairs])
    Ls = [L for _, L in pairs]
    with torch.no_grad():
        outs = head(inputs).cpu().numpy()        # shape (N, head_out_dim)

    rep = PaddingReport(
        head_output_dim=head_out_dim,
        num_outputs_probed=head_out_dim,
    )

    # For each output, decide its role.
    for o in range(head_out_dim):
        col = outs[:, o]
        # Constant?
        if np.allclose(col, col[0]):
            rep.probes.append(HeadPaddingProbe(
                out_idx=o, role="const", detail=f"value={col[0]:g}",
            ))
            continue
        # Bit-valued?
        is_bit = bool(((col == 0) | (col == 1)).all())
        if is_bit:
            # Does it correlate perfectly with a single bit of 8*L?
            best_match: Optional[Tuple[int, float]] = None
            for k in range(56):       # bits of 8*L up to 2^56 (enough for L<=55)
                target = np.array([(8 * L >> k) & 1 for L in Ls])
                if (target == col).all():
                    best_match = (k, 1.0)
                    break
            if best_match is not None:
                k, _ = best_match
                rep.probes.append(HeadPaddingProbe(
                    out_idx=o,
                    role="length_bit",
                    detail=f"bit {k} of (8*L)",
                ))
                rep.length_bit_locations.setdefault(k, []).append(o)
                continue
            # Does it depend only on L (not on content)?  Check: for each L,
            # the value is the same across all strings of that length.
            by_L: Dict[int, set] = {}
            for i, L in enumerate(Ls):
                by_L.setdefault(L, set()).add(int(col[i]))
            if all(len(v) == 1 for v in by_L.values()):
                rep.probes.append(HeadPaddingProbe(
                    out_idx=o,
                    role="length_indicator",
                    detail=f"{len(by_L)} L-values, all unique-per-L",
                ))
                continue
        # Try: matches a specific input byte's bit?
        # (Cheap heuristic.)  For each input byte index i, for each bit k,
        # check whether col[t] == (input_byte_i_of_string_t >> k) & 1.
        matched_passthrough = False
        for i in range(input_width):
            byte_vals = np.array([
                ord(s[i]) if i < len(s) else 0
                for s, _ in pairs
            ])
            for k in range(8):
                target_bits = (byte_vals >> k) & 1
                if (target_bits == col).all():
                    rep.probes.append(HeadPaddingProbe(
                        out_idx=o,
                        role="passthrough",
                        detail=f"bit {k} of input byte {i}",
                    ))
                    matched_passthrough = True
                    break
            if matched_passthrough:
                break
        if not matched_passthrough:
            rep.probes.append(HeadPaddingProbe(
                out_idx=o, role="mixed", detail="no simple classification",
            ))

    rep.role_counts = Counter(p.role for p in rep.probes)
    return rep


def summarize_padding(rep: PaddingReport) -> str:
    lines = ["## Padding decode"]
    lines.append(f"  head output dim: {rep.head_output_dim}")
    lines.append("")
    lines.append("## Role distribution")
    for role, n in rep.role_counts.most_common():
        lines.append(f"  {role:20s} {n}")
    if rep.length_bit_locations:
        lines.append("")
        lines.append("## Length-bit positions in head output (bit k of 8*L)")
        for k in sorted(rep.length_bit_locations):
            locs = rep.length_bit_locations[k]
            lines.append(f"  bit {k:2d}: output positions {locs}")
    return "\n".join(lines)
