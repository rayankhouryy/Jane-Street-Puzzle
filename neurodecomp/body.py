"""Body decompiler -- extract per-iteration deltas from the loop body.

The block finder reports, for each in-block position ``p``, a relative
dispersion ``rho_p`` of the weight matrices across iterations.  Positions
with ``rho_p > 0`` carry per-iteration variation: typically a per-round
additive constant or a per-round shift constant in a hand-compiled
hash/cipher.

This module collects those varying entries and emits a per-iteration table.
When the varying entries look like the bits of an integer (one varying
entry per bit, with values in ``{0, 1}``), we group them into bytes /
words and pretty-print as hex -- the same generic move we used for the
head's IV constants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch.nn as nn

from . import block_finder


@dataclass
class VaryingEntry:
    """A single weight or bias entry that takes different values per iteration."""

    kind: str                   # "weight" or "bias"
    index: Tuple[int, ...]      # (out, in) for weights, (out,) for biases
    values_per_iter: List[float]


def _try_detect_rotate_gadget(
    Ws: np.ndarray,
    var_mask: np.ndarray,
) -> Optional[Dict]:
    """Detect a per-iteration rotate-by-s gadget.

    The "rotation selector" pattern: per output row, per iteration, exactly
    one input col within a contiguous ``n_lanes``-wide window has value
    ``+1`` (a varying entry).  The selected col's offset within the window
    encodes the rotation amount via ``s_t = -bit_pos mod n_lanes``.

    Hand-compiled networks usually store registers at lane-aligned column
    offsets (``col_base`` is a multiple of ``n_lanes``); we use that
    assumption to disambiguate the lane boundary.  Each register may have
    *other* entries (e.g. ``-2`` selectors implementing modular adjustment)
    outside the rotation band; those don't affect the shift recovery.
    """
    n_iters, n_out, n_in = Ws.shape
    var_rows = np.where(var_mask.any(axis=1))[0]
    if var_rows.size == 0:
        return None
    if list(var_rows) != list(range(int(var_rows[0]), int(var_rows[-1]) + 1)):
        return None
    n_rows = int(len(var_rows))
    row_base = int(var_rows[0])

    for n_lanes in (8, 16, 32, 64):
        if n_rows % n_lanes != 0:
            continue
        n_registers = n_rows // n_lanes

        # Build the "+1 varying selector" cols per row at iteration 0.
        col_bases: List[int] = []
        ok_layout = True
        rows_iter0_cols: List[List[int]] = []
        for reg in range(n_registers):
            reg_cols_iter0: List[int] = []
            for b in range(n_lanes):
                r = row_base + reg * n_lanes + b
                ones = np.where((Ws[0, r] == 1.0) & var_mask[r])[0]
                if len(ones) != 1:
                    ok_layout = False
                    break
                reg_cols_iter0.append(int(ones[0]))
            if not ok_layout:
                break
            rows_iter0_cols.append(reg_cols_iter0)
            cmin = min(reg_cols_iter0)
            cmax = max(reg_cols_iter0)
            if cmax - cmin >= n_lanes:
                ok_layout = False
                break
            # Assume the lane is aligned to a multiple of n_lanes.
            col_base = (cmin // n_lanes) * n_lanes
            if cmax >= col_base + n_lanes:
                col_base = cmax - n_lanes + 1   # fallback to "tight" lane
            col_bases.append(col_base)
        if not ok_layout:
            continue

        # Recover shifts from register 0, bit 0 (= row_base).
        shifts: Optional[List[int]] = []
        for t in range(n_iters):
            row = Ws[t, row_base]
            ones = np.where((row == 1.0) & var_mask[row_base])[0]
            if len(ones) != 1:
                shifts = None
                break
            bit_pos = int(ones[0]) - col_bases[0]
            if bit_pos < 0 or bit_pos >= n_lanes:
                shifts = None
                break
            shifts.append((-bit_pos) % n_lanes)
        if shifts is None:
            continue

        # Verify consistency across all rows and registers, all iterations.
        ok = True
        for t in range(n_iters):
            s_t = shifts[t]
            for reg in range(n_registers):
                for b in range(n_lanes):
                    r = row_base + reg * n_lanes + b
                    expected_col = col_bases[reg] + ((b - s_t) % n_lanes)
                    if Ws[t, r, expected_col] != 1.0:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        return {
            "num_lanes": n_lanes,
            "num_registers": n_registers,
            "row_range": (row_base, row_base + n_rows),
            "col_bases": col_bases,
            "shifts_per_iter": shifts,
        }
    return None


@dataclass
class PositionReport:
    p: int
    shape: Tuple[int, int]
    num_iterations: int
    rho: float
    varying_weights: List[VaryingEntry] = field(default_factory=list)
    varying_biases: List[VaryingEntry] = field(default_factory=list)
    packed_words: List[str] = field(default_factory=list)
    rotate_gadget: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class BodyReport:
    matches: bool
    reason: str = ""
    period: int = 0
    num_iterations: int = 0
    positions: List[PositionReport] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _values_to_bits(values: Sequence[float]) -> Optional[List[int]]:
    """Return a list of 0/1 ints if every value is exactly 0 or 1, else None."""
    out: List[int] = []
    for v in values:
        if v == 0:
            out.append(0)
        elif v == 1:
            out.append(1)
        else:
            return None
    return out


def _try_pack_as_bytes_per_iter(
    entries: List[VaryingEntry],
    num_iters: int,
    bits_per_unit: int = 8,
) -> Optional[List[str]]:
    """If `entries` is a set of bit-valued varying biases (and the count is a
    multiple of ``bits_per_unit``), return the per-iteration hex
    interpretation.

    Algorithm: sort entries by their flat index, take their per-iteration
    sequence; treat the i-th entry as bit i.  For each iteration t, read off
    bits[0..N-1] and convert to bytes/hex (LSB first).
    """
    if not entries:
        return None
    # Confirm every entry is bit-valued.
    bit_entries: List[Tuple[int, List[int]]] = []
    for e in entries:
        bits = _values_to_bits(e.values_per_iter)
        if bits is None:
            return None
        # Use the first index component as a sort key (out-dim for bias).
        bit_entries.append((e.index[0], bits))
    n = len(bit_entries)
    if n % bits_per_unit != 0:
        return None
    bit_entries.sort(key=lambda kv: kv[0])

    n_units = n // bits_per_unit
    per_iter_hex: List[str] = []
    for t in range(num_iters):
        chunks: List[str] = []
        for u in range(n_units):
            v = 0
            for k in range(bits_per_unit):
                v |= bit_entries[u * bits_per_unit + k][1][t] << k
            chunks.append(f"{v:02x}")
        per_iter_hex.append(" ".join(chunks))
    return per_iter_hex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def decompile_body(model: nn.Sequential) -> BodyReport:
    layout = block_finder.find_blocks(model)
    if not layout.block_starts:
        return BodyReport(matches=False, reason="no body detected")

    linears = [m for m in model if isinstance(m, nn.Linear)]
    period = layout.period
    n_iters = layout.num_iterations

    reports: List[PositionReport] = []
    for p in range(period):
        rho = layout.position_dispersion[p]
        if rho != rho:    # NaN (shape mismatch)
            continue
        if rho == 0.0:
            continue       # bit-identical; no per-iteration variation

        # Collect weight and bias arrays per iteration.
        Ws: List[np.ndarray] = []
        bs: List[Optional[np.ndarray]] = []
        for s in layout.block_starts:
            idx = s + p
            if idx >= len(linears):
                continue
            lin = linears[idx]
            Ws.append(lin.weight.detach().cpu().numpy())
            bs.append(
                lin.bias.detach().cpu().numpy() if lin.bias is not None else None
            )
        W = np.stack(Ws)                # (n_iters, out, in)
        b: Optional[np.ndarray] = None
        if all(x is not None for x in bs):
            b = np.stack(bs)            # (n_iters, out)
        out_dim, in_dim = W.shape[1], W.shape[2]

        # Find varying entries.
        W_var_mask = (W != W[0:1]).any(axis=0)             # (out, in)
        b_var_mask = (
            (b != b[0:1]).any(axis=0) if b is not None else None
        )

        varying_weights: List[VaryingEntry] = []
        for (o, i) in np.argwhere(W_var_mask):
            varying_weights.append(VaryingEntry(
                kind="weight",
                index=(int(o), int(i)),
                values_per_iter=W[:, o, i].tolist(),
            ))
        varying_biases: List[VaryingEntry] = []
        if b is not None and b_var_mask is not None:
            for o in np.where(b_var_mask)[0]:
                varying_biases.append(VaryingEntry(
                    kind="bias",
                    index=(int(o),),
                    values_per_iter=b[:, o].tolist(),
                ))

        notes: List[str] = []
        n_total = out_dim * in_dim
        notes.append(
            f"{len(varying_weights)} of {n_total} weight entries vary across iterations"
        )
        if b is not None:
            notes.append(
                f"{len(varying_biases)} of {out_dim} bias entries vary across iterations"
            )

        # Try packing biases as bytes-per-iter.
        packed = _try_pack_as_bytes_per_iter(varying_biases, len(layout.block_starts))
        packed_words: List[str] = []
        if packed is not None:
            packed_words = packed
            notes.append(
                f"varying biases are bit-valued and pack into "
                f"{len(packed_words[0].split())} bytes per iteration"
            )

        # Try detecting a rotate gadget in weights.
        rotate = _try_detect_rotate_gadget(W, W_var_mask)
        if rotate is not None:
            notes.append(
                f"detected rotate gadget: {rotate['num_registers']} register(s) "
                f"of {rotate['num_lanes']}-bit lanes, shift amounts s_t in "
                f"[0, {rotate['num_lanes']-1}]"
            )

        reports.append(PositionReport(
            p=p,
            shape=(in_dim, out_dim),
            num_iterations=len(layout.block_starts),
            rho=rho,
            varying_weights=varying_weights,
            varying_biases=varying_biases,
            packed_words=packed_words,
            rotate_gadget=rotate,
            notes=notes,
        ))

    return BodyReport(
        matches=True,
        period=period,
        num_iterations=n_iters,
        positions=reports,
        notes=[
            f"detected body period {period} with {n_iters} iterations",
            f"{len(reports)} positions show per-iteration variation",
        ],
    )


def format_report(rep: BodyReport, max_iters_to_show: int = 16) -> str:
    if not rep.matches:
        return f"Body decompilation failed: {rep.reason}"
    lines = [f"## Body decompilation"]
    for n in rep.notes:
        lines.append(f"  note: {n}")

    for pr in rep.positions:
        lines.append("")
        lines.append(
            f"## Position p={pr.p} of {rep.period}: "
            f"Linear({pr.shape[0]} -> {pr.shape[1]}), rho={pr.rho:.4f}"
        )
        for n in pr.notes:
            lines.append(f"  {n}")
        if pr.rotate_gadget:
            rg = pr.rotate_gadget
            lines.append(
                f"  rotate gadget: {rg['num_registers']} register(s) x "
                f"{rg['num_lanes']}-bit lanes, rows {rg['row_range'][0]}..{rg['row_range'][1]-1}, "
                f"col bases {rg['col_bases']}"
            )
            lines.append(f"  recovered shift schedule s_t:")
            for t0 in range(0, len(rg['shifts_per_iter']), 16):
                chunk = rg['shifts_per_iter'][t0 : t0 + 16]
                lines.append(f"    s[{t0:2d}..{t0+len(chunk)-1:2d}] = {chunk}")
        elif pr.packed_words:
            lines.append(
                f"  per-iteration packed (LSB first, {len(pr.packed_words)} iters):"
            )
            for t, hexstr in enumerate(pr.packed_words[:max_iters_to_show]):
                lines.append(f"    iter[{t:3d}]: {hexstr}")
            if len(pr.packed_words) > max_iters_to_show:
                lines.append(
                    f"    ... ({len(pr.packed_words) - max_iters_to_show} more)"
                )
        elif pr.varying_biases:
            lines.append("  varying bias entries (first 12 by index):")
            for ve in pr.varying_biases[:12]:
                seq = ", ".join(f"{int(v)}" if v == int(v) else f"{v:.3g}"
                                for v in ve.values_per_iter[:8])
                lines.append(f"    bias[{ve.index[0]:3d}] -> [{seq}, ...]")
        if pr.varying_weights:
            lines.append(f"  {len(pr.varying_weights)} varying weight entries "
                         f"(showing first 6):")
            for ve in pr.varying_weights[:6]:
                seq = ", ".join(f"{int(v)}" if v == int(v) else f"{v:.3g}"
                                for v in ve.values_per_iter[:8])
                lines.append(
                    f"    W[{ve.index[0]:3d}, {ve.index[1]:3d}] -> [{seq}, ...]"
                )
    return "\n".join(lines)
