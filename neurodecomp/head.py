"""Head decompiler -- run abstract interpretation on the *head* of a
hand-compiled Sequential and emit a per-output-neuron symbolic summary.

The head is the prefix of layers before the first detected loop block
(see :mod:`neurodecomp.block_finder`).  For models without a detected
body (e.g. small synthetic nets) we interpret the whole model.

The output summary groups neurons by their *symbolic role*:

* ``const(c)``: always-c neurons (often zero pads).
* ``passes x[i]``: identity wires that just carry one input byte.
* ``affine(...)``: pure linear combinations of inputs.
* ``opaque(...)``: neurons whose value involves at least one
  saturating ReLU.  These are the data-dependent decision points.

This is intentionally domain-agnostic: nothing here knows about MD5,
hashing, or any specific algorithm.  We only describe what the head
computes structurally.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch.nn as nn

from . import block_finder, interp
from .domains import Affine, VarName, affine_input_only


@dataclass
class HeadReport:
    head_layer_count: int                       # Linear+ReLU children interpreted
    result: interp.InterpResult
    role_counts: Counter = field(default_factory=Counter)
    by_role: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))


def _role(a: Affine) -> str:
    if a.is_const:
        return f"const({a.bias})"
    if not affine_input_only(a):
        return "opaque"
    # purely over inputs
    if len(a.coefs) == 1 and a.coefs[0][1] == 1 and a.bias == 0:
        return "passthrough"
    if all(c == 1 for _, c in a.coefs) and a.bias == 0:
        return f"sum({len(a.coefs)})"
    if all(c == 1 for _, c in a.coefs):
        return f"sum({len(a.coefs)})+const"
    return "affine"


def decompile_head(
    model: nn.Sequential,
    input_ranges: Sequence[Tuple[int, int]] | None = None,
) -> HeadReport:
    """Interpret the head of ``model`` and produce a role-tagged report.

    Args:
        model: ``nn.Sequential`` of Linear/ReLU.
        input_ranges: per-input ``(lo, hi)``.  If None, infer from the model:
          when a tokenizer is recovered we use ``(0, 255)`` for every input.
    """
    # Decide where the head ends.
    layout = block_finder.find_blocks(model)
    children = list(model)
    if layout.num_iterations > 0:
        # The block_finder reports a Linear index. Convert to a child index
        # (Linear i corresponds to child 2*i in a strictly alternating
        # Linear/ReLU sequence).
        head_lin_end = layout.head_end_linear_idx
        head_child_end = head_lin_end * 2     # exclusive
    else:
        head_child_end = len(children)
    head_children = children[:head_child_end]

    # Determine input ranges.  Default to bytes.
    if input_ranges is None:
        in_dim = head_children[0].in_features if isinstance(head_children[0], nn.Linear) else None
        if in_dim is None:
            raise ValueError("cannot determine input dim")
        input_ranges = [(0, 255)] * in_dim

    res = interp.run(model, list(input_ranges), layers=head_children)

    rep = HeadReport(head_layer_count=len(head_children), result=res)
    for i, a in enumerate(res.values):
        r = _role(a)
        # Bucket all numeric-suffix roles into a coarse family for counting.
        family = r.split("(")[0] if "(" in r else r
        rep.role_counts[family] += 1
        rep.by_role[r].append(i)
    return rep


def _extract_constant_runs(
    values: Sequence[Affine],
    bits_per_unit: int = 8,
) -> List[Tuple[int, int, str]]:
    """Scan the values list for runs of consecutive neurons whose value is a
    constant in {0, 1}, group them into ``bits_per_unit`` chunks (LSB-first),
    and return ``(start_index, end_index_inclusive, hex_string)`` tuples.

    This is a generic structural finding: when a hand-compiled head stamps a
    multi-byte constant into its working state, we should see it as a long
    run of {0, 1} constants.  Reporting the packed hex makes such constants
    easy to recognise (e.g. cryptographic IVs, padding bytes, etc.).
    """
    bits: List[int] = []
    bit_start_index: int | None = None
    runs: List[Tuple[int, int, str]] = []

    def flush(end_idx: int) -> None:
        nonlocal bit_start_index
        if bit_start_index is None or not bits:
            bit_start_index = None
            return
        # Group into ``bits_per_unit`` chunks (LSB first).
        n_full = (len(bits) // bits_per_unit) * bits_per_unit
        if n_full == 0:
            bit_start_index = None
            bits.clear()
            return
        hex_chunks = []
        for off in range(0, n_full, bits_per_unit):
            v = 0
            for k in range(bits_per_unit):
                v |= bits[off + k] << k
            hex_chunks.append(f"{v:02x}")
        runs.append((bit_start_index, bit_start_index + n_full - 1, " ".join(hex_chunks)))
        bit_start_index = None
        bits.clear()

    for i, a in enumerate(values):
        if a.is_const and a.bias in (0, 1):
            if bit_start_index is None:
                bit_start_index = i
            bits.append(int(a.bias))
        else:
            flush(i)
    flush(len(values))
    return runs


def format_report(rep: HeadReport, max_neurons: int = 80) -> str:
    res = rep.result
    lines = []
    lines.append(f"## Head decompilation ({rep.head_layer_count} child layers interpreted)")
    lines.append(f"  total output neurons : {len(res.values)}")
    lines.append(f"  opaque ReLU symbols  : {res.num_opaque}")
    in_only = sum(1 for a in res.values if affine_input_only(a))
    lines.append(f"  expressed purely in inputs : {in_only}/{len(res.values)}")
    lines.append("")
    lines.append("## Role distribution")
    for family, n in rep.role_counts.most_common():
        lines.append(f"  {family:14s} {n:4d}")

    lines.append("")
    lines.append("## Per-layer ReLU resolution")
    for s in res.layer_stats:
        if s["kind"] != "ReLU":
            continue
        lines.append(
            f"  layer {s['layer']:3d} ReLU  out={s['out_dim']:4d}  "
            f"non-sat={s['non_sat']:4d}  dead={s['dead']:3d}  sat={s['sat']:3d}"
        )

    # Constant byte runs -- generic structural finding.
    byte_runs = _extract_constant_runs(res.values, bits_per_unit=8)
    if byte_runs:
        lines.append("")
        lines.append("## Constant {0,1} runs grouped as bytes (LSB first)")
        for s, e, hexstr in byte_runs:
            n_bits = e - s + 1
            n_bytes = n_bits // 8
            lines.append(f"  out[{s:3d}..{e:3d}]  ({n_bits} bits, {n_bytes} bytes): {hexstr}")

    # Show neurons by class.
    lines.append("")
    lines.append(f"## First {max_neurons} non-constant output neurons")
    shown = 0
    for i, a in enumerate(res.values):
        if a.is_const:
            continue
        if shown >= max_neurons:
            break
        lo, hi = a.range(res.var_ranges)
        tag = _role(a)
        lines.append(
            f"  out[{i:3d}]  range=[{lo:>6d}, {hi:>6d}]  role={tag}"
        )
        lines.append(f"              = {a.pretty(max_terms=8)}")
        shown += 1
    return "\n".join(lines)
