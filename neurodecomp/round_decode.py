"""MVP-9 — round-function decoding.

For a single body iteration (one 42-layer block), recover the per-output-bit
boolean function (truth table) that the iteration computes on its register
inputs.

Approach:

1. **Isolate one iteration** as a sub-Sequential.  Its inputs are the 256
   bits of the inter-iteration state (plus 32 control bits).
2. **Probe per output bit.**  For each output bit ``o`` of the iteration's
   output state, find which subset of input bits it depends on by varying
   one input bit at a time on a random base; the bits whose flips change
   ``o`` form ``deps(o)``.
3. **Truth-table extraction.**  Exhaustively evaluate ``o`` on
   ``{0, 1}^|deps(o)|`` to get the 3-input (or k-input) truth table.
4. **Classify** by truth-table identifier against a small named catalog
   (well-known 3-input boolean functions).

The output: per-output-bit ``BitFunction`` records that downstream codegen
can group by register/bit position and pretty-print as named operators.

This module is intentionally *empirical* about one iteration's structure --
it does not rely on the MVP-7 motif scanner. The scanner is good at
detecting AND-shaped rows; this module evaluates each bit's full boolean
behaviour.

The cost is one forward pass per `{0, 1}^k` input combination per probed
output, but the sub-network has only 42 layers (vs the full 5,442), so even
2^16 evaluations are fast.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sub-network construction
# ---------------------------------------------------------------------------

def extract_iteration_subnet(
    model: nn.Sequential,
    iteration_index: int,
    block_starts: Sequence[int],
    period: int,
) -> nn.Sequential:
    """Return the 42-layer (period) sub-Sequential for one body iteration.

    The sub-network's `forward` consumes the iteration's input state and
    returns its output state."""
    s = block_starts[iteration_index]
    # The body is alternating Linear/ReLU; one iteration spans 2*period
    # child modules.  Linear at index s corresponds to child 2s in the
    # parent model (assuming strict Linear/ReLU alternation).
    children = list(model)
    start_child = 2 * s
    end_child = 2 * (s + period)
    return nn.Sequential(*children[start_child:end_child])


def find_iteration_input_dim(subnet: nn.Sequential) -> int:
    """The input dim is the first Linear's in_features."""
    return next(m for m in subnet if isinstance(m, nn.Linear)).in_features


def find_iteration_output_dim(subnet: nn.Sequential) -> int:
    """The output dim is the last Linear's out_features."""
    last = [m for m in subnet if isinstance(m, nn.Linear)][-1]
    return last.out_features


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------

def _forward_int(subnet: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Forward pass returning ints if outputs come out as integers."""
    with torch.no_grad():
        out = subnet(x)
    return out


def find_dependencies(
    subnet: nn.Sequential,
    output_idx: int,
    in_dim: int,
    *,
    base_seed: int = 0,
    num_probes: int = 4,
    input_value_for_one: float = 1.0,
    input_value_for_zero: float = 0.0,
) -> List[int]:
    """Find which input bits affect ``output_idx`` by flipping each bit on
    several random base vectors and seeing whether the output changes.

    To avoid being misled by structural noise we run ``num_probes`` random
    Boolean base vectors and union the dependency sets.  Inputs flipped
    from 0 to 1 with a *non-zero output delta* are considered dependencies.
    """
    rng = random.Random(base_seed)
    deps: set = set()
    for probe in range(num_probes):
        # Random Boolean base vector.
        base = torch.tensor(
            [
                input_value_for_one if rng.random() < 0.5 else input_value_for_zero
                for _ in range(in_dim)
            ],
            dtype=torch.float32,
        )
        baseline = float(_forward_int(subnet, base)[output_idx].item())
        for i in range(in_dim):
            flipped = base.clone()
            flipped[i] = (
                input_value_for_zero
                if base[i].item() == input_value_for_one
                else input_value_for_one
            )
            new = float(_forward_int(subnet, flipped)[output_idx].item())
            if abs(new - baseline) > 1e-9:
                deps.add(i)
    return sorted(deps)


def truth_table(
    subnet: nn.Sequential,
    output_idx: int,
    deps: Sequence[int],
    in_dim: int,
    *,
    other_bits_value: float = 0.0,
    input_value_for_one: float = 1.0,
    max_arity: int = 8,
) -> Optional[Dict[Tuple[int, ...], int]]:
    """Enumerate ``output_idx`` over all 2^|deps| boolean combinations of
    its dependency bits, holding all other inputs to ``other_bits_value``.

    Returns a dict mapping ``(b_0, b_1, ..., b_{k-1})`` to ``int(output)``.
    Returns ``None`` if |deps| > ``max_arity``.
    """
    if len(deps) > max_arity:
        return None
    k = len(deps)
    table: Dict[Tuple[int, ...], int] = {}
    base = torch.full((in_dim,), other_bits_value, dtype=torch.float32)
    for mask in range(1 << k):
        x = base.clone()
        bits = []
        for j in range(k):
            b = (mask >> j) & 1
            bits.append(b)
            if b:
                x[deps[j]] = input_value_for_one
            else:
                x[deps[j]] = other_bits_value
        out = float(_forward_int(subnet, x)[output_idx].item())
        if abs(out - round(out)) > 1e-6:
            return None    # non-integer output → not booleanly recoverable
        table[tuple(bits)] = int(round(out))
    return table


# ---------------------------------------------------------------------------
# Truth-table catalog (named boolean functions on small arities)
# ---------------------------------------------------------------------------

# Encode each truth table as a single int: bit ``i`` of the int holds the
# value at input combination ``i`` (in 0..2^k-1) where input combination
# ``i`` interprets ``i`` as the bit vector (b_0 = i & 1, b_1 = (i >> 1) & 1, ...).
#
# We catalog all the named 1, 2, and 3-input boolean functions we'd expect
# in cryptographic hand-compilation, including MD5's F/G/H/I.

def _table_id_safe(table: Dict[Tuple[int, ...], int]) -> Optional[int]:
    """Return the boolean table id if all values are in {0, 1}, else None.

    For tables with integer-valued outputs (e.g. adder bit-slices), the
    classifier reports a separate ``integer_table_signature`` instead.
    """
    if not all(v in (0, 1) for v in table.values()):
        return None
    k = len(next(iter(table.keys())))
    n = 1 << k
    out = 0
    for mask in range(n):
        bits = tuple((mask >> j) & 1 for j in range(k))
        if table[bits]:
            out |= 1 << mask
    return out


def _table_id(table: Dict[Tuple[int, ...], int]) -> int:
    """Legacy boolean-only id (raises if values are not in {0,1})."""
    tid = _table_id_safe(table)
    if tid is None:
        raise ValueError("truth table has non-boolean values")
    return tid


def _table_signature(table: Dict[Tuple[int, ...], int]) -> str:
    """Compact string signature for non-boolean truth tables.

    Used as a *canonical name* so identical multi-valued tables (e.g. the
    same 2-bit adder bit-slice at multiple positions) cluster together.
    """
    k = len(next(iter(table.keys())))
    n = 1 << k
    vals = []
    for mask in range(n):
        bits = tuple((mask >> j) & 1 for j in range(k))
        vals.append(table[bits])
    return f"int_table[k={k}, values={tuple(vals)}]"


def _build_named_table(k: int, fn) -> int:
    """Build the table id for a boolean function `fn(*bits)` over k inputs."""
    n = 1 << k
    out = 0
    for mask in range(n):
        bits = tuple((mask >> j) & 1 for j in range(k))
        if fn(*bits):
            out |= 1 << mask
    return out


# Named 2-input functions
_K2 = lambda f: _build_named_table(2, f)
NAMED_2: Dict[int, str] = {
    _K2(lambda a, b: 0):            "false",
    _K2(lambda a, b: 1):            "true",
    _K2(lambda a, b: a):            "a",
    _K2(lambda a, b: b):            "b",
    _K2(lambda a, b: 1 - a):        "~a",
    _K2(lambda a, b: 1 - b):        "~b",
    _K2(lambda a, b: a & b):        "a AND b",
    _K2(lambda a, b: a | b):        "a OR b",
    _K2(lambda a, b: a ^ b):        "a XOR b",
    _K2(lambda a, b: 1 - (a & b)):  "NAND(a, b)",
    _K2(lambda a, b: 1 - (a | b)):  "NOR(a, b)",
    _K2(lambda a, b: 1 - (a ^ b)):  "XNOR(a, b)",
    _K2(lambda a, b: (1 - a) & b):  "~a AND b",
    _K2(lambda a, b: a & (1 - b)):  "a AND ~b",
    _K2(lambda a, b: (1 - a) | b):  "~a OR b",
    _K2(lambda a, b: a | (1 - b)):  "a OR ~b",
}

# Named 3-input functions, including the four MD5 round functions
_K3 = lambda f: _build_named_table(3, f)
NAMED_3: Dict[int, str] = {
    # MD5 F: (B AND C) OR (NOT B AND D); with input order (B, C, D)
    _K3(lambda B, C, D: (B & C) | ((1 - B) & D)):  "F(B,C,D) = (B AND C) OR (NOT B AND D)  [MD5 round 1]",
    # MD5 G: (D AND B) OR (NOT D AND C); with input order (B, C, D)
    _K3(lambda B, C, D: (D & B) | ((1 - D) & C)):  "G(B,C,D) = (B AND D) OR (C AND NOT D)  [MD5 round 2]",
    # MD5 H: B XOR C XOR D
    _K3(lambda B, C, D: B ^ C ^ D):                "H(B,C,D) = B XOR C XOR D               [MD5 round 3]",
    # MD5 I: C XOR (B OR NOT D); input order (B, C, D)
    _K3(lambda B, C, D: C ^ (B | (1 - D))):        "I(B,C,D) = C XOR (B OR NOT D)          [MD5 round 4]",
    # Common helpers
    _K3(lambda a, b, c: a & b & c):                "a AND b AND c",
    _K3(lambda a, b, c: a | b | c):                "a OR b OR c",
    _K3(lambda a, b, c: a ^ b ^ c):                "a XOR b XOR c",
    _K3(lambda a, b, c: (a & b) | c):              "(a AND b) OR c",
    _K3(lambda a, b, c: a & (b | c)):              "a AND (b OR c)",
}


def classify_truth_table(table: Dict[Tuple[int, ...], int]) -> Tuple[str, Optional[int]]:
    """Return ``(name, table_id)`` for the truth table.

    For Boolean tables, the name is looked up in the named catalog; the
    ``table_id`` is the integer bit-vector.  For integer-valued tables
    (e.g. modular-adder bit-slices), ``table_id`` is None and the name
    is the canonical signature ``int_table[...]`` so identical adders
    cluster together.
    """
    tid = _table_id_safe(table)
    if tid is None:
        return _table_signature(table), None
    k = len(next(iter(table.keys())))
    catalog = {1: {}, 2: NAMED_2, 3: NAMED_3}.get(k, {})
    return catalog.get(tid, "unknown"), tid


# ---------------------------------------------------------------------------
# Iteration decoder
# ---------------------------------------------------------------------------

@dataclass
class BitFunction:
    """The recovered boolean function for one output bit of one iteration."""

    output_idx: int
    deps: List[int]
    table: Optional[Dict[Tuple[int, ...], int]]
    table_id: Optional[int]
    name: str = "unknown"


@dataclass
class IterationReport:
    iteration_index: int
    in_dim: int
    out_dim: int
    bit_functions: List[BitFunction] = field(default_factory=list)


def find_dependencies_batched(
    subnet: nn.Sequential,
    output_idx: int,
    in_dim: int,
    *,
    base_seed: int = 0,
    num_probes: int = 4,
    input_value_for_one: float = 1.0,
    input_value_for_zero: float = 0.0,
) -> List[int]:
    """Batched variant of :func:`find_dependencies`.

    For each probe, we generate a base vector and ``in_dim + 1`` variants:
    the base plus one variant with each input bit flipped.  All ``in_dim + 1``
    variants are run through the subnet as one batch, then deltas are read
    off in O(in_dim) time.
    """
    rng = random.Random(base_seed)
    deps: set = set()
    for probe in range(num_probes):
        base = torch.tensor(
            [
                input_value_for_one if rng.random() < 0.5 else input_value_for_zero
                for _ in range(in_dim)
            ],
            dtype=torch.float32,
        )
        # Build a batch: row 0 = base, rows 1..in_dim+1 = base with bit i flipped.
        batch = base.unsqueeze(0).expand(in_dim + 1, in_dim).clone()
        for i in range(in_dim):
            batch[i + 1, i] = (
                input_value_for_zero
                if base[i].item() == input_value_for_one
                else input_value_for_one
            )
        with torch.no_grad():
            outs = subnet(batch)[:, output_idx]
        baseline = float(outs[0].item())
        outs_flipped = outs[1:].cpu().numpy()
        for i in range(in_dim):
            if abs(float(outs_flipped[i]) - baseline) > 1e-9:
                deps.add(i)
    return sorted(deps)


def truth_table_batched(
    subnet: nn.Sequential,
    output_idx: int,
    deps: Sequence[int],
    in_dim: int,
    *,
    other_bits_value: float = 0.0,
    input_value_for_one: float = 1.0,
    max_arity: int = 8,
) -> Optional[Dict[Tuple[int, ...], int]]:
    """Batched truth-table enumeration over all 2^|deps| input combinations."""
    if len(deps) > max_arity:
        return None
    k = len(deps)
    n = 1 << k
    base = torch.full((in_dim,), other_bits_value, dtype=torch.float32)
    batch = base.unsqueeze(0).expand(n, in_dim).clone()
    for mask in range(n):
        for j in range(k):
            if (mask >> j) & 1:
                batch[mask, deps[j]] = input_value_for_one
    with torch.no_grad():
        outs = subnet(batch)[:, output_idx]
    table: Dict[Tuple[int, ...], int] = {}
    for mask in range(n):
        v = float(outs[mask].item())
        if abs(v - round(v)) > 1e-6:
            return None
        bits = tuple((mask >> j) & 1 for j in range(k))
        table[bits] = int(round(v))
    return table


def decode_iteration(
    subnet: nn.Sequential,
    iteration_index: int,
    *,
    output_indices: Optional[Sequence[int]] = None,
    max_arity: int = 6,
    num_probes: int = 4,
    batched: bool = True,
) -> IterationReport:
    """Decode (a subset of) the iteration's output bits."""
    in_dim = find_iteration_input_dim(subnet)
    out_dim = find_iteration_output_dim(subnet)
    if output_indices is None:
        output_indices = list(range(out_dim))

    rep = IterationReport(
        iteration_index=iteration_index,
        in_dim=in_dim,
        out_dim=out_dim,
    )
    find_fn = find_dependencies_batched if batched else find_dependencies
    truth_fn = truth_table_batched if batched else truth_table
    for o in output_indices:
        deps = find_fn(subnet, o, in_dim, num_probes=num_probes)
        table = None
        table_id = None
        name = "unknown"
        if 1 <= len(deps) <= max_arity:
            table = truth_fn(subnet, o, deps, in_dim, max_arity=max_arity)
            if table is not None:
                name, table_id = classify_truth_table(table)
        rep.bit_functions.append(BitFunction(
            output_idx=o,
            deps=list(deps),
            table=table,
            table_id=table_id,
            name=name,
        ))
    return rep


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize_iteration(rep: IterationReport) -> str:
    from collections import Counter
    lines = [f"## Iteration {rep.iteration_index} decoding"]
    lines.append(f"  in_dim={rep.in_dim} out_dim={rep.out_dim} "
                 f"probed={len(rep.bit_functions)} outputs")
    arity_counts: Counter = Counter()
    name_counts: Counter = Counter()
    table_id_counts: Counter = Counter()
    constant_outputs = 0
    high_arity = 0
    for bf in rep.bit_functions:
        if not bf.deps:
            constant_outputs += 1
            continue
        if bf.table is None:
            high_arity += 1
            continue
        arity_counts[len(bf.deps)] += 1
        name_counts[bf.name] += 1
        if bf.table_id is not None:
            table_id_counts[(len(bf.deps), bf.table_id)] += 1
    lines.append(f"  constant outputs : {constant_outputs}")
    lines.append(f"  high-arity (>{rep.bit_functions[0].deps if rep.bit_functions else '?'}): {high_arity}")
    lines.append("")
    lines.append("## Arity distribution")
    for k, n in sorted(arity_counts.items()):
        lines.append(f"  {k}-input  {n}")
    lines.append("")
    lines.append("## Named function distribution")
    for name, n in name_counts.most_common():
        lines.append(f"  {n:4d}  {name}")
    lines.append("")
    lines.append("## Top truth-table ids (arity, id, count)")
    for (k, tid), n in table_id_counts.most_common(8):
        lines.append(f"  k={k}, id=0x{tid:0x}  ->  {n} outputs")
    return "\n".join(lines)
