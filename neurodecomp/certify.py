"""MVP-7 — domain-certified motif recognition.

Glue between Stage 3 (abstract interpretation, `neurodecomp.interp`) and
Stage 5 (motif library, `neurodecomp.motifs`).  The motif scanner finds
syntactic row patterns that *could* implement a known operation; this
module *certifies* each candidate by checking that its input neurons
live in the right abstract domain.

Concretely, a candidate ``bool_and`` row ``ReLU(a + b - 1)`` is a true
boolean AND iff both ``a`` and ``b`` are provably in ``{0, 1}``.  We
look at the SSA values produced by the abstract interpreter and read
off the certified domain of each input.

**MVP-7.5 — relational certification.**  The certifier now consults the
*source-expression registry* recorded by the abstract interpreter (one
entry per saturating ReLU).  This lets us certify booleanness for opaque
symbols whose raw range is wider than ``[0, 1]``, as long as the source
affine that fed the ReLU has integer ``hi`` ≤ 1.  This dramatically
reduces false-negative ``bool_and`` rejections that came from the older
range-only check.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from . import interp, motifs
from .domains import (
    Affine, OpaqueSource, VarName, certify_bool_via_source,
)
from .motifs import MotifHit


# ---------------------------------------------------------------------------
# Abstract domain readout (relational)
# ---------------------------------------------------------------------------

def _certify_bool(
    affine: Affine,
    var_ranges: Dict[VarName, Tuple[int, int]],
    opaque_registry: Optional[Dict[VarName, OpaqueSource]] = None,
) -> bool:
    """Return True iff this Affine is provably in ``{0, 1}``.

    Two paths:

    1. **Range path.**  If the affine's integer range is a subset of [0, 1],
       it's directly Boolean.  This is the original MVP-7 check.
    2. **Source-expression path (MVP-7.5).**  If the affine is a single
       opaque ReLU symbol (i.e. ``+1 * r[k]`` with no bias) and the
       pre-ReLU source has integer ``hi ≤ 1``, the post-ReLU value is in
       ``{0, 1}`` -- even when the raw opaque range is wider than [0, 1].
    """
    # Path 1: cheap range check.
    try:
        lo, hi = affine.range(var_ranges)
    except KeyError:
        return False
    if lo >= 0 and hi <= 1:
        return True
    # Path 2: relational lookup through the opaque registry.
    if (
        opaque_registry is not None
        and len(affine.coefs) == 1
        and affine.bias == 0
        and affine.coefs[0][1] == 1
    ):
        v, _ = affine.coefs[0]
        if v[0] == "relu":
            return certify_bool_via_source(v, var_ranges, opaque_registry)
    return False


def _certify_byte(
    affine: Affine,
    var_ranges: Dict[VarName, Tuple[int, int]],
) -> bool:
    try:
        lo, hi = affine.range(var_ranges)
    except KeyError:
        return False
    return lo >= 0 and hi <= 255


# ---------------------------------------------------------------------------
# Per-neuron Affine lookup
# ---------------------------------------------------------------------------

@dataclass
class LayerSnapshot:
    """All Affines after a given Linear's input (= the running tensor that
    will be passed to the next module)."""
    layer_idx: int
    values: List[Affine]


def _snapshot_after_each_relu(
    model: nn.Sequential,
    input_ranges: Sequence[Tuple[int, int]],
) -> Tuple[List[LayerSnapshot], Dict[VarName, Tuple[int, int]], Dict[VarName, OpaqueSource]]:
    """Run abstract interpretation and snapshot the running tensor after
    each ReLU (= the input to the next Linear).  We need these snapshots
    so that, given a motif hit at Linear ``li``, we can look up the
    abstract values of *its inputs*.

    Returns ``(snapshots, var_ranges, opaque_registry)`` where the third
    is the MVP-7.5 source-expression registry used by the certifier to
    refine boolean checks.
    """
    children = list(model)
    snapshots: List[LayerSnapshot] = []
    current = [Affine.of(("in", i), 1, 0) for i in range(len(input_ranges))]
    var_ranges: Dict[VarName, Tuple[int, int]] = {
        ("in", i): (int(lo), int(hi)) for i, (lo, hi) in enumerate(input_ranges)
    }
    opaque_registry: Dict[VarName, OpaqueSource] = {}
    snapshots.append(LayerSnapshot(layer_idx=-1, values=list(current)))

    n_opaque = 0
    linear_idx = -1
    for child_idx, layer in enumerate(children):
        if isinstance(layer, nn.Linear):
            linear_idx += 1
            import numpy as np
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
            new_current: List[Affine] = []
            for o in range(layer.out_features):
                acc = Affine.const(int(b[o]) if b is not None else 0)
                row = W[o]
                for i in np.nonzero(row)[0]:
                    acc = acc + current[i] * int(row[i])
                new_current.append(acc)
            current = new_current
        elif isinstance(layer, nn.ReLU):
            new_current = []
            for src in current:
                lo, hi = src.range(var_ranges)
                if lo >= 0:
                    new_current.append(src)
                elif hi <= 0:
                    new_current.append(Affine.const(0))
                else:
                    sym: VarName = ("relu", n_opaque)
                    n_opaque += 1
                    var_ranges[sym] = (0, int(hi))
                    opaque_registry[sym] = OpaqueSource(expr=src)
                    new_current.append(Affine.of(sym, 1, 0))
            current = new_current
            snapshots.append(LayerSnapshot(layer_idx=linear_idx, values=list(current)))
        else:
            raise ValueError(f"unsupported layer {type(layer).__name__}")
    return snapshots, var_ranges, opaque_registry


def _values_seen_by_linear(
    snapshots: List[LayerSnapshot],
    linear_idx: int,
) -> Optional[List[Affine]]:
    """Return the Affine values that the Linear at ``linear_idx`` reads.

    Linear 0 reads the raw input snapshot (snapshots[0], layer_idx == -1).
    Linear k reads the snapshot taken after the ReLU following Linear k-1.
    """
    if linear_idx == 0:
        return snapshots[0].values
    # Find the snapshot whose layer_idx == linear_idx - 1.
    for s in snapshots:
        if s.layer_idx == linear_idx - 1:
            return s.values
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CertifiedHit:
    hit: MotifHit
    confirmed: bool
    reason: str = ""


@dataclass
class CertifyReport:
    total_candidates: int
    confirmed_counts: Counter = field(default_factory=Counter)
    candidate_counts: Counter = field(default_factory=Counter)
    confirmed_per_region: Dict[str, Counter] = field(default_factory=lambda: {
        "head": Counter(), "body": Counter(), "tail": Counter(),
    })
    candidate_per_region: Dict[str, Counter] = field(default_factory=lambda: {
        "head": Counter(), "body": Counter(), "tail": Counter(),
    })


def _layer_to_child_index(model: nn.Sequential, linear_idx: int) -> int:
    """Map a Linear index to its position in model.children() (strictly
    alternating Linear/ReLU networks have child 2*linear_idx)."""
    return 2 * linear_idx


def _scan_post_relu_neuron(
    model: nn.Sequential,
    linear_idx: int,
    output_neuron: int,
) -> Tuple[int, int]:
    """Locate the SSA index of the post-ReLU neuron that follows the given
    Linear's ``output_neuron`` -- useful for tracking what downstream
    neurons consume that motif's output."""
    # Conceptually: post-ReLU corresponds to the same channel index in the
    # next snapshot.  This helper exists for clarity; the certifier uses
    # snapshot lookup directly.
    return linear_idx + 1, output_neuron


def certify_motifs(
    model: nn.Sequential,
    input_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    head_end: Optional[int] = None,
    body_end: Optional[int] = None,
) -> CertifyReport:
    """Scan ``model`` for motif candidates and certify each one against the
    abstract-interpretation domain of its inputs.

    Returns a :class:`CertifyReport` with confirmed and candidate counts.

    MVP-7.5 — iterative motif-aware certification.  Whenever we confirm a
    Boolean motif (AND, OR, XOR, NOT), the post-ReLU neuron at its output
    is *also* Boolean; we record that and re-run the certification pass
    until a fixed point.  This propagates booleanness through the body's
    XOR-style gadgets that the direct source-expression check can't reach.
    """
    linears = [m for m in model if isinstance(m, nn.Linear)]
    if input_ranges is None:
        in_dim = linears[0].in_features
        input_ranges = [(0, 255)] * in_dim
    if head_end is None or body_end is None:
        from . import block_finder
        layout = block_finder.find_blocks(model)
        head_end = head_end if head_end is not None else layout.head_end_linear_idx
        body_end = body_end if body_end is not None else layout.body_end_linear_idx

    snapshots, var_ranges, opaque_registry = _snapshot_after_each_relu(model, input_ranges)
    hits = motifs.scan_model(model)

    # MVP-7.5: maintain a set of opaque variables that have been confirmed
    # Boolean by an earlier motif.  Iterative certification adds to this
    # set until no new confirmations happen.
    extra_bool_opaques: set = set()

    def _certify_extended(a: Affine) -> bool:
        # First the standard certify_bool (which already handles relational
        # source-expression).  Then if it's a single opaque, check the
        # iterative set.
        if _certify_bool(a, var_ranges, opaque_registry):
            return True
        if len(a.coefs) == 1 and a.bias == 0 and a.coefs[0][1] == 1:
            v, _ = a.coefs[0]
            if v in extra_bool_opaques:
                return True
        return False

    def _output_opaque(hit: MotifHit) -> Optional[VarName]:
        """Return the opaque variable that the motif's output Linear-and-ReLU
        produces, if there is exactly one and it's an opaque."""
        # Linear at hit.layer_idx, output channel hit.output_neurons[0].
        # The post-ReLU snapshot's channel of the same index is the
        # downstream-visible value.
        if len(hit.output_neurons) != 1:
            return None
        out_chan = hit.output_neurons[0]
        # Look up snapshot at layer_idx (= the snapshot taken right after
        # this Linear's ReLU).
        for s in snapshots:
            if s.layer_idx == hit.layer_idx:
                a = s.values[out_chan]
                if (
                    len(a.coefs) == 1
                    and a.bias == 0
                    and a.coefs[0][1] == 1
                    and a.coefs[0][0][0] == "relu"
                ):
                    return a.coefs[0][0]
                return None
        return None

    confirmed_hits: List[MotifHit] = []
    last_confirmed_count = -1

    while True:
        confirmed_hits = []
        for h in hits:
            if h.motif_name == "bool_and":
                inputs = _values_seen_by_linear(snapshots, h.layer_idx)
                if inputs is None:
                    continue
                in0, in1 = h.input_neurons
                a0, a1 = inputs[in0], inputs[in1]
                # Reject degenerate ANDs: one input is the constant 0 or 1
                # (i.e. a dead-neuron pad or a structural constant).  These
                # are syntactically AND-shaped rows but collapse to a unary
                # threshold step.
                if a0.is_const or a1.is_const:
                    continue
                if _certify_extended(a0) and _certify_extended(a1):
                    confirmed_hits.append(h)
            elif h.motif_name == "bool_xor":
                # XOR pattern  out = a + b - 2 * AND(a, b)  is split across
                # two linears: the AND lives in linears[h.layer_idx - 1] and
                # the XOR row in linears[h.layer_idx].  ``input_neurons`` are
                # ``(src0, src1)`` — indices into the input vector of the
                # *upstream* Linear (i.e. the booleans ``a`` and ``b``).
                if h.layer_idx == 0:
                    continue
                up_inputs = _values_seen_by_linear(snapshots, h.layer_idx - 1)
                if up_inputs is None:
                    continue
                in0, in1 = h.input_neurons
                if in0 >= len(up_inputs) or in1 >= len(up_inputs):
                    continue
                a0, a1 = up_inputs[in0], up_inputs[in1]
                if a0.is_const or a1.is_const:
                    continue
                if _certify_extended(a0) and _certify_extended(a1):
                    confirmed_hits.append(h)
        if len(confirmed_hits) == last_confirmed_count:
            break
        last_confirmed_count = len(confirmed_hits)
        for h in confirmed_hits:
            v = _output_opaque(h)
            if v is not None:
                extra_bool_opaques.add(v)

    # Build the final report.
    report = CertifyReport(total_candidates=len(hits))
    for h in hits:
        region = (
            "head" if h.layer_idx < head_end
            else "body" if h.layer_idx < body_end
            else "tail"
        )
        report.candidate_counts[h.motif_name] += 1
        report.candidate_per_region[region][h.motif_name] += 1
    for h in confirmed_hits:
        region = (
            "head" if h.layer_idx < head_end
            else "body" if h.layer_idx < body_end
            else "tail"
        )
        report.confirmed_counts[h.motif_name] += 1
        report.confirmed_per_region[region][h.motif_name] += 1

    return report


def format_report(rep: CertifyReport) -> str:
    lines = []
    lines.append("## Domain-certified motif scan")
    lines.append(f"  total candidates : {rep.total_candidates}")
    lines.append("")
    lines.append("## Confirmed vs. candidate by motif")
    all_names = set(rep.candidate_counts) | set(rep.confirmed_counts)
    for name in sorted(all_names):
        c = rep.candidate_counts[name]
        f = rep.confirmed_counts[name]
        pct = (100.0 * f / c) if c else 0.0
        lines.append(f"  {name:12s} confirmed = {f:6d} / {c:6d} candidates  ({pct:.1f}%)")
    lines.append("")
    lines.append("## Confirmed per region")
    for region in ("head", "body", "tail"):
        cnt = rep.confirmed_per_region[region]
        if not cnt:
            continue
        lines.append(f"  {region}:")
        for name, n in cnt.most_common():
            lines.append(f"    {name:12s} {n:5d}")
    lines.append("")
    lines.append("## Candidate (unconfirmed) per region")
    for region in ("head", "body", "tail"):
        cand = rep.candidate_per_region[region]
        conf = rep.confirmed_per_region[region]
        diff_lines = []
        for name in cand:
            unconf = cand[name] - conf[name]
            if unconf > 0:
                diff_lines.append(f"    {name:12s} {unconf:5d}")
        if diff_lines:
            lines.append(f"  {region}:")
            lines.extend(diff_lines)
    return "\n".join(lines)
