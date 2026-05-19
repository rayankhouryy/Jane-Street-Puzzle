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

The output:

* confirmed motif count -- pattern AND inputs are in the right domain
* unconfirmed candidates -- pattern matches but inputs are wider integers

This module deliberately uses only what abstract interpretation can
prove on the *original* input domain (bytes in ``[0, 255]``).  No
algorithm-specific hints.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch.nn as nn

from . import interp, motifs
from .domains import Affine, VarName
from .motifs import MotifHit


# ---------------------------------------------------------------------------
# Abstract domain readout
# ---------------------------------------------------------------------------

def _certify_bool(
    affine: Affine,
    var_ranges: Dict[VarName, Tuple[int, int]],
) -> bool:
    """Return True iff this Affine is provably in ``{0, 1}``.

    Sufficient condition: the affine's integer range lies in ``[0, 1]``.
    This is conservative but always sound.
    """
    try:
        lo, hi = affine.range(var_ranges)
    except KeyError:
        return False
    return lo >= 0 and hi <= 1


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
) -> List[LayerSnapshot]:
    """Run abstract interpretation and snapshot the running tensor after
    each ReLU (= the input to the next Linear).  We need these snapshots
    so that, given a motif hit at Linear ``li``, we can look up the
    abstract values of *its inputs*.

    The mapping:
      * Linear at child index ``2*li`` consumes the snapshot taken *after*
        the previous ReLU at child index ``2*li - 1`` (i.e. after Linear
        ``li - 1``'s ReLU).
      * Linear at child index 0 consumes the raw inputs.
    """
    children = list(model)
    snapshots: List[LayerSnapshot] = []
    # Take an initial snapshot of the input.
    current = [Affine.of(("in", i), 1, 0) for i in range(len(input_ranges))]
    var_ranges: Dict[VarName, Tuple[int, int]] = {
        ("in", i): (int(lo), int(hi)) for i, (lo, hi) in enumerate(input_ranges)
    }
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
                    new_current.append(Affine.of(sym, 1, 0))
            current = new_current
            snapshots.append(LayerSnapshot(layer_idx=linear_idx, values=list(current)))
        else:
            raise ValueError(f"unsupported layer {type(layer).__name__}")
    return snapshots, var_ranges


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


def certify_motifs(
    model: nn.Sequential,
    input_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    head_end: Optional[int] = None,
    body_end: Optional[int] = None,
) -> CertifyReport:
    """Scan ``model`` for motif candidates and certify each one against the
    abstract-interpretation domain of its inputs.

    Returns a :class:`CertifyReport` with confirmed and candidate counts.
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

    snapshots, var_ranges = _snapshot_after_each_relu(model, input_ranges)
    hits = motifs.scan_model(model)

    report = CertifyReport(total_candidates=len(hits))

    for h in hits:
        region = (
            "head" if h.layer_idx < head_end
            else "body" if h.layer_idx < body_end
            else "tail"
        )
        report.candidate_counts[h.motif_name] += 1
        report.candidate_per_region[region][h.motif_name] += 1

        # Look up the Affines that feed this hit.
        inputs = _values_seen_by_linear(snapshots, h.layer_idx)
        if inputs is None:
            continue
        # For bool_and, both input neurons must be bool.
        if h.motif_name == "bool_and":
            in0, in1 = h.input_neurons
            if (
                _certify_bool(inputs[in0], var_ranges)
                and _certify_bool(inputs[in1], var_ranges)
            ):
                report.confirmed_counts[h.motif_name] += 1
                report.confirmed_per_region[region][h.motif_name] += 1
        # For bool_xor (which we detect at a 2-layer pattern), the certifier
        # would need to look 2 layers back; simplest: skip for now.

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
