"""Stage 3 -- forward abstract interpreter.

Walks an ``nn.Sequential`` of Linear / ReLU modules, maintaining one
:class:`Affine` per neuron of the running tensor.  Each ReLU resolves to one
of three outcomes:

1. **Non-saturating positive.**  The Affine's range on the input domain
   satisfies ``lo >= 0``; ReLU is the identity.  Propagate unchanged.
2. **Always zero.**  ``hi <= 0``; the neuron is dead.  Replace with
   ``Affine.const(0)``.
3. **Saturating.**  ``lo < 0 < hi``.  Introduce a fresh opaque symbol with
   range ``[0, hi]`` so downstream linear layers can still propagate
   exact affine combinations over it.

This is enough to characterise the *symbolic skeleton* of a hand-compiled
head: most ReLUs resolve to branch 1 or 2; the saturating ReLUs are
exactly the points where the head makes data-dependent decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .domains import Affine, OpaqueSource, VarName, affine_input_only


@dataclass
class InterpResult:
    """Per-neuron abstract values + diagnostics after running the interpreter."""

    # Final running tensor — one Affine per output dim of the last layer.
    values: List[Affine] = field(default_factory=list)

    # Variable ranges used during propagation; opaque ReLU symbols accumulate.
    var_ranges: Dict[VarName, Tuple[int, int]] = field(default_factory=dict)

    # Per-layer statistics.
    layer_stats: List[Dict[str, int]] = field(default_factory=list)

    # Total opaque symbols introduced.
    num_opaque: int = 0

    # MVP-7.5: maps each opaque variable id to the source Affine that fed
    # the saturating ReLU which produced it.  Used by the certifier to
    # prove tighter facts than the raw range alone.
    opaque_registry: Dict[VarName, OpaqueSource] = field(default_factory=dict)


@dataclass
class InterpOptions:
    # Maximum value of |coef * range| before we give up exact propagation.
    # We don't currently truncate; this is a future safety knob.
    max_term_magnitude: int = 10**12


def _to_int_array(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=1e-9):
        raise ValueError(
            "non-integer weight detected -- interp currently assumes "
            "integer-valued layers"
        )
    return rounded.astype(np.int64)


def run(
    model: nn.Sequential,
    input_ranges: Sequence[Tuple[int, int]],
    *,
    layers: Sequence[nn.Module] | None = None,
    options: InterpOptions | None = None,
) -> InterpResult:
    """Run abstract interpretation on ``model`` (or the first ``layers``).

    Args:
        model: Sequential to interpret.  Must be Linear/ReLU only.
        input_ranges: ``(lo, hi)`` integer interval for each input dim.
        layers: optional explicit list of modules to interpret.  If None,
            walks every child of ``model``.
        options: tuning knobs.

    Returns:
        :class:`InterpResult` with one Affine per final-tensor dim.
    """
    if options is None:
        options = InterpOptions()

    children = list(layers if layers is not None else model)
    if not children:
        raise ValueError("no layers to interpret")

    in_dim = len(input_ranges)
    # The running tensor: one Affine per dim.  Start as identity x[i].
    current: List[Affine] = [Affine.of(("in", i), 1, 0) for i in range(in_dim)]

    # Variable ranges include each input plus opaque symbols introduced below.
    var_ranges: Dict[VarName, Tuple[int, int]] = {
        ("in", i): (int(lo), int(hi)) for i, (lo, hi) in enumerate(input_ranges)
    }

    layer_stats: List[Dict[str, int]] = []
    num_opaque = 0
    opaque_registry: Dict[VarName, OpaqueSource] = {}

    for layer_idx, layer in enumerate(children):
        stats = {"layer": layer_idx, "kind": type(layer).__name__,
                 "out_dim": 0, "non_sat": 0, "dead": 0, "sat": 0}
        if isinstance(layer, nn.Linear):
            W = _to_int_array(layer.weight)
            b = _to_int_array(layer.bias) if layer.bias is not None else None
            out_dim, in_dim_layer = W.shape
            if in_dim_layer != len(current):
                raise ValueError(
                    f"Linear at layer {layer_idx} expects in={in_dim_layer} "
                    f"but running tensor has width {len(current)}"
                )
            new_current: List[Affine] = []
            for o in range(out_dim):
                acc = Affine.const(int(b[o]) if b is not None else 0)
                row = W[o]
                nz = np.nonzero(row)[0]
                for i in nz:
                    acc = acc + current[i] * int(row[i])
                new_current.append(acc)
            current = new_current
            stats["out_dim"] = len(current)

        elif isinstance(layer, nn.ReLU):
            new_current = []
            for src in current:
                lo, hi = src.range(var_ranges)
                if lo >= 0:
                    new_current.append(src)
                    stats["non_sat"] += 1
                elif hi <= 0:
                    new_current.append(Affine.const(0))
                    stats["dead"] += 1
                else:
                    # Saturating: introduce an opaque symbol *with source*.
                    sym: VarName = ("relu", num_opaque)
                    num_opaque += 1
                    var_ranges[sym] = (0, int(hi))
                    opaque_registry[sym] = OpaqueSource(expr=src)
                    new_current.append(Affine.of(sym, 1, 0))
                    stats["sat"] += 1
            current = new_current
            stats["out_dim"] = len(current)

        else:
            raise ValueError(
                f"unsupported layer at idx {layer_idx}: {type(layer).__name__}"
            )

        layer_stats.append(stats)

    return InterpResult(
        values=current,
        var_ranges=var_ranges,
        layer_stats=layer_stats,
        num_opaque=num_opaque,
        opaque_registry=opaque_registry,
    )


# ---------------------------------------------------------------------------
# Higher-level utilities (used by head decompiler)
# ---------------------------------------------------------------------------

def classify_neuron(a: Affine, input_only: bool) -> str:
    """Return a short tag describing what role this affine plays."""
    if a.is_const:
        return f"const({a.bias})"
    if input_only:
        if len(a.coefs) == 1 and a.coefs[0][1] == 1 and a.bias == 0:
            return f"passes {a.coefs[0][0][0]}[{a.coefs[0][0][1]}]"
        if all(c == 1 for _, c in a.coefs) and a.bias == 0:
            n = len(a.coefs)
            return f"sum of {n} inputs"
        if all(c == 1 for _, c in a.coefs):
            n = len(a.coefs)
            return f"sum of {n} inputs + {a.bias}"
        return f"affine over {len(a.coefs)} inputs"
    return f"opaque (refs {sum(1 for v, _ in a.coefs if v[0] == 'relu')} ReLU symbols)"


def summarize(result: InterpResult, max_show: int = 64) -> str:
    """Pretty per-neuron summary of the abstract interpretation result."""
    lines = []
    lines.append(f"## Abstract interpretation summary")
    lines.append(
        f"  total output neurons : {len(result.values)}"
    )
    lines.append(
        f"  opaque ReLU symbols  : {result.num_opaque}"
    )
    input_only_count = sum(1 for a in result.values if affine_input_only(a))
    lines.append(
        f"  neurons expressed purely in inputs : {input_only_count}/{len(result.values)}"
    )
    n_const = sum(1 for a in result.values if a.is_const)
    lines.append(f"  constant neurons : {n_const}")

    lines.append("")
    lines.append("## Per-layer statistics")
    for s in result.layer_stats:
        if s["kind"] == "ReLU":
            lines.append(
                f"  layer {s['layer']:3d} ReLU   out={s['out_dim']:4d}  "
                f"non-sat={s['non_sat']:4d}  dead={s['dead']:3d}  sat={s['sat']:3d}"
            )
        else:
            lines.append(
                f"  layer {s['layer']:3d} {s['kind']:6s} out={s['out_dim']:4d}"
            )

    lines.append("")
    lines.append(f"## First {max_show} output neurons")
    for i, a in enumerate(result.values[:max_show]):
        tag = classify_neuron(a, affine_input_only(a))
        # Compute the range for this neuron.
        lo, hi = a.range(result.var_ranges)
        lines.append(f"  out[{i:3d}] in [{lo:>6d}, {hi:>6d}]  {tag}")
        lines.append(f"            = {a.pretty(max_terms=10)}")
    if len(result.values) > max_show:
        lines.append(f"  ... ({len(result.values) - max_show} more)")
    return "\n".join(lines)
