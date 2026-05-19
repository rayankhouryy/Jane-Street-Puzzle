"""Stage 5 — motif library.

A catalog of small ReLU-implementable identities, each carrying:

1. A symbolic Python definition (just the math).
2. A Z3 proof of equivalence over its declared domain.  Every motif must
   be *self-verified* on import (or on demand) so that downstream code can
   trust the library.
3. A weight-pattern recogniser that scans a :class:`~neurodecomp.sparse_graph.SSAProgram`
   and reports occurrences of the motif.

The recognisers are *purely structural*: they look at adjacent
``Linear``/``ReLU`` pairs and their integer weight rows, and detect the
canonical algebraic form of each motif.  They do not attempt full abstract
interpretation -- that's the job of :mod:`neurodecomp.interp` -- and they
make no assumptions about the algorithm being decompiled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch.nn as nn

try:
    import z3
    _HAS_Z3 = True
except ImportError:                  # pragma: no cover
    z3 = None                         # type: ignore[assignment]
    _HAS_Z3 = False


# ---------------------------------------------------------------------------
# Motif declarations + Z3 self-verification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Motif:
    name: str
    description: str
    arity: int                          # number of integer inputs
    input_domain: Tuple[int, int]       # (lo, hi) per input
    # The reference Python implementation used both in tests and Z3 proofs.
    ref: Callable[..., int] = field(repr=False)
    # The ReLU-network implementation: same signature.  Both must agree on
    # the declared input domain.
    relu_impl: Callable[..., int] = field(repr=False)


def _relu(x):                           # works for int/float/Z3 expr
    if _HAS_Z3 and isinstance(x, z3.ExprRef):
        return z3.If(x > 0, x, 0)
    return x if x > 0 else 0


# ----- definitions ---------------------------------------------------------

def _ref_delta(z_val):
    # 1 if z_val == 0 else 0 -- written to work for both Python ints and Z3 ints.
    if _HAS_Z3 and isinstance(z_val, z3.ExprRef):
        return z3.If(z_val == 0, 1, 0)
    return 1 if z_val == 0 else 0


def _relu_delta(z_val):
    return _relu(z_val + 1) + _relu(z_val - 1) - 2 * _relu(z_val)


def _ref_and(a, b):
    # For a, b in {0, 1}: AND(a, b) = a * b
    return a * b


def _relu_and(a, b):
    return _relu(a + b - 1)


def _ref_or(a, b):
    # For a, b in {0, 1}: OR(a, b) = a + b - a * b
    return a + b - a * b


def _relu_or(a, b):
    # a + b - AND(a, b) for booleans; using ReLU form:
    return a + b - _relu(a + b - 1)


def _ref_xor(a, b):
    # For a, b in {0, 1}: XOR(a, b) = a + b - 2 * a * b
    return a + b - 2 * a * b


def _relu_xor(a, b):
    # a + b - 2*AND(a, b)
    return a + b - 2 * _relu(a + b - 1)


def _ref_not(a):
    return 1 - a


def _relu_not(a):
    return 1 - a    # trivially -- no ReLU needed -- but kept for completeness


def _ref_thresh(x, k):
    # max(0, x - k); avoid Python's max() so this works for Z3 too.
    return _relu(x - k)


# A tiny ROTL-by-s recogniser is in body.py; rotation isn't expressed at the
# SSA-row granularity so we don't include it here.

CATALOG: List[Motif] = [
    Motif("delta",       "1 if z == 0 else 0  (Kronecker delta on integers)",
          arity=1, input_domain=(-8, 8),
          ref=_ref_delta, relu_impl=_relu_delta),
    Motif("bool_and",    "a AND b  for booleans",
          arity=2, input_domain=(0, 1),
          ref=_ref_and, relu_impl=_relu_and),
    Motif("bool_or",     "a OR b  for booleans",
          arity=2, input_domain=(0, 1),
          ref=_ref_or, relu_impl=_relu_or),
    Motif("bool_xor",    "a XOR b  for booleans",
          arity=2, input_domain=(0, 1),
          ref=_ref_xor, relu_impl=_relu_xor),
    Motif("bool_not",    "NOT a  for boolean a",
          arity=1, input_domain=(0, 1),
          ref=_ref_not, relu_impl=_relu_not),
    Motif("threshold",   "max(0, x - k)  (thermometer step)",
          arity=2, input_domain=(-8, 8),
          ref=_ref_thresh, relu_impl=_ref_thresh),
]


# ---------------------------------------------------------------------------
# Z3 self-verification
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    motif_name: str
    proved: bool
    sat: bool                           # True if counterexample found
    counterexample: Optional[dict] = None
    error: Optional[str] = None


def verify_motif(m: Motif) -> VerificationResult:
    """Prove `m.ref(x) == m.relu_impl(x)` over the integer input domain via Z3.

    Returns a :class:`VerificationResult`.  If Z3 isn't installed we report
    that as the error.
    """
    if not _HAS_Z3:
        return VerificationResult(m.name, proved=False, sat=False,
                                  error="z3 not installed")
    s = z3.Solver()
    args = [z3.Int(f"x{i}") for i in range(m.arity)]
    lo, hi = m.input_domain
    for a in args:
        s.add(a >= lo, a <= hi)
    diff = m.ref(*args) - m.relu_impl(*args)
    s.add(diff != 0)
    res = s.check()
    if res == z3.unsat:
        return VerificationResult(m.name, proved=True, sat=False)
    if res == z3.sat:
        model = s.model()
        cex = {str(a): model[a].as_long() for a in args}
        return VerificationResult(m.name, proved=False, sat=True,
                                  counterexample=cex)
    return VerificationResult(m.name, proved=False, sat=False,
                              error=f"z3 returned {res}")


def verify_all() -> List[VerificationResult]:
    return [verify_motif(m) for m in CATALOG]


# ---------------------------------------------------------------------------
# Weight-pattern recogniser on Sequential models
# ---------------------------------------------------------------------------

@dataclass
class MotifHit:
    """A motif detected in a model's structure."""

    motif_name: str
    layer_idx: int                  # the Linear layer where the motif terminates
    output_neurons: Tuple[int, ...]  # which output neurons embody the motif
    input_neurons: Tuple[int, ...]  # which input neurons it consumes


def _detect_delta_motifs(
    linears: List[nn.Linear],
) -> List[MotifHit]:
    """Look for Kronecker-delta triples.

    Pattern: an output is computed as ``1 * ReLU(z+1) + 1 * ReLU(z-1)
    - 2 * ReLU(z)``.  We detect this by looking at a ``Linear`` whose row
    contains exactly three nonzero coefficients ``(+1, -2, +1)``, with the
    corresponding three input neurons being the post-ReLU of an upstream
    Linear whose triples differ in bias by ``(+1, 0, -1)``.
    """
    hits: List[MotifHit] = []
    if len(linears) < 2:
        return hits
    for li in range(1, len(linears)):
        last = linears[li]
        prev = linears[li - 1]
        # Quick scan: count rows in `last` matching the (+1, +1, -2) pattern
        # (any permutation).
        W = last.weight.detach().cpu().numpy()
        for out in range(last.out_features):
            row = W[out]
            nz = [(int(i), float(row[i])) for i in range(last.in_features)
                  if row[i] != 0]
            if len(nz) != 3:
                continue
            vals = sorted(c for _, c in nz)
            if vals != [-2.0, 1.0, 1.0]:
                continue
            # Identify the three input neuron indices in the upstream
            # Linear and their biases.
            ones = [i for i, c in nz if c == 1.0]
            twos = [i for i, c in nz if c == -2.0]
            if len(ones) != 2 or len(twos) != 1:
                continue
            # Check that all three reference the same upstream "expression"
            # by comparing row vectors of `prev` (modulo bias offsets).
            Wp = prev.weight.detach().cpu().numpy()
            bp = (
                prev.bias.detach().cpu().numpy()
                if prev.bias is not None
                else None
            )
            r0, r1, r2 = Wp[ones[0]], Wp[ones[1]], Wp[twos[0]]
            if not (r0 == r1).all() or not (r1 == r2).all():
                continue
            if bp is None:
                continue
            biases = {bp[ones[0]], bp[ones[1]], bp[twos[0]]}
            ref_b = bp[twos[0]]
            if biases != {ref_b - 1, ref_b, ref_b + 1}:
                continue
            hits.append(MotifHit(
                motif_name="delta",
                layer_idx=li,
                output_neurons=(out,),
                input_neurons=tuple(sorted(ones + twos)),
            ))
    return hits


def _detect_bool_xor_motifs(
    linears: List[nn.Linear],
) -> List[MotifHit]:
    """Look for boolean XOR via ``a + b - 2 * ReLU(a + b - 1)``.

    Pattern: an output row has weights ``[+1, +1, -2]`` (two pre-ReLU
    sources + one post-ReLU intermediate), bias 0, where the third input
    is the ReLU of ``a + b - 1`` (rows-sum-equals-two pattern in the
    upstream Linear).
    """
    hits: List[MotifHit] = []
    if len(linears) < 2:
        return hits
    for li in range(1, len(linears)):
        last = linears[li]
        prev = linears[li - 1]
        W = last.weight.detach().cpu().numpy()
        Wp = prev.weight.detach().cpu().numpy()
        bp = (
            prev.bias.detach().cpu().numpy()
            if prev.bias is not None
            else None
        )
        if bp is None:
            continue
        for out in range(last.out_features):
            row = W[out]
            nz = [(int(i), float(row[i])) for i in range(last.in_features)
                  if row[i] != 0]
            if len(nz) != 3:
                continue
            vals = sorted(c for _, c in nz)
            if vals != [-2.0, 1.0, 1.0]:
                continue
            ones = [i for i, c in nz if c == 1.0]
            twos = [i for i, c in nz if c == -2.0]
            and_in = twos[0]
            r_and = Wp[and_in]
            b_and = bp[and_in]
            # XOR motif: the (-2) source should have weight pattern that
            # is exactly the sum of the two ones-sources upstream, with bias -1.
            # In other words, r_and equals indicator(ones[0]) + indicator(ones[1])
            # *as inputs to prev*.  We require ones[0] and ones[1] to be
            # passthrough rows of prev:  Wp[ones[i]] is a one-hot.
            for k in (0, 1):
                row_k = Wp[ones[k]]
                if (row_k != 0).sum() != 1:
                    # The two "ones" rows must be one-hots (i.e. passthroughs).
                    break
            else:
                src0 = int((Wp[ones[0]] != 0).argmax())
                src1 = int((Wp[ones[1]] != 0).argmax())
                # The "AND" row should be all zero except +1 at src0 and src1.
                expected = [0.0] * prev.in_features
                expected[src0] = 1.0
                expected[src1] = 1.0
                if (
                    list(r_and) == expected
                    and b_and == -1.0
                    and bp[ones[0]] == 0.0
                    and bp[ones[1]] == 0.0
                ):
                    hits.append(MotifHit(
                        motif_name="bool_xor",
                        layer_idx=li,
                        output_neurons=(out,),
                        input_neurons=(src0, src1),
                    ))
    return hits


def _detect_bool_and_motifs(
    linears: List[nn.Linear],
) -> List[MotifHit]:
    """Look for boolean AND via ``ReLU(a + b - 1)``.

    Pattern: a Linear row with weights ``[+1, +1]`` and bias ``-1``
    followed by a ReLU.
    """
    hits: List[MotifHit] = []
    for li, last in enumerate(linears):
        W = last.weight.detach().cpu().numpy()
        b = (
            last.bias.detach().cpu().numpy()
            if last.bias is not None
            else None
        )
        if b is None:
            continue
        for out in range(last.out_features):
            row = W[out]
            nz = [(int(i), float(row[i])) for i in range(last.in_features)
                  if row[i] != 0]
            if len(nz) != 2:
                continue
            if {c for _, c in nz} != {1.0}:
                continue
            if b[out] != -1.0:
                continue
            hits.append(MotifHit(
                motif_name="bool_and",
                layer_idx=li,
                output_neurons=(out,),
                input_neurons=tuple(sorted(i for i, _ in nz)),
            ))
    return hits


def scan_model(model: nn.Sequential) -> List[MotifHit]:
    """Return all motif hits found in ``model``."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    return (
        _detect_bool_and_motifs(linears)
        + _detect_bool_xor_motifs(linears)
        + _detect_delta_motifs(linears)
    )


def summarize_scan(hits: List[MotifHit]) -> str:
    from collections import Counter
    by_name = Counter(h.motif_name for h in hits)
    lines = ["## Motif scan results"]
    lines.append(f"  total hits: {len(hits)}")
    for name, n in by_name.most_common():
        lines.append(f"  {name:12s} {n:5d}")
    return "\n".join(lines)


def hits_per_layer_range(
    hits: List[MotifHit],
    head_end: int,
    body_end: int,
) -> dict:
    """Group hits by head / body / tail region.  Inputs are Linear-layer
    indices (i.e. the index of the Linear *of which* the motif consumes
    output or terminates at)."""
    from collections import Counter
    region: dict = {"head": Counter(), "body": Counter(), "tail": Counter()}
    for h in hits:
        if h.layer_idx < head_end:
            region["head"][h.motif_name] += 1
        elif h.layer_idx < body_end:
            region["body"][h.motif_name] += 1
        else:
            region["tail"][h.motif_name] += 1
    return region
