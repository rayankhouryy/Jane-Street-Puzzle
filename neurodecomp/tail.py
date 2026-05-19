"""Stage-2 decompiler for the *tail* of an `nn.Sequential`: a hand-compiled
suffix that ANDs together $N$ integer-equality predicates via the
"Kronecker-delta" ReLU motif.

Given a model whose final Linear matches the AND-of-deltas template

    W = ( +1, ..., +1, -2, ..., -2, +1, ..., +1 )      (3N entries)
    b = -(N - 1)

this module:

1.  Verifies that the penultimate Linear has *triple-replicated* rows, so the
    three Kronecker components really do depend on the same scalar expression
    `z_i`.
2.  Recovers the per-predicate target value `t_i` from the bias of the middle
    row of each triple.
3.  Recognises each row's nonzero weights as a *binary-to-integer decoder*
    if their absolute values are powers of two (the canonical "bit -> byte"
    gadget).  Reports the bit-address vector and sign per bit.

No algorithm-specific knowledge is hard-coded.  Any network that compares N
integer-valued sums of binary signals against fixed targets will be
recognised.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch.nn as nn


# Tolerances.  Hand-compiled weights are exact integers so 1e-9 is generous.
_EPS = 1e-9


@dataclass
class Predicate:
    """One recovered integer-equality predicate `z_i(x) == target`.

    `bit_addresses` is a list of (upstream_index, coef) pairs.  When all
    coefs are +/- powers of two, `bit_kind` is "powers_of_two" and the
    predicate is recognised as a signed binary-to-integer decoded byte
    compared with `target`.
    """

    i: int
    target: float                                # integer target (post-cast if exact)
    bit_addresses: List[Tuple[int, float]]
    bit_kind: str                                # "powers_of_two" | "general"
    num_terms: int


@dataclass
class TailReport:
    matches: bool
    reason: str = ""
    N: Optional[int] = None
    predicates: List[Predicate] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # Convenient summary fields
    targets: List[float] = field(default_factory=list)
    targets_are_bytes: bool = False
    targets_hex: Optional[str] = None


# ---------------------------------------------------------------------------
# Final-layer template detection
# ---------------------------------------------------------------------------

def _detect_final_template(last: nn.Linear) -> Tuple[bool, str, int]:
    """Return (matched, reason, N) where the template is

        W = (+1)*N, (-2)*N, (+1)*N         (3N entries)
        b = -(N - 1).

    """
    if last.out_features != 1:
        return False, f"last layer is not scalar (out_features={last.out_features})", 0
    W = last.weight.detach().cpu().numpy().flatten()
    b = float(last.bias.detach().cpu().numpy().item()) if last.bias is not None else 0.0
    n = len(W)
    if n % 3 != 0:
        return False, f"last in-dim {n} not divisible by 3", 0
    N = n // 3
    if not np.allclose(W[:N], 1.0, atol=_EPS):
        return False, "first third of weights is not +1*N", 0
    if not np.allclose(W[N : 2 * N], -2.0, atol=_EPS):
        return False, "second third of weights is not -2*N", 0
    if not np.allclose(W[2 * N :], 1.0, atol=_EPS):
        return False, "third third of weights is not +1*N", 0
    if abs(b - (-(N - 1))) > _EPS:
        return False, f"bias is {b}, expected {-(N - 1)}", 0
    return True, "ok", N


# ---------------------------------------------------------------------------
# Penultimate-layer triple check + byte decoder
# ---------------------------------------------------------------------------

def _check_triple_replication(W: np.ndarray, b: np.ndarray, N: int) -> Tuple[bool, str, int]:
    """The penultimate layer is expected to produce 3N outputs grouped so that
    rows ``i``, ``i + N``, ``i + 2N`` compute ``z_i + s_0``, ``z_i``, ``z_i + s_2``
    where ``(s_0, s_2) ∈ {(+1, -1), (-1, +1)}`` (commutative Kronecker delta).

    Returns ``(ok, reason, sign_top)`` where ``sign_top`` is the offset of the
    first row's bias relative to the middle row (``+1`` or ``-1``).
    """
    if N == 0:
        return True, "ok", +1

    # Determine the sign convention from row 0; require all rows to agree.
    first_offset = b[0] - b[N]
    if abs(first_offset - 1.0) < _EPS:
        sign_top, sign_bot = +1.0, -1.0
    elif abs(first_offset + 1.0) < _EPS:
        sign_top, sign_bot = -1.0, +1.0
    else:
        return False, (
            f"row 0 top bias offset is {first_offset:+.4f}, expected +/-1"
        ), 0

    for i in range(N):
        r0, r1, r2 = W[i], W[N + i], W[2 * N + i]
        if not (np.allclose(r0, r1, atol=_EPS) and np.allclose(r1, r2, atol=_EPS)):
            return False, f"row {i} not triple-replicated (W differs)", 0
        if abs((b[i] - b[N + i]) - sign_top) > _EPS:
            return False, (
                f"row {i} top bias offset is {b[i] - b[N + i]:+.4f}, "
                f"expected {sign_top:+.0f}"
            ), 0
        if abs((b[2 * N + i] - b[N + i]) - sign_bot) > _EPS:
            return False, (
                f"row {i} bottom bias offset is {b[2 * N + i] - b[N + i]:+.4f}, "
                f"expected {sign_bot:+.0f}"
            ), 0
    return True, "ok", int(sign_top)


_POW2 = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}


def _classify_row(row: np.ndarray) -> Tuple[List[Tuple[int, float]], str]:
    """Return (sparse_terms, kind) for a single row of the penultimate Linear.

    kind == "powers_of_two" when every nonzero |coef| is a power of two.
    """
    nz = np.nonzero(np.abs(row) > _EPS)[0]
    terms: List[Tuple[int, float]] = [(int(j), float(row[j])) for j in nz]
    if all(abs(c) in _POW2 for _, c in terms):
        kind = "powers_of_two"
    else:
        kind = "general"
    return terms, kind


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompile_tail(model: nn.Sequential) -> TailReport:
    """Recover the AND-of-deltas tail of a hand-compiled Sequential.

    No prior knowledge of the underlying algorithm is used.  The recogniser
    operates purely on the last two Linear layers' weights and biases.
    """
    linears: List[nn.Linear] = [m for m in model if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return TailReport(matches=False, reason="need at least 2 Linear layers")

    last = linears[-1]
    matched, reason, N = _detect_final_template(last)
    if not matched:
        return TailReport(matches=False, reason=reason)

    penu = linears[-2]
    if penu.out_features != 3 * N:
        return TailReport(
            matches=False,
            reason=f"penultimate Linear has out={penu.out_features}, expected 3*N = {3*N}",
        )
    Wp = penu.weight.detach().cpu().numpy()
    bp = (
        penu.bias.detach().cpu().numpy()
        if penu.bias is not None
        else np.zeros(3 * N)
    )

    ok, reason2, _sign_top = _check_triple_replication(Wp, bp, N)
    if not ok:
        return TailReport(
            matches=False,
            reason=f"triple-replication check failed: {reason2}",
            N=N,
        )

    # Decode each row.
    predicates: List[Predicate] = []
    for i in range(N):
        middle_row = Wp[N + i]
        terms, kind = _classify_row(middle_row)
        target = -float(bp[N + i])     # because the row computes z_i = (decoded value) - target
        predicates.append(
            Predicate(
                i=i,
                target=target,
                bit_addresses=terms,
                bit_kind=kind,
                num_terms=len(terms),
            )
        )

    # Summary metrics.
    targets = [p.target for p in predicates]
    all_int = all(t == int(t) for t in targets)
    all_byte = all_int and all(0 <= int(t) <= 255 for t in targets)
    targets_hex = (
        "".join(f"{int(t):02x}" for t in targets) if all_byte else None
    )
    notes: List[str] = []
    if all_byte:
        notes.append(
            f"all {N} targets are bytes -> packed hex: {targets_hex}"
        )
    n_pow2 = sum(1 for p in predicates if p.bit_kind == "powers_of_two")
    notes.append(f"{n_pow2}/{N} rows are powers-of-two decoders (binary -> integer)")

    # Bit-count consistency: if every powers-of-two row has the same number
    # of nonzero terms, that's the integer width (8 -> bytes, 16 -> shorts...).
    if n_pow2 == N:
        bit_widths = {p.num_terms for p in predicates}
        if len(bit_widths) == 1:
            (bw,) = bit_widths
            notes.append(f"every predicate is a sum of {bw} signed bits "
                         f"-> matches a {bw}-bit register interpretation")

    return TailReport(
        matches=True,
        reason="ok",
        N=N,
        predicates=predicates,
        notes=notes,
        targets=targets,
        targets_are_bytes=all_byte,
        targets_hex=targets_hex,
    )


def format_report(report: TailReport, max_predicates: int = 16) -> str:
    """Pretty-print a TailReport for human consumption."""
    lines: List[str] = []
    if not report.matches:
        lines.append(f"Tail does NOT match AND-of-deltas template: {report.reason}")
        return "\n".join(lines)
    lines.append(f"## Tail decompilation -- AND-of-deltas, N = {report.N}")
    for note in report.notes:
        lines.append(f"  note: {note}")
    lines.append("")
    lines.append("## Per-predicate recovery")
    for p in report.predicates[:max_predicates]:
        addr_summary = ", ".join(
            f"{int(c):+d}*bit[{j}]" for j, c in p.bit_addresses
        )
        if p.target == int(p.target):
            tgt = (
                f"0x{int(p.target):02x}"
                if 0 <= int(p.target) <= 255
                else f"{int(p.target)}"
            )
        else:
            tgt = f"{p.target}"
        lines.append(
            f"  pred[{p.i:2d}] : decoded_int == {tgt}   "
            f"({p.num_terms} terms; {p.bit_kind})"
        )
        lines.append(f"            decoded_int = {addr_summary}")
    if len(report.predicates) > max_predicates:
        lines.append(f"  ... ({len(report.predicates) - max_predicates} more)")
    if report.targets_hex:
        lines.append("")
        lines.append(f"## Concatenated target (byte order as found): {report.targets_hex}")
    return "\n".join(lines)
