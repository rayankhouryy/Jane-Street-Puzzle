"""Stage 3 — abstract value lattice.

For hand-compiled integer-domain ReLU networks, every neuron's pre-activation
is an exact integer affine combination of "variables" — where a variable is
either an input dimension or an "opaque" ReLU output (a ReLU we couldn't
resolve symbolically).

The single abstract value type is :class:`Affine`.  It subsumes:

* constants (empty ``coefs`` + integer ``bias``)
* "identity passes" (one coef equal to 1, bias 0)
* genuine linear combinations
* expressions over opaque symbols introduced by earlier saturating ReLUs.

When a ReLU is provably non-saturating on the input domain we can keep
propagating an exact Affine.  When it does saturate, we introduce a fresh
``Opaque`` variable with known integer bounds and continue.

This lattice is intentionally narrow.  We do not track full Intervals or
Bit/UInt domains here -- those will arrive once the head is mapped out and
we can reason about specific neurons.  For abstract interpretation the
Affine lattice is enough to recover the head's symbolic structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


# A "variable name" is a typed pair so we can distinguish inputs from opaque
# ReLU symbols when pretty-printing.
#
#   ("in",     i)  — input byte i
#   ("relu",   j)  — opaque ReLU symbol j
#   ("const",  0)  — never used (the bias slot handles constants)
VarName = Tuple[str, int]


# ---------------------------------------------------------------------------
# Affine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Affine:
    """Exact integer affine combination::

        value = sum(coef * var for var, coef in coefs) + bias

    ``coefs`` is a sorted tuple to make the value hashable and comparable.
    Zero coefficients are not stored.
    """

    coefs: Tuple[Tuple[VarName, int], ...]
    bias: int

    # ----- factories ------------------------------------------------------

    @classmethod
    def const(cls, c: int) -> "Affine":
        return cls(coefs=(), bias=int(c))

    @classmethod
    def of(cls, var: VarName, coef: int = 1, bias: int = 0) -> "Affine":
        return cls(coefs=((var, int(coef)),), bias=int(bias))

    # ----- predicates -----------------------------------------------------

    @property
    def is_const(self) -> bool:
        return len(self.coefs) == 0

    @property
    def const_value(self) -> int:
        if not self.is_const:
            raise ValueError(f"not a constant: {self}")
        return self.bias

    # ----- algebra --------------------------------------------------------

    def __add__(self, other: "Affine | int") -> "Affine":
        if isinstance(other, int):
            return Affine(self.coefs, self.bias + other)
        merged: Dict[VarName, int] = dict(self.coefs)
        for v, c in other.coefs:
            merged[v] = merged.get(v, 0) + c
        cleaned = tuple(sorted(((v, c) for v, c in merged.items() if c != 0)))
        return Affine(cleaned, self.bias + other.bias)

    __radd__ = __add__

    def __neg__(self) -> "Affine":
        return Affine(tuple((v, -c) for v, c in self.coefs), -self.bias)

    def __sub__(self, other: "Affine | int") -> "Affine":
        return self + (-other if isinstance(other, Affine) else -int(other))

    def __mul__(self, scalar: int) -> "Affine":
        s = int(scalar)
        if s == 0:
            return Affine.const(0)
        return Affine(
            tuple((v, c * s) for v, c in self.coefs),
            self.bias * s,
        )

    __rmul__ = __mul__

    # ----- range ----------------------------------------------------------

    def range(self, var_ranges: Dict[VarName, Tuple[int, int]]) -> Tuple[int, int]:
        """Compute the integer interval [lo, hi] this Affine takes on, given the
        variable ranges.  Unknown variables default to (-inf, +inf) -> we
        raise instead, to force the caller to be explicit."""
        lo = hi = self.bias
        for v, c in self.coefs:
            if v not in var_ranges:
                raise KeyError(f"variable {v} has no range in {var_ranges.keys()}")
            vlo, vhi = var_ranges[v]
            if c >= 0:
                lo += c * vlo
                hi += c * vhi
            else:
                lo += c * vhi
                hi += c * vlo
        return lo, hi

    # ----- display --------------------------------------------------------

    def __repr__(self) -> str:
        return self.pretty()

    def pretty(self, max_terms: int = 8) -> str:
        if self.is_const:
            return str(self.bias)
        parts = []
        for v, c in self.coefs[:max_terms]:
            name = _fmt_var(v)
            if c == 1:
                parts.append(f"+{name}")
            elif c == -1:
                parts.append(f"-{name}")
            elif c > 0:
                parts.append(f"+{c}*{name}")
            else:
                parts.append(f"-{abs(c)}*{name}")
        if len(self.coefs) > max_terms:
            parts.append(f"+...(+{len(self.coefs) - max_terms} more)")
        if self.bias > 0:
            parts.append(f"+{self.bias}")
        elif self.bias < 0:
            parts.append(f"-{abs(self.bias)}")
        body = " ".join(parts)
        if body.startswith("+"):
            body = body[1:]
        return body


def _fmt_var(v: VarName) -> str:
    kind, idx = v
    if kind == "in":
        return f"x[{idx}]"
    if kind == "relu":
        return f"r[{idx}]"
    return f"{kind}[{idx}]"


# ---------------------------------------------------------------------------
# Helpers for collections of Affines
# ---------------------------------------------------------------------------

def affine_input_only(a: Affine) -> bool:
    """True if `a` references only input variables ("in") -- no opaque symbols."""
    return all(v[0] == "in" for v, _ in a.coefs)


def affine_uses(a: Affine, var: VarName) -> bool:
    return any(v == var for v, _ in a.coefs)


def iter_vars(affines: Iterable[Affine]) -> Iterable[VarName]:
    seen = set()
    for a in affines:
        for v, _ in a.coefs:
            if v not in seen:
                seen.add(v)
                yield v
