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

**MVP-7.5 upgrade — relational opaque symbols.** Each opaque variable
optionally carries the affine *source expression* that produced it (i.e. the
expression that was passed to the saturating ReLU).  Downstream reasoning can
then re-examine the source expression to prove tighter facts than the raw
range alone — most importantly, **booleanness**:

>  ReLU(z) ∈ {0, 1}    iff    hi(z) ≤ 1

so any opaque whose source has integer hi ≤ 1 is certified Boolean, even when
its lo is far below zero.  This is the single change that converts most of
our "candidate" boolean ANDs into "confirmed" boolean ANDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


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
# Relational opaque-symbol tracking (MVP-7.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpaqueSource:
    """Records what an opaque ReLU symbol was generated from.

    ``expr`` is the Affine that was passed into the saturating ReLU; the
    opaque variable's value is ``ReLU(expr)`` on the original input domain.

    Storing this lets us certify tighter facts downstream, e.g.:

      ReLU(a + b - 1) ∈ {0, 1}   iff   hi(a + b - 1) ≤ 1
    """

    expr: "Affine"


# A registry of OpaqueSource per opaque variable id.  Carried alongside
# variable ranges by the interpreter.
OpaqueRegistry = Dict[VarName, OpaqueSource]


def certify_bool_via_source(
    var: VarName,
    var_ranges: Dict[VarName, Tuple[int, int]],
    opaque_registry: Optional[OpaqueRegistry],
    *,
    _seen: Optional[set] = None,
    _max_depth: int = 6,
) -> bool:
    """Return True iff opaque ``var`` is provably in ``{0, 1}``.

    Two levels of evidence:

    1. **Raw range.**  If ``var_ranges[var] ⊆ [0, 1]``, immediately Boolean.
    2. **Refined source range (recursive).**  If ``var`` has a recorded
       source affine ``expr``, refine the integer ``hi`` of ``expr`` by
       treating any opaque variable inside ``expr`` whose *own* source has
       a refined ``hi ≤ 1`` as Boolean -- i.e. use ``hi = 1`` instead of the
       raw opaque range when bounding the source.  If the refined ``hi ≤
       1``, ``var`` is Boolean.

    Recursion is bounded by ``_max_depth`` to keep the certifier total even
    on deeply opaque-chained networks.
    """
    if _seen is None:
        _seen = set()
    if var in _seen or _max_depth <= 0:
        return False
    _seen.add(var)

    lo_r, hi_r = var_ranges.get(var, (None, None))
    if lo_r is not None and hi_r is not None and lo_r >= 0 and hi_r <= 1:
        return True
    if opaque_registry is None or var not in opaque_registry:
        return False
    expr = opaque_registry[var].expr

    # Compute refined hi of expr: for each variable, use 1 if it's certified
    # Boolean (recursively), otherwise its raw upper bound.
    refined_hi = expr.bias
    for v, c in expr.coefs:
        if v not in var_ranges:
            return False
        v_lo, v_hi = var_ranges[v]
        # Check whether v is itself Boolean (recurse only for opaque vars).
        v_is_bool = (
            v_lo >= 0 and v_hi <= 1
        ) or (
            v[0] == "relu"
            and certify_bool_via_source(
                v, var_ranges, opaque_registry,
                _seen=_seen, _max_depth=_max_depth - 1,
            )
        )
        eff_v_hi = 1 if v_is_bool else v_hi
        eff_v_lo = 0 if v_is_bool else v_lo
        if c >= 0:
            refined_hi += c * eff_v_hi
        else:
            refined_hi += c * eff_v_lo
    return refined_hi <= 1


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
