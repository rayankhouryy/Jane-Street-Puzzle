"""Stage 1 — SSA / sparse weighted DAG extraction.

We unroll an ``nn.Sequential`` of ``Linear``/``ReLU`` modules into a flat list
of SSA statements.  Every value gets a unique id; pre-ReLU values carry their
sparse affine expression over earlier ids.  Zero-weight edges are dropped.

This is intentionally tiny: downstream stages only see SSA ids and sparse
affine combinations, never the original PyTorch modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

ZERO_TOL = 1e-12


@dataclass(frozen=True)
class AffineExpr:
    """Sparse affine combination over previously-defined SSA ids.

    Attributes:
        terms: tuple of (coef, src_sid) pairs, sorted by src_sid, no zero coefs.
        bias:  constant term.
    """

    terms: Tuple[Tuple[float, int], ...]
    bias: float

    def as_dict(self) -> Dict[int, float]:
        return {sid: c for c, sid in self.terms}

    def nonzero_count(self) -> int:
        return len(self.terms)


@dataclass
class Stmt:
    """One SSA statement."""

    kind: str                       # 'input' | 'affine' | 'relu' | 'output'
    sid: int                        # SSA id assigned here
    affine: AffineExpr | None = None
    relu_src: int | None = None
    input_index: int | None = None
    output_src: int | None = None
    layer_idx: int | None = None    # original Sequential layer index (debug)


@dataclass
class SSAProgram:
    statements: List[Stmt] = field(default_factory=list)
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)
    source_module: nn.Module | None = None

    # Convenience -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.statements)

    def get(self, sid: int) -> Stmt:
        return self.statements[sid]

    def affine_count(self) -> int:
        return sum(1 for s in self.statements if s.kind == "affine")

    def relu_count(self) -> int:
        return sum(1 for s in self.statements if s.kind == "relu")


def extract_ssa(model: nn.Sequential, input_dim: int | None = None) -> SSAProgram:
    """Lower an ``nn.Sequential`` of Linear/ReLU into SSA form."""

    children = list(model)
    if not children:
        raise ValueError("empty Sequential")
    if not isinstance(children[0], nn.Linear):
        raise ValueError("first child must be Linear")

    if input_dim is None:
        input_dim = children[0].in_features

    prog = SSAProgram(source_module=model)

    for i in range(input_dim):
        sid = len(prog.statements)
        prog.statements.append(Stmt(kind="input", sid=sid, input_index=i))
        prog.input_ids.append(sid)

    current: List[int] = list(prog.input_ids)

    for layer_idx, layer in enumerate(children):
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy() if layer.bias is not None else None
            out_dim, in_dim = W.shape
            if in_dim != len(current):
                raise ValueError(
                    f"Linear at layer {layer_idx} expects in={in_dim} but "
                    f"running tensor width is {len(current)}"
                )
            new_current: List[int] = []
            for o in range(out_dim):
                terms_list: List[Tuple[float, int]] = []
                for i in range(in_dim):
                    w = float(W[o, i])
                    if abs(w) > ZERO_TOL:
                        terms_list.append((w, current[i]))
                terms_list.sort(key=lambda t: t[1])
                bias = float(b[o]) if b is not None else 0.0
                expr = AffineExpr(terms=tuple(terms_list), bias=bias)
                sid = len(prog.statements)
                prog.statements.append(
                    Stmt(kind="affine", sid=sid, affine=expr, layer_idx=layer_idx)
                )
                new_current.append(sid)
            current = new_current

        elif isinstance(layer, nn.ReLU):
            new_current = []
            for src in current:
                sid = len(prog.statements)
                prog.statements.append(
                    Stmt(kind="relu", sid=sid, relu_src=src, layer_idx=layer_idx)
                )
                new_current.append(sid)
            current = new_current

        else:
            raise ValueError(
                f"unsupported layer at idx {layer_idx}: {type(layer).__name__}"
            )

    for c in current:
        sid = len(prog.statements)
        prog.statements.append(Stmt(kind="output", sid=sid, output_src=c))
        prog.output_ids.append(sid)

    return prog


# ---------------------------------------------------------------------------
# Stats + reference evaluator (small, for tests / profile reports).
# ---------------------------------------------------------------------------

def weight_alphabet(prog: SSAProgram, max_size: int = 64) -> List[float]:
    """Return the (small) set of distinct nonzero coefficients used in the SSA.

    Returns at most ``max_size`` values; if the true alphabet is larger we
    truncate and the caller can decide how to interpret that."""
    vals = set()
    for s in prog.statements:
        if s.kind == "affine":
            for c, _ in s.affine.terms:
                vals.add(c)
            if s.affine.bias != 0:
                vals.add(s.affine.bias)
        if len(vals) > max_size:
            return sorted(vals)[:max_size]
    return sorted(vals)


def evaluate(prog: SSAProgram, input_values: Sequence[float]) -> List[float]:
    """Reference evaluator for SSAProgram (slow but obvious-correct)."""
    if len(input_values) != len(prog.input_ids):
        raise ValueError(
            f"expected {len(prog.input_ids)} inputs, got {len(input_values)}"
        )
    val: Dict[int, float] = {}
    outputs: List[float] = []
    for s in prog.statements:
        if s.kind == "input":
            val[s.sid] = float(input_values[s.input_index])
        elif s.kind == "affine":
            v = s.affine.bias
            for c, sid in s.affine.terms:
                v += c * val[sid]
            val[s.sid] = v
        elif s.kind == "relu":
            v = val[s.relu_src]
            val[s.sid] = v if v > 0 else 0.0
        elif s.kind == "output":
            outputs.append(val[s.output_src])
    return outputs


def pretty_print(prog: SSAProgram, max_terms: int = 6) -> str:
    """Compact human-readable SSA dump.  Truncates long affine combinations."""
    def fmt_coef(c: float) -> str:
        return str(int(c)) if c == int(c) else f"{c:g}"

    lines: List[str] = []
    for s in prog.statements:
        if s.kind == "input":
            lines.append(f"v{s.sid:<5} = INPUT[{s.input_index}]")
        elif s.kind == "affine":
            terms = s.affine.terms
            shown = ", ".join(
                f"{fmt_coef(c)}*v{sid}" for c, sid in terms[:max_terms]
            )
            if len(terms) > max_terms:
                shown += f", ...(+{len(terms) - max_terms})"
            bias = s.affine.bias
            bias_str = f" + {fmt_coef(bias)}" if bias != 0 else ""
            lines.append(f"v{s.sid:<5} = affine({shown}){bias_str}")
        elif s.kind == "relu":
            lines.append(f"v{s.sid:<5} = relu(v{s.relu_src})")
        elif s.kind == "output":
            lines.append(f"OUTPUT = v{s.output_src}")
    return "\n".join(lines)
