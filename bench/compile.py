"""Compile small Boolean / integer circuits into sparse-integer ReLU networks.

The point of the bench is **ground truth**: every circuit produced here
ships with the exact Python function it implements *and* the exact list
of motifs the compiler placed.  NeuroDecomp's motif scanner, certifier,
and codegen can then be measured against an oracle instead of against
itself.

The compiler is intentionally low-level.  You queue *rows* (sparse
integer-weighted Linear rows) into the current stage, then `commit_stage`
to materialise one ``nn.Linear`` followed by optional ``nn.ReLU``.
This keeps the produced networks structurally identical to the
hand-compiled Jane Street model (strictly alternating Linear/ReLU,
integer weights, canonical motif shapes), which is what the existing
NeuroDecomp scanners expect.

Canonical motif shapes (mirroring ``neurodecomp/motifs.py``):

    bool_and   :  ReLU(a + b - 1)
                  row = {a:+1, b:+1}, bias = -1

    bool_xor   :  prev stage emits 3 rows -- {a:+1, bias 0},
                  {b:+1, bias 0}, AND(a,b);
                  final stage row = {pa:+1, pb:+1, pand:-2}, bias 0

    delta(z)   :  prev stage emits 3 rows -- {z:+1, bias +1}, {z:+1, 0},
                  {z:+1, bias -1};
                  final stage row = {pos:+1, mid:-2, neg:+1}, bias 0

(``bool_or``, ``bool_not``, ``threshold`` have no syntactic detector
yet -- compiling them is supported but they will not register against
the scanner.  That's a measurable gap the bench will surface.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Ground-truth records
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PlacedMotif:
    """A motif explicitly placed by the compiler -- the ground truth.

    ``layer_idx`` is the index of the ``nn.Linear`` (NOT the position in
    the Sequential, which also contains ReLUs) at which the motif
    *terminates*.  ``out_neuron`` is the output channel of that Linear.
    """
    kind: str
    layer_idx: int
    out_neuron: int
    in_neurons: Tuple[int, ...]


@dataclass
class CompiledCircuit:
    """A hand-compiled ReLU network with ground truth attached."""
    model: nn.Sequential
    n_inputs: int
    n_outputs: int
    fn: Callable[[Sequence[int]], List[int]]
    motifs: List[PlacedMotif]
    name: str = ""
    input_domain: Tuple[int, int] = (0, 1)

    # ----- equivalence checking --------------------------------------

    def run_model(self, x: Sequence[int]) -> List[int]:
        with torch.no_grad():
            t = torch.tensor(list(x), dtype=torch.float32)
            y = self.model(t)
            return [int(round(v.item())) for v in y]

    def enumerate_inputs(self, max_count: int = 1 << 14) -> List[List[int]]:
        """Enumerate the input domain exhaustively if small enough,
        else uniformly random-sample ``max_count`` points (seeded)."""
        lo, hi = self.input_domain
        domain_size = (hi - lo + 1) ** self.n_inputs
        if domain_size <= max_count:
            return list(_cartesian(lo, hi, self.n_inputs))
        rng = random.Random(0)
        return [
            [rng.randint(lo, hi) for _ in range(self.n_inputs)]
            for _ in range(max_count)
        ]

    def check_equivalence(self, max_count: int = 1 << 14) -> Tuple[bool, int, int]:
        """Verify the compiled model agrees with ``self.fn`` on the input
        domain.  Returns (all_match, n_tested, n_mismatch)."""
        xs = self.enumerate_inputs(max_count=max_count)
        n_mismatch = 0
        for x in xs:
            if list(self.fn(x)) != self.run_model(x):
                n_mismatch += 1
        return (n_mismatch == 0, len(xs), n_mismatch)

    # ----- structural ------------------------------------------------

    @property
    def n_linears(self) -> int:
        return sum(1 for m in self.model if isinstance(m, nn.Linear))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


def _cartesian(lo: int, hi: int, n: int) -> Iterable[List[int]]:
    if n == 0:
        yield []
        return
    for v in range(lo, hi + 1):
        for rest in _cartesian(lo, hi, n - 1):
            yield [v] + rest


# ----------------------------------------------------------------------
# Compiler
# ----------------------------------------------------------------------

class CircuitCompiler:
    """Stage-by-stage builder for sparse-integer ReLU circuits.

    Usage::

        c = CircuitCompiler(n_inputs=2, name="and1")
        a, b = c.inputs()
        y = c.row({a: 1, b: 1}, bias=-1, motif="bool_and", inputs=(a, b))
        c.commit_stage()              # Linear + ReLU
        out = c.row({y: 1}, bias=0)
        c.commit_stage()              # final Linear + ReLU
        circuit = c.build(fn=lambda x: [int(bool(x[0]) and bool(x[1]))])

    The compiler maintains a *pending* list of rows for the next stage.
    Each ``row`` call queues one output neuron and returns its index
    *in the next stage's output vector*; that index is your wire handle
    for downstream rows in the same stage and (after ``commit_stage``)
    for the following stage.
    """

    def __init__(
        self,
        n_inputs: int,
        name: str = "",
        input_domain: Tuple[int, int] = (0, 1),
    ):
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        self.n_inputs = n_inputs
        self.name = name
        self.input_domain = input_domain
        self.current_width = n_inputs
        self.linears: List[nn.Linear] = []
        self.has_relu: List[bool] = []
        self.placed: List[PlacedMotif] = []
        # Each pending entry: (coefs, bias, motif_kind_or_None, in_neurons_or_None)
        self._pending: List[
            Tuple[Dict[int, int], int, Optional[str], Optional[Tuple[int, ...]]]
        ] = []

    # ----- input handles --------------------------------------------

    def inputs(self) -> List[int]:
        return list(range(self.n_inputs))

    # ----- raw row API ----------------------------------------------

    def row(
        self,
        coefs: Dict[int, int],
        bias: int = 0,
        motif: Optional[str] = None,
        inputs: Optional[Tuple[int, ...]] = None,
    ) -> int:
        """Queue one output neuron for the next stage.

        ``coefs`` maps current-stage input index -> integer coefficient.
        ``bias`` is the integer bias.  ``motif`` optionally tags this row
        as the terminating Linear of a known motif (recorded as ground
        truth).  Returns the wire index for the queued output.
        """
        for src in coefs:
            if not (0 <= src < self.current_width):
                raise ValueError(
                    f"row references wire {src} outside current width "
                    f"{self.current_width}"
                )
        idx = len(self._pending)
        self._pending.append((dict(coefs), int(bias), motif,
                              tuple(inputs) if inputs is not None else None))
        return idx

    def passthrough(self, a: int) -> int:
        """Convenience: a no-op row that propagates ``a`` to the next
        stage.  Valid for any wire produced post-ReLU (non-negative)."""
        return self.row({a: 1}, bias=0)

    def commit_stage(self, do_relu: bool = True) -> List[int]:
        """Materialise the pending rows into a Linear (+ optional ReLU)
        and return the list of new wire indices."""
        if not self._pending:
            raise ValueError("commit_stage called with no pending rows")
        out_dim = len(self._pending)
        W = torch.zeros(out_dim, self.current_width, dtype=torch.float32)
        b = torch.zeros(out_dim, dtype=torch.float32)
        layer_idx = len(self.linears)
        for o, (coefs, bias, motif, in_neurons) in enumerate(self._pending):
            for src, coef in coefs.items():
                W[o, src] = float(coef)
            b[o] = float(bias)
            if motif:
                ins = in_neurons if in_neurons is not None else tuple(coefs.keys())
                self.placed.append(PlacedMotif(
                    kind=motif,
                    layer_idx=layer_idx,
                    out_neuron=o,
                    in_neurons=tuple(sorted(ins)),
                ))
        lin = nn.Linear(self.current_width, out_dim)
        with torch.no_grad():
            lin.weight.copy_(W)
            lin.bias.copy_(b)
        self.linears.append(lin)
        self.has_relu.append(do_relu)
        self.current_width = out_dim
        self._pending = []
        return list(range(out_dim))

    # ----- finalise --------------------------------------------------

    def build(
        self,
        fn: Callable[[Sequence[int]], List[int]],
    ) -> CompiledCircuit:
        if self._pending:
            self.commit_stage(do_relu=True)
        layers: List[nn.Module] = []
        for lin, hr in zip(self.linears, self.has_relu):
            layers.append(lin)
            if hr:
                layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        return CompiledCircuit(
            model=model,
            n_inputs=self.n_inputs,
            n_outputs=self.current_width,
            fn=fn,
            motifs=list(self.placed),
            name=self.name,
            input_domain=self.input_domain,
        )
