"""Tests for the synthetic-benchmark compiler.

These tests are the compiler's correctness contract: every circuit
``build_*`` function must produce a network whose forward pass exactly
matches its declared Python reference function on every enumerable
input.  If this drifts, the benchmark metrics are meaningless.
"""
from __future__ import annotations

import pytest

from bench.circuits.bool_gates import (
    build_and1,
    build_and_nbit,
    build_mux1,
    build_not1,
    build_or1,
    build_or_nbit,
    build_xor1,
    build_xor_nbit,
)
from bench.compile import CircuitCompiler


# ----------------------------------------------------------------------
# CircuitCompiler unit tests
# ----------------------------------------------------------------------

def test_row_rejects_out_of_range_wire():
    c = CircuitCompiler(n_inputs=2)
    with pytest.raises(ValueError):
        c.row({5: 1}, bias=0)


def test_commit_stage_empty_raises():
    c = CircuitCompiler(n_inputs=1)
    with pytest.raises(ValueError):
        c.commit_stage()


def test_records_placed_motif():
    c = CircuitCompiler(n_inputs=2, name="t")
    a, b = c.inputs()
    c.row({a: 1, b: 1}, bias=-1, motif="bool_and", inputs=(a, b))
    c.commit_stage()
    c.row({0: 1})
    circ = c.build(fn=lambda x: [int(bool(x[0]) and bool(x[1]))])
    assert len(circ.motifs) == 1
    m = circ.motifs[0]
    assert m.kind == "bool_and"
    assert m.layer_idx == 0
    assert m.out_neuron == 0
    assert m.in_neurons == (0, 1)


# ----------------------------------------------------------------------
# Per-circuit equivalence: compiled model agrees with oracle on every
# enumerable input.
# ----------------------------------------------------------------------

ONE_BIT_BUILDERS = [build_and1, build_not1, build_or1, build_xor1, build_mux1]
N_BIT_BUILDERS = [
    lambda: build_and_nbit(8),
    lambda: build_or_nbit(8),
    lambda: build_xor_nbit(8),
]


@pytest.mark.parametrize("builder", ONE_BIT_BUILDERS,
                         ids=lambda b: b.__name__)
def test_one_bit_circuit_equivalence(builder):
    circuit = builder()
    ok, n_tested, n_mismatch = circuit.check_equivalence()
    assert ok, f"{circuit.name}: {n_mismatch}/{n_tested} mismatches"


@pytest.mark.parametrize("builder", N_BIT_BUILDERS,
                         ids=lambda b: getattr(b, "__name__", repr(b)))
def test_n_bit_circuit_equivalence(builder):
    circuit = builder()
    ok, n_tested, n_mismatch = circuit.check_equivalence()
    assert ok, f"{circuit.name}: {n_mismatch}/{n_tested} mismatches"


# ----------------------------------------------------------------------
# Ground-truth placement: every placed motif lives on a real Linear,
# and the placed (layer_idx, out_neuron) is in range.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("builder",
                         ONE_BIT_BUILDERS + N_BIT_BUILDERS,
                         ids=lambda b: getattr(b, "__name__", repr(b)))
def test_placed_motifs_are_well_formed(builder):
    import torch.nn as nn

    circuit = builder()
    linears = [m for m in circuit.model if isinstance(m, nn.Linear)]
    for motif in circuit.motifs:
        assert 0 <= motif.layer_idx < len(linears), (
            f"{circuit.name}: motif layer {motif.layer_idx} out of range "
            f"({len(linears)} linears)"
        )
        lin = linears[motif.layer_idx]
        assert 0 <= motif.out_neuron < lin.out_features
        for n in motif.in_neurons:
            assert 0 <= n < lin.in_features
