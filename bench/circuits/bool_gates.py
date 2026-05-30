"""Boolean-gate benchmarks: ground-truth-known small circuits.

Each ``build_*`` function returns a :class:`CompiledCircuit` ready for
the bench runner.  The Python reference function ``fn`` is the oracle;
the placed-motif list is the structural ground truth NeuroDecomp's
scanner / certifier should recover.

Canonical motif shapes (so the existing motif scanner picks them up):

    AND(a, b) :  one stage   -- row {a:+1, b:+1, bias:-1}      [bool_and]
    NOT(a)    :  one stage   -- row {a:-1, bias:+1}            [no detector]
    OR(a, b)  :  two stages  -- s1: passthroughs(a,b) + AND;
                                s2: row {pa:+1, pb:+1, pand:-1} bias 0
                                                              [no detector]
    XOR(a, b) :  two stages  -- s1: passthroughs(a,b) + AND;
                                s2: row {pa:+1, pb:+1, pand:-2} bias 0
                                                              [bool_xor]
    MUX(s, a, b) = (NOT s AND a) OR (s AND b)
                 : three stages of mixed gates                [partial: 2 ANDs]
"""
from __future__ import annotations

from typing import List, Sequence

from bench.compile import CircuitCompiler, CompiledCircuit


# ----------------------------------------------------------------------
# 1-bit primitives
# ----------------------------------------------------------------------

def build_and1() -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2, name="and1")
    a, b = c.inputs()
    y = c.row({a: 1, b: 1}, bias=-1, motif="bool_and", inputs=(a, b))
    c.commit_stage()
    c.row({y: 1})
    return c.build(fn=lambda x: [int(bool(x[0]) and bool(x[1]))])


def build_not1() -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=1, name="not1")
    (a,) = c.inputs()
    y = c.row({a: -1}, bias=1)   # 1 - a, ReLU passes since a in {0,1}
    c.commit_stage()
    c.row({y: 1})
    return c.build(fn=lambda x: [int(not bool(x[0]))])


def build_or1() -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2, name="or1")
    a, b = c.inputs()
    pa = c.passthrough(a)
    pb = c.passthrough(b)
    pand = c.row({a: 1, b: 1}, bias=-1, motif="bool_and", inputs=(a, b))
    c.commit_stage()
    y = c.row({pa: 1, pb: 1, pand: -1}, bias=0)  # a + b - AND(a, b) = OR
    c.commit_stage()
    c.row({y: 1})
    return c.build(fn=lambda x: [int(bool(x[0]) or bool(x[1]))])


def build_xor1() -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2, name="xor1")
    a, b = c.inputs()
    pa = c.passthrough(a)
    pb = c.passthrough(b)
    pand = c.row({a: 1, b: 1}, bias=-1, motif="bool_and", inputs=(a, b))
    c.commit_stage()
    y = c.row({pa: 1, pb: 1, pand: -2}, bias=0,
              motif="bool_xor", inputs=(pa, pb))
    c.commit_stage()
    c.row({y: 1})
    return c.build(fn=lambda x: [int(bool(x[0]) ^ bool(x[1]))])


def build_mux1() -> CompiledCircuit:
    """mux(s, a, b) = (NOT s AND a) OR (s AND b).

    Layout (5 stages -> 4 Linears + final passthrough):
       s1: ns = NOT(s) ; pa, pb propagated  (1 Linear)
       s2: x = AND(ns, a) ; y = AND(s, b) ; passthroughs   (1 Linear, 2 ANDs)
       s3: OR(x, y) via passthroughs + AND  (1 Linear, 1 AND for OR-inner)
       s4: OR output row  (1 Linear)
       s5: passthrough output
    """
    c = CircuitCompiler(n_inputs=3, name="mux1")
    s, a, b = c.inputs()
    # Stage 1: NOT(s) and propagate s, a, b
    ns = c.row({s: -1}, bias=1)
    ps = c.passthrough(s)
    pa = c.passthrough(a)
    pb = c.passthrough(b)
    c.commit_stage()
    # Stage 2: x = ns AND pa, y = ps AND pb, propagate x and y for OR
    x = c.row({ns: 1, pa: 1}, bias=-1, motif="bool_and", inputs=(ns, pa))
    y = c.row({ps: 1, pb: 1}, bias=-1, motif="bool_and", inputs=(ps, pb))
    c.commit_stage()
    # Stage 3: OR(x, y) -- requires passthroughs + inner AND
    px = c.passthrough(x)
    py = c.passthrough(y)
    pand = c.row({x: 1, y: 1}, bias=-1, motif="bool_and", inputs=(x, y))
    c.commit_stage()
    # Stage 4: out = px + py - pand
    out = c.row({px: 1, py: 1, pand: -1}, bias=0)
    c.commit_stage()
    c.row({out: 1})

    def fn(xin: Sequence[int]) -> List[int]:
        sv, av, bv = bool(xin[0]), bool(xin[1]), bool(xin[2])
        return [int(av if not sv else bv)]

    return c.build(fn=fn)


# ----------------------------------------------------------------------
# n-bit lane-parallel versions
# ----------------------------------------------------------------------

def build_and_nbit(n: int = 8) -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2 * n, name=f"and{n}")
    ins = c.inputs()
    a_bits, b_bits = ins[:n], ins[n:]
    ys = []
    for i in range(n):
        y = c.row({a_bits[i]: 1, b_bits[i]: 1}, bias=-1,
                  motif="bool_and", inputs=(a_bits[i], b_bits[i]))
        ys.append(y)
    c.commit_stage()
    for y in ys:
        c.row({y: 1})

    def fn(xin: Sequence[int]) -> List[int]:
        return [int(bool(xin[i]) and bool(xin[n + i])) for i in range(n)]

    return c.build(fn=fn)


def build_xor_nbit(n: int = 8) -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2 * n, name=f"xor{n}")
    ins = c.inputs()
    a_bits, b_bits = ins[:n], ins[n:]
    pas, pbs, pands = [], [], []
    for i in range(n):
        pas.append(c.passthrough(a_bits[i]))
        pbs.append(c.passthrough(b_bits[i]))
        pands.append(c.row({a_bits[i]: 1, b_bits[i]: 1}, bias=-1,
                           motif="bool_and",
                           inputs=(a_bits[i], b_bits[i])))
    c.commit_stage()
    ys = []
    for i in range(n):
        y = c.row({pas[i]: 1, pbs[i]: 1, pands[i]: -2}, bias=0,
                  motif="bool_xor", inputs=(pas[i], pbs[i]))
        ys.append(y)
    c.commit_stage()
    for y in ys:
        c.row({y: 1})

    def fn(xin: Sequence[int]) -> List[int]:
        return [int(bool(xin[i]) ^ bool(xin[n + i])) for i in range(n)]

    return c.build(fn=fn)


def build_or_nbit(n: int = 8) -> CompiledCircuit:
    c = CircuitCompiler(n_inputs=2 * n, name=f"or{n}")
    ins = c.inputs()
    a_bits, b_bits = ins[:n], ins[n:]
    pas, pbs, pands = [], [], []
    for i in range(n):
        pas.append(c.passthrough(a_bits[i]))
        pbs.append(c.passthrough(b_bits[i]))
        pands.append(c.row({a_bits[i]: 1, b_bits[i]: 1}, bias=-1,
                           motif="bool_and",
                           inputs=(a_bits[i], b_bits[i])))
    c.commit_stage()
    ys = []
    for i in range(n):
        y = c.row({pas[i]: 1, pbs[i]: 1, pands[i]: -1}, bias=0)
        ys.append(y)
    c.commit_stage()
    for y in ys:
        c.row({y: 1})

    def fn(xin: Sequence[int]) -> List[int]:
        return [int(bool(xin[i]) or bool(xin[n + i])) for i in range(n)]

    return c.build(fn=fn)


# ----------------------------------------------------------------------
# Registry for the bench runner
# ----------------------------------------------------------------------

ALL_CIRCUITS = [
    build_and1,
    build_not1,
    build_or1,
    build_xor1,
    build_mux1,
    lambda: build_and_nbit(8),
    lambda: build_or_nbit(8),
    lambda: build_xor_nbit(8),
]
