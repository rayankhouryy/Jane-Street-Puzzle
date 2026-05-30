"""NeuroDecomp synthetic benchmark suite.

Hand-compiled ReLU circuits with known ground truth, used to measure
NeuroDecomp's motif precision/recall, codegen equivalence, and runtime
scaling on networks where the symbolic program is known a priori.
"""

from bench.compile import CircuitCompiler, CompiledCircuit, PlacedMotif

__all__ = ["CircuitCompiler", "CompiledCircuit", "PlacedMotif"]
