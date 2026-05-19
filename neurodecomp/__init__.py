"""NeuroDecomp — automatically recovering symbolic programs from
hand-compiled ReLU networks.

Quick start::

    import torch
    from neurodecomp import profile, decompile

    model = torch.load("model_3_11.pt", weights_only=False, map_location="cpu")
    report = profile(model)             # MVP-1: structural facts
    program = decompile(model)          # MVP-5: full decompilation (WIP)
    print(program.to_python())
    program.verify(model)

Subpackage map (see docs/02_neurodecomp_design.md)::

    model_loader   load .pt + recover any cloudpickle-attached tokenizer
    sparse_graph   Stage 1: SSA / DAG extraction
    canonical      Stage 2: affine canonicalisation
    domains        Stage 3a: abstract value lattice
    interp         Stage 3b: forward abstract interpretation
    block_finder   Stage 4: periodicity + weight-tying detection
    motifs         Stage 5: motif library (Z3-verified)
    registers      Stage 6: byte / word grouping
    emit           Stage 7: Python code emission
    validate       Stage 8: equivalence checks
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import block_finder, body, head, interp, model_loader, sparse_graph, tail
from .block_finder import find_blocks
from .body import decompile_body
from .head import decompile_head
from .model_loader import load_model, recover_tokenizer
from .sparse_graph import extract_ssa
from .tail import decompile_tail

__all__ = [
    "extract_ssa",
    "find_blocks",
    "load_model",
    "recover_tokenizer",
    "decompile_tail",
    "decompile_head",
    "decompile_body",
    "block_finder",
    "body",
    "head",
    "interp",
    "model_loader",
    "sparse_graph",
    "tail",
]
