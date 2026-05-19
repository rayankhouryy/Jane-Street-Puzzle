# 02 — NeuroDecomp design

## Goal

Given an arbitrary `nn.Sequential` of `Linear` / `ReLU` modules whose weights
are sparse and drawn from a small integer alphabet, recover a Python program
that is functionally equivalent on the network's intended input domain.

## Inputs / outputs

```
INPUT
  model: nn.Sequential               (Linear / ReLU only)
  input_domain: List[AbstractValue]  (default: byte ∈ [0,255] per input dim)

OUTPUT
  Program object, with:
    - .to_python()   readable source string
    - .run(x)        Python interpreter, returns same value as model(x)
    - .verify(model, n_random)  end-to-end equivalence check
```

## Pipeline (8 stages)

```
PyTorch Sequential
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1 — SSA / sparse graph extraction (neurodecomp.sparse_graph)  │
│  - one SSA value per neuron                                          │
│  - each pre-ReLU encoded as sparse AffineExpr over earlier ids       │
│  - zero-weight edges dropped                                         │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 2 — canonicalisation (neurodecomp.canonical)                  │
│  - sort terms, gcd-reduce, sign-normalise                            │
│  - hash AffineExpr for dedup / motif matching                        │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 3 — abstract interpretation (neurodecomp.domains + .interp)   │
│  Lattice:                                                            │
│      Const(c)                                                        │
│      Dead     ≡ Const(0)                                             │
│      Bool     V ⊆ {0,1}                                              │
│      Bit(k)   V ⊆ {0, 2^k}                                           │
│      SmallSet V finite, |V| ≤ 16                                     │
│      Byte     V ⊆ {0..255}                                           │
│      UInt(b)  V ⊆ {0..2^b-1}                                         │
│      Interval [lo, hi] ⊂ ℤ                                           │
│      Affine   sparse linear combination of prior abstract values     │
│      Top                                                             │
│  Propagate from input_domain forward.                                │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 4 — block discovery (neurodecomp.block_finder)                │
│  - width-sequence periodicity                                        │
│  - per-position weight-tying via Frobenius dispersion ρ_p            │
│  - emits: head range, body block size + count + per-iter deltas, tail│
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 5 — motif library (neurodecomp.motifs)                        │
│  Catalog of recognisers, each:                                       │
│    (a) matches a canonical SSA subgraph,                             │
│    (b) emits a higher-level op (XOR, ROTL, eq, etc.),                │
│    (c) is locally verified by Z3 on a small bounded domain.          │
│  Motifs included:                                                    │
│    delta_eq, threshold_step, bit_extract, bit_pack,                  │
│    bool_and, bool_or, bool_xor, bool_not,                            │
│    mod2pow_add, rotl32, byte_pack, uint32_from_bits                  │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 6 — register inference (neurodecomp.registers)                │
│  - group bits into bytes (groups of 8 contiguous Bit lattice nodes)  │
│  - group bytes into words (32-bit lanes)                             │
│  - assign symbolic names (e.g. md5_A, md5_B, …) only by structural   │
│    role, not by string matching                                      │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 7 — codegen (neurodecomp.emit)                                │
│  Pretty-print decompiled Python with:                                │
│    - extracted per-iteration tables                                  │
│    - named registers                                                 │
│    - high-level ops from Stage 5                                     │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 8 — validation (neurodecomp.validate)                         │
│  - random input equivalence on the network's intended domain          │
│  - intermediate trace comparison (per-block / per-byte)              │
│  - Z3 motif equivalence proofs on bounded domains                    │
│  - if the network is identified as a hash, compare digest to hashlib │
└──────────────────────────────────────────────────────────────────────┘
```

## Design rules

1. **Never use finite value sets for wide neurons.** Enumeration is only
   permitted when $|V| \le 16$. Use abstract lattices otherwise.
2. **Always carry the input domain.** Without a domain you'll reason over
   $\mathbb{R}^{55}$ and see junk piecewise-linear extrapolations.
3. **Canonicalise affine expressions before matching motifs.** Equivalent
   circuits may be syntactically rearranged (sign flips, ordering, gcd).
4. **Use Z3 locally, never globally.** Z3 proves motifs match their specs on
   bounded domains; it does not solve the whole network.
5. **Block discovery comes early.** Period detection + weight-tying lets us
   decompile one representative block and fold 64 copies.
6. **The decompiler should be domain-agnostic.** No MD5-specific recognisers
   in the core. MD5 (or any other hash, padding scheme, cipher round, …)
   should fall out of generic motif composition.
