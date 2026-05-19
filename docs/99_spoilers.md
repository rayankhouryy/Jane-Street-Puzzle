# 99 — Spoilers ⚠️

**This file contains the public solution. Read it only for validation. The
goal of NeuroDecomp is to rediscover this information automatically; using it
as input defeats the purpose.**

---

The Jane Street March 2025 puzzle was solved publicly. The model computes

$$
f(s) \;=\; \mathbb{1}\big[\,\mathrm{MD5}(s) \;=\; t\,\big],
\qquad t = \mathtt{c7ef65233c40aa32c2b9ace37595fa7c}.
$$

A valid preimage is the string `"bitter lesson"` — Rich Sutton's essay
on scaling laws beating hand-engineered methods.

```python
>>> hashlib.md5(b"bitter lesson").hexdigest()
'c7ef65233c40aa32c2b9ace37595fa7c'
>>> model("bitter lesson")
tensor([1.])
```

`scripts/09_verify_md5.py` confirms this end-to-end.

## Roles of the head, body, and tail

| Region | Linears | Layers | Role |
|---|---|---|---|
| **Head** | $L_1 \ldots L_{18}$         | 18    | MD5 padding: length-prefix → `0x80` → zero pad → 64-bit length suffix |
| **Body** | $L_{19} \ldots L_{2664}$    | 2646  | 64 rounds of MD5; one 42-layer block per round, weight-tied |
| **Tail** | $L_{2665} \ldots L_{2721}$  | 57    | Binary-to-byte hash decoder + 16-byte equality AND |

## How the structure encodes MD5

- **42-layer block, 64 iterations** ↔ the 64 MD5 round transformations
  $B \leftarrow B + \mathrm{ROTL}_{s_i}(A + F_i(B,C,D) + M_{g_i} + K_i)$.
- **Per-iteration delta at position 28** ↔ the per-round left-rotate constants
  $s_i \in \{7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21\}$.
- **Per-iteration delta at other positions** ↔ the per-round additive
  constants $K_i$ derived from $\sin$ values.
- **Recurring widths $\{304, 312, 316, 318, 319\} = \{320 - 2^k\}_{k=0..4}$**
  ↔ bit-level extraction inside 32-bit word lanes.
- **Final $W = ([+1]^{16}, [-2]^{16}, [+1]^{16})$ with bias $-15$** ↔ AND of
  16 Kronecker deltas, one per byte of the 128-bit digest.

## Use as validation only

`neurodecomp.validate` will compare any program emitted by the decompiler to
`hashlib.md5(s.encode()).hexdigest() == t` on 10 000 random byte strings.
That's the win condition.

Credit for the original solve: <https://github.com/adirk0/Jane-Street-Puzzle>.
