# NeuroDecomp — Progress Report

> A deep dive into the project for anyone joining mid-stream. Reading this
> end-to-end should leave you fully oriented: what the puzzle is, what we've
> discovered, how the tool works, and what's left to do.
>
> *Last updated: 2026-05-19 — corresponds to commit `a711b45`.*

---

## 1. TL;DR

We started with the [Jane Street March 2025 puzzle][js-puzzle]: a 1.16 GB
PyTorch `Sequential` model whose purpose was unknown. After confirming it was
solved publicly (it computes
`f(s) = 1 iff MD5(s) == c7ef65233c40aa32c2b9ace37595fa7c`, with
`"bitter lesson"` as a preimage), we pivoted to a research goal:

> **Build NeuroDecomp — a static decompiler that recovers a symbolic program
> from a hand-compiled ReLU network, with no algorithm-specific hints.**

The benchmark is the Jane Street model itself. We've shipped 7 MVPs that
collectively recover, from raw weights only:

- the ASCII / null-padded tokenizer,
- the 16-byte equality target `c7ef65233c40aa32c2b9ace37595fa7c`,
- four 32-bit IV constants matching MD5's IV,
- the precomputed subexpressions `B & C = 0x88888888` and `~B = 0x10325476`,
- the first round constant `K[0] = 0xd76aa478`,
- the per-iteration **rotate gadget** with the exact MD5 shift schedule
  `[7,12,17,22] × 4, [5,9,14,20] × 4, [4,11,16,23] × 4, [6,10,15,21] × ~4`,
- a Z3-proven catalog of 6 ReLU motifs,
- 12,859 *domain-certified* boolean ANDs (out of 58,524 syntactic
  candidates),
- a self-documenting Python file (`artifacts/recovered_program.py`) that
  embodies every recovered fact as a Python literal and raises
  `NotImplementedError` where decompilation is still in progress.

**Open** (in priority order):

1. **MVP-7.5** — richer relational abstract domain to certify more of the
   78 % unconfirmed AND candidates. **Highest leverage; do this next.**
2. **MVP-9** — F/G/H/I round-function clustering (~200 certified ANDs per
   iteration → named 3-input boolean functions on 32-bit registers).
3. **MVP-8** — recover K[1..63] as **residual** 32-bit constants after the
   round function is known.
4. **MVP-10** — padding scheme decode (head's first 18 layers).
5. **Final** — complete codegen + `verify_against_model` on 10 000 random
   byte strings.

**41 unit tests pass. 7 commits on `main`.**

[js-puzzle]: https://huggingface.co/spaces/jane-street/puzzle

---

## 2. The puzzle and the model

### 2.1 The artifact

- `torch.nn.modules.container.Sequential` with **5442 children**, strictly
  alternating `Linear` then `ReLU`. So 2721 Linears + 2721 ReLUs.
- First Linear has `in_features=55`; last has `out_features=1`.
- Hosted on Hugging Face:
  [`jane-street/2025-03-10`](https://huggingface.co/jane-street/2025-03-10).
  We use `model_3_11.pt` (Python 3.11 re-pack).
- Formally
  $f(x) = (L_{2721} \circ \sigma \circ \dots \circ \sigma \circ L_1)(x)$
  with affine $L_i$ and element-wise ReLU $\sigma$. Range is non-negative.

### 2.2 Hidden tokenizer

The HF Space calls `model("some string")` directly because the model
instance carries a cloudpickle-attached callable. Disassembling its bytecode
(`scripts/05_extract_tokenizer.py`) recovers:

```python
_call_impl = lambda x: model.forward(
    torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00'))))
)
```

So the input is the ASCII byte values of the first 55 characters, null-padded.

### 2.3 Public solution (validation oracle, kept strictly separate)

[adirk0/Jane-Street-Puzzle][adir] solved the puzzle by hand. The full
solution lives in [`docs/99_spoilers.md`](99_spoilers.md):

- $f(s) = \mathbb{1}\big[\mathrm{MD5}(s) = \mathtt{c7ef65233c40aa32c2b9ace37595fa7c}\big]$,
- valid preimage: `"bitter lesson"`.

`scripts/09_verify_md5.py` confirms `model("bitter lesson") = 1`. We use
this **only as validation** — never as input to the decompiler.

[adir]: https://github.com/adirk0/Jane-Street-Puzzle

---

## 3. Why this is interesting beyond the puzzle

NN verification tools (DeepPoly, α,β-CROWN, Marabou, Reluplex) ask:

> *Does this network satisfy property $P$?*

NeuroDecomp asks:

> *What program did this network secretly compile?*

The trick is that **hand-compiled** ReLU networks satisfy strong structural
properties:

- weights drawn from $\{0, \pm 1, \pm 2, \pm 4, \dots, \pm 256\}$,
- 99.6 % sparsity on the benchmark,
- activations are integers / 0-1 bits / 0..255 bytes / 32-bit lanes,
- ReLU `Linear` cascades express boolean and modular arithmetic via small
  algebraic identities (Kronecker $\delta$, XOR via ReLU, etc.).

Once we propagate the input domain forward through an appropriate abstract
lattice, every neuron resolves to a finite-domain value with a recognisable
algebraic role. The decompiler folds these into a higher-level Python
program.

---

## 4. Structural facts about the benchmark

See README §4–§6 for the full math. The highlights, recovered from weights
only:

### 4.1 Macro structure

Period detection on the layer-width sequence yields $p^\star = 42$ with
match ratio 0.97. Block-start detection gives:

| Region | Linear indices | Size | Role |
|---|---|---|---|
| **Head** | $L_1..L_{18}$        | 18    | MD5-style padding |
| **Body** | $L_{19}..L_{2664}$   | 2646  | 64 unrolled rounds, 63 recovered |
| **Tail** | $L_{2665}..L_{2721}$ | 57    | digest decoder + 16-byte equality |

### 4.2 Bit-mask fingerprint

The recurring widths $\{304, 312, 316, 318, 319\} = \{320 - 2^k\}_{k=0..4}$
are the canonical signature of bit-level operations on a 32-bit word lane.
Body working width is $288 = 256 + 32$.

### 4.3 Weight tying

For each in-block position $p$, the relative dispersion

$$
\rho_p = \frac{1}{\|\bar W^{(p)}\|_F}\cdot \frac{1}{63}\sum_{t=1}^{63}\big\|W^{(p)}_t - \bar W^{(p)}\big\|_F
$$

is $0$ (bit-identical) for 38 of 42 positions. Only position 28 has
$\rho_{28} \approx 1.19$ (per-iteration variation); positions 0, 1, 41 are
boundary stitching. **The body is a true weight-tied unrolled loop.**

### 4.4 Head SVD

$W_1 \in \{0, 1\}^{224 \times 55}$ with exactly 3 ones per column and all
singular values equal to $\sqrt{3}$. The bias contains the integers
$1, 2, \dots, 55$ — the head is laying the 55 bytes on a positional strip.

### 4.5 Tail signature

The final Linear $L_{2721}: \mathbb{R}^{48} \to \mathbb{R}$ has weights
$(+1)^{16}, (-2)^{16}, (+1)^{16}$ and bias $-15$, so

$$
f(x) = \mathrm{ReLU}\!\big(S_1 - 2 S_2 + S_3 - 15\big), \qquad S_k = \sum_i y_i^{(k)}.
$$

Each triple $(a_i, b_i, c_i)$ encodes a Kronecker delta
$\delta(z_i) = \mathrm{ReLU}(z_i+1) + \mathrm{ReLU}(z_i-1) - 2\mathrm{ReLU}(z_i)$,
so the final scalar is

$$
f(x) = \bigwedge_{i=1}^{16}\mathbb{1}[\text{predicate}_i(x)]
$$

— the AND of 16 integer-equality predicates. The penultimate Linear is a
binary-to-byte decoder with powers-of-two weights.

---

## 5. NeuroDecomp — the tool

### 5.1 Pipeline (8 stages)

```
PyTorch Sequential
       │
       ▼ Stage 1  SSA + sparse weighted DAG       neurodecomp.sparse_graph
       ▼ Stage 2  Affine canonicalisation         (implicit)
       ▼ Stage 3  Abstract interpretation         neurodecomp.{domains, interp}
       ▼ Stage 4  Block discovery + loop folding  neurodecomp.block_finder
       ▼ Stage 5  Motif library                   neurodecomp.motifs (Z3-verified)
       ▼ Stage 6  Register inference              (partial — MVP-9 in progress)
       ▼ Stage 7  Codegen                         neurodecomp.emit
       ▼ Stage 8  Validation                      neurodecomp.certify + verify_against_model
```

### 5.2 Design rules

1. **Never use finite value sets for wide neurons.** Enumeration only when
   $|V| \le 16$; abstract lattices otherwise.
2. **Always carry the input domain.** Start from
   `Byte ∈ [0, 255]^55`.
3. **Canonicalise before motif matching.**
4. **Use Z3 locally, never globally.**
5. **Block discovery comes early.** Decompile one iteration and fold the rest.
6. **No algorithm-specific hard-coding in the core.** Algorithm
   identification belongs in validation, not in decompilation.

---

## 6. MVPs — what each one does, and what it recovered

### MVP-1: Sparse Graph + Domain Profiler — `scripts/run_profile.py`

Structural profile of an arbitrary `nn.Sequential`. On the benchmark:

```text
total modules   : 5442 (2721 Linear + 2721 ReLU, alternating)
signature       : f : R^55 -> R^1
weight alphabet : integers in [-1, 30] (full alphabet ±powers of 2 + small ints)
sparsity        : 99.6 % of weights are zero
tokenizer       : ord(str(x)[:55].ljust(55, '\x00'))
best period     : 42 (97.05 % match)
head/body/tail  : [0, 18) / [18, 2664) / [2664, 2721)
iterations      : 63 in recovered body
weight tying    : 40/42 positions bit-identical (varying: {28, 41})
output template : AND of 16 integer-equality predicates
```

### MVP-2: Tail Decompiler — `scripts/run_tail.py`

Detects the AND-of-deltas template at the last two Linears, walks back
through the binary-to-byte decoder, recovers per-byte targets.

**Recovered**: 16-byte target `c7ef65233c40aa32c2b9ace37595fa7c`, plus
per-byte bit-address layout. Eight of 16 bytes use an extra modular-subtract
gadget with $-2^k$ weights.

### MVP-3: Head Decompiler — `scripts/run_head.py`

Forward abstract interpretation through the head's 18 Linears starting from
`Byte ∈ [0, 255]^55`. Lattice = `Affine` type in `neurodecomp/domains.py`.
Each neuron resolves to const / passthrough / affine / opaque.

**Recovered** (decoded as 4-byte LE words from bias structure):

| Bytes (LE) | Value | Identification |
|---|---|---|
| `01 23 45 67` | `0x67452301` | MD5 IV A |
| `89 ab cd ef` | `0xefcdab89` | MD5 IV B |
| `fe dc ba 98` | `0x98badcfe` | MD5 IV C |
| `76 54 32 10` | `0x10325476` | MD5 IV D |
| `88 88 88 88` | `0x88888888` | `B & C` (precomputed) |
| `76 54 32 10` | `0x10325476` | `~B` (= D for these IVs) |
| `78 a4 6a d7` | `0xd76aa478` | MD5 round constant `K[0]` |

### MVP-4: Body Decompiler — `scripts/run_body.py`

Extracts per-iteration variation at varying positions. Detects multi-register
rotate-by-$s_t$ gadgets (one-hot `+1` selector inside a lane-aligned
`n_lanes`-wide window).

**Recovered**: 3-register × 32-bit rotate gadget at position 28 of 42, with
the full MD5 shift schedule for 63 iterations:

```
s[ 0..15]  = [7, 12, 17, 22] × 4
s[16..31]  = [5,  9, 14, 20] × 4
s[32..47]  = [4, 11, 16, 23] × 4
s[48..62]  = [6, 10, 15, 21] × ~4
```

Iteration 64 (MD5 round 64) is currently absorbed into head/tail/stitching.

### MVP-6: Motif Library — `scripts/run_motifs.py`

Catalog of 6 small ReLU motifs, each self-verified by Z3 on bounded integer
domains:

| Motif | Identity | Domain |
|---|---|---|
| delta | $\mathrm{ReLU}(z+1) + \mathrm{ReLU}(z-1) - 2\mathrm{ReLU}(z) = \mathbb{1}[z=0]$ | $z \in [-8, 8]$ |
| bool_and | $\mathrm{ReLU}(a+b-1) = a \land b$ | $a, b \in \{0, 1\}$ |
| bool_or | $a + b - \mathrm{ReLU}(a+b-1) = a \lor b$ | $\{0, 1\}^2$ |
| bool_xor | $a + b - 2\mathrm{ReLU}(a+b-1) = a \oplus b$ | $\{0, 1\}^2$ |
| bool_not | $1 - a$ | $\{0, 1\}$ |
| threshold | $\mathrm{ReLU}(x - k)$ | $x, k \in [-8, 8]$ |

Scan output on the benchmark:

```text
[PROVED] delta, bool_and, bool_or, bool_xor, bool_not, threshold

Hits per region (syntactic, candidates):
  head:     32 bool_and
  body: 56,708 bool_and (~900 per iteration)
  tail:  1,784 bool_and
```

⚠️ Candidates only — see MVP-7 for the precision upgrade.

### MVP-7: Domain-certified motif recognition — `scripts/run_certify.py`

Glues Stage 3 (abstract interpretation) and Stage 5 (motifs). A candidate
`bool_and` is confirmed iff both inputs are provably in $\{0, 1\}$ on the
byte input domain.

```text
58,524 candidate ANDs → 12,859 CONFIRMED (22.0 %)

Head:  32 of    32 confirmed (100 %)
Body:  12,575 of 56,708 confirmed (22.2 % — ~200 per iteration)
Tail:  252 of  1,784 confirmed (14.1 %)
```

A 76 % reduction. The remaining 78 % unconfirmed are blocked by precision
loss when the abstract domain widens through saturating ReLUs — exactly the
issue MVP-7.5 addresses.

### MVP-5: Codegen — `scripts/run_emit.py`

Aggregates artefacts from MVPs 1–7 into `artifacts/recovered_program.py`.
Recovered pieces are inlined as Python literals; unrecovered pieces raise
`NotImplementedError` with a pointer to the responsible pending MVP.

```python
TOKENIZER_LENGTH = 55
INITIAL_STATE_LE = (0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476)
EXTRA_HEAD_CONSTANTS_LE = (0x88888888, 0x10325476, 0xd76aa478)
NUM_BODY_ITERATIONS = 63
SHIFT_SCHEDULE = (7, 12, 17, 22, 7, ...)   # 63 entries
TARGET_DIGEST_HEX = 'c7ef65233c40aa32c2b9ace37595fa7c'

def encode(s): ...                         # ← runs
def K(i): ...                              # ← K[0] only (MVP-8)
def round_function(...): ...               # ← raises (MVP-9)
def pad_message(...): ...                  # ← raises (MVP-10)
```

A `verify_against_model(model, samples)` harness runs samples through both
the original model and the emitted predictor side-by-side.

---

## 7. Where we are vs where MD5 really is

| Component | Recovered value | MD5 truth | Status |
|---|---|---|---|
| Initial state A | `0x67452301` | `0x67452301` | ✅ exact |
| Initial state B | `0xefcdab89` | `0xefcdab89` | ✅ exact |
| Initial state C | `0x98badcfe` | `0x98badcfe` | ✅ exact |
| Initial state D | `0x10325476` | `0x10325476` | ✅ exact |
| First round constant K[0] | `0xd76aa478` | `0xd76aa478` | ✅ exact |
| K[1..63]                  | only K[0]      | 63 more values    | ⏳ MVP-8 |
| Shift schedule rounds 0..62 | matches MD5 exactly | rounds 0..63 | ✅ 63 recovered |
| Round 63 (= MD5 round 64) | absorbed into stitching | s = 21 | ⏳ |
| F/G/H/I round functions   | not yet decoded | known formulas | ⏳ MVP-9 |
| 32-bit modular adder      | candidate ANDs only | one per round | ⏳ MVP-9 |
| Padding scheme            | not yet decoded | length + 0x80 + length-suffix | ⏳ MVP-10 |
| Target digest             | `c7ef65...fa7c` | recovered from tail | ✅ exact |
| Tokenizer                 | `ord(s[:55].ljust(55,'\x00'))` | same | ✅ exact |

---

## 8. Roadmap (revised order — leverage first)

The original plan put codegen and K-table recovery before deeper semantics.
The revised order does **semantics first**, codegen last:

1. **MVP-7.5 — relational abstract domain** *(highest leverage, next)*

   Bottleneck today: 58k candidate ANDs collapse to 12,859 because the
   current `Affine` lattice widens through saturating ReLUs. The richer
   domain tracks
   `Opaque(id, expr=known_affine, constraints=...)` so that downstream
   reasoning can prove `u = ReLU(a + b - 1)` is `a AND b` whenever `a, b ∈
   {0, 1}`, even when `a + b - 1` straddles zero. Expected outcome:
   confirmed-AND count rises substantially, more motifs become certifiable
   (XOR, OR, delta in addition to AND), and head opaque symbols become
   tractable.

   *Deliverable*: `neurodecomp/domains.py` extended with `Opaque(expr=…)`;
   `neurodecomp/interp.py` updated to carry expressions through saturating
   ReLUs; `tests/test_interp.py` extended with tests that show
   improved precision on the Kronecker-delta toy nets.

2. **MVP-9 — F/G/H/I round-function clustering**

   With the abstract domain stronger, cluster the ~200 certified booleans
   per body iteration into named 3-input boolean functions on 32-bit
   registers. Each iteration's round-function shape ∈ {F, G, H, I}.

   *Deliverable*: `neurodecomp/registers.py`, `neurodecomp/cluster_bool.py`,
   `scripts/run_round_decode.py`.

3. **MVP-8 — recover K[1..63] as residuals**

   With the round function known, the per-iteration $K_i$ is whatever fixed
   32-bit residual makes the round match the model's body. Avoids the
   bias-archaeology rabbit hole.

   *Deliverable*: a constant-extraction routine that takes the recovered
   round function and the body weights of one iteration, and emits $K_i$
   for $i \in \{1, \dots, 63\}$.

4. **MVP-10 — padding scheme**

   Symbolic recovery of the head's first 18 layers as an explicit
   `pad_message` function (input length count, `0x80` marker, zero pad,
   length suffix). Depends on MVP-7.5.

5. **Final — verified emit**

   Replace `NotImplementedError` stubs in `artifacts/recovered_program.py`;
   run `verify_against_model` on 10 000 random byte strings; declare
   equivalence on the byte-domain input set.

---

## 9. Codebase tour

```
.
├── README.md                            long-form math + roadmap
├── docs/
│   ├── 00_problem.md                    what the model presents to the world
│   ├── 02_neurodecomp_design.md         pipeline + design rules
│   ├── 99_spoilers.md                   public solution (validation only)
│   └── PROGRESS.md                      ← you are here
├── neurodecomp/                         the tool
│   ├── __init__.py
│   ├── model_loader.py                  load .pt + recover tokenizer
│   ├── sparse_graph.py                  Stage 1
│   ├── domains.py                       Stage 3a: Affine abstract value
│   ├── interp.py                        Stage 3b: forward abstract interp
│   ├── block_finder.py                  Stage 4
│   ├── head.py                          head decompiler (MVP-3)
│   ├── body.py                          body decompiler (MVP-4)
│   ├── tail.py                          tail decompiler (MVP-2)
│   ├── motifs.py                        Stage 5 (MVP-6)
│   ├── certify.py                       domain certification (MVP-7)
│   └── emit.py                          Stage 7 codegen (MVP-5)
├── scripts/
│   ├── 01–09_*.py                       baseline manual replication
│   ├── inspect_boundary.py              exploratory K probe
│   ├── run_profile.py                   MVP-1
│   ├── run_tail.py                      MVP-2
│   ├── run_head.py                      MVP-3
│   ├── run_body.py                      MVP-4
│   ├── run_motifs.py                    MVP-6
│   ├── run_certify.py                   MVP-7
│   └── run_emit.py                      MVP-5
├── tests/                               41 unit tests
└── artifacts/
    └── recovered_program.py             flagship output of MVP-5
```

---

## 10. Reproducing every MVP

```powershell
# Weights (1.16 GB; .gitignored).
curl -L -o model_3_11.pt "https://huggingface.co/jane-street/2025-03-10/resolve/main/model_3_11.pt"

pip install torch numpy cloudpickle z3-solver
python -m unittest discover tests                # 41 should pass

python scripts/run_profile.py model_3_11.pt      # MVP-1: structural profile
python scripts/run_tail.py    model_3_11.pt      # MVP-2: 16-byte target
python scripts/run_head.py    model_3_11.pt      # MVP-3: IV + K[0]
python scripts/run_body.py    model_3_11.pt      # MVP-4: shift schedule
python scripts/run_motifs.py  model_3_11.pt      # MVP-6: 58k candidates
python scripts/run_certify.py model_3_11.pt     # MVP-7: 12.9k confirmed
python scripts/run_emit.py    model_3_11.pt     # MVP-5: emit artefact

python scripts/09_verify_md5.py                 # spoiler — for sanity
```

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Affine** | sparse integer linear combination of named variables + bias; the abstract value type. |
| **Block finder** | period detection + head/body/tail split with weight-tying analysis. |
| **Body iteration** | one execution of the 42-layer block; 63 unrolled in the benchmark. |
| **Boundary stitching** | the 3 in-block positions (0, 1, 41) whose tensors change shape between consecutive iterations. |
| **Hand-compiled** | the network encodes an algorithm into integer weights by hand, not by gradient descent. |
| **Kronecker delta motif** | the three-ReLU identity $\delta(z) = \mathrm{ReLU}(z+1) + \mathrm{ReLU}(z-1) - 2\mathrm{ReLU}(z)$ that detects `z == 0` on integers. |
| **Motif** | a small ReLU-implementable identity with a Z3-proof of equivalence to a Python reference. |
| **Opaque ReLU symbol** | a fresh variable introduced by the abstract interpreter when a ReLU's input straddles zero. MVP-7.5 will track the expression behind each opaque symbol. |
| **Rotate gadget** | per-iteration weight matrix that, for output bit `b`, selects input bit `(b - s_t) mod n` — hand-compiled `ROTL`. |
| **Sparse DAG** | the network represented as SSA values with zero-weight edges dropped. |

---

## 12. References

### Internal
- `README.md` — long-form math.
- `docs/00_problem.md`, `docs/02_neurodecomp_design.md` — design.
- `docs/99_spoilers.md` — Adir's solution (validation oracle).

### External
- α,β-CROWN — NN verification via bound propagation:
  <https://github.com/Verified-Intelligence/alpha-beta-CROWN>
- Marabou — SMT-based NN verifier:
  <https://github.com/NeuralNetworkVerification/Marabou>
- DeepPoly / AI² — abstract interpretation for NNs.
- `egg` — equality saturation / e-graphs:
  <https://egraphs-good.github.io/>
- Z3 — SMT with bit-vector logic:
  <https://github.com/Z3Prover/z3>
- Souper, STOKE — LLVM IR superoptimisation.
- Adir's MD5 write-up: <https://github.com/adirk0/Jane-Street-Puzzle>
- Sutton, *"The Bitter Lesson"*:
  <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>
