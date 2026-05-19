# NeuroDecomp

> **Automatically recovering symbolic programs from hand-compiled ReLU
> networks.**

Some neural networks aren't really neural networks. When their weights live in
$\{0, \pm 1, \pm 2, \pm 4, \dots\}$ and 99% of them are zero, they were
**hand-compiled**: someone wrote an algorithm by hand and arranged a
`Linear → ReLU → Linear → ReLU → …` stack to evaluate it. NeuroDecomp is a
tool that, given such a network, recovers the underlying algorithm as
readable Python — without being told what algorithm to look for.

The validation target for this repo is the [Jane Street March 2025 puzzle][js-puzzle]:
a 1.16 GB PyTorch model that takes a string and returns 0 or 1.
The puzzle ships only the weights. **Our job is to make NeuroDecomp tell us
what the model computes, with no human in the loop.**

> **Spoiler warning.** The puzzle has been solved publicly. To keep the
> decompiler honest we keep the known answer in [`docs/99_spoilers.md`](docs/99_spoilers.md)
> and use it only for after-the-fact validation. The README and design docs
> reason only from observable facts about the weights.

[js-puzzle]: https://huggingface.co/spaces/jane-street/puzzle

---

## 1. What this is, what it isn't

| | |
|---|---|
| **Is** | a static decompiler for sparse, integer-domain ReLU networks |
| **Is** | a research probe into algorithm recovery from circuit-style nets |
| **Is** | benchmarked against a real, fully reverse-engineerable target |
| **Isn't** | a general decompiler for trained networks (open problem) |
| **Isn't** | an NN verifier (DeepPoly / α,β-CROWN / Marabou do that) |
| **Isn't** | a SAT/SMT engine bolted onto Linear/ReLU (doesn't scale) |

Standard NN verification asks *"does this network satisfy property $P$?"*
NeuroDecomp asks *"what program did this network secretly compile?"*

---

## 2. The benchmark target

The model presents the following type and structure to a fresh observer.

### 2.1 Container and signature

- `torch.nn.modules.container.Sequential`.
- 5442 children, strictly alternating: 2721 `Linear` + 2721 `ReLU`.
- First `Linear`: `in_features = 55`. Last `Linear`: `out_features = 1`.
- Custom `_call_impl` attached via cloudpickle so `model("string")` works
  directly. Disassembling the bytecode (`scripts/05_extract_tokenizer.py`)
  yields

  ```python
  _call_impl = lambda x: model.forward(
      torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00'))))
  )
  ```

  so the input is the ASCII byte values of the first 55 characters,
  null-padded:

  $$x_i = \mathrm{ord}(s_i), \qquad i = 0, \dots, 54.$$

- Effective type:

  $$f : \Sigma^{\le 55} \;\longrightarrow\; \{0, 1\}, \qquad \Sigma = \{0, \dots, 255\}.$$

  Output is non-negative (final activation is `ReLU`) and behaves like a 0/1
  indicator on random ASCII inputs.

### 2.2 Sparsity and weight alphabet

- 99.6% of `Linear.weight` entries are exactly zero.
- The set of distinct nonzero values is small and integer-valued (powers of 2
  plus small integers, $\{0, \pm 1, \pm 2, \pm 4, \dots, \pm 256\}$).
- $\Rightarrow$ **the network was not trained; it is a hand-compiled circuit.**

### 2.3 Macro structure: head / body / tail

Let $d_i$ denote the output width of the $i$-th `Linear` and $d_0 = 55$ the
input width. Best-matching period of the width sequence:

$$
p^\star = 42, \qquad \text{match ratio } = 0.97.
$$

The longest contiguous range that obeys that period gives the loop body:

| Region | Linear indices | Layers | Width signature |
|---|---|---|---|
| **Head** | $L_1 \ldots L_{18}$         | 18   | $55 \to 224 \to 232 \to 64 \to \ldots \to 336$ |
| **Body** | $L_{19} \ldots L_{m}$       | $42 N$ | 42-layer block, $N$ iterations |
| **Tail** | $L_{m+1} \ldots L_{2721}$   | rest | $\ldots \to 192 \to 48 \to 1$ |

So the network factorises as

$$f \;=\; \mathrm{Tail}\;\circ\;\underbrace{B \circ B \circ \cdots \circ B}_{N \text{ times}}\;\circ\;\mathrm{Head}.$$

### 2.4 Bit-mask fingerprint inside the body

The recurring widths $\{304, 312, 316, 318, 319\}$ satisfy

$$\{304, 312, 316, 318, 319\} \;=\; \{320 - 2^k\}_{k=0,1,2,3,4}.$$

A direct fingerprint of bit-level extraction inside a 32-bit word lane.
"Working width" $288 = 256 + 32$ — consistent with 32 bytes of state plus
32 control bits, or alternatively $8\cdot 32 + 32$ = "32-bit word lanes plus
status bits".

### 2.5 Weight tying

For each in-block position $p \in \{0, \dots, 41\}$, with iterations
$t = 1, \dots, N$, define the per-position dispersion

$$
\rho_p \;=\; \frac{1}{\|\bar W^{(p)}\|_F} \cdot \frac{1}{N} \sum_{t=1}^{N}
\big\|\, W^{(p)}_t - \bar W^{(p)}\,\big\|_F,
\qquad \bar W^{(p)} = \frac{1}{N}\sum_t W^{(p)}_t.
$$

Empirically, $\rho_p = 0$ for 40 of 42 positions — those matrices are
**bit-identical** across iterations. Only two positions vary: one carries
per-iteration deltas, the other is a stitching layer with a shape change.

**The body is a true weight-tied unrolled loop.** Modulo per-iteration
constants, all $N$ iterations apply the same function $B$.

### 2.6 The head's first layer

SVD of $W_1 \in \mathbb{R}^{224 \times 55}$ gives $W_1 = U \Sigma V^{\!\top}$
with

$$\sigma_1 = \sigma_2 = \cdots = \sigma_{55} = \sqrt{3}, \qquad V = I_{55}.$$

Equivalently, $W_1 \in \{0, 1\}^{224 \times 55}$ with **exactly 3 ones per
input column**, and the 55 input dims are mutually orthogonal. The bias
$b_1 \in \mathbb{R}^{224}$ contains the integers $1, 2, \dots, 55$ (one entry
each) plus 56 entries of $-1$ and 113 entries of $0$ — i.e. the head is
laying the input onto a positional strip indexed by byte position.

### 2.7 The output layer (decoded analytically)

The final `Linear(48 \to 1)` has explicit weights

$$W_{2721} = \big(\underbrace{+1,\dots,+1}_{16},\;\underbrace{-2,\dots,-2}_{16},\;\underbrace{+1,\dots,+1}_{16}\big), \qquad b_{2721} = -15.$$

Letting $y = (a, b, c)$ partition the 48 inputs into three 16-tuples,

$$f(x) \;=\; \mathrm{ReLU}\!\Big(\textstyle\sum_{i=1}^{16}(a_i - 2 b_i + c_i)\; -\; 15\Big).$$

The motif

$$\delta(z) := \mathrm{ReLU}(z+1) + \mathrm{ReLU}(z-1) - 2\,\mathrm{ReLU}(z)
\;=\; \mathbb{1}[z = 0] \quad \text{for } z \in \mathbb{Z}$$

is the **integer Kronecker delta** — three ReLUs that detect $z = 0$. So
each triple $(a_i, b_i, c_i)$ is exactly a Kronecker delta on some integer
expression, and the final layer reduces to

$$f(x) \;=\; \bigwedge_{i=1}^{16} \mathbb{1}\big[\text{some integer predicate}_i(x)\big].$$

The network's job is therefore to compute 16 integers from the 55-byte input
and check that all 16 equal fixed targets. The penultimate `Linear(192 \to 48)`
has weight magnitudes that are powers of 2 with $\|W_{2720}\|_F \approx 1774$
— a binary-to-integer decoder for an internal $128$-bit register.

---

## 3. Approach

Hand-compiled networks satisfy strong structural properties:

- **Sparse + small alphabet** (§2.2) → enumerable weight values.
- **Discrete activations.** Integers, bytes, 0/1 bits, or 32-bit word lanes
  (§2.4).
- **Recognisable algebraic identities.** Examples we look for generically:

  $$\mathbb{1}[x = 0] \;=\; \mathrm{ReLU}(x+1) + \mathrm{ReLU}(x-1) - 2\,\mathrm{ReLU}(x) \qquad (\text{Kronecker } \delta)$$

  $$\mathrm{ReLU}(x - k) \;=\; \max(0, x - k) \qquad (\text{thermometer step})$$

  $$a \oplus b \;=\; a + b - 2\,\mathrm{ReLU}(a + b - 1) \qquad \text{for } a, b \in \{0, 1\}$$

  $$\mathrm{ROTL}_s(x) \;=\; ((x \ll s) \;|\; (x \gg (32-s))) \bmod 2^{32}$$

Once we propagate the input domain forward through the network with an
abstract lattice (Stage 3 below), every neuron resolves to a finite-domain
value with a recognisable algebraic role. The decompiler walks the network
and folds those structures into a higher-level Python program.

### 3.1 Pipeline

```
PyTorch Sequential
       │
       ▼  Stage 1  SSA + sparse weighted DAG       neurodecomp.sparse_graph
       │
       ▼  Stage 2  Affine canonicalisation          neurodecomp.canonical
       │
       ▼  Stage 3  Abstract interpretation          neurodecomp.{domains,interp}
       │            Lattice:                        
       │              Const(c) | Dead | Bool | Bit(k)
       │              | SmallSet | Byte | UInt(b)
       │              | Interval | Affine | Top
       │
       ▼  Stage 4  Block discovery + loop folding   neurodecomp.block_finder
       │            (periodicity, weight tying)
       │
       ▼  Stage 5  Motif library                    neurodecomp.motifs
       │            δ, XOR, AND/OR/NOT, bit-extract,
       │            bit-pack, mod-2^n add, rotL/rotR
       │
       ▼  Stage 6  Register inference               neurodecomp.registers
       │            bits → bytes → 32-bit words
       │
       ▼  Stage 7  Codegen                          neurodecomp.emit
       │
       ▼  Stage 8  Validation                       neurodecomp.validate
                    random equivalence,
                    intermediate trace comparison,
                    Z3 proofs for motifs on
                    bounded domains.
```

### 3.2 Design rules

1. **Never use finite value sets for wide neurons.** Enumeration is only
   allowed when $|V| \le 16$; abstract lattices otherwise.
2. **Always carry the input domain.** Reasoning over $\mathbb{R}^{55}$
   produces junk piecewise-linear extrapolation.
3. **Canonicalise before motif matching.** Equivalent circuits may be
   syntactically rearranged (sign flips, ordering, gcd, dead ReLUs).
4. **Use Z3 locally, never globally.** Z3 proves a motif matches its spec
   on a small domain; it does not solve the whole network.
5. **Block discovery comes early.** Decompile one representative iteration
   and fold the rest; the period detector reports per-iteration deltas.
6. **No algorithm-specific hard-coding in the core.** Anything that looks
   like *"this looks like algorithm X"* belongs in a separate validation
   pass, not in the decompiler.

---

## 4. Status

| Phase | Description | Status |
|---|---|---|
| 0 | Baseline scripts (replicating manual analysis) | ✅ `scripts/01-09` |
| 1 | NeuroDecomp scaffolding | ✅ `neurodecomp/` |
| MVP-1 | Sparse Graph + Domain Profiler | ✅ `scripts/run_profile.py` |
| MVP-2 | Tail Decompiler (AND-of-deltas → predicate list) | ⏳ |
| MVP-3 | Head Decompiler | ⏳ |
| MVP-4 | One-iteration Body Decompiler | ⏳ |
| MVP-5 | Full loop folding & codegen | ⏳ |
| MVP-6 | Z3-backed motif library | ⏳ |

### What MVP-1 currently reports on the benchmark target

(All findings purely structural — no algorithm-specific heuristics.)

```text
total modules   : 5442  (2721 Linear + 2721 ReLU, alternating)
signature       : f : R^55 -> R^1
weight alphabet : integers in [-1, 30]  (truncated; full alphabet is
                  ±powers of 2 plus small ints)
sparsity        : 99.6% of weights are exactly 0
tokenizer       : ord(str(x)[:55].ljust(55, '\x00'))   (recovered from pickle)
best period     : 42  (97.05% match)
head/body/tail  : [0, 18) / [18, 1320) / [1320, 2721)
iterations      : 31 in longest periodic run
weight tying    : 40 of 42 in-block positions bit-identical across iterations
                  (varying positions: 28 and 41)
output template : AND of 16 predicates of shape (a + c == 2b - 1) on integers
```

The output template tells us, without naming the algorithm: the model is a
boolean **AND of 16 integer-equality predicates**. The body looks like a
weight-tied recurrent computation with two per-iteration deltas. That's a lot
of information about the algorithm, recovered purely from weights.

---

## 5. Reproducing

```powershell
# Weights (.gitignored; 1.16 GB)
curl -L -o model_3_11.pt "https://huggingface.co/jane-street/2025-03-10/resolve/main/model_3_11.pt"

# Deps
pip install torch numpy cloudpickle z3-solver

# Tests
python -m unittest discover tests

# Profile (MVP-1)
python scripts/run_profile.py model_3_11.pt
```

## 6. Layout

```
.
├── README.md
├── docs/
│   ├── 00_problem.md              what the model presents to the world
│   ├── 02_neurodecomp_design.md   pipeline, IR stack, design rules
│   └── 99_spoilers.md             spoilers from the public solution (validation only)
├── neurodecomp/
│   ├── __init__.py
│   ├── model_loader.py            load .pt + recover the tokenizer
│   ├── sparse_graph.py            Stage 1
│   └── block_finder.py            Stage 4
├── scripts/                       reproducible exploratory analysis
│   ├── 01_arch_summary.py
│   ├── 02_deep_analysis.py
│   ├── 03_encoding_probe.py
│   ├── 05_extract_tokenizer.py
│   ├── 06_verify_tokenizer.py
│   ├── 07_find_nonzero.py
│   ├── 09_verify_md5.py           validation only (spoiler)
│   └── run_profile.py             MVP-1 entry point
├── tests/
│   ├── toy_circuits.py
│   ├── test_sparse_graph.py
│   └── test_block_finder.py
└── artifacts/                     analysis outputs (.json / .txt)
```

## 7. References

### NN verification & analysis
- α,β-CROWN — bound propagation: <https://github.com/Verified-Intelligence/alpha-beta-CROWN>
- Marabou — SMT-based NN verifier: <https://github.com/NeuralNetworkVerification/Marabou>
- DeepPoly / AI² — abstract interpretation for NNs.

### Program reasoning
- `egg` — equality saturation / e-graphs: <https://egraphs-good.github.io/>
- Z3 — SMT solver with bit-vector logic: <https://github.com/Z3Prover/z3>
- Souper — superoptimization for LLVM IR: <https://github.com/google/souper>
- STOKE — stochastic superoptimization.

### Background
- Sutton, *"The Bitter Lesson"*: <http://www.incompleteideas.net/IncIdeas/BitterLesson.html>
