# NeuroDecomp

> **Automatically recovering symbolic programs from hand-compiled ReLU
> networks.**

Some neural networks aren't really neural networks. When their weights live in
$\{0, \pm 1, \pm 2, \pm 4, \dots\}$ and 99 % of them are zero, they were
**hand-compiled**: someone wrote an algorithm by hand and arranged a
`Linear → ReLU → Linear → ReLU → …` stack to evaluate it. NeuroDecomp is a
tool that, given such a network, recovers the underlying algorithm as
readable Python — without being told what algorithm to look for.

The validation target for this repo is the [Jane Street March 2025 puzzle][js-puzzle]:
a 1.16 GB PyTorch model that takes a string and returns 0 or 1. The puzzle
ships only the weights. **Our job is to make NeuroDecomp tell us what the
model computes, with no human in the loop.**

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

## 2. Methodology

We borrow techniques from three traditions:

1. **Architectural-motif analysis** (ResNet / Inception). Identify repeating
   sub-structures so we can collapse a $K$-layer net to "1 motif × $N$
   iterations".
2. **Mechanistic interpretability** (Olah, Anthropic).
   - For any input $x$, a ReLU MLP is a piecewise-linear function; locally
     it is a single affine map $J(x)\,x + c(x)$ where $J(x)$ is the
     input–output Jacobian on the active linear region.
   - SVD of an individual layer matrix reveals the *natural basis* used by
     that layer: $W = U\Sigma V^{\!\top}$, with right singular vectors $V$
     telling us which input directions matter.
3. **Decompilation / circuit reading** (program synthesis tradition). If
   weights take values from a tiny set such as $\{0, \pm 1, \pm 2, \dots\}$
   or powers of 2, the layer is almost certainly a hand-written
   boolean/arithmetic circuit, not a trained net. Read it like an opcode.

---

## 3. The benchmark target

### 3.1 Container and signature

- `torch.nn.modules.container.Sequential`.
- 5442 children, strictly alternating: **2721 `Linear`** + **2721 `ReLU`**.
- First `Linear`: `in_features = 55`. Last `Linear`: `out_features = 1`.
- Formally, the network is

  $$
  f(x) \;=\; (L_{2721} \circ \sigma \circ L_{2720} \circ \sigma \circ \cdots \circ L_2 \circ \sigma \circ L_1)(x),
  $$

  where each $L_i$ is an affine map $L_i(h) = W_i h + b_i$ and $\sigma$ is
  the elementwise ReLU $\sigma(z) = \max(0, z)$.
- Range is non-negative because the final activation is a `ReLU`.

### 3.2 Hidden tokenizer

The Space's `app.py` calls `model("some string")` directly. The model
instance carries a custom callable attached via cloudpickle. Recovering it
from the pickle (`scripts/05_extract_tokenizer.py`) gives:

```python
_call_impl = lambda x: model.forward(
    torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00'))))
)
```

So the input is the ASCII byte values of the first 55 characters of the
string, null-padded:

$$
x_i \;=\; \mathrm{ord}(s_i), \qquad i = 0, \dots, 54.
$$

The effective type of the model is therefore

$$
f : \Sigma^{\le 55} \;\longrightarrow\; \{0, 1\}, \qquad \Sigma = \{0, \dots, 255\}.
$$

The output behaves like a 0/1 indicator on random ASCII inputs.

### 3.3 Sparsity and weight alphabet

Across all 2721 `Linear` layers:

| | |
|---|---|
| total weights | 288,122,268 |
| zero weights  | 287,047,559  (**99.6 %**) |
| weight alphabet (truncated) | $\{-1, 0, 1, 2, \dots, 30\}$ — full alphabet is small integers plus $\pm$ powers of 2 |

$\Rightarrow$ **the network was not trained; it is a hand-compiled circuit.**

---

## 4. Architecture decomposition

Let $d_i$ denote the output width of $L_i$ and $d_0 = 55$ the input width.

### 4.1 Macro structure

A period detector minimising $\sum_i \mathbb{1}[d_i \ne d_{i+p}]$ over
$p \in [2, 200]$ finds

$$
p^{\star} = 42, \qquad \text{match ratio } = 0.97.
$$

Augmenting period detection with "block-start" markers (a `Linear` whose
$(d_\text{in}, d_\text{out})$ pair appears repeatedly and reduces width)
gives the head/body/tail decomposition:

| Region | Linear indices | Size | Width pattern |
|---|---|---|---|
| **Head** | $L_1 \ldots L_{18}$         | 18 layers   | $55 \to 224 \to 232 \to 64 \to \ldots \to 336$ |
| **Body** | $L_{19} \ldots L_{2664}$    | 2646 layers | 42-layer block repeated **63×** |
| **Tail** | $L_{2665} \ldots L_{2721}$  | 57 layers   | $\ldots \to 192 \to 48 \to 1$ |

So the network factorises as

$$
f \;=\; \mathrm{Tail} \;\circ\; \underbrace{B \circ B \circ \cdots \circ B}_{63 \text{ times}} \;\circ\; \mathrm{Head}.
$$

### 4.2 The 42-layer body $B$

One period of width transitions decomposes into three sub-phases:

$$
\underbrace{336 \to 296 \to 340 \to \ldots \to 352 \to 288}_{\text{WIDE (15 layers)}} \;\to\;
\underbrace{288 \to 256 \to 319 \to 288 \to 318 \to \ldots \to 256}_{\text{INNER}_1\text{ (13 layers)}} \;\to\;
\underbrace{288 \to 256 \to 319 \to \ldots \to 336}_{\text{INNER}_2\text{ (14 layers)}}.
$$

The WIDE phase mixes the working state through a wider scratch space; the
two INNER phases share an almost identical width pattern, suggesting a
repeated sub-operation per outer iteration.

### 4.3 Bit-mask fingerprint

The recurring widths inside the INNER phases satisfy

$$
\{304, 312, 316, 318, 319\} \;=\; \{320 - 2^k\}_{k=0,1,2,3,4}.
$$

This is the canonical signature of **bit-level extraction inside a 32-bit
word**: width $320$ holds a 32-bit register's lanes, and dropping $2^k$ from
that width corresponds to masking out bit $k$. The body reads / writes
individual bits of a 32-bit word, five bits in a row inside each INNER
phase.

### 4.4 State decomposition candidates

The "working width" inside the body is $288 = 256 + 32$. The structurally
plausible readings are:

- **Turing-machine flavour**: a 256-cell binary tape plus a 32-position
  one-hot head pointer.
- **Register-file flavour**: $32 \times 8 = 256$ data bits plus 32 control
  bits — i.e., 8 bytes of state plus a 32-bit status word.
- **Word-lane flavour**: $288 = 9 \times 32$, i.e. nine 32-bit lanes —
  consistent with the bit-mask fingerprint in §4.3 (32-bit word
  manipulation).

Combined with the period-42 unrolled loop, the most parsimonious
description so far is: *a weight-tied iterative compression function that
mutates a 288-bit state and is run 63 times*.

### 4.5 Weight tying: the loop is real

For each in-block position $p \in \{0, 1, \dots, 41\}$, with iterations
$t = 1, \dots, 63$, let

$$
W^{(p)}_t \;=\; \text{weight matrix of the } p\text{-th Linear in iteration } t,
$$

and define the relative dispersion across iterations

$$
\rho_p \;=\; \frac{1}{\|\bar W^{(p)}\|_F}\;\cdot\;\frac{1}{63}\sum_{t=1}^{63}
\big\| W^{(p)}_t - \bar W^{(p)} \big\|_F,
\qquad \bar W^{(p)} = \frac{1}{63} \sum_t W^{(p)}_t.
$$

| Position $p$ | $\rho_p$ | Interpretation |
|---|---|---|
| $0, 1, 41$ | shape mismatch | boundary stitching between consecutive iterations |
| $2 \le p \le 40$, $p \ne 28$ | $\rho_p = 0$ (bit-exact) | identical across all 63 iterations |
| $p = 28$ | $\rho_p \approx 1.19$ with per-element std $0.015$ | a small number of iterations differ from the rest — almost certainly the slot that carries a per-iteration constant |

**Conclusion.** 38/42 of the body's weight matrices are *literally
identical* across all 63 iterations. The network is an explicit unrolling
of a recurrent program

$$
B(h)\; = \;B(h;\, W, b) \qquad \text{(same parameters every step)}.
$$

This is not something gradient descent produces; the only sensible
explanation is **hand-compilation**.

---

## 5. The head ($L_1$)

### 5.1 SVD

For $W_1 \in \mathbb{R}^{224 \times 55}$ we compute $W_1 = U \Sigma V^{\!\top}$
and find

$$
\sigma_1 = \sigma_2 = \cdots = \sigma_{55} \;=\; \sqrt{3},
\qquad V \;=\; I_{55}.
$$

Every singular value is identical and equal to $\sqrt 3$, and the right
singular vectors are the standard basis of $\mathbb{R}^{55}$. Consequences:

- The 55 input coordinates are **mutually orthogonal and equally weighted**
  as far as $L_1$ is concerned.
- Each column $W_{1,\,:,\,j}$ has $\|W_{1,\,:,\,j}\|_2 = \sqrt 3$.

### 5.2 Direct inspection of $W_1$ and $b_1$

- $W_1 \in \{0, 1\}^{224 \times 55}$. Of the $224 \times 55 = 12{,}320$
  entries, exactly **165 are 1** and **12,155 are 0**.
- **Each column has exactly 3 ones.** (Confirming $\|W_{1,\,:,\,j}\|^2 = 3$
  ↔ matching the SVD.)
- All 55 columns are distinct — no two input bytes share the same
  fingerprint in the head.
- Bias $b_1 \in \mathbb{R}^{224}$ has only 57 distinct values:

  | Value | Count |
  |---|---|
  | $-1$ | 56 |
  | $0$  | 113 |
  | $1, 2, 3, \dots, 55$ | one entry each |

> The bias contains exactly the integers $\{1, 2, \dots, 55\}$ — one per
> input byte position. This is a hand-rolled enumeration: the head is
> laying out the input on a "1-indexed strip" of 55 byte positions,
> presumably so the body can iterate over byte indices later (e.g. to
> count the input length, or pick a particular byte position by
> comparison).

### 5.3 Forward probes

- $f(\mathbf 0) = 0$ and $f(\mathbf 1) = 0$.
- $f(e_j) = 0$ for every standard basis vector $e_j$, $j = 1, \dots, 55$.
- $f(e_j + e_k) = 0$ for 50 random pairs $(j, k)$.
- 50,000 random ASCII strings of varying length: **all** give $f = 0$.
- Gradient ascent on a continuous relaxation $x \in [0, 127]^{55}$ from
  multiple random restarts converges to the all-zero pre-ReLU output and
  fails to escape.

So $f$ is **strongly zero**: the final ReLU absorbs almost every input.
Finding inputs with $f(x) > 0$ requires either symbolic reasoning (the
decompiler's job) or knowing the algorithm in advance.

---

## 6. The output layer and tail (last 57 layers)

### 6.1 Final Linear $L_{2721}$

$L_{2721} : \mathbb{R}^{48} \to \mathbb{R}$ has the explicit weights

$$
W_{2721} \;=\; \big(\underbrace{+1,\dots,+1}_{16},\;\underbrace{-2,\dots,-2}_{16},\;\underbrace{+1,\dots,+1}_{16}\big),
\qquad b_{2721} = -15.
$$

Partitioning the 48-dim input as $y = (a, b, c)$ with each component in
$\mathbb{R}^{16}$,

$$
f(x) \;=\; \mathrm{ReLU}\!\Big(\mathbf{1}^{\!\top} a - 2\,\mathbf{1}^{\!\top} b + \mathbf{1}^{\!\top} c - 15\Big)
\;=\; \mathrm{ReLU}\!\big(S_1 - 2 S_2 + S_3 - 15\big),
$$

where $S_k = \sum_i y_i^{(k)}$.

### 6.2 The Kronecker-delta identity

For any integer $z \in \mathbb{Z}$, the three-ReLU motif

$$
\delta(z) \;:=\; \mathrm{ReLU}(z + 1) + \mathrm{ReLU}(z - 1) - 2\,\mathrm{ReLU}(z)
\;=\; \mathbb{1}[z = 0]
$$

is exactly the integer-supported Kronecker delta (proof: case analysis on
$z \in \{<-1, -1, 0, 1, >1\}$). Each triple $(a_i, b_i, c_i)$ in §6.1 is
this motif applied to some integer expression — i.e. **each triple fires
exactly when one integer predicate holds**.

Therefore

$$
f(x) \;=\; \mathrm{ReLU}\!\Big(\textstyle\sum_{i=1}^{16}\delta_i(x) - 15\Big)
\;=\; \bigwedge_{i=1}^{16} \delta_i(x)
\;=\; \bigwedge_{i=1}^{16} \mathbb{1}[\text{integer predicate}_i \text{ holds}].
$$

The model's output is the **AND of 16 integer-equality predicates**. What
each predicate compares we will discover by walking backward through the
tail.

### 6.3 Penultimate Linear $L_{2720}$

$L_{2720} : \mathbb{R}^{192} \to \mathbb{R}^{48}$ has

$$
\|W_{2720}\|_F \;\approx\; 1774, \qquad
\min W = -256, \qquad
\max W = +128.
$$

The weight magnitudes are exact powers of two: $\{0, \pm 1, \pm 2, \pm 4,
\pm 8, \pm 16, \pm 32, \pm 64, \pm 128, \pm 256\}$ — the **hallmark of a
binary-to-integer decoder**. The 192 inputs are best read as a stack of
binary signals (24 bytes' worth), and the 48 outputs are 48 weighted
integer accumulators of those bits.

In particular: $\log_2(256) = 8$ matches the byte-width of an integer
register. So $L_{2720}$ is decoding bit lanes into integer-valued bytes,
and $L_{2721}$ is comparing **16 bytes** against fixed targets.

### 6.4 Tail outline

Reading widths from $L_{2665}$ to $L_{2721}$, the tail breaks into
recognisable sub-blocks:

```
256 → 287 → 319 → 318 → 316 → 312 → 304 → 256       (one more WIDE-style
                                                      compression of the
                                                      288-dim body state
                                                      down to 256)
256 → 192 → 192 → 160 → 223 → 192 → 222 → 192 → 220
       → 192 → 216 → 192 → 208 → 192 → 160           (mini-decoder pass A:
                                                      192 ↔ 160/208/.../223)
160 → 192 → 192 → 160 → 223 → 192 → ... → 160        (mini-decoder pass B,
                                                      same shape)
160 → 320 → 320 → 382 → 446 → 444 → ...              (re-expansion to a
       → 416 → 416 → 320 → 192 → 48 → 1              48-input comparator)
```

Two near-identical "mini-decoder" passes immediately precede the comparator,
strongly suggesting a **two-stage operation**: (i) extract bits from the
working state into bytes; (ii) re-expand those bytes into the 48-input
shape needed by $L_{2721}$.

### 6.5 Output bounds

If the 48 features emerging from the tail's last `ReLU` lie in $[0, 1]$
(plausible since they come through a long ReLU chain),

$$
f(x) \in \big[\max(0,\; -2 \cdot 16 - 15),\;\; \max(0,\; 16 + 16 - 15)\big]
\;=\; [0,\; 17].
$$

If instead the 16 deltas are exact $\{0, 1\}$ booleans, then $S_1 + S_3 -
2 S_2 \in \{-32, -31, \dots, 16\}$ and the bias $-15$ pushes the ReLU's
threshold to "all 16 deltas equal 1":

$$
f(x) \;=\;
\begin{cases}
1 & \text{if all 16 predicates hold} \\
0 & \text{otherwise}.
\end{cases}
$$

This matches §6.2's algebraic derivation and the empirical $f \in \{0, 1\}$
observation.

---

## 7. Working summary (no algorithm identification)

From §3–§6, purely from weights:

1. The model is a hand-compiled, sparse, integer-domain ReLU circuit
   ($\rho_p = 0$ for 38/42 body positions; 99.6 % sparsity; weights ∈
   $\{0, \pm 1, \pm 2, \pm 4, \dots, \pm 256\}$).
2. The input is interpreted as **55 ASCII bytes** (`ord(s[:55].ljust(55, '\x00'))`).
3. The body $B$ is a **weight-tied iterative compression**: an unrolled
   loop of 63 iterations, each manipulating a 288-bit working state with
   bit-level operations on 32-bit word lanes (bit-mask fingerprint, §4.3).
4. The tail extracts **16 bytes** from the final 288-bit state and ANDs
   together 16 integer-equality predicates against fixed targets
   (Kronecker-delta proof, §6.2; powers-of-two decoder, §6.3).
5. The output is exactly

   $$
   f(s) \;=\; \mathbb{1}\big[\,\mathcal{H}(s) = t\,\big]
   $$

   where $\mathcal{H} : \Sigma^{\le 55} \to \{0, 1\}^{128}$ is some
   hand-compiled 128-bit compression function with 63 unrolled rounds and
   $t \in \{0, 1\}^{128}$ is a fixed 16-byte target encoded in
   $L_{2720}, L_{2721}$.

We have not yet identified $\mathcal{H}$ — that's the decompiler's
remaining job. Candidates that match the structural signature include
MD5-family hashes (64 rounds, 128-bit state, 32-bit lanes), Murmur-family
hashes, custom finalists, etc. Distinguishing them requires recovering
the per-iteration constants and the round function $B$.

---

## 8. Approach (the decompiler)

Hand-compiled networks satisfy strong structural properties:

- **Sparse + small alphabet** (§3.3) → enumerable weight values.
- **Discrete activations.** Integers, bytes, 0/1 bits, or 32-bit word
  lanes (§4.3).
- **Recognisable algebraic identities.** Examples we look for
  generically:

  $$
  \mathbb{1}[x = 0] \;=\; \mathrm{ReLU}(x+1) + \mathrm{ReLU}(x-1) - 2\,\mathrm{ReLU}(x) \qquad (\text{Kronecker }\delta)
  $$

  $$
  \mathbb{1}[x = c] \;=\; \delta(x - c)
  $$

  $$
  \mathrm{ReLU}(x - k) \;=\; \max(0,\, x - k) \qquad (\text{thermometer step})
  $$

  $$
  a \;\mathrm{AND}\; b \;=\; \mathrm{ReLU}(a + b - 1) \qquad \text{for } a, b \in \{0, 1\}
  $$

  $$
  a \;\mathrm{OR}\; b \;=\; \min(1,\, a + b) \;=\; a + b - (a \;\mathrm{AND}\; b)
  $$

  $$
  a \oplus b \;=\; a + b - 2(a \;\mathrm{AND}\; b)
  $$

  $$
  \mathrm{bit}_k(x) \;=\; \big(x \bmod 2^{k+1}\big) - \big(x \bmod 2^k\big)
  $$

  $$
  (a + b) \bmod 2^n \;=\; a + b - 2^n \cdot \mathrm{ReLU}(a + b - 2^n) / (\text{leading 1})
  $$

  $$
  \mathrm{ROTL}_s(x) \;=\; \big((x \ll s) \;\vert\; (x \gg (32 - s))\big) \bmod 2^{32}
  $$

Once we propagate the input domain forward through the network with an
abstract lattice, every neuron resolves to a finite-domain value with a
recognisable algebraic role. The decompiler walks the network and folds
those structures into a higher-level Python program.

### 8.1 Pipeline

```
PyTorch Sequential
       │
       ▼  Stage 1  SSA + sparse weighted DAG       neurodecomp.sparse_graph
       │
       ▼  Stage 2  Affine canonicalisation         neurodecomp.canonical
       │
       ▼  Stage 3  Abstract interpretation         neurodecomp.{domains,interp}
       │            Lattice:
       │              Const(c) | Dead | Bool | Bit(k)
       │              | SmallSet | Byte | UInt(b)
       │              | Interval | Affine | Top
       │
       ▼  Stage 4  Block discovery + loop folding  neurodecomp.block_finder
       │            (periodicity, weight tying)
       │
       ▼  Stage 5  Motif library                   neurodecomp.motifs
       │            δ, AND/OR/NOT/XOR, bit-extract,
       │            bit-pack, mod-2^n add, rotL/rotR
       │
       ▼  Stage 6  Register inference              neurodecomp.registers
       │            bits → bytes → 32-bit words
       │
       ▼  Stage 7  Codegen                         neurodecomp.emit
       │
       ▼  Stage 8  Validation                      neurodecomp.validate
                    random equivalence,
                    intermediate trace comparison,
                    Z3 proofs for motifs on bounded domains.
```

### 8.2 Design rules

1. **Never use finite value sets for wide neurons.** Enumeration is only
   allowed when $|V| \le 16$; abstract lattices otherwise.
2. **Always carry the input domain.** Reasoning over $\mathbb{R}^{55}$
   produces junk piecewise-linear extrapolation.
3. **Canonicalise before motif matching.** Equivalent circuits may be
   syntactically rearranged (sign flips, ordering, gcd, dead ReLUs).
4. **Use Z3 locally, never globally.** Z3 proves a motif matches its
   spec on a small domain; it does not solve the whole network.
5. **Block discovery comes early.** Decompile one representative
   iteration and fold the rest; the period detector reports
   per-iteration deltas.
6. **No algorithm-specific hard-coding in the core.** Anything that
   looks like *"this looks like algorithm X"* belongs in a separate
   validation pass, not in the decompiler.

---

## 9. Status

### Headline claim (precise)

NeuroDecomp has independently recovered a strong cryptographic fingerprint
from the model weights — purely from structural analysis, with no
algorithm-specific code:

* the ASCII/null-padded tokenizer (recovered from the pickled `_call_impl`),
* a **16-byte equality comparator** with the exact target
  `c7ef65233c40aa32c2b9ace37595fa7c`,
* four **32-bit initial constants** matching the MD5 IV
  ``A = 0x67452301, B = 0xefcdab89, C = 0x98badcfe, D = 0x10325476``,
* the precomputed subexpressions ``B & C = 0x88888888`` and ``~B`` stamped
  into the head's working state,
* the first round constant ``K[0] = 0xd76aa478``,
* a 3-register × 32-bit **rotate gadget** with the per-iteration shift
  schedule matching MD5's rotation table for the **63 recovered body
  iterations** (one round absorbed into head/tail/stitching),
* a **Z3-proven catalog of 6 ReLU motifs** (Kronecker delta, AND, OR, XOR,
  NOT, threshold), with **58,524 candidate boolean-AND row patterns**
  located across the network and **12,859 of them (22 %) confirmed as
  genuine booleans** by domain certification through abstract
  interpretation.

> **From candidates to confirmed.**
> The motif scanner identifies the row pattern `ReLU(a + b - 1)`, which
> is a *true* AND only when both inputs are certified booleans.
> MVP-7's domain certifier runs abstract interpretation over the network
> starting from the byte domain `[0, 255]^55`, and checks each candidate
> against its inputs' certified abstract range.  This drops the count
> from 58 k → 13 k, and yields a much more honest per-region breakdown:
>
> | region | candidates | confirmed | rate |
> |---|---|---|---|
> | head | 32 | 32 | 100 % |
> | body | 56,708 | 12,575 | 22.2 % |
> | tail | 1,784 | 252 | 14.1 % |
>
> The remaining 78 % candidates lose certification because our Affine
> abstract domain widens its bounds through saturating ReLUs.  Richer
> domains (e.g. bit-vector tracking of opaque ReLU symbols) are listed
> as future work in MVP-7.5.

Independent confirmation that the algorithm is MD5 comes from
[adirk0/Jane-Street-Puzzle](https://github.com/adirk0/Jane-Street-Puzzle);
we use that as a *validation oracle* only. MD5(`"bitter lesson"`) =
`c7ef65233c40aa32c2b9ace37595fa7c`, and `scripts/09_verify_md5.py`
confirms `model("bitter lesson") = 1`.

### What remains open

* per-round constant table $K_i$ for $i \in \{1, \dots, 63\}$ (only $K[0]$
  recovered so far);
* iteration 63 / 64 of the body (block finder reports 63; full MD5 has 64
  rounds, so one round is currently absorbed into head/tail/stitching);
* symbolic decode of one representative body block as Python (F/G/H/I
  round functions);
* loop folding & full codegen (`g(s) = model(s)` for every $s \in \Sigma^{\le 55}$);
* domain-certified motif recognition (turn the 58k *candidate* AND patterns
  into *confirmed* boolean ANDs by carrying input domains through the
  network).

### Pipeline progress table

| Phase | Description | Status |
|---|---|---|
| 0 | Baseline scripts (replicating manual analysis) | ✅ `scripts/01–09` |
| 1 | NeuroDecomp scaffolding | ✅ `neurodecomp/` |
| MVP-1 | Sparse Graph + Domain Profiler | ✅ `scripts/run_profile.py` |
| MVP-2 | Tail Decompiler (AND-of-deltas → predicate list) | ✅ `scripts/run_tail.py` |
| MVP-3 | Head Decompiler (abstract interpretation) | ✅ `scripts/run_head.py` |
| MVP-4 | One-iteration Body Decompiler (rotate gadget) | ✅ `scripts/run_body.py` |
| MVP-6 | Z3-backed motif library | ✅ `scripts/run_motifs.py` |
| MVP-7 | Domain-certified motif recognition | ✅ `scripts/run_certify.py` |
| MVP-5 | Full loop folding & codegen | ⏳ next |

### What MVP-1 currently reports

```text
total modules   : 5442  (2721 Linear + 2721 ReLU, alternating)
signature       : f : R^55 -> R^1
weight alphabet : integers in [-1, 30]  (truncated; full alphabet is
                  ±powers of 2 plus small ints)
sparsity        : 99.6% of weights are exactly 0
tokenizer       : ord(str(x)[:55].ljust(55, '\x00'))   (recovered from pickle)
best period     : 42  (97.05% match)
head/body/tail  : [0, 18) / [18, 2664) / [2664, 2721)
iterations      : 63 in the recovered body
weight tying    : 40 of 42 in-block positions bit-identical across iterations
                  (varying positions: 28 and 41)
output template : AND of 16 predicates of shape (a + c == 2b - 1) on integers
```

The output template tells us, without naming the algorithm: the model is a
boolean **AND of 16 integer-equality predicates**. The body is a
weight-tied recurrent computation with two per-iteration deltas. That's a
lot of information about the algorithm, recovered purely from weights.

---

## 10. Reproducing

```powershell
# Weights (.gitignored; 1.16 GB)
curl -L -o model_3_11.pt "https://huggingface.co/jane-street/2025-03-10/resolve/main/model_3_11.pt"

# Deps
pip install torch numpy cloudpickle z3-solver

# Tests
python -m unittest discover tests

# Profile (MVP-1)
python scripts/run_profile.py model_3_11.pt

# Tail decompilation (MVP-2) -- recovers the 16-byte target value
python scripts/run_tail.py model_3_11.pt

# Head decompilation (MVP-3) -- abstract interp, recovers initial constants
python scripts/run_head.py model_3_11.pt

# Body decompilation (MVP-4) -- recovers per-iteration rotate schedule
python scripts/run_body.py model_3_11.pt

# Motif library scan (MVP-6) -- Z3-verified motif catalog + pattern matcher
python scripts/run_motifs.py model_3_11.pt

# Domain-certified motif scan (MVP-7) -- confirms which candidates are
# genuine boolean gates by carrying input domains through the network
python scripts/run_certify.py model_3_11.pt
```

---

## 11. Roadmap (open questions)

- [x] Locate head / body / tail; find period $p^\star = 42$.
- [x] Verify weight tying across the 63 iterations.
- [x] SVD the head; recover the standard-basis byte encoding.
- [x] Decode the final Linear $L_{2721}$ analytically.
- [x] Identify the input tokenizer (ASCII bytes, length 55, null pad).
- [x] Decode the penultimate Linear $L_{2720}$ symbolically: which
  16 bytes does it extract, from where in the 288-bit state, with which
  binary-to-integer mapping?  → `scripts/run_tail.py` recovers the
  16-byte target ``c7ef65233c40aa32c2b9ace37595fa7c`` and the bit-address
  vector for each byte.
- [x] Decode the head: recover the initial constants stamped into the
  working state.  → `scripts/run_head.py` runs abstract interpretation
  over bytes ∈ [0,255]^55 and emits the head's 336 output neurons
  with role tags.  On the benchmark model it independently recovers
  the four 32-bit constants
  ``A = 0x67452301``, ``B = 0xefcdab89``, ``C = 0x98badcfe``,
  ``D = 0x10325476``, plus the precomputed ``B & C = 0x88888888`` and
  ``~B = 0x10325476`` and the first round constant ``0xd76aa478``,
  packed as little-endian bytes in the head's working state.
- [x] Decode one representative body block $B$.  Detect the rotate
  gadget at the (only) varying in-block position and recover the
  per-iteration shift schedule.  → `scripts/run_body.py` reports a
  3-register × 32-bit rotate gadget at position 28 of 42 with the shift
  schedule

  ```
  s[ 0..15]  = [7, 12, 17, 22] x 4
  s[16..31]  = [5,  9, 14, 20] x 4
  s[32..47]  = [4, 11, 16, 23] x 4
  s[48..62]  = [6, 10, 15, 21] x ~4
  ```

  matching MD5's rotation table for the **63 recovered body iterations**.
  (Iteration 63 — i.e. round 64 in MD5 — is currently absorbed into the
  head/tail/stitching layers; reuniting it is a follow-up.)
- [x] Build a Z3-verified motif catalog covering the
  Kronecker $\delta$, boolean AND/OR/XOR/NOT, and threshold gadgets;
  every motif is proved equivalent to its reference on import.  Scan
  the model for instances.  → `scripts/run_motifs.py` finds **58,524
  `bool_and`-pattern rows** (32 in the head, ~900 per body iteration,
  1,784 in the tail).
- [x] **Domain-certify the motif candidates.** Run abstract
  interpretation over the model from the byte input domain and check
  each candidate AND against its inputs' certified domain.  →
  `scripts/run_certify.py` confirms **12,859 of 58,524** candidates
  (22 %): all 32 in the head, 12,575 of 56,708 in the body (~200
  per iteration -- consistent with MD5's per-round F/G/H/I + modular
  adders), 252 of 1,784 in the tail.
- [ ] Recover the per-iteration constant table $K_i$ for
  $i \in \{1, \dots, 63\}$.  Only $K[0]$ has been recovered so far
  (from the head's IV stamping).  Candidates: position-41 shape-368
  bias deltas (16 iterations of 32-bit constants observed
  exploratorily), or symbolic propagation through the body.
- [ ] Decode the F/G/H/I non-linear round functions of one body
  iteration into Python via motif-driven simplification of the
  abstract-interpretation output.
- [ ] Glue all stages; emit a single Python program $g$ such that
  $g(s) = \mathrm{model}(s)$ for every $s \in \Sigma^{\le 55}$.
- [ ] Identify the algorithm: compare $g$ against known
  16-byte-output 128-bit compression functions.  (External validation
  oracle [`docs/99_spoilers.md`](docs/99_spoilers.md) confirms MD5;
  we use it only to check the decompiler's output.)
- [ ] Decode one representative body block $B$. Express it as a Python
  function $B(\text{state}, K_t, s_t) \to \text{state}$ where $K_t, s_t$
  are the per-iteration constants extracted from position 28 (and other
  varying slots).
- [ ] Fold the 63 unrolled iterations; recover the tables of
  per-iteration constants.
- [ ] Decode the head: recover the input padding scheme.
- [ ] Glue all stages; emit a single Python program $g$ such that
  $g(s) = \mathrm{model}(s)$ for every $s \in \Sigma^{\le 55}$.
- [ ] Identify the algorithm: compare $g$ against known
  16-byte-output 128-bit compression functions.

---

## 12. Layout

```
.
├── README.md                       ← this document
├── docs/
│   ├── 00_problem.md               what the model presents to the world
│   ├── 02_neurodecomp_design.md    pipeline, IR stack, design rules
│   └── 99_spoilers.md              public solution (validation only)
├── neurodecomp/
│   ├── __init__.py
│   ├── model_loader.py             load .pt + recover the tokenizer
│   ├── sparse_graph.py             Stage 1
│   └── block_finder.py             Stage 4
├── scripts/                        reproducible exploratory analysis
│   ├── 01_arch_summary.py
│   ├── 02_deep_analysis.py
│   ├── 03_encoding_probe.py
│   ├── 05_extract_tokenizer.py
│   ├── 06_verify_tokenizer.py
│   ├── 07_find_nonzero.py
│   ├── 09_verify_md5.py            validation only (spoiler)
│   └── run_profile.py              MVP-1 entry point
├── tests/
│   ├── toy_circuits.py
│   ├── test_sparse_graph.py
│   └── test_block_finder.py
└── artifacts/                      analysis outputs (.json / .txt)
```

---

## 13. References

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
- Adir's hand decompilation: <https://github.com/adirk0/Jane-Street-Puzzle>
