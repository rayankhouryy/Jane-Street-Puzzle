# Jane Street Puzzle — March 2025: Decompiling `model.pt`

> *"Today I went on a hike and found a pile of tensors hidden underneath a neolithic burial mound!  
> I sent it over to the local neural plumber, and they managed to cobble together this. — model.pt"*

A reverse-engineering log of the [Jane Street March 2025 puzzle](https://huggingface.co/spaces/jane-street/puzzle).
The artifact is a 1.16 GB PyTorch `Sequential` model whose purpose is unknown.  Goal: figure out what
function it computes and submit a correct answer.

---

## 1. The artifact

- **Source:** [`jane-street/2025-03-10`](https://huggingface.co/jane-street/2025-03-10) on Hugging Face.
- **Files used:** `model_3_11.pt` (the original `model.pt` is pickled with a Python ≤ 3.10 cloudpickle and refuses to load on Python 3.11; the maintainers shipped a re-packed `model_3_11.pt` for newer interpreters).
- **Module type:** `torch.nn.modules.container.Sequential` (5442 children: 2721 `Linear` + 2721 `ReLU`, strictly alternating).
- **Signature:** $f : \mathbb{R}^{55} \rightarrow \mathbb{R}_{\ge 0}$ (because the final activation is a `ReLU`).
- **Example given in the puzzle:** `f(\text{"vegetable dog"}) = 0`.

Formally, the network is

$$
f(x) \;=\; (L_{2721} \circ \sigma \circ L_{2720} \circ \sigma \circ \cdots \circ L_{2} \circ \sigma \circ L_{1})(x),
$$

where each $L_i$ is an affine map $L_i(h) = W_i h + b_i$ and $\sigma$ is the elementwise ReLU
$\sigma(z) = \max(0, z)$.

---

## 2. Methodology

We borrow techniques from three traditions:

1. **Architectural-motif analysis** (ResNet/Inception). Identify repeating sub-structures so we can collapse a $K$-layer net to "1 motif × $N$ iterations".
2. **Mechanistic interpretability** (Olah, Anthropic).
   - For any input $x$, a ReLU MLP is a piecewise-linear function; locally it is a single affine map $J(x)\, x + c(x)$, where $J(x)$ is the input-output Jacobian on the active linear region.
   - Singular value decomposition of individual layer matrices reveals the *natural basis* used by that layer:
     $W = U\Sigma V^{\!\top}$, with right singular vectors $V$ telling us which input directions matter.
3. **Decompilation / circuit reading** (program synthesis tradition). If weights take values from a tiny set such as $\{0,\pm 1, \pm 2, \dots\}$ or powers of 2, the layer is almost certainly a hand-written boolean/arithmetic circuit, not a trained net. Read it like an opcode.

---

## 3. Architecture decomposition

Let $d_i$ denote the output width of $L_i$ and $d_0 = 55$ the input width.

### 3.1 Macro structure
A robust period detector (best $p$ minimising $\sum_i \mathbb{1}[d_i \ne d_{i+p}]$) finds

$$
p^{\star} = 42, \qquad \text{match ratio } = 0.97.
$$

Augmenting period detection with "block-start" markers (Linear with $(d_{\text{in}}, d_{\text{out}}) \in \{(336,296), (368,328), (336,328)\}$) gives an exact decomposition:

| Region | Linear indices | Size | Width pattern |
|---|---|---|---|
| **Head**      | $L_1 \ldots L_{18}$            | 18 layers   | $55 \to 224 \to 232 \to 64 \to \ldots \to 336$ |
| **Body**      | $L_{19} \ldots L_{2664}$       | 2646 layers | 42-layer block repeated **63×** |
| **Tail**      | $L_{2665} \ldots L_{2721}$     | 57 layers   | $\ldots \to 192 \to 48 \to 1$ |

So the network is

$$
f \;=\; \mathrm{Tail} \;\circ\; \underbrace{B \circ B \circ \cdots \circ B}_{63 \text{ times}} \;\circ\; \mathrm{Head}.
$$

### 3.2 The 42-layer body $B$
One period of width transitions:

$$
\underbrace{336 \to 296 \to 340 \to \ldots \to 352 \to 288}_{\text{WIDE (15 layers)}} \;\to\;
\underbrace{288 \to 256 \to 319 \to 288 \to 318 \to \ldots \to 256}_{\text{INNER}_1\text{ (13 layers)}} \;\to\;
\underbrace{288 \to 256 \to 319 \to \ldots \to 336}_{\text{INNER}_2\text{ (14 layers)}}.
$$

### 3.3 Telling widths
The recurring widths $\{304, 312, 316, 318, 319\}$ satisfy

$$
\{304, 312, 316, 318, 319\} \;=\; \{320 - 2^k\}_{k=0,1,2,3,4}.
$$

This **bit-mask pattern** is a strong signature of bit-level computation: the body reads/writes individual bits of a 5-bit register five times in sequence — once per `INNER` half — exactly as if the inner loops were unrolling a 5-bit symbol encode/decode.

### 3.4 Plausible state decomposition
The "working width" inside the body is $288 = 256 + 32$. Two natural readings:

- **Turing-machine flavour:** 256-cell binary tape + 32-position one-hot head pointer.
- **Register-file flavour:** $32 \times 8 = 256$ data bits plus 32 control bits.

Combined with the bit-mask pattern, **the most parsimonious hypothesis** is:
> The body $B$ is one iteration of a *virtual machine* that consumes/produces a 5-bit symbol per step, mutating a 288-dim state, executed for 63 iterations.

---

## 4. Weight-tying: the loop is real

For each position $p \in \{0, 1, \dots, 41\}$ in the body, let

$$
W^{(p)}_t \;=\; \text{weight matrix of the } p\text{-th Linear in iteration } t, \quad t=1,\dots,63,
$$

and define the relative dispersion

$$
\rho_p \;=\; \frac{1}{\|\bar W^{(p)}\|_F}\;\cdot\;\frac{1}{63}\sum_{t=1}^{63} \big\| W^{(p)}_t - \bar W^{(p)} \big\|_F,
\quad \bar W^{(p)} = \frac{1}{63} \sum_t W^{(p)}_t.
$$

| Position $p$ | $\rho_p$ |
|---|---|
| $0, 1, 41$ | shapes mismatch — these are *boundary stitching* layers between consecutive iterations |
| $2 \le p \le 40$, $p \ne 28$ | $\rho_p = 0$ (bit-exact) |
| $p = 28$ | $\rho_p \approx 1.19$ with per-element std $0.015$ — at most a few iterations differ from the rest |

**Conclusion.** 38/42 of the body's weight matrices are *literally identical* across all 63 iterations. The network is an explicit unrolling of a recurrent program:

$$
B(h) \;=\; B(h;\; W, b) \qquad \text{(same parameters every step)}.
$$

This is essentially impossible to obtain by gradient descent; the only sensible explanation is that the model was **hand-compiled from an algorithm**.

---

## 5. The head $\mathrm{Head}$ at $L_1$

### 5.1 SVD

For $W_1 \in \mathbb{R}^{224 \times 55}$ we compute $W_1 = U\Sigma V^{\!\top}$ and find

$$
\sigma_1 = \sigma_2 = \cdots = \sigma_{55} \;=\; \sqrt{3}, \qquad V \;=\; I_{55}.
$$

Every singular value is identical and equal to $\sqrt 3$, and the right singular vectors are the standard basis of $\mathbb{R}^{55}$. Consequently:

- The 55 input coordinates are **mutually orthogonal and equally weighted** as far as the first layer is concerned.
- Each column $W_{1,:,j}$ has $\|W_{1,:,j}\|_2 = \sqrt 3$.

### 5.2 Direct inspection of $W_1$ and $b_1$
- **$W_1 \in \{0, 1\}^{224 \times 55}$**: only $165$ entries are $1$, the other $12{,}155$ are $0$.
- **Each column has exactly 3 ones.** (Hence $\|W_{1,:,j}\|^2 = 3$ → matches the SVD.)
- All 55 columns are **distinct** (no two input dimensions share the same fingerprint).
- Bias $b_1 \in \mathbb{R}^{224}$:

| Value | Count |
|---|---|
| $-1$ | 56 |
| $0$  | 113 |
| $1, 2, 3, \dots, 55$ | one entry each |

> The bias contains exactly the integers $\{1, 2, \dots, 55\}$ — one per input dimension. This is a hand-rolled enumeration: the head is laying out the input on a "1-indexed strip" of 55 positions, presumably so the body can iterate over it.

### 5.3 Forward probes
- $f(\mathbf 0) = 0$, $f(\mathbf 1) = 0$.
- $f(e_j) = 0$ for every standard basis vector $e_j$, $j = 1, \dots, 55$.
- $f(e_j + e_k) = 0$ for 50 random pairs.

So the function is **strongly zero**: the final ReLU absorbs nearly every "obvious" input. Finding inputs with $f(x) > 0$ requires either domain knowledge of the encoding or gradient-based search.

---

## 6. The tail (last 57 layers)

### 6.1 Final Linear $L_{2721}$
$L_{2721}: \mathbb{R}^{48} \to \mathbb{R}$, with explicit weights

$$
W_{2721} \;=\; \big(\underbrace{+1,\dots,+1}_{16},\; \underbrace{-2,\dots,-2}_{16},\; \underbrace{+1,\dots,+1}_{16}\big),
\qquad b_{2721} = -15.
$$

If $y = (y^{(1)}, y^{(2)}, y^{(3)}) \in \mathbb{R}^{16+16+16}$ is the input, then

$$
f(x) \;=\; \mathrm{ReLU}\!\Big( \mathbf 1^{\!\top} y^{(1)} \;-\; 2\,\mathbf 1^{\!\top} y^{(2)} \;+\; \mathbf 1^{\!\top} y^{(3)} \;-\; 15 \Big) \;=\; \mathrm{ReLU}\!\big( S_1 + S_3 - 2 S_2 - 15 \big),
$$

where $S_k = \sum_i y^{(k)}_i$.

### 6.2 Penultimate Linear $L_{2720}$
$L_{2720}: \mathbb{R}^{192} \to \mathbb{R}^{48}$ has

$$
\|W_{2720}\|_F \approx 1774, \quad \min W = -256, \quad \max W = +128.
$$

The weight magnitudes are powers of two, hallmark of a **binary-to-integer decoder**. The 192 inputs are best read as a stack of binary signals, and the 48 outputs are 48 weighted integer accumulators.

### 6.3 Tail outline
The first ~14 layers of the tail look like another `WIDE` block (256 → 287 → 319 → 318 → 316 → …), suggesting the tail finishes one last computation step before two "mini-decoder" passes (widths 192 ↔ 160/208/216/220/222/223 and then 320 → 382 → 446 → … → 48 → 1) that compress the state down to the scalar output.

### 6.4 Output bounds
If the 48 features are all in $[0, 1]$ (a plausible assumption given they emerge through ReLUs), the output ranges over

$$
f(x) \in \big[\, \max(0,\; -2 \cdot 16 - 15),\;\; \max(0,\; 16 + 16 - 15)\,\big] = [\,0, \,17\,].
$$

If the features were thermometer-encoded integers in $\{0,\dots,16\}$, the output would be $\max\!\big(0,\, A + C - 2B - 15\big)$ for three small integers $A, B, C$.

---

## 7. Working hypothesis

Putting it all together:

1. The input is a **55-symbol bag** (the puzzle's "vegetable dog" picks 2 of the 55 entries).
2. The head fans this out into a 224-wide canvas with explicit positional biases $1, 2, \dots, 55$ — preparing a "tape" for the body.
3. The body $B$ is a **weight-tied unrolled program**, executed 63 times. The bit-mask widths $\{320-2^k\}$ inside $B$ indicate 5-bit symbol processing.
4. The tail collapses the final 288-dim state into 3 small integers $A, B, C$ and returns

$$
f(x) \;=\; \mathrm{ReLU}\!\big( A + C - 2B - 15 \big).
$$

Reading this as a puzzle output, $f(x)$ is most naturally a small non-negative integer score (probably $\in \{0, 1, \dots, 17\}$). The submission almost certainly asks for an input $x$ — i.e. a choice of words from the 55-symbol vocabulary — that maximises $f$.

---

## 8. Roadmap (open questions)

- [x] Locate head/body/tail; find period $p^\star = 42$.
- [x] Verify weight tying across the 63 iterations.
- [x] SVD the head; recover the standard-basis encoding.
- [x] Decode the final Linear $L_{2721}$.
- [ ] **Identify the 55-word vocabulary.** Bias $b_1$ enumerates $1..55$ — what assignment of words to indices fits the puzzle hints?
- [ ] **Find inputs with $f(x) > 0$.** Strategy: gradient ascent on a continuous relaxation $x \in [0,1]^{55}$, then round; alternatively beam-search over small subsets.
- [ ] Once we have a non-zero example, run intermediate-activation probes to map the body's state semantics.
- [ ] Submit.

---

## 9. Repository layout

```
.
├── README.md                 ← this document
├── model_3_11.pt             ← Python-3.11 repacked weights (1.16 GB)
├── model.pt                  ← original weights (Python ≤ 3.10 cloudpickle)
├── scripts/
│   ├── 01_arch_summary.py    ← head/body/tail split + period detector
│   ├── 02_deep_analysis.py   ← tail inspection, weight-tying, head SVD
│   └── 03_encoding_probe.py  ← layer-0 columns/bias, forward probes
└── artifacts/
    ├── arch_summary.txt
    ├── arch_summary.json
    ├── widths.json
    ├── block_starts.json
    ├── tail_layers.txt
    ├── weight_tying.json
    ├── head_svd.txt
    ├── head_layer0_columns.json
    └── head_layer0_bias_unique.json
```

## 10. Reproducing

```powershell
# 1. Download
curl -L -o model_3_11.pt "https://huggingface.co/jane-street/2025-03-10/resolve/main/model_3_11.pt"

# 2. Dependencies (Python 3.11)
pip install torch numpy cloudpickle

# 3. Run the analysis pipeline
python scripts/01_arch_summary.py
python scripts/02_deep_analysis.py
python scripts/03_encoding_probe.py
```
