# NeuroDecomp — Executive Summary

> A one-page-ish synthesis of *what we've built*, *what's novel about it*,
> and *what prior work it relates to*. For the long version, see
> [`PROGRESS.md`](PROGRESS.md) and [`02_neurodecomp_design.md`](02_neurodecomp_design.md).
>
> *Last updated: 2026-05-19.*

---

## 1. What was done

### 1.1 The artifact

The Jane Street March 2025 puzzle ships a `torch.nn.Sequential` with **5,442
modules** (2,721 `Linear` + 2,721 `ReLU`, alternating), 1.16 GB on disk,
$f : \mathbb{R}^{55} \to \mathbb{R}^1$, 99.6 % sparse, weights drawn from
$\{0, \pm 1, \pm 2, \pm 4, \dots\}$. The "model" is a hand-compiled
algorithmic circuit, not a trained network.

We confirmed (publicly, via [adirk0][adir]) that it computes

$$
f(s) \;=\; \mathbb{1}\Big[\mathrm{MD5}(s) = \mathtt{c7ef65233c40aa32c2b9ace37595fa7c}\Big],
$$

with `"bitter lesson"` as a valid preimage. That oracle is kept strictly
isolated in [`docs/99_spoilers.md`](99_spoilers.md) and used *only* to
validate the decompiler's output — **never as input** to the pipeline.

The interesting work then started: **rebuild that fact from raw weights
with no algorithm-specific hints.**

### 1.2 Pipeline shipped (MVPs 0–7.5, 9a–d, 10 initial, 5 initial)

| Stage | Module | What it does |
|---|---|---|
| 1 | `sparse_graph.py` | SSA + sparse weighted DAG over the 5,442 modules |
| 2 | (implicit) | Affine canonicalisation |
| 3 | `domains.py` + `interp.py` | Forward abstract interpretation over an `Affine` lattice, extended in MVP-7.5 with **relational opaque tracking** so origin info survives saturating ReLUs |
| 4 | `block_finder.py` | Period detection ($p^\star = 42$, 0.97 match), head/body/tail split, weight tying confirmed (40/42 positions bit-identical across 63 iterations) |
| 5 | `motifs.py` | Z3-verified catalog of 6 ReLU motifs: delta, AND, OR, XOR, NOT, threshold |
| 6 | `registers.py` (MVP-9d) | Register inference via inter-iteration state deltas (∼32 changed positions per round → reveals all 4 registers across 4 iterations) |
| 7 | `certify.py` | Domain-certifies motif candidates — only accepts an AND if both inputs are *provably* in $\{0,1\}$ |
| 8 | `head.py`, `body.py`, `tail.py`, `padding.py` | Region-specific decompilers |
| 9 | `round_decode.py` | Per-output-bit truth-table extraction; round-function building-block clustering (MVP-9a/b/c) |
| 10 | `emit.py` → `artifacts/recovered_program.py` | Codegen that inlines every recovered fact as a Python literal; remaining holes raise `NotImplementedError` with a pointer to the responsible MVP |

### 1.3 Concrete facts recovered from weights alone

| Component | Recovered value | MD5 ground truth |
|---|---|---|
| Tokenizer | `ord(str(x)[:55].ljust(55,'\x00'))` | exact match |
| Tail target | `c7ef65233c40aa32c2b9ace37595fa7c` | exact match |
| IV A, B, C, D | `0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476` | exact match |
| Round constant `K[0]` | `0xd76aa478` | exact match |
| Precomputed `B & C`, `~B` | `0x88888888`, `0x10325476` | exact match |
| Block structure | Head 18 / Body 63×42 / Tail 57 | matches MD5 padding + 64-round body + digest decoder |
| Shift schedule | `[7,12,17,22]×4, [5,9,14,20]×4, [4,11,16,23]×4, [6,10,15,21]×~4` | exact MD5 schedule for 63 of 64 rounds |
| AND gates | **12,859 certified** (out of 58,524 syntactic candidates) | structurally consistent with F/G/H/I round functions |
| Motifs | 6, each Z3-proved on bounded integer domains | — |
| Register inference | All 4 MD5 registers identified by inter-iteration deltas | — |

### 1.4 What is *not* yet recovered

* **K[1..63]** as residual 32-bit constants (only `K[0]` confirmed).
* Iteration 63/64 boundary: block finder reports 63 of MD5's 64 rounds; the
  last round may be absorbed into head/tail stitching.
* Clean per-iteration **F/G/H/I** round-function identification (building
  blocks `a∧¬b` / `¬a∧b` are visible at iteration 20, but the clean
  3-input boolean output is hidden inside the modular-adder carry chain).
* **Full padding semantics** (the head's first 18 layers).
* Stub-free `artifacts/recovered_program.py` + a verified
  `verify_against_model(model, 10000_random_byte_strings)` equivalence claim.

[adir]: https://github.com/adirk0/Jane-Street-Puzzle

---

## 2. What is novel

NeuroDecomp is best framed as a **static decompiler for sparse integer
ReLU networks** — a new niche between three established fields.

> **Mainstream NN verification** asks: *"does this network satisfy property
> $P$?"*  **NeuroDecomp asks:** *"what program did this network secretly
> compile?"*

Concretely, the contribution is the **combination**, not any single piece:

1. **Decompilation rather than verification.** Existing tools (Marabou,
   α,β-CROWN, DeepPoly) certify input/output properties, but leave the
   network as an opaque function. We *recover the source program* — IVs,
   round constants, rotation schedule, register layout, boolean motifs —
   as readable artefacts.

2. **Domain-certified motif recognition.** Mechanistic-interpretability
   work typically *names* motifs that humans recognise. We require every
   motif claim to be either (a) **Z3-proved** as an algebraic identity on
   bounded integer domains, or (b) **domain-certified** by forward
   abstract interpretation showing the inputs actually inhabit the
   required set. The 12,859 confirmed-vs-58,524 candidate AND figure is
   the headline of this discipline: **no motif claim without a proof.**

3. **Sparsity- and integer-aware abstract domains.** The `Affine` lattice
   plus the MVP-7.5 **relational opaque domain** (which keeps
   `Opaque(id, expr=known_affine, constraints)` alive across saturating
   ReLUs) is built for hand-compiled circuits, not the continuous
   robustness questions that drive `DeepPoly`/`Zonotope`.

4. **Loop folding for unrolled circuits.** The benchmark is a 63× unrolled
   MD5 body. We *recognise* the loop structurally (period detection +
   weight-tying analysis) and intend to decompile one iteration plus the
   loop, not all 5,442 layers in isolation.

5. **A genuinely hard, fully reverse-engineerable benchmark.** Most NN
   interpretability papers benchmark on toy circuits or partial
   explanations of large trained models. The Jane Street model is
   adversarially obfuscated, real (1.16 GB of weights), and has a
   **single ground-truth program** to compare against — yet the
   decompiler is given none of that.

A precise headline claim:

> NeuroDecomp is a static decompiler for sparse integer ReLU networks. On
> a 5,442-layer real hand-compiled PyTorch network, it automatically
> recovers — purely from weight structure — the tokenizer, the 16-byte
> output comparator, the four IV constants, the first round constant,
> the full 63-round rotation schedule, a Z3-proved motif catalog, and
> 12,859 domain-certified boolean ANDs.

The **stretch claim** (next milestones): a complete `recovered_program.py`
with no stubs, plus equivalence to the model on all byte strings of
length ≤ 55.

---

## 3. Related work

There is no existing tool that targets the *exact* niche (decompiling
hand-compiled ReLU circuits as programs). The closest neighbours sit in
five communities. We borrow techniques from each but our problem
statement differs in each case.

### 3.1 Neural-network verification

| Tool | What it does | How NeuroDecomp differs |
|---|---|---|
| **DeepPoly / ERAN** (Singh et al., 2019) | Symbolic bound propagation through ReLU networks via abstract domains | We use abstract interpretation too, but to **recover program structure**, not to check robustness. The domain is integer-/bit-typed, not continuous-noise-typed. |
| **α,β-CROWN** (Wang et al., 2021; Xu et al., 2020) | Branch-and-bound with linear relaxations; SOTA on VNN-COMP | Verification asks `∀x: f(x) ∈ Y?`; we ask `f ≡ g for some readable g?` |
| **Marabou / Reluplex** (Katz et al., 2017, 2019) | SMT-style solving for ReLU networks | We use Z3 **locally on small motifs**, never globally on the full network — explicit design rule. |
| **AI²** (Gehr et al., 2018) | Abstract interpretation for verifying NN safety | Same lineage; different goal. |

### 3.2 Mechanistic interpretability

| Work | What it does | How NeuroDecomp differs |
|---|---|---|
| **Olah et al. — Circuits** (Distill, 2020) | Identifies named circuits in vision models | We *prove* motifs with Z3 rather than naming them by inspection. Our target is a hand-compiled cryptographic circuit, not learned features. |
| **Anthropic — Sparse autoencoders, attribution graphs** | Decomposes trained transformer features | We exploit weight sparsity that *exists by construction* (99.6 %) rather than imposing it via SAE training. |
| **Conmy et al. — ACDC** (Conmy et al., 2023) | Automated circuit discovery in transformers | We do circuit discovery on `Linear/ReLU` cascades with integer weights, and emit *code*, not subgraphs. |
| **Nanda et al. — Grokking modular arithmetic** (2023) | Reverse-engineers a 1-layer transformer's modular-addition algorithm by hand | Same flavour ("what algorithm did the network learn?"), but we want this **automated** and on hand-compiled (not trained) targets. |

### 3.3 Abstract interpretation / program analysis

| Work | How it relates |
|---|---|
| **Cousot & Cousot — Abstract Interpretation** (1977) | Theoretical foundation for the `Affine`/`Opaque` lattices in `domains.py`. |
| **Astrée, Frama-C, IKOS** | Industrial abstract interpreters for numeric programs. Same techniques, applied to a network instead of source code. |

### 3.4 Traditional binary decompilation

| Tool | How it relates |
|---|---|
| **Ghidra, IDA Pro, Binary Ninja** | Static decompilers for x86/ARM. Architecturally similar pipeline (lift → SSA → simplify → recognise idioms → emit C). We replay this pattern for `Linear/ReLU` IR. |
| **angr, BAP** | Binary analysis frameworks with symbolic execution. Z3 use mirrors theirs. |
| **RetDec, Hex-Rays** | Idiom/motif libraries for compiler-emitted patterns. Our `motifs.py` is the analogue for ReLU motifs. |

### 3.5 Program synthesis

| Work | How it relates |
|---|---|
| **Sketch** (Solar-Lezama, 2008) | Synthesis with templates and verification. We *don't* synthesise from scratch; we extract structure that's already encoded. |
| **Rosette** (Torlak & Bodik, 2014) | Solver-aided programming language. Z3 motif verification follows the same "tiny SMT problems with strong locality" philosophy. |
| **NeuralPS / Karel program synthesis** | Generates code from input-output examples. We use the *weights themselves*, not I/O behaviour, which makes the search vastly more constrained. |

### 3.6 Circuit reverse engineering (hardware)

| Work | How it relates |
|---|---|
| **Netlist reverse engineering** (e.g., HAL, Subramanyan et al.) | Recovers logical functions from gate-level netlists. The problem is structurally analogous: identify standard cells (motifs), recover registers and FSMs. |
| **PUF/cryptographic circuit recovery** | Same flavour: low-level boolean substrate hiding a known crypto primitive. |

### 3.7 Closest single comparable

There is no published tool with NeuroDecomp's exact problem statement.
The closest single point of comparison is **mechanistic-interp work on
hand-compiled toy transformers** (e.g., Vaintrob et al. on
addition-circuit transformers, Olsson et al. on induction heads), but
those decompile by hand and target small, learned networks. NeuroDecomp
aims to be **automatic, motif-certified, and applicable to ≥1 GB
adversarially obfuscated circuits**.

---

## 4. Three success levels

| Level | Goal | Status |
|---|---|---|
| **L1 — Algorithm ID** | "This model computes an MD5 digest comparator." | ✅ Done — supported by IVs, K[0], rotation schedule, target digest. |
| **L2 — Program recovery** | `artifacts/recovered_program.py` with zero `NotImplementedError`, matching the model on randomised tests. | ⏳ Stubs remain for `K[1..63]`, `F/G/H/I`, `pad_message`. |
| **L3 — Certified decompilation** | Prove $g(s) = \mathrm{model}(s)$ for all $s \in \Sigma^{\le 55}$. | 🎯 SOTA endgame. |

Next-step priority (semantics-first, codegen-last):
**MVP-7.5 → MVP-9 → MVP-8 → MVP-10 → final emit/verify**.

---

## 5. One-sentence elevator pitch

> NeuroDecomp recovers a human-readable symbolic program from a 5,442-layer
> sparse integer ReLU network, with motif-level formal proofs and
> input-domain certification — no algorithm-specific hints, no
> human-in-the-loop.
