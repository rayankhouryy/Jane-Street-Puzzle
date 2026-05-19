# 00 — The puzzle and the model (no spoilers)

## The puzzle

The [Jane Street March 2025 puzzle](https://huggingface.co/spaces/jane-street/puzzle)
ships a Hugging Face Space that runs a PyTorch model on user-typed strings:

> *Today I went on a hike and found a pile of tensors hidden underneath a
> neolithic burial mound! I sent it over to the local neural plumber, and they
> managed to cobble together this — `model.pt`. Anyway, I'm not sure what it
> does yet, but it must have been important to this past civilization. Maybe
> start by looking at the last two layers.*

The default input `"vegetable dog"` returns `0`. Find an input that returns
something non-zero.

## What the model presents to the world

These are observations available to anyone with the weights — no algorithm
knowledge required.

### Container

- `torch.nn.modules.container.Sequential`.
- **5442 children**, strictly alternating `Linear` then `ReLU`.
- Hence **2721 `Linear` layers** and **2721 `ReLU` layers**.
- First `Linear`: `in_features = 55`. Last `Linear`: `out_features = 1`.

### Hidden tokenizer

The Space's `app.py` calls `model("some string")` on raw strings, which
ordinarily would fail. The model instance carries a custom callable attached
via cloudpickle. Recovering it from the pickle (see
`scripts/05_extract_tokenizer.py`) gives:

```python
_call_impl = lambda x: model.forward(
    torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00'))))
)
```

So the input is interpreted as the ASCII byte values of the first 55
characters of the string, null-padded:

$$
x_i \;=\; \mathrm{ord}(s_i), \qquad i = 0, \dots, 54.
$$

### Type signature (effective)

$$
f : \Sigma^{\le 55} \;\longrightarrow\; \{0, 1\}, \qquad \Sigma = \{0, \dots, 255\}.
$$

The output is non-negative (final activation is `ReLU`) and behaves like a
0/1 indicator on random ASCII inputs.

### Weight alphabet & sparsity

Across all 2721 `Linear` layers:

- **99.6 %** of weights are exactly zero.
- The set of distinct nonzero values is small and integer-valued
  (powers of 2 plus small integers, e.g.
  $\{0, \pm 1, \pm 2, \pm 4, \dots, \pm 256\}$).

> The network was not trained. It is a hand-compiled circuit.

### Repetition

The sequence of layer widths has dominant period **42** with match ratio
$\approx 0.97$. The longest contiguous range obeying that period spans 31
iterations (the full body contains more iterations separated by
stitching layers we have not yet merged). Of the 42 in-block positions,
**40 are bit-identical across all iterations** — the body is a true
weight-tied unrolled loop. Two positions (28 and 41) carry per-iteration
deltas; one of them is a shape transition between blocks.

### Output-layer algebraic template

The final `Linear(48 → 1)` has the explicit weights

$$
W = \big([+1]^{16},\;[-2]^{16},\;[+1]^{16}\big), \qquad b = -15.
$$

Therefore

$$
f = \mathrm{ReLU}\!\Big(\textstyle\sum_{i=1}^{16}(a_i - 2 b_i + c_i) - 15\Big),
$$

which detects exactly when each of 16 independent integer-equality predicates
holds. So the final layer encodes the boolean

$$
f = \bigwedge_{i=1}^{16} \mathbb{1}[a_i + c_i = 2 b_i - 1].
$$

What those 16 predicates actually compare we will discover by walking
backward through the tail.

## What we want NeuroDecomp to discover

A symbolic Python program $g$ such that:

1. $g(s) = \mathrm{model}(s)$ for every $s \in \Sigma^{\le 55}$.
2. The program is human-readable: named registers, an explicit loop instead of
   31+ unrolled iterations, an explicit hash-like compression function instead
   of 2700 ReLU dispatches.
3. The algorithm name (if there is one) falls out of comparing $g$ against
   well-known compression functions.
