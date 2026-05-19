"""MVP-1 — Sparse Graph + Domain Profiler.

Algorithm-agnostic structural profile of a hand-compiled `nn.Sequential`:

* layer counts, widths, alternation
* weight alphabet + sparsity
* recovered tokenizer (if cloudpickle attached one)
* block discovery (period, iterations, per-position dispersion)
* output-layer signature: detect generic algebraic templates like
  "AND of N delta-encoded equality predicates"

The profiler must remain ignorant of the actual algorithm. It only describes
*structural* features; interpretation is left to later stages or the user.

Usage::

    python scripts/run_profile.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import block_finder, model_loader  # noqa: E402


# ---------------------------------------------------------------------------
# Per-model statistics
# ---------------------------------------------------------------------------

def weight_alphabet(model: nn.Sequential, max_size: int = 32) -> list:
    seen = set()
    for m in model:
        if isinstance(m, nn.Linear):
            for arr in (m.weight, m.bias):
                if arr is None:
                    continue
                seen.update(np.unique(arr.detach().cpu().numpy()).tolist())
                if len(seen) > max_size:
                    return sorted(seen)[:max_size]
    return sorted(seen)


def sparsity_stats(model: nn.Sequential) -> dict:
    total_weight = zero_weight = total_bias = zero_bias = 0
    for m in model:
        if isinstance(m, nn.Linear):
            W = m.weight.detach().cpu().numpy()
            total_weight += W.size
            zero_weight += int(np.sum(W == 0))
            if m.bias is not None:
                b = m.bias.detach().cpu().numpy()
                total_bias += b.size
                zero_bias += int(np.sum(b == 0))
    return {
        "weight_total": total_weight,
        "weight_zero": zero_weight,
        "weight_sparsity": zero_weight / max(1, total_weight),
        "bias_total": total_bias,
        "bias_zero": zero_bias,
    }


# ---------------------------------------------------------------------------
# Algebraic templates on the output layer
# ---------------------------------------------------------------------------

def detect_and_of_deltas(model: nn.Sequential) -> dict:
    """Detect whether the *last* Linear computes ``ReLU(sum_delta - (N-1))``
    where each delta is a Kronecker-delta-style triple ``(+1, -2, +1)``.

    A "template" of this form is:

        weight row  = (+1)*N, (-2)*N, (+1)*N   (3N entries)
        bias        = -(N - 1)

    and the upstream layer has organised features into triples ``a, b, c``
    where ``a - 2b + c == 1`` iff some integer equality holds.

    The function reports the detection without naming what the equality is
    (that's an interpretation, not a structural fact).
    """
    linears = [m for m in model if isinstance(m, nn.Linear)]
    if not linears or linears[-1].out_features != 1:
        return {"matches": False, "reason": "last layer is not scalar"}
    last = linears[-1]
    W = last.weight.detach().cpu().numpy().flatten()
    b = float(last.bias.detach().cpu().numpy().item()) if last.bias is not None else 0.0
    n = len(W)
    if n % 3 != 0:
        return {"matches": False, "reason": f"last in-dim {n} not divisible by 3"}
    third = n // 3
    expected_w = (
        list(np.ones(third)) + list(-2.0 * np.ones(third)) + list(np.ones(third))
    )
    if not np.allclose(W, expected_w, atol=1e-9):
        return {"matches": False, "reason": "weight pattern not (+1,-2,+1)*N"}
    if abs(b + (third - 1)) > 1e-9:
        return {
            "matches": False,
            "reason": f"bias is {b}, expected -(N-1) = {-(third - 1)}",
        }
    return {
        "matches": True,
        "N": third,
        "interpretation": (
            f"out = ReLU(sum_{{i=1..{third}}}(a_i - 2 b_i + c_i) - {third - 1})"
        ),
        "boolean_form": (
            f"AND of {third} predicates of shape (a + c == 2 b - 1) on integers"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument(
        "--out",
        type=str,
        default="artifacts/profile_report.json",
        help="Where to write the JSON report.",
    )
    args = ap.parse_args()

    print(f"# NeuroDecomp profile: {args.model_path}\n")

    model = model_loader.load_model(args.model_path)
    children = list(model)
    linears = [m for m in children if isinstance(m, nn.Linear)]
    relus = [m for m in children if isinstance(m, nn.ReLU)]

    # 1. Modules
    print("## Module summary")
    print(f"  total modules : {len(children)}")
    print(f"  nn.Linear     : {len(linears)}")
    print(f"  nn.ReLU       : {len(relus)}")
    alternating = (
        len(children) >= 2
        and all(
            isinstance(children[2 * i], nn.Linear)
            and isinstance(children[2 * i + 1], nn.ReLU)
            for i in range(min(len(linears), len(relus)))
        )
    )
    print(f"  alternating Linear/ReLU? {alternating}")
    print(f"  signature      : f : R^{linears[0].in_features} -> R^{linears[-1].out_features}")

    # 2. Weight alphabet & sparsity
    print("\n## Weight alphabet (truncated to 32 entries)")
    alphabet = weight_alphabet(model, max_size=32)
    print(f"  {alphabet}")
    sp = sparsity_stats(model)
    print("\n## Sparsity")
    print(f"  total weights : {sp['weight_total']:,}")
    print(f"  zero weights  : {sp['weight_zero']:,}  ({sp['weight_sparsity']:.1%})")
    print(f"  total biases  : {sp['bias_total']:,}")
    print(f"  zero biases   : {sp['bias_zero']:,}")
    if sp["weight_sparsity"] > 0.9 and len(alphabet) < 30:
        print(
            "  >> network is sparse with small integer alphabet -- "
            "consistent with a hand-compiled circuit, not gradient-trained."
        )

    # 3. Tokenizer
    print("\n## Tokenizer")
    tok = model_loader.recover_tokenizer(model)
    print(f"  {tok.summary}")
    if tok.callable_ is not None:
        sample = tok.callable_("test")
        print(
            f"  tok('test') = tensor[{sample.shape[0]}], "
            f"first 8 ords = {[int(v) for v in sample[:8].tolist()]}"
        )

    # 4. Block discovery
    print("\n## Block discovery (algorithm-agnostic)")
    layout = block_finder.find_blocks(model)
    print(f"  best period            : {layout.period}  (match ratio {layout.period_match_ratio:.4f})")
    print(
        f"  head linears           : [0, {layout.head_end_linear_idx})  "
        f"({layout.head_end_linear_idx} layers)"
    )
    print(
        f"  body linears           : [{layout.head_end_linear_idx}, {layout.body_end_linear_idx})  "
        f"({layout.body_end_linear_idx - layout.head_end_linear_idx} layers)"
    )
    print(
        f"  tail linears           : [{layout.body_end_linear_idx}, {layout.num_linear})  "
        f"({layout.num_linear - layout.body_end_linear_idx} layers)"
    )
    print(f"  num body iterations    : {layout.num_iterations}")
    tied = [p for p, d in enumerate(layout.position_dispersion) if d == 0.0]
    untied_nan = [p for p, d in enumerate(layout.position_dispersion) if d != d]    # NaN
    untied_real = [
        p for p, d in enumerate(layout.position_dispersion)
        if d not in (0.0,) and d == d
    ]
    print(
        f"  bit-identical positions: {len(tied)} of {layout.period}  "
        f"-> {tied[:16]}{'...' if len(tied) > 16 else ''}"
    )
    if untied_real:
        print(
            f"  varying positions      : {untied_real}  "
            "(per-iteration deltas live here)"
        )
    if untied_nan:
        print(
            f"  shape-changing positions: {untied_nan}  "
            "(stitching layers between iterations)"
        )
    for note in layout.notes:
        print(f"  note: {note}")

    # 5. Output-layer algebraic template
    print("\n## Output-layer signature")
    sig = detect_and_of_deltas(model)
    if sig["matches"]:
        print(f"  *** MATCH: AND-of-deltas template with N = {sig['N']}")
        print(f"      interpretation: {sig['interpretation']}")
        print(f"      boolean form:   {sig['boolean_form']}")
    else:
        last_lin = linears[-1]
        print(f"  no known template matched: {sig.get('reason', '')}")
        print(
            f"  raw last Linear: in={last_lin.in_features}, "
            f"weight value counts = {Counter(last_lin.weight.detach().cpu().numpy().flatten().tolist())}"
        )

    # 6. JSON dump
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_path": str(args.model_path),
        "num_modules": len(children),
        "num_linear": len(linears),
        "num_relu": len(relus),
        "alternating": bool(alternating),
        "input_dim": linears[0].in_features,
        "output_dim": linears[-1].out_features,
        "weight_alphabet": alphabet,
        "sparsity": sp,
        "tokenizer": {
            "summary": tok.summary,
            "width": tok.width,
            "pad_char": tok.pad_char,
            "byte_domain": tok.byte_domain,
        },
        "layout": {
            "period": layout.period,
            "period_match_ratio": layout.period_match_ratio,
            "head_end_linear_idx": layout.head_end_linear_idx,
            "body_end_linear_idx": layout.body_end_linear_idx,
            "block_starts": layout.block_starts,
            "num_iterations": layout.num_iterations,
            "position_dispersion": [
                None if d != d else d for d in layout.position_dispersion
            ],
            "notes": layout.notes,
        },
        "output_layer_template": sig,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
