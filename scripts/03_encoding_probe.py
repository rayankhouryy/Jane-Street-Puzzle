"""Probe input encoding: look at layer-0 weight structure, layer-0 bias, and head activations.

Key questions:
  - What does each of the 55 input dims look like? (column pattern of W0)
  - What's special about the bias vector?
  - Does feeding the all-zero input give a deterministic output? Single-dim inputs?
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
MODEL_PATH = ROOT / "model_3_11.pt"

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
linears = [m for m in model if isinstance(m, torch.nn.Linear)]

# ---------------------------------------------------------------------------
# 1. Layer 0 column patterns.
# ---------------------------------------------------------------------------
W0 = linears[0].weight.detach().cpu().numpy()       # (224, 55)
b0 = linears[0].bias.detach().cpu().numpy()         # (224,)

cols = W0.T                                          # (55, 224)
col_pattern_counts = Counter()
col_signatures = []
for j in range(55):
    nz = np.nonzero(cols[j])[0]
    sig = tuple(sorted([(int(k), float(cols[j, k])) for k in nz]))
    col_signatures.append(sig)
    col_pattern_counts[len(nz)] += 1
print(f"col non-zero count distribution: {dict(col_pattern_counts)}")
print(f"unique values present in W0: {sorted(set(W0.flatten().tolist()))[:20]} ...")
unique_vals, val_counts = np.unique(W0, return_counts=True)
print(f"all unique W0 values: {dict(zip(unique_vals.tolist(), val_counts.tolist()))}")

# How many distinct column patterns?
print(f"distinct columns: {len(set(col_signatures))} (of 55)")

# ---------------------------------------------------------------------------
# 2. Layer 0 bias: print sorted unique values + which row indices have each.
# ---------------------------------------------------------------------------
print("\nLayer-0 bias unique values + counts:")
ub, cb = np.unique(b0, return_counts=True)
for v, c in zip(ub.tolist(), cb.tolist()):
    print(f"  bias = {v:6.2f}  count = {c}")

# ---------------------------------------------------------------------------
# 3. Run actual forwards.
# ---------------------------------------------------------------------------
with torch.no_grad():
    # Baseline: all zeros input
    z = torch.zeros(1, 55)
    print(f"\nall-zero input  -> output = {model(z).item():.4f}")

    # Each single-dim input (value 1)
    one_hot_outs = []
    for j in range(55):
        x = torch.zeros(1, 55); x[0, j] = 1.0
        one_hot_outs.append(model(x).item())
    print(f"single-1 input outputs: min={min(one_hot_outs):.2f} max={max(one_hot_outs):.2f}")
    print(f"  non-zero outputs (j -> y):")
    for j, y in enumerate(one_hot_outs):
        if abs(y) > 1e-6:
            print(f"    dim {j:2d}: y = {y:.4f}")

    # All-ones (every word present)
    x = torch.ones(1, 55)
    print(f"\nall-ones input  -> output = {model(x).item():.4f}")

    # Pairs (j, k) - test a few random ones
    rng = np.random.default_rng(0)
    pair_results = []
    for _ in range(50):
        j, k = rng.choice(55, 2, replace=False)
        x = torch.zeros(1, 55); x[0, j] = 1; x[0, k] = 1
        y = model(x).item()
        pair_results.append((int(j), int(k), y))
    nonzero_pairs = [t for t in pair_results if abs(t[2]) > 1e-6]
    print(f"\nrandom 2-hot pairs: {len(nonzero_pairs)}/50 non-zero")
    for j, k, y in nonzero_pairs[:20]:
        print(f"  dims ({j:2d}, {k:2d}) -> y = {y:.4f}")

# Save column patterns for reference
sig_summary = []
for j, sig in enumerate(col_signatures):
    sig_summary.append({"dim": j, "nonzero_rows_and_vals": [list(p) for p in sig]})
(ART / "head_layer0_columns.json").write_text(json.dumps(sig_summary, indent=2))
(ART / "head_layer0_bias_unique.json").write_text(json.dumps({"unique": ub.tolist(), "counts": cb.tolist()}))
