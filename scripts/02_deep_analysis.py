"""Deep analysis: last layers, weight tying across loop iterations, head SVD.

Outputs:
  artifacts/tail_layers.txt     - last-layer weights/biases inspection
  artifacts/weight_tying.json   - per-position cross-iteration similarity stats
  artifacts/head_svd.txt        - SVD of head Linear(55->224)
  artifacts/block_starts.json   - indices of each loop iteration's first Linear
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "model_3_11.pt"

# ---------------------------------------------------------------------------
# 0. Load
# ---------------------------------------------------------------------------
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
linears = [m for m in model if isinstance(m, torch.nn.Linear)]
N = len(linears)
print(f"loaded {N} linear layers")

# ---------------------------------------------------------------------------
# 1. Robustly locate loop-body block boundaries.
# A "block start" is any Linear whose in_features is in {336, 368} that begins
# a wide-phase: in_features in {336, 368} and out_features in {296, 328}.
# Decoder will look different.
# ---------------------------------------------------------------------------
block_starts: list[int] = []
for i, lin in enumerate(linears):
    if (
        lin.in_features in (336, 368)
        and lin.out_features in (296, 328)
        and i > 10  # skip head
    ):
        block_starts.append(i)
print(f"detected {len(block_starts)} loop-body block starts")
print(f"first 5 starts: {block_starts[:5]}, last 5: {block_starts[-5:]}")
# expected spacing 42
spacings = [block_starts[i + 1] - block_starts[i] for i in range(len(block_starts) - 1)]
print(f"spacings unique: {set(spacings)}")
(ART / "block_starts.json").write_text(
    json.dumps({"block_starts": block_starts, "spacings_unique": sorted(set(spacings))})
)

head_end = block_starts[0]                       # first body block start (inclusive)
body_end = block_starts[-1] + 42                 # exclusive
print(f"head: linears[0:{head_end}] ({head_end} layers)")
print(f"body: linears[{head_end}:{body_end}] ({body_end - head_end} layers, {len(block_starts)} iterations)")
print(f"tail: linears[{body_end}:{N}] ({N - body_end} layers)")

# ---------------------------------------------------------------------------
# 2. Inspect tail (decoder) layers in detail.
# ---------------------------------------------------------------------------
tail_lin = linears[body_end:]
print("\n=== TAIL layers ===")
tail_txt: list[str] = []
for k, lin in enumerate(tail_lin):
    W = lin.weight.detach().cpu().numpy()
    b = lin.bias.detach().cpu().numpy() if lin.bias is not None else None
    line = (
        f"[{body_end + k:4d}] tail[{k:2d}] Linear({lin.in_features:4d} -> {lin.out_features:4d}) "
        f"||W||F={np.linalg.norm(W):.3f}  W min/max={W.min():.3f}/{W.max():.3f}  "
        f"b min/max={b.min():.3f}/{b.max():.3f}" if b is not None else ""
    )
    tail_txt.append(line)
    print(line)

# Last Linear: 48 -> 1.  Print its full weight row and bias.
last = tail_lin[-1]
W_last = last.weight.detach().cpu().numpy().flatten()
b_last = float(last.bias.detach().cpu().numpy().flatten()[0])
tail_txt.append("")
tail_txt.append("LAST Linear weight row (48,):")
tail_txt.append(np.array2string(W_last, precision=4, threshold=200))
tail_txt.append(f"LAST Linear bias: {b_last:.6f}")
print("\nLAST Linear weight row:", W_last)
print(f"LAST Linear bias: {b_last:.6f}")

# Penultimate Linear: 192 -> 48.  Look at its structure / row norms.
penu = tail_lin[-2]
W_pen = penu.weight.detach().cpu().numpy()
row_norms = np.linalg.norm(W_pen, axis=1)
col_norms = np.linalg.norm(W_pen, axis=0)
tail_txt.append("")
tail_txt.append("Penultimate Linear(192->48) row norms:")
tail_txt.append(np.array2string(row_norms, precision=3))
tail_txt.append("Penultimate Linear(192->48) col norms (first 50 of 192):")
tail_txt.append(np.array2string(col_norms[:50], precision=3))

(ART / "tail_layers.txt").write_text("\n".join(tail_txt))

# ---------------------------------------------------------------------------
# 3. Weight tying: compare layer-k of iteration i vs iteration j.
# If iterations share weights => values close to identical. If not => similar
# norms but maybe different patterns.
# ---------------------------------------------------------------------------
# For each position p in [0, 42), get the Linear at every iteration and
# measure pairwise differences.
period = 42
positions = list(range(period))
tying_stats: dict[int, dict] = {}
for p in positions:
    layers_at_p = []
    for s in block_starts:
        idx = s + p
        if idx < body_end:
            layers_at_p.append(linears[idx])
    # confirm all share same shape
    shapes = {(l.in_features, l.out_features) for l in layers_at_p}
    if len(shapes) != 1:
        tying_stats[p] = {"shapes": list(shapes), "comparable": False}
        continue
    Ws = np.stack([l.weight.detach().cpu().numpy() for l in layers_at_p])
    bs = np.stack([l.bias.detach().cpu().numpy() for l in layers_at_p])
    W_mean = Ws.mean(0)
    b_mean = bs.mean(0)
    W_std = Ws.std(0)
    # mean per-iteration L2 distance from the mean weight matrix
    diff_W = np.linalg.norm(Ws - W_mean, axis=(1, 2))   # (num_iters,)
    diff_b = np.linalg.norm(bs - b_mean, axis=1)
    tying_stats[p] = {
        "shape": list(next(iter(shapes))),
        "num_iters": len(layers_at_p),
        "W_mean_norm": float(np.linalg.norm(W_mean)),
        "W_std_mean": float(W_std.mean()),
        "W_diff_mean": float(diff_W.mean()),
        "W_diff_max": float(diff_W.max()),
        "b_diff_mean": float(diff_b.mean()),
        "b_diff_max": float(diff_b.max()),
        # ratio: if tied, ratio approx 0; if independent, ratio ~ O(1)
        "W_relative_diff": float(diff_W.mean() / (np.linalg.norm(W_mean) + 1e-9)),
    }
(ART / "weight_tying.json").write_text(json.dumps(tying_stats, indent=2))

print("\n=== WEIGHT TYING summary (all positions) ===")
for p in positions:
    s = tying_stats[p]
    if not s.get("comparable", True):
        print(f"pos {p:2d}: NOT COMPARABLE (shapes vary): {s['shapes']}")
        continue
    print(
        f"pos {p:2d}: shape={s['shape']} iters={s['num_iters']}  "
        f"W_rel_diff={s['W_relative_diff']:.4f}  W_std_mean={s['W_std_mean']:.4f}"
    )

avg_rel = np.mean([tying_stats[p]["W_relative_diff"] for p in positions if "W_relative_diff" in tying_stats[p]])
print(f"\nAverage relative weight diff across positions: {avg_rel:.4f}")
print("If << 0.05 => loop is weight-TIED.  If ~0.5-1.5 => iterations have independent weights.")

# ---------------------------------------------------------------------------
# 4. SVD of head Linear(55->224).
# ---------------------------------------------------------------------------
W0 = linears[0].weight.detach().cpu().numpy()        # shape (224, 55)
b0 = linears[0].bias.detach().cpu().numpy()
U, S, Vt = np.linalg.svd(W0, full_matrices=False)
svd_txt = []
svd_txt.append(f"Layer 0 Linear({linears[0].in_features} -> {linears[0].out_features})")
svd_txt.append(f"  ||W||F = {np.linalg.norm(W0):.4f}, rank-tol singular values:")
svd_txt.append(f"  singular values: {S}")
svd_txt.append(f"  cumulative energy: {np.cumsum(S**2) / np.sum(S**2)}")
svd_txt.append("")
svd_txt.append("Top-5 right singular vectors (Vt rows, length 55):")
for k in range(5):
    svd_txt.append(f"  v[{k}] (σ={S[k]:.3f}):")
    svd_txt.append(f"    {np.array2string(Vt[k], precision=3, suppress_small=True)}")
svd_txt.append("")
svd_txt.append(f"Layer-0 bias (224 dims): mean={b0.mean():.4f} std={b0.std():.4f} min={b0.min():.4f} max={b0.max():.4f}")
(ART / "head_svd.txt").write_text("\n".join(svd_txt), encoding="utf-8")
print("\n=== HEAD SVD ===")
print(f"singular values: {S}")
print(f"cumulative energy (cum sum of σ²/Σσ²):\n{np.cumsum(S**2) / np.sum(S**2)}")
