"""Plot the per-position state-change frequency to find 32-bit register windows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch.nn as nn

from neurodecomp import block_finder, model_loader
from neurodecomp.registers import _snapshot_states, _encode

m = model_loader.load_model("model_3_11.pt")
layout = block_finder.find_blocks(m)
probes = [
    "bitter lesson", "abcdefgh" * 4, "x" * 7,
    "the quick brown fox", "0123456789", "AAAAA",
]
pos_change_count = None
max_dim = 0
for s in probes:
    snaps = _snapshot_states(m, layout.block_starts, layout.period, s, 55)
    max_dim = max(max_dim, max(snap.shape[0] for snap in snaps))
    if pos_change_count is None:
        pos_change_count = np.zeros(max_dim, dtype=np.int64)
    n_iters = len(snaps) - 1
    for t in range(n_iters):
        a, b = snaps[t], snaps[t+1]
        if a.shape != b.shape:
            m_ = max(a.shape[0], b.shape[0])
            a = np.concatenate([a, np.zeros(m_ - a.shape[0])])
            b = np.concatenate([b, np.zeros(m_ - b.shape[0])])
        diff = (a != b).astype(np.int64)
        if diff.shape[0] > pos_change_count.shape[0]:
            pos_change_count = np.concatenate([pos_change_count, np.zeros(diff.shape[0] - pos_change_count.shape[0], dtype=np.int64)])
        pos_change_count[: diff.shape[0]] += diff

# Per 32-bit window: total change frequency
print("Change frequency by 32-bit window:")
for start in range(0, len(pos_change_count), 32):
    end = min(start + 32, len(pos_change_count))
    total = int(pos_change_count[start:end].sum())
    unique = int(np.unique(pos_change_count[start:end]).size)
    avg = total / (end - start) if end > start else 0
    bar = "#" * min(60, total // 5)
    print(f"  [{start:3d}, {end:3d}): total={total:5d}  avg={avg:5.1f}  unique_freqs={unique}  {bar}")

# Per-position: print which positions have which counts
print()
print("Positions sorted by change frequency (lowest first):")
order = np.argsort(pos_change_count)
print(f"  most stable (lowest 16): {order[:16].tolist()}")
print(f"  most variable (highest 8): {order[-8:].tolist()}")
unique, counts = np.unique(pos_change_count, return_counts=True)
print()
print("Change-count distribution (count -> #positions):")
for c, n in zip(unique[:30], counts[:30]):
    print(f"  count={c:4d}: {n} positions")
