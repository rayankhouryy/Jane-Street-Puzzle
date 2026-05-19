"""Inspect ALL varying entries at boundary positions, all iterations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch.nn as nn
from collections import defaultdict

from neurodecomp import model_loader, block_finder

m = model_loader.load_model("model_3_11.pt")
linears = [x for x in m if isinstance(x, nn.Linear)]
layout = block_finder.find_blocks(m)


def show_position(target_p: int):
    print(f"\n=== Position {target_p} ===")
    biases = []
    for s in layout.block_starts:
        idx = s + target_p
        if idx >= len(linears):
            continue
        lin = linears[idx]
        biases.append((s, lin.bias.detach().cpu().numpy() if lin.bias is not None else None))
    by_shape = defaultdict(list)
    for it_idx, (s, b) in enumerate(biases):
        if b is None:
            continue
        by_shape[b.shape].append((it_idx, b))
    for shape, items in by_shape.items():
        first_iters = [i for i, _ in items[:3]]
        last_iters = [i for i, _ in items[-3:]]
        print(f"\n  Shape {shape}: {len(items)} iterations (iters {first_iters}..{last_iters})")
        arr = np.stack([b for _, b in items])
        var = (arr != arr[0:1]).any(axis=0)
        n_var = int(var.sum())
        if n_var == 0:
            print(f"    no variation within this shape group")
            continue
        var_indices = np.where(var)[0]
        unique_vals = sorted(set(arr[:, var_indices].flatten().tolist()))
        preview = unique_vals[:12]
        suffix = "..." if len(unique_vals) > 12 else ""
        print(f"    {n_var} varying bias entries; unique values: {preview}{suffix}")
        all_bit = bool(((arr[:, var_indices] == 0) | (arr[:, var_indices] == 1)).all())
        if all_bit and n_var % 8 == 0:
            n_bytes = n_var // 8
            print(f"    bit-valued. Packing as {n_bytes} bytes per iteration (LSB first):")
            for i in range(arr.shape[0]):
                bits = arr[i, var_indices].astype(int)
                chunks = []
                for u in range(n_bytes):
                    v = 0
                    for k in range(8):
                        v |= int(bits[u * 8 + k]) << k
                    chunks.append(f"{v:02x}")
                print(f"      iter {items[i][0]:2d}: {' '.join(chunks)}")
        else:
            print(f"    integer-valued. All iterations:")
            for i in range(arr.shape[0]):
                row = arr[i, var_indices]
                # Try interpreting groups of 4 entries as little-endian 32-bit ints.
                if len(row) % 4 == 0 and all(v == int(v) and abs(v) < 2**32 for v in row):
                    n_groups = len(row) // 4
                    words = []
                    for g in range(n_groups):
                        # Big or little endian? Try little.
                        bs = row[g * 4 : g * 4 + 4]
                        w_le = (int(bs[0]) & 0xff) | ((int(bs[1]) & 0xff) << 8) | ((int(bs[2]) & 0xff) << 16) | ((int(bs[3]) & 0xff) << 24)
                        words.append(f"{w_le:08x}")
                    print(f"      iter {items[i][0]:2d}: LE-words = {' '.join(words[:6])}{'...' if len(words) > 6 else ''}")
                else:
                    preview = ", ".join(f"{int(v) if v == int(v) else v:>4d}" for v in row[:12])
                    print(f"      iter {items[i][0]:2d}: [{preview}, ...] ({len(row)} vals)")


for p in [0, 1, 41]:
    show_position(p)
