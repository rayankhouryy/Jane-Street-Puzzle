"""MVP-10: padding scheme decoder.

Probe the head with varying-length inputs and classify each output as
constant / passthrough / length-bit / length-indicator / mixed.

Usage::

    python scripts/run_padding.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch.nn as nn

from neurodecomp import block_finder, model_loader, padding  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--num-inputs", type=int, default=400)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    layout = block_finder.find_blocks(model)
    head_end_child = layout.head_end_linear_idx * 2
    head = nn.Sequential(*list(model)[:head_end_child])

    t0 = time.time()
    rep = padding.decode_padding(head, input_width=55, num_inputs=args.num_inputs)
    elapsed = time.time() - t0
    print(f"decoded {rep.head_output_dim} head outputs in {elapsed:.1f}s")
    print()
    print(padding.summarize_padding(rep))


if __name__ == "__main__":
    main()
