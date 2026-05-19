"""MVP-9d: register inference.

Identify which 32-bit windows of the body's inter-iteration state hold the
A/B/C/D registers (or analogues), by observing per-iteration state diffs.

Usage::

    python scripts/run_register_inference.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import block_finder, model_loader, registers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--lane-width", type=int, default=32)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    layout = block_finder.find_blocks(model)
    t0 = time.time()
    rep = registers.infer_registers(
        model, layout.block_starts, layout.period,
        lane_width=args.lane_width,
    )
    elapsed = time.time() - t0
    print(f"inferred in {elapsed:.1f}s")
    print()
    print(registers.format_report(rep))


if __name__ == "__main__":
    main()
