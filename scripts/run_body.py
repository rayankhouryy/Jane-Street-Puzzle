"""MVP-4: body decompiler.

Extract per-iteration deltas from the unrolled loop body and report them
as a per-iteration table.

Usage::

    python scripts/run_body.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import body, model_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--max-iters", type=int, default=64)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    rep = body.decompile_body(model)
    print(body.format_report(rep, max_iters_to_show=args.max_iters))


if __name__ == "__main__":
    main()
