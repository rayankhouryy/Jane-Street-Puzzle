"""MVP-9: round-function decoder.

For one body iteration of a hand-compiled `nn.Sequential`, recover per-output-bit
boolean functions empirically (probe + truth-table) and classify against a
catalog of named functions including MD5's F/G/H/I.

Usage::

    python scripts/run_round_decode.py model_3_11.pt --iteration 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import block_finder, model_loader, round_decode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--max-outputs", type=int, default=32,
                    help="probe at most this many output neurons (start small)")
    ap.add_argument("--max-arity", type=int, default=6)
    ap.add_argument("--num-probes", type=int, default=4)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    layout = block_finder.find_blocks(model)
    if args.iteration >= layout.num_iterations:
        raise SystemExit(
            f"iteration {args.iteration} >= num_iterations {layout.num_iterations}"
        )
    subnet = round_decode.extract_iteration_subnet(
        model, args.iteration, layout.block_starts, layout.period,
    )
    out_dim = round_decode.find_iteration_output_dim(subnet)
    output_indices = list(range(min(out_dim, args.max_outputs)))
    rep = round_decode.decode_iteration(
        subnet,
        args.iteration,
        output_indices=output_indices,
        max_arity=args.max_arity,
        num_probes=args.num_probes,
    )
    print(round_decode.summarize_iteration(rep))


if __name__ == "__main__":
    main()
