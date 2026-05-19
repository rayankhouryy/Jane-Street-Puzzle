"""MVP-9c: identify the round function per iteration by probing register-update bits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from neurodecomp import block_finder, model_loader, round_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--iterations", type=str, default="0,16,32,48,62")
    ap.add_argument("--max-outputs", type=int, default=64)
    ap.add_argument("--output-offset", type=int, default=32)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    layout = block_finder.find_blocks(model)

    print(f"Block layout: period={layout.period}, iterations={layout.num_iterations}")
    print()

    for it in [int(x) for x in args.iterations.split(",")]:
        if it >= layout.num_iterations:
            print(f"Iteration {it}: out of range")
            continue
        subnet = round_decode.extract_iteration_subnet(
            model, it, layout.block_starts, layout.period,
        )
        t0 = time.time()
        info = round_decode.identify_iteration_round(
            subnet,
            max_outputs_to_try=args.max_outputs,
            output_offset=args.output_offset,
        )
        elapsed = time.time() - t0
        if info["evidence_count"] > 0:
            print(f"Iteration {it:2d}: {info['round_name']}"
                  f" (evidence_count={info['evidence_count']}, {elapsed:.1f}s)")
            for h in info["found_subsets"][:3]:
                print(f"    output #{h['output_idx']:3d}: subset={h['selected_subset']}"
                      f" -> {h['name']}")
        else:
            print(f"Iteration {it:2d}: unknown ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
