"""MVP-9b: identify the round function (F/G/H/I) for each body iteration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

from neurodecomp import block_finder, model_loader, round_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--iterations", type=str, default="0,16,32,48",
                    help="comma-separated iteration indices to probe")
    ap.add_argument("--min-cluster-size", type=int, default=4)
    ap.add_argument("--verbose", action="store_true",
                    help="print every snapshot's top clusters")
    args = ap.parse_args()

    iter_idxs = [int(x) for x in args.iterations.split(",")]
    model = model_loader.load_model(args.model_path)
    layout = block_finder.find_blocks(model)

    print(f"Block layout: period={layout.period}, iterations={layout.num_iterations}")
    print()

    for it in iter_idxs:
        if it >= layout.num_iterations:
            print(f"Iteration {it}: out of range")
            continue
        subnet = round_decode.extract_iteration_subnet(
            model, it, layout.block_starts, layout.period,
        )
        t0 = time.time()
        info = round_decode.find_round_function(
            subnet,
            max_arity=3,
            min_cluster_size=args.min_cluster_size,
        )
        elapsed = time.time() - t0
        print(f"Iteration {it:2d}: {info['round_name']}"
              f" (cluster_size={info['cluster_size']}, {elapsed:.1f}s)")
        # Show top clusters at each snapshot.
        for snap_stat in info["snapshot_stats"]:
            top = snap_stat["top_clusters"]
            if not top and not args.verbose:
                continue
            top_names = ", ".join(
                f"{c['name']}({c['count']})" for c in top[:5]
            ) or "(no 3-input clusters)"
            print(f"           snapshot child#{snap_stat['snapshot_child_idx']:3d}"
                  f" (out_dim={snap_stat['out_dim']:3d}): {top_names}")
        print()


if __name__ == "__main__":
    main()
