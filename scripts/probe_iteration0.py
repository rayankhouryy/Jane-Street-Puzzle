"""Probe a body iteration in detail with the upgraded round-decoder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import Counter

from neurodecomp import block_finder, model_loader, round_decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=20)
    ap.add_argument("--max-arity", type=int, default=8)
    args = ap.parse_args()

    m = model_loader.load_model("model_3_11.pt")
    layout = block_finder.find_blocks(m)
    subnet = round_decode.extract_iteration_subnet(
        m, args.iteration, layout.block_starts, layout.period,
    )

    import time
    t0 = time.time()
    rep = round_decode.decode_iteration(
        subnet, args.iteration, max_arity=args.max_arity, num_probes=4,
    )
    elapsed = time.time() - t0
    print(f"decoded {len(rep.bit_functions)} outputs of iteration {args.iteration} in {elapsed:.1f}s")

    arity_counts: Counter = Counter()
    name_counts: Counter = Counter()
    high_arity = []
    for bf in rep.bit_functions:
        if not bf.deps:
            arity_counts[0] += 1
            name_counts["const"] += 1
            continue
        if bf.table is None:
            arity_counts[len(bf.deps)] += 1
            high_arity.append((bf.output_idx, len(bf.deps)))
            name_counts[f"high-arity (>={args.max_arity + 1} deps)"] += 1
            continue
        arity_counts[len(bf.deps)] += 1
        name_counts[bf.name] += 1

    print()
    print("Arity distribution:")
    for k in sorted(arity_counts):
        print(f"  k={k}: {arity_counts[k]:4d}")
    print()
    print("Top function classes:")
    for name, n in name_counts.most_common(20):
        print(f"  {n:4d}  {name}")
    print()
    print("First 10 high-arity outputs:")
    for o, k in high_arity[:10]:
        print(f"  out[{o:3d}]  {k} deps")


if __name__ == "__main__":
    main()


