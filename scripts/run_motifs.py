"""MVP-6: motif scanner.

Verify the motif catalog symbolically (Z3) and then scan a model for
occurrences of each motif.

Usage::

    python scripts/run_motifs.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import model_loader, motifs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    args = ap.parse_args()

    # 1. Self-verify the motif catalog with Z3.
    print("## Motif catalog verification (Z3)")
    for r in motifs.verify_all():
        if r.proved:
            print(f"  [PROVED] {r.motif_name}")
        elif r.error:
            print(f"  [ERROR ] {r.motif_name}: {r.error}")
        else:
            print(f"  [FAIL  ] {r.motif_name}: counterexample {r.counterexample}")

    # 2. Scan the model.
    print()
    print(f"## Scanning {args.model_path}")
    model = model_loader.load_model(args.model_path)
    hits = motifs.scan_model(model)
    print(motifs.summarize_scan(hits))

    # 3. Per-region summary using block_finder.
    from neurodecomp import block_finder
    layout = block_finder.find_blocks(model)
    if layout.block_starts:
        regions = motifs.hits_per_layer_range(
            hits, layout.head_end_linear_idx, layout.body_end_linear_idx
        )
        print()
        print("## Hits per region")
        for region in ("head", "body", "tail"):
            print(f"  {region}:")
            for name, n in regions[region].most_common():
                print(f"    {name:12s} {n:5d}")
        # Body iteration size + hits per iteration.
        body_lin_count = layout.body_end_linear_idx - layout.head_end_linear_idx
        n_iters = layout.num_iterations
        if n_iters > 0:
            print(
                f"\n  body iterations: {n_iters}, "
                f"approx {sum(regions['body'].values()) // n_iters} hits per iter"
            )


if __name__ == "__main__":
    main()
