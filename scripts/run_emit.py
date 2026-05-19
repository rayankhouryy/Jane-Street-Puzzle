"""MVP-5: emit decompiled Python program.

Runs every NeuroDecomp stage on a model and writes the recovered structure
to a self-documenting Python file.

Usage::

    python scripts/run_emit.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import emit, model_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument(
        "--out",
        default="artifacts/recovered_program.py",
        help="Path to write the emitted Python file.",
    )
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    summary = emit.emit_to_file(model, args.out)

    print(f"Wrote {summary['bytes_written']} bytes to {summary['path']}")
    print()
    rec = summary["recovered"]
    print("## Recovered facts")
    print(f"  tokenizer length        : {rec.tokenizer_length}")
    print(f"  initial state words     : {len(rec.initial_words_le)}")
    print(f"  extra head constants    : {len(rec.extra_head_constants_le)}")
    print(f"  body iterations         : {rec.num_body_iterations}")
    print(f"  rotate gadget           : {rec.rotate_num_registers} x "
          f"{rec.rotate_num_lanes}-bit lanes")
    print(f"  shift schedule entries  : {len(rec.shift_schedule)}")
    print(f"  target digest           : {rec.target_digest_hex}")
    print(f"  AND patterns confirmed  : {rec.confirmed_and_count} / "
          f"{rec.candidate_and_count}")
    print()
    print("## Pending milestones")
    for p in summary["pending"].items:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
