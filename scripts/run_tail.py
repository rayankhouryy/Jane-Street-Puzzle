"""MVP-2: decompile the tail of a hand-compiled Sequential.

For a model whose tail computes ``ReLU(sum_i delta(z_i - t_i) - (N-1))``,
recover:
  * N (the number of predicates),
  * each target t_i,
  * the bit-address vector for the integer that each predicate compares,
  * the packed concatenation of all targets when they're all bytes.

Usage::

    python scripts/run_tail.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import model_loader, tail  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--out", default="artifacts/tail_report.json")
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    rep = tail.decompile_tail(model)

    print(tail.format_report(rep))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "matches": rep.matches,
        "reason": rep.reason,
        "N": rep.N,
        "targets": rep.targets,
        "targets_are_bytes": rep.targets_are_bytes,
        "targets_hex": rep.targets_hex,
        "notes": rep.notes,
        "predicates": [
            {
                "i": p.i,
                "target": p.target,
                "bit_kind": p.bit_kind,
                "num_terms": p.num_terms,
                "bit_addresses": p.bit_addresses,
            }
            for p in rep.predicates
        ],
    }, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
