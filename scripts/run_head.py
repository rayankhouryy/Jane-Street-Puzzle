"""MVP-3: head decompiler.

Run abstract interpretation on the head of a hand-compiled Sequential and
print a per-neuron symbolic summary.

Usage::

    python scripts/run_head.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import head, model_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    ap.add_argument("--max-show", type=int, default=80)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    rep = head.decompile_head(model)
    print(head.format_report(rep, max_neurons=args.max_show))


if __name__ == "__main__":
    main()
