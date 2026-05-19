"""MVP-7: domain-certified motif scan.

Run abstract interpretation + motif scanning together; certify each motif
candidate against the abstract domain of its inputs.

Usage::

    python scripts/run_certify.py model_3_11.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neurodecomp import certify, model_loader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path", type=str)
    args = ap.parse_args()

    model = model_loader.load_model(args.model_path)
    rep = certify.certify_motifs(model)
    print(certify.format_report(rep))


if __name__ == "__main__":
    main()
