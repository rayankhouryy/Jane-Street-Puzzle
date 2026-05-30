"""Run the synthetic benchmark suite and emit a report.

Usage:
    python -m bench.run_bench
    python -m bench.run_bench --out artifacts/bench/report.md
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

from bench.circuits.bool_gates import ALL_CIRCUITS as BOOL_GATES
from bench.metrics import CircuitResult, compute_motif_pr, format_results

# Aggregate all circuit families here as the bench grows.
ALL_FAMILIES = {
    "bool_gates": BOOL_GATES,
}


def _try_certify(circuit) -> tuple[Counter, Counter]:
    """Run the domain certifier if it tolerates the tiny network shape.

    For a small synthetic circuit we can't always rely on block_finder to
    return useful head/body boundaries, so we pass head_end=0,
    body_end=n_linears (every layer is body) which is a no-op as far as
    the certifier's region-tagging is concerned.
    """
    from neurodecomp.certify import certify_motifs

    n_lin = circuit.n_linears
    n_in = circuit.n_inputs
    lo, hi = circuit.input_domain
    input_ranges = [(lo, hi)] * n_in
    try:
        report = certify_motifs(
            circuit.model,
            input_ranges=input_ranges,
            head_end=0,
            body_end=n_lin,
        )
        return report.confirmed_counts, report.candidate_counts
    except Exception as e:  # pragma: no cover -- surface but don't crash bench
        print(f"  [warn] certifier failed on {circuit.name}: {e}",
              file=sys.stderr)
        return Counter(), Counter()


def run_family(family_name: str, builders) -> List[CircuitResult]:
    results: List[CircuitResult] = []
    print(f"\n=== {family_name} ===")
    for build in builders:
        t0 = time.time()
        circuit = build()
        equiv_ok, n_tested, n_mismatch = circuit.check_equivalence()
        pr = compute_motif_pr(circuit)
        conf, cand = _try_certify(circuit)
        elapsed = time.time() - t0
        result = CircuitResult(
            name=circuit.name,
            n_inputs=circuit.n_inputs,
            n_outputs=circuit.n_outputs,
            n_linears=circuit.n_linears,
            n_params=circuit.n_params,
            equiv_ok=equiv_ok,
            n_tested=n_tested,
            n_mismatch=n_mismatch,
            motif_pr=pr,
            confirmed_by_kind=Counter(conf),
            candidate_by_kind=Counter(cand),
        )
        results.append(result)
        flag = "OK " if equiv_ok else "BAD"
        print(f"  [{flag}] {circuit.name:10s}  "
              f"in={circuit.n_inputs:3d} out={circuit.n_outputs:3d} "
              f"layers={circuit.n_linears:3d}  "
              f"placed={dict(pr.placed_by_kind)} "
              f"detected={dict(pr.detected_by_kind)} "
              f"confirmed={dict(conf)} "
              f"({elapsed:.2f}s)")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/bench/report.md"),
                   help="Where to write the markdown report.")
    p.add_argument("--family", choices=list(ALL_FAMILIES), default=None,
                   help="Run only one family (default: all).")
    args = p.parse_args(argv)

    families = (
        {args.family: ALL_FAMILIES[args.family]} if args.family
        else ALL_FAMILIES
    )

    all_results: List[CircuitResult] = []
    for name, builders in families.items():
        all_results.extend(run_family(name, builders))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = format_results(all_results)
    args.out.write_text(report, encoding="utf-8")
    print(f"\nWrote {args.out}")
    n_bad = sum(1 for r in all_results if not r.equiv_ok)
    if n_bad:
        print(f"  {n_bad}/{len(all_results)} circuits failed equivalence")
        return 1
    print(f"  {len(all_results)} circuits passed equivalence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
