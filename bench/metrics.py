"""Per-benchmark metrics: equivalence + motif precision/recall."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from bench.compile import CompiledCircuit, PlacedMotif
from neurodecomp.motifs import MotifHit, scan_model


def _placed_set(motifs: List[PlacedMotif]) -> set:
    """Identity of a placed motif for matching: (kind, layer_idx, out_neuron)."""
    return {(m.kind, m.layer_idx, m.out_neuron) for m in motifs}


def _hit_set(hits: List[MotifHit]) -> set:
    """Identity of a scan hit: (motif_name, layer_idx, output_neurons[0])."""
    out = set()
    for h in hits:
        if len(h.output_neurons) != 1:
            continue
        out.add((h.motif_name, h.layer_idx, h.output_neurons[0]))
    return out


@dataclass
class MotifPRReport:
    """Motif-level precision / recall vs ground truth.

    Computed at the *syntactic* level (motif scan output), per kind."""
    placed_by_kind: Counter = field(default_factory=Counter)
    detected_by_kind: Counter = field(default_factory=Counter)
    tp_by_kind: Counter = field(default_factory=Counter)
    fp_by_kind: Counter = field(default_factory=Counter)
    fn_by_kind: Counter = field(default_factory=Counter)

    def kinds(self):
        return sorted(set(self.placed_by_kind) | set(self.detected_by_kind))

    def precision(self, kind: str) -> float:
        d = self.detected_by_kind[kind]
        return (self.tp_by_kind[kind] / d) if d else float("nan")

    def recall(self, kind: str) -> float:
        p = self.placed_by_kind[kind]
        return (self.tp_by_kind[kind] / p) if p else float("nan")


def compute_motif_pr(circuit: CompiledCircuit) -> MotifPRReport:
    placed = circuit.motifs
    detected = scan_model(circuit.model)
    placed_ids = _placed_set(placed)
    detected_ids = _hit_set(detected)

    rep = MotifPRReport()
    for m in placed:
        rep.placed_by_kind[m.kind] += 1
    for h in detected:
        if len(h.output_neurons) != 1:
            continue
        rep.detected_by_kind[h.motif_name] += 1

    tp_ids = placed_ids & detected_ids
    for kind, _, _ in tp_ids:
        rep.tp_by_kind[kind] += 1
    for kind, _, _ in detected_ids - placed_ids:
        rep.fp_by_kind[kind] += 1
    for kind, _, _ in placed_ids - detected_ids:
        rep.fn_by_kind[kind] += 1

    return rep


@dataclass
class CircuitResult:
    name: str
    n_inputs: int
    n_outputs: int
    n_linears: int
    n_params: int
    equiv_ok: bool
    n_tested: int
    n_mismatch: int
    motif_pr: MotifPRReport
    # Confirmed counts from the domain-certifier (kind -> count).
    # Population is optional; the runner fills this in.
    confirmed_by_kind: Counter = field(default_factory=Counter)
    candidate_by_kind: Counter = field(default_factory=Counter)


def format_results(results: List[CircuitResult]) -> str:
    """Markdown table with one row per circuit."""
    lines = []
    lines.append("# Synthetic benchmark suite — results")
    lines.append("")
    lines.append("Equivalence column: ✅ if model(x) == oracle(x) on the tested "
                 "input set (exhaustive when ≤2^14 points, else 16k random "
                 "samples seeded at 0).")
    lines.append("")
    header = ("| circuit | in | out | linears | params | equiv | "
              "placed | detected | confirmed | TP | FP | FN |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        placed = ", ".join(f"{k}:{v}" for k, v in
                           sorted(r.motif_pr.placed_by_kind.items())) or "-"
        detected = ", ".join(f"{k}:{v}" for k, v in
                             sorted(r.motif_pr.detected_by_kind.items())) or "-"
        confirmed = ", ".join(f"{k}:{v}" for k, v in
                              sorted(r.confirmed_by_kind.items())) or "-"
        tp = sum(r.motif_pr.tp_by_kind.values())
        fp = sum(r.motif_pr.fp_by_kind.values())
        fn = sum(r.motif_pr.fn_by_kind.values())
        equiv = "✅" if r.equiv_ok else f"❌ ({r.n_mismatch}/{r.n_tested})"
        lines.append(
            f"| {r.name} | {r.n_inputs} | {r.n_outputs} | {r.n_linears} | "
            f"{r.n_params} | {equiv} | {placed} | {detected} | "
            f"{confirmed} | {tp} | {fp} | {fn} |"
        )
    lines.append("")
    lines.append("## Per-kind precision / recall (syntactic scanner)")
    lines.append("")
    agg = MotifPRReport()
    for r in results:
        for k, v in r.motif_pr.placed_by_kind.items():
            agg.placed_by_kind[k] += v
        for k, v in r.motif_pr.detected_by_kind.items():
            agg.detected_by_kind[k] += v
        for k, v in r.motif_pr.tp_by_kind.items():
            agg.tp_by_kind[k] += v
        for k, v in r.motif_pr.fp_by_kind.items():
            agg.fp_by_kind[k] += v
        for k, v in r.motif_pr.fn_by_kind.items():
            agg.fn_by_kind[k] += v
    lines.append("| kind | placed | detected | TP | FP | FN | precision | recall |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for kind in agg.kinds():
        p = agg.precision(kind)
        r = agg.recall(kind)
        ps = f"{p:.3f}" if p == p else "n/a"
        rs = f"{r:.3f}" if r == r else "n/a"
        lines.append(
            f"| {kind} | {agg.placed_by_kind[kind]} | "
            f"{agg.detected_by_kind[kind]} | {agg.tp_by_kind[kind]} | "
            f"{agg.fp_by_kind[kind]} | {agg.fn_by_kind[kind]} | {ps} | {rs} |"
        )
    return "\n".join(lines)
