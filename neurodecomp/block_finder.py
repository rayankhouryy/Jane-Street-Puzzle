"""Stage 4 — block discovery.

Goal: from the weight sequence alone (no width hard-coding, no algorithm
knowledge), detect:

1. The dominant period in the layer-width sequence.
2. The longest contiguous range of layers that obey that period — i.e., the
   "body" of an unrolled loop.
3. For each in-block position p, the cross-iteration dispersion of weights.

Output: a `LayoutReport` with head / body / tail layer ranges and per-position
tying statistics.

The block finder must remain **algorithm-agnostic**: it sees only layer
widths and weight tensors, and must work equally well on MD5, SHA-1, AES,
or any other hand-compiled circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch.nn as nn


@dataclass
class LayoutReport:
    num_linear: int
    widths: List[int]                # length == num_linear + 1
    period: int
    period_match_ratio: float
    head_end_linear_idx: int         # exclusive
    body_end_linear_idx: int         # exclusive
    block_starts: List[int]          # one entry per body iteration
    num_iterations: int
    position_dispersion: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Period detection
# ---------------------------------------------------------------------------

def _find_period(widths: List[int], min_p: int = 2, max_p: int = 200) -> Tuple[int, float]:
    """Best-matching period of the width sequence."""
    n = len(widths)
    best: Tuple[int, float] = (0, 0.0)
    for p in range(min_p, min(max_p, n)):
        matches = sum(1 for i in range(n - p) if widths[i] == widths[i + p])
        ratio = matches / max(1, n - p)
        if ratio > best[1]:
            best = (p, ratio)
        if ratio > 0.999:
            break
    return best


def _longest_periodic_run(widths: List[int], period: int) -> Tuple[int, int]:
    """Longest contiguous [s, e) such that widths[i] == widths[i+period] for
    all s <= i < e. Returns (s, e); empty range is (0, 0)."""
    if period <= 0 or period >= len(widths):
        return (0, 0)
    runs: List[Tuple[int, int]] = []
    cur_s: int | None = None
    for i in range(len(widths) - period):
        if widths[i] == widths[i + period]:
            if cur_s is None:
                cur_s = i
        else:
            if cur_s is not None:
                runs.append((cur_s, i))
                cur_s = None
    if cur_s is not None:
        runs.append((cur_s, len(widths) - period))
    if not runs:
        return (0, 0)
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs[0]


# ---------------------------------------------------------------------------
# Block-start phase alignment
# ---------------------------------------------------------------------------

def _phase_align_block_starts(
    head_offset: int,
    body_end: int,
    period: int,
) -> List[int]:
    """Given the head/body/tail split and the period, emit one block-start
    per iteration. Block starts are at head_offset, head_offset + period, …

    The width index space and the Linear index space agree because widths[0]
    is the input width and widths[i+1] is the output of Linear i.
    """
    return list(range(head_offset, body_end, period))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_blocks(model: nn.Sequential, min_iterations: int = 3) -> LayoutReport:
    """Detect head / body / tail purely from the width sequence and weights.

    `min_iterations` is the minimum number of body iterations required to
    consider the body "real"; below that the whole network is reported as
    head-only.
    """
    linears: List[nn.Linear] = [m for m in model if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("model has no Linear layers")

    # widths[i] = input width of linear i; widths[-1] = output width of last.
    widths = [linears[0].in_features] + [l.out_features for l in linears]

    period, ratio = _find_period(widths)
    body_run = _longest_periodic_run(widths, period)
    head_end, body_end = body_run

    # Number of iterations covered by the run.
    n_iters = (body_end - head_end) // period if period else 0

    if period <= 1 or n_iters < min_iterations:
        return LayoutReport(
            num_linear=len(linears),
            widths=widths,
            period=period,
            period_match_ratio=ratio,
            head_end_linear_idx=0,
            body_end_linear_idx=len(linears),
            block_starts=[],
            num_iterations=0,
            notes=[
                "no repeated block detected "
                f"(period={period}, iterations={n_iters})"
            ],
        )

    block_starts = _phase_align_block_starts(head_end, body_end, period)

    # Per-position dispersion of weights across iterations.
    dispersions: List[float] = []
    for p in range(period):
        shapes = set()
        Ws = []
        for s in block_starts:
            idx = s + p
            if idx >= len(linears):
                continue
            lin = linears[idx]
            shapes.add((lin.in_features, lin.out_features))
            Ws.append(lin.weight.detach().cpu().numpy())
        if len(shapes) != 1 or len(Ws) <= 1:
            dispersions.append(float("nan"))
            continue
        W = np.stack(Ws)
        W_mean = W.mean(0)
        diff = np.linalg.norm(W - W_mean, axis=(1, 2))
        denom = float(np.linalg.norm(W_mean)) + 1e-12
        dispersions.append(float(diff.mean() / denom))

    n_tied = sum(1 for d in dispersions if d == 0.0)
    notes = [
        f"period={period} with match ratio {ratio:.4f}",
        f"body spans linears [{head_end}, {body_end}) = {n_iters} iterations",
        f"{n_tied} of {period} in-block positions are bit-identical across iterations",
    ]

    return LayoutReport(
        num_linear=len(linears),
        widths=widths,
        period=period,
        period_match_ratio=ratio,
        head_end_linear_idx=head_end,
        body_end_linear_idx=body_end,
        block_starts=block_starts,
        num_iterations=n_iters,
        position_dispersion=dispersions,
        notes=notes,
    )
