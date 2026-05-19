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

def _find_block_starts_phase(
    linears: List[nn.Linear],
    period: int,
    *,
    head_skip: int = 2,
    width_jump_ratio: float = 1.3,
) -> List[int]:
    """Phase-aligned block-start finder that survives stitching layers.

    Body block starts are characterised by:

    * a *strict jump* in input width relative to the previous Linear's input
      (``cur_in / prev_in >= width_jump_ratio``),
    * a compression at the current Linear (``in_features > out_features``).

    All such Linears are projected onto their offset modulo ``period`` and the
    dominant phase is the loop's block-start phase.  This survives stitching
    boundaries where minor widths differ.
    """
    from collections import Counter

    candidates: List[int] = []
    for i, l in enumerate(linears):
        if i < head_skip:
            continue
        prev_in = linears[i - 1].in_features if i > 0 else 0
        if (
            prev_in > 0
            and l.in_features >= width_jump_ratio * prev_in
            and l.in_features > l.out_features
        ):
            candidates.append(i)
    if not candidates:
        return []

    offsets = Counter(c % period for c in candidates)
    dominant_offset, _ = offsets.most_common(1)[0]
    starts = sorted(c for c in candidates if c % period == dominant_offset)
    return starts


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

    # Phase-grouped block-start enumeration.  This survives stitching layers
    # where the longest contiguous-run heuristic gives up.
    if period > 1:
        block_starts = _find_block_starts_phase(linears, period)
    else:
        block_starts = []

    if not block_starts:
        head_end, body_end = 0, len(linears)
        n_iters = 0
    else:
        head_end = block_starts[0]
        body_end = block_starts[-1] + period
        n_iters = len(block_starts)

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
