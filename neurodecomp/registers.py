"""MVP-9d — register inference.

Identify which 32-bit windows of the body's 288-dim inter-iteration state
hold registers A, B, C, D (and which carry message words / round
constants) by observing how the state changes between consecutive
iterations.

Key insight: a hand-compiled MD5-style round modifies *exactly one*
32-bit register per iteration (say B = B + ROTL(F(B,C,D) + A + M + K, s)),
and the four registers cycle (A' = D, C' = B, D' = C).  Therefore:

  state(t+1) differs from state(t) in ~32 positions    [the active register]

Across 4+ consecutive iterations, all four registers are visited; we
then cluster the changed-bit positions into 4 contiguous 32-bit windows.

Recovery output:
  - register_positions: a dict ``{name -> (start_idx, end_idx)}`` for
    the 4 registers (named ``A``, ``B``, ``C``, ``D`` by convention,
    ordered by their per-iteration write index)
  - message_positions / round_constant_positions (heuristic from
    bit-passthrough analysis of the head)
  - per_iteration_active_register: which register name was updated at
    each iteration
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class RegisterReport:
    state_dim: int
    register_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    per_iteration_active_register: List[Optional[str]] = field(default_factory=list)
    per_iteration_diff_size: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _encode(s: str, width: int = 55) -> torch.Tensor:
    s = str(s)[:width].ljust(width, "\x00")
    return torch.Tensor([ord(c) for c in s])


def _snapshot_states(
    model: nn.Sequential,
    block_starts: Sequence[int],
    period: int,
    s: str,
    input_width: int = 55,
) -> List[np.ndarray]:
    """Run the model on ``s`` and return one state snapshot per iteration
    boundary (= input to each body block).  The first snapshot is the head's
    output; subsequent snapshots are the outputs of each iteration."""
    children = list(model)
    # The boundaries are: child index 2*block_starts[i] (input to iteration i).
    boundaries = [2 * s for s in block_starts]
    boundaries.append(2 * (block_starts[-1] + period))

    snapshots: List[np.ndarray] = []
    x = _encode(s, input_width)
    cur_idx = 0
    cur = x
    for b in boundaries:
        # Run from cur_idx to b (exclusive).
        with torch.no_grad():
            for i in range(cur_idx, b):
                cur = children[i](cur)
        snapshots.append(cur.detach().cpu().numpy().copy())
        cur_idx = b
    return snapshots


def infer_registers(
    model: nn.Sequential,
    block_starts: Sequence[int],
    period: int,
    *,
    probe_inputs: Optional[Sequence[str]] = None,
    input_width: int = 55,
    lane_width: int = 32,
    num_register_lanes: int = 4,
) -> RegisterReport:
    """Infer register windows by per-position change-frequency profiling.

    Strategy:
      * Compute the total change-count per position across many probe
        inputs and all iteration boundaries.
      * Slice into ``lane_width``-aligned windows starting at offset 0.
      * The top ``num_register_lanes`` windows by total change count are
        the candidate registers; the bottom windows are stable / constant
        storage (often where per-round constants K_i live).
    """
    if probe_inputs is None:
        probe_inputs = [
            "bitter lesson",
            "abcdef" * 4,
            "x" * 7,
            "the quick brown fox",
            "1234567890",
            "Z" * 30,
            "AAAA",
        ]

    n_iters = len(block_starts)
    pos_change_count: np.ndarray = np.zeros(0, dtype=np.int64)
    per_iter_change_mask: np.ndarray = np.zeros((n_iters, 0), dtype=np.int64)
    per_iter_diff_size_total: List[int] = []
    state_dim_max = 0

    for s_idx, s in enumerate(probe_inputs):
        snaps = _snapshot_states(model, block_starts, period, s, input_width)
        for t in range(n_iters):
            a = snaps[t]
            b = snaps[t + 1]
            if a.shape != b.shape:
                m = max(a.shape[0], b.shape[0])
                if a.shape[0] < m:
                    a = np.concatenate([a, np.zeros(m - a.shape[0])])
                if b.shape[0] < m:
                    b = np.concatenate([b, np.zeros(m - b.shape[0])])
            if pos_change_count.shape[0] < a.shape[0]:
                pad = np.zeros(a.shape[0] - pos_change_count.shape[0], dtype=np.int64)
                pos_change_count = np.concatenate([pos_change_count, pad])
            if per_iter_change_mask.shape[1] < a.shape[0]:
                pad2 = np.zeros((n_iters, a.shape[0] - per_iter_change_mask.shape[1]), dtype=np.int64)
                per_iter_change_mask = np.concatenate([per_iter_change_mask, pad2], axis=1)
            diff = (a != b).astype(np.int64)
            pos_change_count[: diff.shape[0]] += diff
            per_iter_change_mask[t, : diff.shape[0]] += diff
            state_dim_max = max(state_dim_max, a.shape[0])
            if s_idx == 0:
                per_iter_diff_size_total.append(int(diff.sum()))

    state_dim = pos_change_count.shape[0]
    notes: List[str] = []

    # Compute total change per lane_width-aligned window.
    n_windows = state_dim // lane_width
    window_totals = []
    for w in range(n_windows):
        s_pos = w * lane_width
        e_pos = s_pos + lane_width
        window_totals.append((s_pos, e_pos, int(pos_change_count[s_pos:e_pos].sum())))

    # Sort windows by total change descending.
    window_totals_sorted = sorted(window_totals, key=lambda t: -t[2])

    # Top num_register_lanes are register candidates.
    register_positions: Dict[str, Tuple[int, int]] = {}
    register_candidates = window_totals_sorted[:num_register_lanes]
    # Sort by start position so registers are named in left-to-right order.
    register_candidates.sort(key=lambda t: t[0])
    for i, (s_pos, e_pos, total) in enumerate(register_candidates):
        name = f"reg_{chr(ord('A') + i)}"
        register_positions[name] = (s_pos, e_pos)
        notes.append(f"{name}: positions [{s_pos}, {e_pos}) total change = {total}")

    # Stable tail positions: bottom-most windows by change count.
    stable_windows: List[Tuple[int, int]] = []
    for (s_pos, e_pos, total) in window_totals_sorted[::-1][:num_register_lanes]:
        if total < 200:    # quasi-stable; reflects per-round constant injection
            stable_windows.append((s_pos, e_pos))
    if stable_windows:
        notes.append(
            f"{len(stable_windows)} quasi-stable {lane_width}-bit windows "
            "(candidate per-round constant storage K_i / M_i)"
        )
        for s_pos, e_pos in stable_windows:
            notes.append(f"  stable lane: [{s_pos}, {e_pos})")

    # Assign "active register" per iteration: which register window had the
    # most position changes at that iteration.
    per_iteration_active_register: List[Optional[str]] = []
    for t in range(n_iters):
        best_name = None
        best_count = 0
        for name, (s_pos, e_pos) in register_positions.items():
            changed_in_window = int(per_iter_change_mask[t, s_pos:e_pos].sum())
            if changed_in_window > best_count:
                best_count = changed_in_window
                best_name = name
        per_iteration_active_register.append(best_name)

    return RegisterReport(
        state_dim=state_dim,
        register_positions=register_positions,
        per_iteration_active_register=per_iteration_active_register,
        per_iteration_diff_size=per_iter_diff_size_total,
        notes=notes,
    )


def format_report(rep: RegisterReport) -> str:
    lines = ["## Register inference"]
    lines.append(f"  state dim: {rep.state_dim}")
    for note in rep.notes:
        lines.append(f"  {note}")
    lines.append("")
    lines.append("## Recovered register windows")
    for name, (s_idx, e_idx) in rep.register_positions.items():
        lines.append(f"  {name}: positions [{s_idx}, {e_idx})  width={e_idx-s_idx}")
    lines.append("")
    lines.append("## Per-iteration active register (first 32)")
    for t, name in enumerate(rep.per_iteration_active_register[:32]):
        diff = (
            rep.per_iteration_diff_size[t]
            if t < len(rep.per_iteration_diff_size)
            else "?"
        )
        lines.append(f"  iter {t:2d}: active = {name}  (total state diff = {diff})")
    if len(rep.per_iteration_active_register) > 32:
        lines.append(f"  ... ({len(rep.per_iteration_active_register) - 32} more)")
    return "\n".join(lines)
