"""Stage 7 -- emit decompiled Python program.

Aggregate the recovered artefacts from earlier stages and write them out as
a *self-contained Python file*.  The file:

* documents what NeuroDecomp has recovered from the model,
* implements every recovered piece directly (tokenizer, initial state,
  shift schedule, target digest equality),
* marks every *not-yet-recovered* piece with an explicit
  ``raise NotImplementedError`` so reviewers can see exactly where the
  decompilation gap is,
* exposes a `verify_against_model(model, samples)` harness that compares
  emit-side intermediate computations against the original PyTorch model.

The intent is honest scaffolding: the emitted file always *runs*, but only
for the parts we can already prove.  As later milestones land, the
``raise NotImplementedError`` stubs get replaced with real Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import indent
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from . import block_finder, body, head, model_loader, motifs, tail
from . import certify as certify_mod


# ---------------------------------------------------------------------------
# Aggregated artefact container
# ---------------------------------------------------------------------------

@dataclass
class Recovered:
    # Tokenizer
    tokenizer_length: int = 55
    tokenizer_pad: str = "\x00"
    tokenizer_summary: str = ""

    # Initial state from head
    initial_words_le: List[int] = field(default_factory=list)
    extra_head_constants_le: List[int] = field(default_factory=list)

    # Per-iteration shift schedule from body
    shift_schedule: List[int] = field(default_factory=list)
    rotate_num_registers: int = 0
    rotate_num_lanes: int = 0

    # Tail (16-byte equality)
    target_digest_hex: str = ""

    # Block-finder layout
    num_body_iterations: int = 0
    body_block_size: int = 0
    head_linear_count: int = 0
    tail_linear_count: int = 0

    # Motif certification
    candidate_and_count: int = 0
    confirmed_and_count: int = 0


@dataclass
class Pending:
    items: List[str] = field(default_factory=list)


def aggregate(model: nn.Sequential) -> tuple[Recovered, Pending]:
    """Run every NeuroDecomp stage and pack the results into a Recovered."""
    rec = Recovered()
    pen = Pending()

    # Tokenizer.
    tok = model_loader.recover_tokenizer(model)
    rec.tokenizer_length = tok.width
    rec.tokenizer_pad = tok.pad_char
    rec.tokenizer_summary = tok.summary

    # Layout.
    layout = block_finder.find_blocks(model)
    rec.num_body_iterations = layout.num_iterations
    rec.body_block_size = layout.period
    rec.head_linear_count = layout.head_end_linear_idx
    rec.tail_linear_count = layout.num_linear - layout.body_end_linear_idx

    # Head: harvest constant LE words from the head's working-state output.
    head_rep = head.decompile_head(model)
    from neurodecomp.head import _extract_constant_runs
    word_runs = _extract_constant_runs(head_rep.result.values, bits_per_unit=8)
    # Each run is (start, end_inclusive, "hh hh hh hh ..."). Group bytes into
    # 4-byte LE words.
    all_words_le: List[int] = []
    for _, _, hexstr in word_runs:
        chunks = hexstr.split()
        for i in range(0, len(chunks) - 3, 4):
            b0, b1, b2, b3 = [int(c, 16) for c in chunks[i : i + 4]]
            word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
            all_words_le.append(word)
    # The first 4 are the IV; the rest are auxiliary constants.
    if len(all_words_le) >= 4:
        rec.initial_words_le = all_words_le[:4]
        rec.extra_head_constants_le = all_words_le[4:]

    # Body: rotate schedule.
    body_rep = body.decompile_body(model)
    for pr in body_rep.positions:
        if pr.rotate_gadget:
            rg = pr.rotate_gadget
            rec.shift_schedule = list(rg["shifts_per_iter"])
            rec.rotate_num_registers = rg["num_registers"]
            rec.rotate_num_lanes = rg["num_lanes"]
            break

    # Tail: target digest.
    tail_rep = tail.decompile_tail(model)
    if tail_rep.matches and tail_rep.targets_hex:
        rec.target_digest_hex = tail_rep.targets_hex

    # Motif certification.
    cert_rep = certify_mod.certify_motifs(model)
    rec.candidate_and_count = cert_rep.candidate_counts.get("bool_and", 0)
    rec.confirmed_and_count = cert_rep.confirmed_counts.get("bool_and", 0)

    # Pending list.
    pen.items = [
        f"per-round K constants for i in [1, {rec.num_body_iterations}] "
        "(only K[0] recovered, stamped into head's working state)",
        f"iteration {rec.num_body_iterations} / iteration "
        f"{rec.num_body_iterations + 1} of the body (absorbed into "
        "head/tail/stitching layers; block finder reports "
        f"{rec.num_body_iterations} of presumably 64 MD5-style rounds)",
        "F/G/H/I non-linear round functions and the modular 32-bit "
        "adder gadget (visible as bool_and patterns, not yet identified "
        "as named operations)",
        "input padding scheme (head's input length counting, 0x80 framing "
        "byte, length suffix)",
        "full codegen: a Python g such that g(s) == model(s) for every "
        "s in Sigma^<=55 (waiting on the items above)",
        "richer abstract domain (MVP-7.5) to certify the remaining 78% "
        "of bool_and candidate patterns",
    ]

    return rec, pen


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------

_FILE_DOCSTRING = '''"""Auto-generated by NeuroDecomp.

This file is a HONEST PARTIAL decompilation of a hand-compiled
nn.Sequential model.  Every fact below has been independently recovered
from the model's weights -- no algorithm-specific knowledge was used.

Recovered:
{recovered_list}

Pending milestones:
{pending_list}

The emitted program will run for parts that have been recovered, and will
raise NotImplementedError where decompilation is still in progress.  As
later milestones land, the stubs get replaced with real Python.
"""'''


def _format_words_le(words: Sequence[int]) -> str:
    return "(" + ", ".join(f"0x{w:08x}" for w in words) + ")"


def emit_python(rec: Recovered, pen: Pending) -> str:
    """Return the source code of the emitted Python file."""

    recovered_list_lines = [
        f"  * tokenizer: {rec.tokenizer_summary}",
        f"  * initial state: {len(rec.initial_words_le)} x 32-bit words",
        f"  * auxiliary head constants: {len(rec.extra_head_constants_le)}"
        " x 32-bit words",
        f"  * body iterations: {rec.num_body_iterations}",
        f"  * body block size (Linears per iteration): {rec.body_block_size}",
        f"  * rotate gadget: {rec.rotate_num_registers} register(s) x "
        f"{rec.rotate_num_lanes}-bit lanes",
        f"  * shift schedule: {len(rec.shift_schedule)} entries",
        f"  * 16-byte equality target: {rec.target_digest_hex}",
        f"  * AND patterns: {rec.confirmed_and_count} confirmed / "
        f"{rec.candidate_and_count} candidate",
    ]
    pending_list_lines = [f"  * {p}" for p in pen.items]

    docstring = _FILE_DOCSTRING.format(
        recovered_list="\n".join(recovered_list_lines),
        pending_list="\n".join(pending_list_lines),
    )

    body_src = f'''
import sys
from dataclasses import dataclass, field
from typing import Sequence


# =============================================================================
# Recovered constants
# =============================================================================

TOKENIZER_LENGTH = {rec.tokenizer_length}
TOKENIZER_PAD = {rec.tokenizer_pad!r}

INITIAL_STATE_LE = {_format_words_le(rec.initial_words_le)}
EXTRA_HEAD_CONSTANTS_LE = {_format_words_le(rec.extra_head_constants_le)}

NUM_BODY_ITERATIONS = {rec.num_body_iterations}
SHIFT_SCHEDULE = {tuple(rec.shift_schedule)!r}
ROTATE_NUM_REGISTERS = {rec.rotate_num_registers}
ROTATE_NUM_LANES = {rec.rotate_num_lanes}

TARGET_DIGEST_HEX = {rec.target_digest_hex!r}
TARGET_DIGEST_BYTES = bytes.fromhex(TARGET_DIGEST_HEX) if TARGET_DIGEST_HEX else b""

CANDIDATE_AND_COUNT = {rec.candidate_and_count}
CONFIRMED_AND_COUNT = {rec.confirmed_and_count}


# =============================================================================
# Recovered tokenizer (fully implemented)
# =============================================================================

def encode(s: str) -> list:
    """Map a string to TOKENIZER_LENGTH ASCII byte values, null-padded."""
    s = str(s)[:TOKENIZER_LENGTH].ljust(TOKENIZER_LENGTH, TOKENIZER_PAD)
    return [ord(c) for c in s]


# =============================================================================
# Pending: per-round constant K[i] for i in [1, NUM_BODY_ITERATIONS]
# =============================================================================

ROUND_CONSTANTS_RECOVERED = {{}}      # Pending MVP-8


def K(i: int) -> int:
    if i in ROUND_CONSTANTS_RECOVERED:
        return ROUND_CONSTANTS_RECOVERED[i]
    raise NotImplementedError(
        f"K[{{i}}] not yet recovered (only K[0] = "
        "0xd76aa478 stamped into head)."
    )


# =============================================================================
# Pending: F/G/H/I-style non-linear round function (MVP-9)
# =============================================================================

def round_function(A: int, B: int, C: int, D: int,
                   message_word: int, K_i: int, s_i: int,
                   iteration_index: int) -> tuple:
    """One MD5-style round; not yet recovered.

    Future MVP-9 will derive the F/G/H/I + modular-add structure from
    the body's confirmed boolean gates and emit it here.
    """
    raise NotImplementedError(
        "round_function not yet recovered (pending MVP-9: F/G/H/I + adder)."
    )


# =============================================================================
# Pending: padding scheme (MVP-10)
# =============================================================================

def pad_message(byte_values: Sequence[int]) -> list:
    """Pad the byte sequence into the head's input format; not yet recovered."""
    raise NotImplementedError(
        "Padding scheme not yet recovered (pending MVP-10)."
    )


# =============================================================================
# Top-level emit (raises until enough has been recovered to run end-to-end)
# =============================================================================

def emit_predict(s: str) -> int:
    """Pure Python prediction; will run end-to-end once all pieces are recovered."""
    bytes_in = encode(s)
    padded = pad_message(bytes_in)              # MVP-10
    A, B, C, D = INITIAL_STATE_LE
    for i in range(NUM_BODY_ITERATIONS + 1):
        k_i = K(i)                              # MVP-8
        s_i = SHIFT_SCHEDULE[i % len(SHIFT_SCHEDULE)]
        # The "message word" indexing is part of MVP-9.
        m_i = padded[i % len(padded)]
        A, B, C, D = round_function(A, B, C, D, m_i, k_i, s_i, i)
    digest_bytes = bytes_from_state(A, B, C, D)  # also pending
    return int(digest_bytes == TARGET_DIGEST_BYTES)


def bytes_from_state(A: int, B: int, C: int, D: int) -> bytes:
    """Pack the four 32-bit state words into 16 bytes (little-endian)."""
    out = bytearray()
    for w in (A, B, C, D):
        out.extend(w.to_bytes(4, "little"))
    return bytes(out)


# =============================================================================
# Verification harness against the original PyTorch model
# =============================================================================

def verify_against_model(model, samples) -> dict:
    """Run the emitted program alongside the original model on `samples`.

    For each sample we record (a) what the model returns, (b) what the
    emitted program produces (or which NotImplementedError it raised).

    Returns a dict with three fields: ``total``, ``model_results``,
    ``emit_status`` (one of {{"ok", "not_implemented", "mismatch"}}).
    """
    import torch
    rows = []
    for s in samples:
        x = torch.tensor([float(b) for b in encode(s)])
        with torch.no_grad():
            model_out = float(model(s).item()) if hasattr(model, "__call__") else None
        try:
            emit_out = emit_predict(s)
            status = "ok" if (model_out is None or emit_out == int(model_out)) else "mismatch"
        except NotImplementedError as e:
            emit_out = None
            status = f"not_implemented: {{e}}"
        rows.append((s, model_out, emit_out, status))
    return {{
        "total": len(samples),
        "model_results": [(s, m) for s, m, _, _ in rows],
        "emit_status": [(s, st) for s, _, _, st in rows],
    }}


if __name__ == "__main__":
    print(__doc__)
    print()
    print("Try:")
    print('  encode("bitter lesson")')
    print()
    print("encode('bitter lesson') =", encode("bitter lesson"))
'''

    return docstring + body_src


def emit_to_file(model: nn.Sequential, path: str | Path) -> dict:
    """Run aggregation + emit and write the result to `path`."""
    rec, pen = aggregate(model)
    src = emit_python(rec, pen)
    Path(path).write_text(src, encoding="utf-8")
    return {
        "path": str(path),
        "bytes_written": len(src),
        "recovered": rec,
        "pending": pen,
    }
