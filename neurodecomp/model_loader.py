"""Model + tokenizer loading.

Hand-compiled puzzle models often attach a custom ``_call_impl`` to the
``Sequential`` instance via cloudpickle so that ``model("some string")`` works
directly. We recover that tokenizer so downstream stages can reason about the
true input domain (bytes, ASCII, etc.) instead of arbitrary floats.
"""
from __future__ import annotations

import dis
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class TokenizerInfo:
    """What we recovered about how strings become tensors."""

    callable_: Optional[Callable[[str], torch.Tensor]]
    width: int                      # padded input width (e.g. 55)
    pad_char: str                   # padding char (e.g. '\x00')
    byte_domain: tuple[int, int]    # min/max byte values (e.g. (0, 255))
    bytecode_dump: Optional[str] = None
    summary: str = ""


def load_model(path: str | Path) -> nn.Sequential:
    """Load a ``.pt`` model containing a Sequential.  Returns the Sequential."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, nn.Sequential):
        raise TypeError(f"expected nn.Sequential, got {type(obj).__name__}")
    obj.eval()
    return obj


def recover_tokenizer(model: nn.Module) -> TokenizerInfo:
    """Recover the cloudpickle-attached tokenizer, if any.

    Heuristically supports the puzzle's lambda

        lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:N].ljust(N, '\\x00')))))

    by disassembling its bytecode and extracting the constants ``N`` and the
    pad character.
    """
    ci = model.__dict__.get("_call_impl", None)
    fn = ci.__func__ if (ci is not None and hasattr(ci, "__func__")) else ci

    # Default: input is a raw float vector of length = first Linear's in_features.
    first_linear = next(m for m in model if isinstance(m, nn.Linear))
    fallback_width = first_linear.in_features

    if fn is None or not callable(fn) or not hasattr(fn, "__code__"):
        return TokenizerInfo(
            callable_=None,
            width=fallback_width,
            pad_char="\x00",
            byte_domain=(0, 255),
            summary="No custom tokenizer found; assuming raw float input.",
        )

    # Pull constants from the bytecode.
    code = fn.__code__
    consts = list(code.co_consts)
    buf = io.StringIO()
    dis.dis(fn, file=buf)
    disasm = buf.getvalue()

    # Width = max int in co_consts that looks like an ASCII width.
    int_consts = [c for c in consts if isinstance(c, int) and 1 <= c <= 4096]
    width = max(int_consts) if int_consts else fallback_width

    # Pad char = any 1-char string in co_consts.
    pad_candidates = [c for c in consts if isinstance(c, str) and len(c) == 1]
    pad_char = pad_candidates[0] if pad_candidates else "\x00"

    # Bag of names mentioned in the bytecode tells us what ops were used.
    uses_ord = "ord" in code.co_names
    uses_ljust = "ljust" in code.co_names
    uses_str = "str" in code.co_names
    uses_torch_tensor = "Tensor" in code.co_names

    if uses_ord and uses_ljust and uses_torch_tensor:
        def tok(s: str) -> torch.Tensor:
            s = str(s)[:width].ljust(width, pad_char)
            return torch.Tensor([ord(c) for c in s])

        summary = (
            f"Recovered ASCII tokenizer: ord(str(x)[:{width}]"
            f".ljust({width}, {pad_char!r}))"
        )
        byte_domain = (0, 255)
    else:
        tok = None
        summary = (
            f"Custom _call_impl detected (width={width}, pad={pad_char!r}) but "
            "structure not recognised; falling back to identity."
        )
        byte_domain = (0, 255)

    return TokenizerInfo(
        callable_=tok,
        width=width,
        pad_char=pad_char,
        byte_domain=byte_domain,
        bytecode_dump=disasm,
        summary=summary,
    )
