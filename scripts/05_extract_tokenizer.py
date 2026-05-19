"""Extract the custom _call_impl that tokenises the input string.

This callable is stored as `m.__dict__['_call_impl']`. It must convert a string
to a 55-dim Tensor before passing it to the Sequential's first Linear.
"""
from __future__ import annotations

import dis
import inspect
import io
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
m = torch.load(ROOT / "model_3_11.pt", map_location="cpu", weights_only=False)

ci = m.__dict__["_call_impl"]
print(f"_call_impl type: {type(ci)}")
print(f"_call_impl repr: {ci}")

# It's likely a function or bound method
fn = ci.__func__ if hasattr(ci, "__func__") else ci
print(f"underlying fn: {fn}")
print(f"qualname: {getattr(fn, '__qualname__', '?')}")
print(f"module  : {getattr(fn, '__module__', '?')}")

# Try inspect.getsource — usually fails for cloudpickled funcs, fall back to dis.
try:
    src = inspect.getsource(fn)
    print("--- source ---")
    print(src)
    (ART / "tokenizer_source.py").write_text(src, encoding="utf-8")
except (OSError, TypeError) as exc:
    print(f"getsource failed ({exc}); disassembling instead")

# Disassemble bytecode
buf = io.StringIO()
dis.dis(fn, file=buf)
disasm = buf.getvalue()
print("--- bytecode ---")
print(disasm[:6000])
(ART / "tokenizer_bytecode.txt").write_text(disasm, encoding="utf-8")

# Code object constants & names (often contain vocabulary / regex / etc.)
code = fn.__code__
print()
print(f"co_filename : {code.co_filename}")
print(f"co_name     : {code.co_name}")
print(f"co_varnames : {code.co_varnames}")
print(f"co_freevars : {code.co_freevars}")
print(f"co_cellvars : {code.co_cellvars}")
print(f"co_names    : {code.co_names}")
print()
print("co_consts (truncated repr):")
for i, c in enumerate(code.co_consts):
    r = repr(c)
    print(f"  [{i}] {type(c).__name__}: {r[:300]}")

# Closure variables
if fn.__closure__:
    print()
    print("--- closure ---")
    for i, cell in enumerate(fn.__closure__):
        try:
            v = cell.cell_contents
            r = repr(v)
            print(f"  closure[{i}] ({fn.__code__.co_freevars[i] if i < len(fn.__code__.co_freevars) else '?'}): "
                  f"{type(v).__name__} = {r[:400]}")
        except Exception as exc:
            print(f"  closure[{i}] error: {exc}")

# Globals referenced
print()
print("--- relevant globals ---")
for name in code.co_names:
    if name in fn.__globals__:
        v = fn.__globals__[name]
        print(f"  {name}: {type(v).__name__} = {repr(v)[:200]}")
