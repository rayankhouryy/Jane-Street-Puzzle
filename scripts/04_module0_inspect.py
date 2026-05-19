"""Inspect what the first module of the Sequential really is.

The HF Space calls model("vegetable dog") on a raw string, so the first module
cannot be a vanilla Linear. Either the Sequential's forward is overridden, or
m[0] only *looks* like a Linear but is actually a tokenizer-like callable.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
m = torch.load(ROOT / "model_3_11.pt", map_location="cpu", weights_only=False)

print(f"m type            : {type(m)}")
print(f"m forward qualname: {m.forward.__qualname__}")
print(f"m[0] type         : {type(m[0])}")
print(f"m[0] MRO          : {[c.__name__ for c in type(m[0]).__mro__]}")
print(f"m[0] forward      : {m[0].forward}")
print(f"m[0] forward qual : {m[0].forward.__qualname__}")
try:
    print("--- m[0].forward source ---")
    print(inspect.getsource(m[0].forward))
except Exception as exc:
    print(f"getsource failed: {exc}")

print()
print(f"m[0].__dict__ keys: {list(m[0].__dict__.keys())}")
print(f"m[0] dir (non-_)  : {[x for x in dir(m[0]) if not x.startswith('_')]}")

# Probe behaviour with a string.
print()
print('TEST: calling m[0]("hello")')
try:
    out = m[0]("hello")
    print(f"  out type   : {type(out)}")
    print(f"  out shape  : {getattr(out, 'shape', None)}")
    print(f"  out sample : {out[:20] if hasattr(out, '__getitem__') else out}")
except Exception as exc:
    print(f"  raised {type(exc).__name__}: {exc}")

# Probe with the canonical example.
print()
print('TEST: calling m("vegetable dog")')
try:
    out = m("vegetable dog")
    print(f"  out type : {type(out)}")
    print(f"  out value: {out}")
except Exception as exc:
    print(f"  raised {type(exc).__name__}: {exc}")
