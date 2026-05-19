"""Final verification: the model computes MD5 and checks against a fixed digest.

Adir's solution: the puzzle reduces to finding s such that
    MD5(s) == c7ef65233c40aa32c2b9ace37595fa7c
and the answer is "bitter lesson" (Rich Sutton's essay).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
m = torch.load(ROOT / "model_3_11.pt", map_location="cpu", weights_only=False)
m.eval()

TARGET = "c7ef65233c40aa32c2b9ace37595fa7c"

cases = [
    "bitter lesson",
    "vegetable dog",
    "",
    "hello",
    "BITTER LESSON",
    "bitter  lesson",
    "bitter_lesson",
    "the bitter lesson",
]

for s in cases:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    y = float(m(s).item())
    tick = "[OK]" if h == TARGET else "    "
    print(f"{tick}  f({s!r:25s}) = {y:.4f}   MD5 = {h}")
