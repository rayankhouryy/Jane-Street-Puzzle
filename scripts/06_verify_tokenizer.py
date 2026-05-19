"""Verify the recovered tokenizer reproduces the puzzle's behaviour."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
m = torch.load(ROOT / "model_3_11.pt", map_location="cpu", weights_only=False)


def encode(s: str) -> torch.Tensor:
    """Re-implementation of the puzzle's tokenizer."""
    s = str(s)[:55].ljust(55, "\x00")
    return torch.Tensor([ord(c) for c in s])


def predict_string(s: str) -> float:
    with torch.no_grad():
        return float(m.forward(encode(s)).item())


# Sanity: match the canonical example
print(f'f("vegetable dog") = {predict_string("vegetable dog")}')
print(f'f("") = {predict_string("")}')
print(f'f("a") = {predict_string("a")}')
print(f'f("the quick brown fox") = {predict_string("the quick brown fox")}')

# Compare with calling model directly on the string (using the cloudpickled path)
print()
print("direct via model('vegetable dog'):", float(m("vegetable dog").item()))
print("our encode-then-forward equals direct call:",
      predict_string("vegetable dog") == float(m("vegetable dog").item()))
