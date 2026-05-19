"""Batched search for non-zero outputs.

Per-forward overhead is dominated by 5442 Python-level module dispatches, so we
batch heavily: each forward processes 1024+ candidates in ~0.5 s.
"""
from __future__ import annotations

import random
import string
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
m = torch.load(ROOT / "model_3_11.pt", map_location="cpu", weights_only=False)
m.eval()


def encode_batch(strings: list[str]) -> torch.Tensor:
    out = torch.zeros(len(strings), 55)
    for i, s in enumerate(strings):
        s = str(s)[:55].ljust(55, "\x00")
        for j, ch in enumerate(s):
            out[i, j] = ord(ch)
    return out


@torch.no_grad()
def predict_batch(strings: list[str]) -> torch.Tensor:
    return m.forward(encode_batch(strings)).squeeze(-1)


@torch.no_grad()
def call_f_pre_relu_batch(x: torch.Tensor) -> torch.Tensor:
    out = x
    for i, mod in enumerate(m):
        if i == len(m) - 1:
            break
        out = mod(out)
    return out.squeeze(-1)


# ---------------------------------------------------------------------------
# (A) Structured sweeps (cheap).
# ---------------------------------------------------------------------------
print("=== STRUCTURED SWEEPS ===")
samples: list[str] = []
samples += list(string.printable)
samples += [ch * L for ch in string.ascii_lowercase + string.digits + " " for L in (1, 2, 5, 10, 20, 55)]
samples += [
    "hello", "world", "the", "a", "an", "and", "or", "yes", "no",
    "puzzle", "model", "tensor", "neural", "math", "code",
    "jane street", "vegetable dog", "vegetable", "dog",
    string.ascii_lowercase, string.ascii_uppercase, string.digits,
    " ".join(string.ascii_lowercase),
    "the quick brown fox", "lorem ipsum dolor sit amet",
    "0", "1", "2", "9", "00", "99",
]
t = time.time()
ys = predict_batch(samples)
nz = [(samples[i], float(ys[i])) for i in range(len(samples)) if float(ys[i]) > 0]
print(f"sweeps: {len(samples)} samples in {time.time()-t:.2f}s, non-zero: {len(nz)}")
for s, y in nz[:20]:
    print(f"  {s!r:50s} -> {y}")

# ---------------------------------------------------------------------------
# (B) Random brute-force.
# ---------------------------------------------------------------------------
print()
print("=== RANDOM SEARCH (printable ASCII) ===")
random.seed(0)
chars = string.ascii_lowercase + " " + string.digits
best = ("", 0.0)
TOTAL = 50_000
BATCH = 1024
hits: list[tuple[str, float]] = []
t = time.time()
for start in range(0, TOTAL, BATCH):
    batch = []
    for _ in range(BATCH):
        L = random.randint(1, 55)
        batch.append("".join(random.choice(chars) for _ in range(L)))
    ys = predict_batch(batch)
    mx = float(ys.max())
    if mx > best[1]:
        i = int(ys.argmax())
        best = (batch[i], mx)
        print(f"  trial {start + BATCH:6d} new best: {batch[i]!r} -> {mx}")
    for i in range(BATCH):
        if float(ys[i]) > 0:
            hits.append((batch[i], float(ys[i])))
print(f"random search: {TOTAL} samples in {time.time()-t:.1f}s, non-zero hits: {len(hits)}")
print(f"best random: {best}")

# ---------------------------------------------------------------------------
# (C) Gradient ascent on continuous input.
# ---------------------------------------------------------------------------
print()
print("=== GRADIENT ASCENT ===")
torch.manual_seed(0)
best_grad = ("", -1e9)
N_RESTARTS = 8
STEPS = 200
for trial in range(N_RESTARTS):
    x = torch.empty(55).uniform_(32.0, 126.0).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=2.0)
    for step in range(STEPS):
        opt.zero_grad()
        out = x
        for i, mod in enumerate(m):
            if i == len(m) - 1:    # skip final ReLU
                break
            out = mod(out)
        loss = -out
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0.0, 127.0)
    with torch.no_grad():
        y_pre = float(call_f_pre_relu_batch(x.unsqueeze(0)).item())
    rounded = x.detach().round().clamp(0, 127).int().tolist()
    s = "".join(chr(v) if 32 <= v <= 126 else f"\\x{v:02x}" for v in rounded)
    y_round = float(predict_batch([s])[0])
    print(f"  trial {trial}: pre={y_pre:+8.3f}  rounded f={y_round}  s={s!r}")
    if y_pre > best_grad[1]:
        best_grad = (s, y_pre)

print(f"\nbest gradient: {best_grad}")
