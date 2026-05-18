"""Architecture summary: load model, dump widths, head/body/tail split, period.

Outputs:
  artifacts/widths.json
  artifacts/arch_summary.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model_3_11.pt"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def find_period(widths: list[int], min_p: int = 2, max_p: int = 200) -> tuple[int, float]:
    """Return (period, fraction_of_matches) for the most-matching period."""
    n = len(widths)
    best = (0, 0.0)
    for p in range(min_p, max_p):
        matches = sum(1 for i in range(n - p) if widths[i] == widths[i + p])
        ratio = matches / (n - p)
        if ratio > best[1]:
            best = (p, ratio)
        if ratio > 0.97:
            break
    return best


def longest_periodic_run(widths: list[int], period: int) -> tuple[int, int]:
    """Find the longest contiguous range [s, e) where widths[i]==widths[i+period]."""
    n = len(widths)
    runs: list[tuple[int, int]] = []
    cur_s: int | None = None
    for i in range(n - period):
        if widths[i] == widths[i + period]:
            if cur_s is None:
                cur_s = i
        else:
            if cur_s is not None:
                runs.append((cur_s, i))
                cur_s = None
    if cur_s is not None:
        runs.append((cur_s, n - period))
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return runs[0]


def main() -> None:
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    widths = [linears[0].in_features] + [l.out_features for l in linears]

    period, ratio = find_period(widths)
    body_s, body_e = longest_periodic_run(widths, period)

    summary = {
        "num_linear": len(linears),
        "input_dim": widths[0],
        "output_dim": widths[-1],
        "min_width": min(widths),
        "max_width": max(widths),
        "period": period,
        "period_match_ratio": ratio,
        "body_start_idx": body_s,
        "body_end_idx_inclusive": body_e + period,
        "head_widths": widths[: body_s + 1],
        "one_period_widths": widths[body_s : body_s + period + 1],
        "tail_widths": widths[body_e + period :],
    }

    (ART / "widths.json").write_text(json.dumps(widths))
    (ART / "arch_summary.json").write_text(json.dumps(summary, indent=2))

    txt = ART / "arch_summary.txt"
    with txt.open("w") as f:
        f.write(f"num Linear        : {summary['num_linear']}\n")
        f.write(f"input/output dim  : {summary['input_dim']} -> {summary['output_dim']}\n")
        f.write(f"min/max width     : {summary['min_width']} / {summary['max_width']}\n")
        f.write(f"period            : {summary['period']} (match ratio {ratio:.4f})\n")
        f.write(f"body range        : [{body_s}, {body_e + period}) (linear-layer indices)\n")
        f.write(f"num iterations    : {(body_e - body_s) / period:.3f}\n\n")
        f.write(f"head widths : {summary['head_widths']}\n\n")
        f.write(f"one period  : {summary['one_period_widths']}\n\n")
        f.write(f"tail widths : {summary['tail_widths']}\n")
    print(txt.read_text())


if __name__ == "__main__":
    main()
