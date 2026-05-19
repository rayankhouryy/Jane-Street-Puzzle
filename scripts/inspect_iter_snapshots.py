"""Deep inspection of one iteration's intermediate snapshots."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import Counter

import torch
from neurodecomp import block_finder, model_loader, round_decode

m = model_loader.load_model("model_3_11.pt")
layout = block_finder.find_blocks(m)
subnet = round_decode.extract_iteration_subnet(
    m, 16, layout.block_starts, layout.period,
)
relus = round_decode.list_intermediate_linears(subnet)
print(f"iteration 16 subnet: {len(list(subnet))} children, {len(relus)} ReLUs")
print()

# Promising late ReLU snapshots — probe a wider channel range there.
for relu_idx in [55, 61, 67, 73]:
    sub = round_decode.truncate_subnet(subnet, relu_idx + 1)
    with torch.no_grad():
        test_out = sub(torch.zeros(round_decode.find_iteration_input_dim(subnet)))
    out_dim = test_out.shape[0]
    # Probe 4 different channel windows.
    print(f"ReLU child#{relu_idx:3d}  (out_dim={out_dim})")
    for start in [0, 64, 128, 192, 256]:
        if start >= out_dim:
            break
        end = min(start + 64, out_dim)
        rep = round_decode.decode_iteration(
            sub, iteration_index=-1,
            output_indices=list(range(start, end)),
            max_arity=4, num_probes=4,
        )
        arity_counts: Counter = Counter()
        name_counts: Counter = Counter()
        for bf in rep.bit_functions:
            if not bf.deps:
                arity_counts[0] += 1
            elif bf.table is None:
                arity_counts[">4"] += 1
            else:
                arity_counts[len(bf.deps)] += 1
                name_counts[bf.name] += 1
        top = name_counts.most_common(4)
        named = ", ".join(f"{n}({c})" for n, c in top if n != "unknown") or "-"
        print(f"  channels [{start:3d}..{end:3d}): "
              f"arity={dict(arity_counts)}  named={named}")
    print()
