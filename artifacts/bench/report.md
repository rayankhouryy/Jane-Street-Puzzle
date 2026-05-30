# Synthetic benchmark suite — results

Equivalence column: ✅ if model(x) == oracle(x) on the tested input set (exhaustive when ≤2^14 points, else 16k random samples seeded at 0).

| circuit | in | out | linears | params | equiv | placed | detected | confirmed | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| and1 | 2 | 1 | 2 | 5 | ✅ | bool_and:1 | bool_and:1 | bool_and:1 | 1 | 0 | 0 |
| not1 | 1 | 1 | 2 | 4 | ✅ | - | - | - | 0 | 0 | 0 |
| or1 | 2 | 1 | 3 | 15 | ✅ | bool_and:1 | bool_and:1 | bool_and:1 | 1 | 0 | 0 |
| xor1 | 2 | 1 | 3 | 15 | ✅ | bool_and:1, bool_xor:1 | bool_and:1, bool_xor:1 | bool_and:1, bool_xor:1 | 2 | 0 | 0 |
| mux1 | 3 | 1 | 5 | 41 | ✅ | bool_and:3 | bool_and:3 | bool_and:3 | 3 | 0 | 0 |
| and8 | 16 | 8 | 2 | 208 | ✅ | bool_and:8 | bool_and:8 | bool_and:8 | 8 | 0 | 0 |
| or8 | 16 | 8 | 3 | 680 | ✅ | bool_and:8 | bool_and:8 | bool_and:8 | 8 | 0 | 0 |
| xor8 | 16 | 8 | 3 | 680 | ✅ | bool_and:8, bool_xor:8 | bool_and:8, bool_xor:8 | bool_and:8, bool_xor:8 | 16 | 0 | 0 |

## Per-kind precision / recall (syntactic scanner)

| kind | placed | detected | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|---|
| bool_and | 30 | 30 | 30 | 0 | 0 | 1.000 | 1.000 |
| bool_xor | 9 | 9 | 9 | 0 | 0 | 1.000 | 1.000 |