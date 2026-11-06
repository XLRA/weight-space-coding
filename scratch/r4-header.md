# The starting edit: an earlier run's final edit

This edit is the final edit of run C. Run C started from the combination of two runs' edits (runs A and B), and both of those were built on a third run's edit (E2 and E3).

After this header come two reports:
- run C's own final report;
- then the report of run C's starting edit, the combination, which contains runs A's and B's reports.

How to read "the starting edit" in the reports below:
- In run C's report, it means the combination.
- In the combination's reports, it means E2 and E3.
- For you, the starting edit is run C's final edit: all of the below together.

| tensor | from | what it does |
|---|---|---|
| layers.10.mlp.down_proj.weight | E2 | loop-exit detector columns 59 and 120 |
| layers.13.mlp.down_proj.weight | E2, E3 | detector columns 7 and 16; answer-end column 1420 |
| layers.14.mlp.down_proj.weight | E2 | detector column 239 |
| layers.23.mlp.gate_proj / up_proj / down_proj.weight | run A, run C | neuron 3109: the end-of-turn brake (run A). Neuron 2664: the degenerate-cycle stopper. Neurons 1965 and 2621: box the answer statement (both run C). |
| layers.15 / 19 / 23.self_attn.q_proj.weight | run B | function-name gates: rank-1 edits of name-copying heads' output gates, active where a function name is written after "def" |

Run C's final edit was evaluated on held-out suites you do not have, with this run's sampling settings. Paired against its starting edit (the combination), with standard errors in brackets:

| Suite | Combination | Run C | Paired diff (SE) | Cut at token limit | No answer |
|---|---|---|---|---|---|
| Competition math | 54.1% | 56.9% | +2.75 (0.93) | 298 → 90 of 2,000 | 250 → 289 |
| Multiple choice | 42.9% | 45.2% | +2.30 (1.33) | 13 → 3 of 1,000 | 6 → 20 |
| Instruction following (prompt-level) | 60.8% | 58.2% | −2.59 (1.60) | 63 → 29 of 1,082 | 8 → 44 |
| Code benchmarks | 32.6% | 32.3% | −0.28 (0.96) | 67 → 39 | |
| Code, HumanEval-style prompts only | 29.6% | 26.4% | −3.20 (1.61) | | |
| Competitive programming | 7.4% | 7.0% | −0.43 (0.82) | 247 → 19 of 700 | |

On competitive programming, the answers that used to be cut became syntax errors (213 → 389), so the pass rate did not change.

---

# Run C's report

