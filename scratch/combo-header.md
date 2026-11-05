# The starting edit: two runs' edits combined

This edit is the union of two runs' final edits. Both runs started from the same earlier edit (E2 and E3: three mlp.down_proj tensors of layers 10, 13 and 14, 6,144 changed weights) and kept it byte-identical. No tensor was changed by both runs, so combining them needed no arithmetic: each tensor comes from the run that changed it.

| tensor | from | what it does |
|---|---|---|
| layers.10.mlp.down_proj.weight | the earlier edit | E2: loop-exit detector columns 59 and 120 |
| layers.13.mlp.down_proj.weight | the earlier edit | E2: detector columns 7 and 16; E3: answer-end column 1420 |
| layers.14.mlp.down_proj.weight | the earlier edit | E2: detector column 239 |
| layers.23.mlp.gate_proj / up_proj / down_proj.weight | run A (first report below) | neuron 3109: cancels the detectors' push on <\|im_end\|> where they fire strongly |
| layers.15 / 19 / 23.self_attn.q_proj.weight | run B (second report below) | output-gate edits of name-copying heads, active where a function name is written after "def" |

The two edits act in different places (run A in the answer's end-of-turn token, run B at function names after "def"), and a held-out evaluation of the combination found each one's gains kept: run A's on prose and math answers, run B's on function-writing problems.

Run A's report and run B's report follow, unchanged. Where they say "the starting edit" or "/start", they mean the earlier edit (E2 and E3), not this combination.

---

