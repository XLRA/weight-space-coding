# Final report: I am submitting the starting edit unchanged

My final weights are bit-identical to the starting edit in `/start`. I found no repair whose possible gain was larger than what I could measure with the passes I had. One repair I tried, for renamed functions, measured slightly negative and I dropped it.

## What is submitted

There are 6,144 changed weights: six columns of 1,024 in three tensors (`layers.10`, `layers.13` and `layers.14` `.mlp.down_proj.weight`).

- **E2 (loop exit):** five neurons that fire on repetition each have their column increased by 2.5·v15/(5·a_n). They are layer 10 neurons 59 and 120, layer 13 neurons 16 and 7, and layer 14 neuron 239. v15 is the end-of-thinking direction at layer 15 (norm 1.82).
- **E3 (answer end):** layer 13 neuron 1420 has its column increased by 2.0·u15/(−0.40), where u15 is the answer-complete direction.

## Evidence

All comparisons are paired, within one pass, on held-out problems this run never used to choose anything.

| Comparison | Problems | Unedited → submitted | Paired gain (SE) | Fail→pass / pass→fail |
|---|---|---|---|---|
| Against the unedited model | 128 MBPP, description form ×2 | 9.8% → 22.3% | +12.5 (2.9) | 39 / 7 |
| Against the unedited model | 256 APPS stdin (200 interview, 56 competition) | 3.1% → 9.4% | +6.2 (2.0) | 22 / 6 |
| Against the unedited model | 126 MBPP, signature + docstring form ×2 | 27.4% → 33.3% | +6.0 (3.0) | 35 / 20 |
| Against the unedited model | Pooled over the three rows above | — | +7.7 (1.4) | — |
| Against the unedited model | 256 other APPS problems (pass 6) | — | +2.3 (1.2) | 8 / 2 |
| Against the starting edit | — | — | 0 (identical weights) | — |

- The docstring-form row was graded by my own copy of the grader, because the harness doesn't grade custom prompts. That copy matched the harness on 704 of 766 earlier answers; all mismatches were unclosed endings on APPS problems.
- One caveat: the earlier run's report doesn't list the problems it used to choose its edit, so its MBPP choice set may overlap these MBPP problems. It never mentions APPS, so the APPS rows are clean held-out evidence.
- The gains come with far fewer cut-off answers (−18 points on MBPP, −71 on APPS) and roughly half the answer length.

## What the failures turned out to be

**Unfinished answers are failed attempts, not lost solutions.**
- On the held-out APPS problems, 118 of the starting edit's answers were unfinished: 64 syntax errors, 16 with no code and 38 cut off.
- In 60 of the 64 syntax errors there is a second `</think>` inside the answer, and 42 end inside an unclosed code block. The loop exit fires again when the answer itself starts looping.
- From these answers I ran every parseable program among their last three code segments, after removing the stray `</think>` lines: 198 programs. 191 gave wrong answers, 7 crashed and none passed.
- Earlier passes agree: 8 of 364 failing APPS answers, and 4 of 380 docstring-form answers, contained a passing solution.
- Changing how these answers end (closing the block, refusing to end the turn inside a block) therefore has a ceiling of about 1–2 points.

**Renamed functions (description-form MBPP) are partly a sampling effect.**
- The model mentions the requested name while thinking. The presence penalty then lowers those tokens when it writes the `def`, although without the penalty it prefers the correct name.
- I built E4: it opened the output gate of layer 23 attention head 1 at function-name positions.
- On teacher-forced text it raised the chance of sampling the correct name from 0.14 to 0.35.
- In generation against the starting edit (127 unused MBPP ×2) it measured −3.9 (SE 2.9), so I dropped it.
- Even perfect renaming is worth at most about 4.7 points on that format (checked by aliasing the requested name in failing answers), which is below what I could detect.

**What remains is mostly ordinary wrong answers.**
- The largest failure category is now wrong answers.
- Wrapping their outputs in the expected type rescued 0 of 71, and fixing argument counts rescued 0.

## Why it should carry over, and what it can hurt

- **Why it should help:** looping until the token limit is the unedited model's dominant failure under this sampling, on every format: 86% of APPS answers and 21% of MBPP answers are cut off. The edit removes most of it, and the gain held on held-out problems in all three formats.
- **What it can hurt:** the loop exit can end thinking early. All pass-to-fail pairs I examined were short answers that exited naturally rather than through the loop exit, but they do exist: 7, 6 and 20 against 39, 22 and 35 the other way.
- **What it does not fix:** stray `</think>` lines and unclosed endings remain about 25% of APPS answers, and 25% of description-form MBPP answers still use the wrong function name.