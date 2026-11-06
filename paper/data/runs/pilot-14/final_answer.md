# Final report: making answers settle after the thinking closes (layer-23 MLP edit on top of the starting edit)

I submitted v4b: three repurposed neurons in the layer-23 MLP. Answers now cut at the token limit much less often and end with a final box much more often. Accuracy changes are within noise on every suite. All four parts of the starting edit are kept unchanged.

## What I changed

Everything is in the layer-23 MLP: `gate_proj` rows, `up_proj` rows and `down_proj` columns of neurons 2664, 1965 and 2621. That is 9,216 weights changed relative to the starting edit, in tensors the brief allows. The submitted /work/edited holds 9 tensors: the starting edit plus these changes (copy in /work/best_final).

- **Unchanged from the starting edit (verified identical):**
  - the loop-exit detectors in layers 10, 13 and 14;
  - E3, the answer-end neuron;
  - the end-of-turn brake, neuron 3109;
  - the function-name gates in the q_proj tensors of layers 15, 19 and 23.
- **Readout convention for all three neurons:**
  - They read the exact layer-23 MLP input, z = rmsnorm(X)·(1+g). I measured it in teacher-forced passes with the layer-23 down projection zeroed.
  - b is a fitted direction with b·z ≈ 1 ± 2%, which serves as a constant.
  - All pushes assume a final rms of 0.5.

| Neuron | Original | Gate row | Up row | Down column (push per unit) |
|---|---|---|---|---|
| 2664 (stop degenerate cycles) | near-dead, max \|act\| 0.28 | 100·(w_F − (a0+0.95(a1−a0))·b)/(a1−a0) | b | <\|im_end\|> +4, end-of-text +1, </think> −4; zero on 10 frequent tokens |
| 1965 (box the answer statement) | near-dead, max \|act\| ≈ 0.3 | 3·(w_A − 12.10·b) | +b | " $\\" +8, " " −3; zero on 21 protected tokens |
| 2621 (pair with 1965) | near-dead, max \|act\| ≈ 0.3 | 3·w_A − (3·12.10+1)·b | −b | same as 1965 |

- **w_F** is a shrinkage-LDA readout of the degenerate state: the thinking closed at least twice and the previous segment is under 60 tokens.
- **w_A** is a shrinkage-LDA readout of "answer statement after the thinking closed, value comes next", at the position after "answer is" or "Answer:" in math answers.
  - Its negatives were thinking statements, multiple-choice statements (letter and numeric), code statements and random positions.
  - The threshold is the 99.97th percentile of the negatives.
- **The 1965/2621 pair** gives a combined activation of silu(g) − silu(g−1). This is a step from 0 to about 1 that saturates.
- **Protected tokens** for the pair include " (", digits, ".", ":", newlines, </think>, <|im_end|> and the other box openers.

## Mechanism

- **Restarts come from the starting edit's own loop exit.**
  - When an answer loops, the E2 detectors push </think>, and neuron 3109 keeps <|im_end|> from winning.
  - At re-close points the teacher-forced log-probability of </think> is −1.1 with the starting edit against −10.8 with the original model at the 2nd close, and −0.95 against −12.6 at the 3rd.
  - After about 8 closes the cycle sustains itself by copying (original model −0.6).
  - The 2nd and 3rd closes are useful and hold many correct answers. Degenerate cycles almost never are: 1 of 618 start answers that reached that state ended with a correct box.
  - Neuron 2664 recognises the degenerate state and ends the turn. Held-out it caught 92–98% of these positions and never fired in thinking or in normal answers.
- **Restarted segments are written in thinking style.**
  - Their conclusions come out as prose without a box ("Okay, I'll stop.\nThe answer is 41."), which the grader counts as no answer.
  - Offline, boxing the first such statement would have added 12 correct answers and lost 2 out of 245.
  - The 1965/2621 pair fires at those positions. Cross-validated it caught 70% of them. It fired on 0.01% of random positions, 1% of numeric multiple-choice statements and 3% of code statements.
  - Teacher-forced on held-out text, it made " $\\" the top token at 14 of 14 statement positions and left multiple-choice "The answer is (" untouched.

## Evidence: paired differences against the starting edit

All numbers are edited minus start, one sample per item. Brackets give the standard error.

| Suite (problems not used to choose v4b) | Start | v4b | Difference |
|---|---|---|---|
| **Confirmation math (246 own problems, pass 17)** | | | |
| Boxed accuracy | 24.0 | 25.6 | +1.6 (2.6) |
| No final box | 52.4 | 37.8 | −14.6 (4.0) |
| Cut at token limit | 37.0 | 12.2 | −24.8 (3.6) |
| More than one </think> | 87.8 | 82.5 | −5.3 (2.8) |
| **APPS, 92 held-out (pass 18)** | | | |
| Pass rate | 4.3 | 5.4 | +1.1 (2.4) |
| Syntax errors (count) | 17 | 48 | |
| **MBPP not used to choose** | | | |
| 256 problems (pass 19) | 23.0 | 25.4 | +2.3 (2.8) |
| of which 197 never generated with an edit | 21.8 | 26.4 | +4.6 (3.1) |
| 96 problems (pass 18) | 33.3 | 22.9 | −10.4 (4.0) |
| **Instruction following, 64 held-out own prompts** | 60.9 | 65.6 | +4.7 (5.2) |
| **Multiple choice, 256 own questions** | 63.3 | 66.0 | +2.7 (2.2) |

- **Choice set (math A, 160):** accuracy −2.5 (2.3), no final box −11.2 (4.4), cut −19.4 (4.4).
- **APPS syntax errors:** 38 of the 48 are final blocks that span a </think>. These are stopped fence cycles that start mostly ran to the token limit, so they were failures either way. Cut answers across APPS and MBPP fell from 19.1% to 1.6%.
- **The MBPP −10.4 is drift, not the edit.**
  - No code answer contained an induced box.
  - The texts split from the v2 answers early inside the thinking, at ordinary tokens.
  - Start answers were byte-identical across passes.
  - Pooled over the three v4b MBPP comparisons (367 problems), the difference is about −1.9.
- **Instruction following:** cut answers went from 3 to 1; answers with more than one </think> stayed at 5.
- **Multiple choice:** answers with no extractable letter were unchanged (±0) and cut answers fell by 7.0 (2.5). Only 1 of 256 answers got an induced box, and it kept its letter.
- **Against v2 alone (neuron 2664 only), on the same math problems:** v2 gave accuracy −1.6 and no final box only −4.1. So the pair adds about 10 points of final boxes without costing accuracy.

## Why it should carry over, and what it might hurt

Both changes act on states the network itself represents, not on particular problems:

- the degenerate re-close cycle;
- a conclusion being written after the thinking has closed.

They were validated on held-out texts, with explicit negatives from the other suites.

What it might hurt:

- **Syntax-error counts rise** in code, because cycles that used to run to the limit now stop. Those answers were already failing.
- **Rare late recoveries** after a long degenerate cycle could be lost. This is bounded at about 0.2%.
- **The pair may box a value inside the thinking.** This is harmless: natural boxes in thinking never closed it early.
- **The pair may box a mid-derivation value that later changes.** 19 of 27 induced boxes were followed by more reasoning.
- **Rarely, the pair may box a numeric statement in a non-math answer.**
- **Not fixed:** loops whose lines keep changing within one segment, which still cut about 12% of math answers, and ordinary algorithmic mistakes.

The accuracy changes are within noise. The clear gains are fewer answers cut at the limit and many more answers ending with a final box.