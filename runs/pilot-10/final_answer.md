# Final report: two weight edits that let Qwen3.5-0.8B escape thinking loops and stop after its solution

The final edit is in `/work/edited` and is identical to the snapshot `/work/e2e3_g2`. On the 191 held-out MBPP problems, never used to choose anything, it adds +12.3 points of pass rate (SE 2.1). On the evaluation's second format, signature plus docstring, the gain is +4.0 points (SE 3.8), which is not significant.

## What I changed

Six columns of MLP `down_proj` were changed, 6 × 1,024 = 6,144 weights in three tensors (`layers.10/13/14.mlp.down_proj.weight`). Each changed column now has an added norm of 0.84–3.19; the original columns have norms of about 0.25.

**E2, loop exit.** Five repetition-detector neurons are wired to the model's own end-of-thinking state.
- **Neurons and typical in-loop activation a_n:** layer 10 neuron 59 (a_n = −0.453), layer 10 neuron 120 (−0.619), layer 13 neuron 16 (+1.085), layer 13 neuron 7 (−0.777), layer 14 neuron 239 (−0.761).
- **Change:** column n += 2.5 · v15 / (5 · a_n).
- **v15 (the end-of-thinking direction):** the mean residual at hidden index 15, taken at the newline just before `</think>`, minus the mean at the same token type elsewhere in thinking. It was measured on 30 thinking ends against 940 controls; its norm is 1.82.
- **Net effect:** when the five neurons fire at their typical loop level, together they add 2.5 · v15.

**E3, answer end.** Layer 13 neuron 1420, column += 2.0 · u15 / (−0.40).
- **u15 (the answer-complete direction):** the mean h15 at answer code-block closes that end the turn, minus closes that are followed by more text.
- **Where the neuron fires:** at 74% of answer closing fences, and on 1 of 120,460 thinking tokens.

## What goes wrong and why

**Diagnosis on 512 original answers to dev problems:**
- Pass rate is 9.0%; 69.5% of answers are cut at the token limit.
- 98% of the cut answers are verbatim loops, 93% of them inside thinking.
- Loops start after a median of about 1,600 tokens, with a period of about 180 tokens. They occur on 121 of 128 problems, and no answer escapes once a loop starts.

**How the loop locks in (teacher-forced passes):**
- Per-token log-probability rises from −0.38 before the loop to −0.01 by the third copy.
- Lock-in is carried by induction heads in the full-attention layers 15, 19 and 23. For example, head L15H7 puts 0.82 of its attention on the token after the previous copy.

**Why I did not damp those heads.** Zeroing them (E1) broke copying of the requested function name: entry-point token log-probability fell from −0.01 to −0.84. E1 was rejected.

**What E2 does instead.** The network already contains repetition detectors that are near zero on normal text and large inside loops. They are passive: they never trigger an exit. E2 connects them to the end-of-thinking direction.
- Measured at strength 1.5: normal text changes by at most 0.02 nats, and the chance of leaving a loop on each repetition rises from 0.02 to 0.45.
- The final strength is 2.5, chosen by extrapolating that dose-response (expected loop exit about 0.73 per repetition, premature exit in normal thinking about 0.04).

**What E3 fixes.** After E2, a major failure was extra usage or test blocks after the solution; the last block is the one graded.
- E3 raises the chance of ending after the first solution close from 0.004–0.18 to about 0.98.
- Positions before the solution stay at most 0.005, and thinking at most 0.02.

## Evidence

Paired differences, edited minus original, on held-out problems:

| Problems | Original → edited | Paired gain (SE) | Fail→pass / pass→fail |
|---|---|---|---|
| Pass 9: 128 MBPP × 2 | 9.4% → 19.1% | +9.8 (2.5) | 31 / 6 |
| Pass 10: 63 other MBPP × 2 | 11.1% → 28.6% | +17.5 (3.9) | 25 / 3 |
| All 191 held-out MBPP | — | +12.3 (2.1) | — |
| Docstring form of the pass-10 problems | 27.8% → 31.7% | +4.0 (3.8) | 16 / 11 |

- **Docstring grading:** these answers were scored with my own grader, which matched the official grader on 252/252 answers.
- **The 11 docstring pass→fail cases look like resampling noise.** The edited model reproduces 69% of the original's passing samples; the original reproduces its own only 57%.
- **Length:** answers cut at the limit drop from 55–64% to 4–8%, and mean length from about 21k to about 4–5k tokens.
- **On the dev problems used to choose the edit:** E2 alone gave +6.6 (SE 2.4); E2+E3 gave +15.2 (SE 2.8).

## Why it should carry over, and what it might hurt

**Why it should help on unseen problems:**
- Looping until the token limit is the dominant, content-independent failure, affecting about two thirds of answers across both formats.
- The detectors respond to verbatim repetition, not to problem content, and write only the model's own end-of-thinking direction. No token is boosted regardless of context, and thinking that does not loop is not cut.
- E3 acts only at answer code-block closes.

**Risks:**
- **Forced exits are mid-reasoning.** Only 13–22% of formerly cut answers become passes. The rest become wrong answers, or syntax errors (14–18% of exits happen inside an open code block), or the model resumes reasoning inside its answer.
- **Legitimately repetitive reasoning can be cut short.** Measured on teacher-forced text, premature exits in finished thinking roughly double, from 0.017 to an estimated 0.04.
- **E3 can stop an answer too early,** before a revision the model would have written. I found one genuine pass→fail of this kind on dev.
- **Not addressed:** renamed functions (about 7% of edited answers) and ordinary algorithmic wrong answers, which are now the largest failure category and show no single responsible component.