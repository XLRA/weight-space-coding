# Final report: a loop exit on top of the GRPO-trained model (submitted edit "v4")

The edit closes the thinking when the model is stuck in a loop, and stays silent everywhere else. Truncated answers on practice code problems fell from 18.5% to 7.4% and pass rate held (+0.9, SE 1.7). Instruction following rose by about 3 points. Two side effects are real: in math, answers that close the thinking twice rose by +3.9 points, and syntax errors on APPS-style problems rose by +3.6.

## What I changed
Eight tensors in layers 12–13: 669,696 values differ from the trained model. /work/edited is identical to /work/cand_v4.

**1. A "thinking has closed" flag** (layer 12, `linear_attn` head 7, the head with the smallest output). I rebuilt this head as a write-once memory:
- `in_proj_qkv` rows of head 7 (q, k and v blocks, 3×128 rows): a constant direction.
- `in_proj_b[7]`: a detector that makes the head write only at the `</think>` token (logit about +70 there, at most −51 elsewhere).
- `in_proj_a[7]`: set so the memory never decays.
- `in_proj_z` rows 896–1023: output gate held open.
- `out_proj` columns 896–1023: after the first `</think>`, the head writes a flag of size 0.193 along one fixed direction.

**2. The exit unit** (layer 13 MLP, quiet neurons 2369 and 2705; `gate_proj`/`up_proj` rows and `down_proj` columns).
- **Gate:** a step function of a linear loop probe read from hidden layers 13 and 14. It switches on between probe scores 12 and 27.
- **Up projection:** reads the flag. It is about 1 while the thinking is open and about −0.2 after `</think>`, so the unit cannot act in answers.
- **Down projection:** adds a push of size 1.5 along the model's own "about to close the thinking" direction.

**3. A flag eraser** (layer 13 MLP, neurons 1527 and 3493). It removes the flag from the residual stream after layer 13, so later layers never see it.

## What goes wrong in a loop
- **The model never reaches its own exit state.** All 35 natural closes come right after a newline, with median P(`</think>`) 0.70 there. At loop newlines that probability is about 0.0000. Along the natural-exit direction, loop newlines sit at −0.06 and natural exits at +1.14.
- **The loop is readable mid-network.** A linear probe on hidden layers 13–14 separates loop text from normal text (held-out AUC 0.994–0.998) and rises within about 300 tokens of a loop's start. Non-loop text peaked at a score of 2.9; loops have median score 64.
- **The push has a clear dose-response.** In teacher-forced loop text, P(`</think>`) at loop newlines rises with the size of the push:

| push along the exit direction | <0.1 | 0.3–0.6 | 0.6–0.9 | 0.9–1.2 | ≥1.2 |
|---|---|---|---|---|---|
| P(`</think>`) at loop newlines | 0.0007 | 0.02 | 0.11 | 0.24 | 0.64 |

  The earlier edit E pushed about 4–5, far beyond this natural scale.
- **Without the flag, the exit fires again inside answers.** In loop-prone items, answers with two or more `</think>` went 0 → 8 (APPS) and 9 → 17 (math). With the flag they were 0 → 0 and 9 → 9. A memory was needed because:
  - the model's own "answer mode" signal fails in degenerate answers (AUC 0.84–0.86);
  - no attention head looks back at `</think>`;
  - repurposing a full-attention head cost −0.011 nats per token, against −0.0019 for the linear-attention head I used.
- **The exit is specific:**
  - The unit never fired on any of the 8 natural closes after enumerations.
  - It never fires in natural answers.
  - It never fired on long finished math (17–21k tokens).
  - v4 keeps as many correct math answers over 16k tokens as the trained model (68 vs 62).

## Paired differences against the trained model
Every row is v4 minus the trained model on the same items; numbers in parentheses are standard errors.

| Set | Pass / accuracy | Cut at the limit | Other | 2+ `</think>` |
|---|---|---|---|---|
| Practice never used to choose (80 problems, two passes, 480 answers per model) | 28.3 → 29.2 (+0.8, 2.1) | 11.0 → 4.0 (−7.1, 1.8) | — | 0.4 → 0.4 (+0.0, 0.4) |
| Practice B (111 problems)¹ | 17.1 → 18.0 (+0.9, 2.6) | 23.9 → 9.9 (−14.0, 3.2) | — | 0.5 → 1.4 (+0.9, 1.1) |
| All 191 practice problems | +0.9 (1.7) | −11.1 (2.0) | syntax errors 1.4 → 3.6 (+2.2, 1.1) | +0.5 (0.7) |
| Math confirmation set (122 × 2) | 25.4 → 23.4 (−2.0, 2.5) | 44.7 → 38.9 (−5.7, 3.7) | no final box 44.7 → 38.9 (−5.7, 3.7) | 5.7 → 11.5 (+5.7, 2.5) |
| Fresh math (119 × 2) | 11.8 → 13.4 (+1.7, 2.1) | 54.2 → 49.2 (−5.0, 4.0) | no final box 54.6 → 47.9 (−6.7, 4.0) | 9.2 → 11.3 (+2.1, 2.8) |
| Own math, 3 sets pooled | −0.7 (1.4) | — | — | +3.9 (1.4) |
| Instruction following B (80 × 3) | 72.9 → 76.7 (+3.8, 3.4) | 9.2 → 3.8 (−5.4, 2.1) | — | 0 → 0 |
| Fresh instruction following (80 × 3) | 63.7 → 68.3 (+4.6, 2.7)² | 12.5 → 5.4 (−7.1, 2.0) | — | 0 → 0 |
| Panel: choice | +0.0 (2.6) | — | — | — |
| Panel: math | −0.4 (2.4) | 18.8 → 17.6 | — | — |
| Panel: instructions | +2.0 (2.5) | 5.6 → 2.0 | — | — |
| Panel: code | +0.8 (2.9) | 2.0 → 1.2 | — | — |
| Panel: mean of suites | +0.6 (1.3) | — | — | — |

¹ Practice B was run once with an earlier version (v3) before I added the eraser, so it is only partly held out.
² In the half of the prompts designed to invite legitimate repetition: +7.5 (4.2).

## Expected effect and risks
**Expected gains:**
- **Competitive programming:** answers there run to the limit 33% of the time, so I expect far fewer truncated answers.
- **Pass rate:** only a small gain, because exits on hard problems rarely become correct. Of 46 previously truncated APPS answers that now closed, none passed.
- **Instruction following:** I expect gains of about 3 points.

**Risks:**
- **Math re-closing (+3.9, SE 1.4).** This is not driven by the exit: in 11 of 14 re-closing answers I sampled, the unit never fired during the thinking. Teacher-forced checks show v4 neither raises P(`</think>`) in answers nor weakens the answer-mode state. The most likely cause is the diffuse cost of giving up layer 12's head 7 (−0.0019 nats per token).
- **APPS syntax errors (+3.6, SE 1.2).** The common patterns are dangling try/except blocks and unterminated strings. They didn't lower the APPS pass rate (+0.2), and I couldn't trace them to a specific component.
- **Loops the exit misses.** Loops with no newlines (numbers or prose) can't exit, because a close only follows a newline.