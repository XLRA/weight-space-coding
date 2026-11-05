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

# Run A

# Final report: the starting edit plus one repurposed layer-23 neuron, which stops prose answers from ending early

I kept the starting edit unchanged and added one layer-23 neuron. On 160 new instruction-following prompts it raises full instruction compliance from 36.2% to 54.4% against the starting edit. Math held up and code did not clearly fall, though a small code loss can't be ruled out. It also introduces one new failure mode.

## Final edit (in /work/edited, copy in /work/best_L)

9,216 weights differ from the original model.

- **Starting edit, unchanged (6,144 weights):**
  - Five loop-exit detector columns (E2): layer 10 down_proj columns 59 and 120, layer 13 columns 7 and 16, layer 14 column 239.
  - The answer-end column (E3): layer 13 column 1420.
- **New (3,072 weights): layer 23 MLP neuron 3109 repurposed.** This was the least-used neuron in my teacher-forced (TF) data (largest activation 0.24).
  - **Gate row:** 20 × (wR + (bR − 0.3)·wk), divided by the post-attention norm weight. wR and bR are a ridge readout r of the detectors' combined drive, fitted on 38 teacher-forced texts (held-out correlation 0.95). wk is a nearly constant direction that supplies the bias.
  - **Up row:** wk, divided by the same norm weight.
  - **Down column:** 0.8 × w_cancel. Per unit, w_cancel lowers `<|im_end|>` by 1.0 nat and `<|endoftext|>` by 0.67. It was built orthogonal to the `</think>` output direction and to the 200 most frequent tokens.
  - **Net effect:** the `<|im_end|>` logit drops by 16 × (r − 0.3) nats, and only where r exceeds 0.3. At premature stops (r about 0.85) that is about −9 nats.
  - Nothing upstream of the last MLP changes, so the loop exit's `</think>` route, the code-block answer end and the math answer end all work as in the starting edit.

## Mechanism

1. **The answers copy a draft.** The starting edit's prose answers mostly transcribe a draft written in the thinking: a median 75% of answer 8-grams appear in the thinking.
2. **Copying looks like a loop.** The verbatim repetition makes the five repetition detectors fire at loop level (drive about 1.0, against about 0 in non-copied prose, code and math answers).
3. **The push lands on end-of-turn.** In the answer, `</think>` is held down by the presence penalty, so the loop exit's push lands on `<|im_end|>` at the next sentence end. The turn ends mid-copy, so the closing quote, the postscript or the remaining paragraphs are never written. The correlation between copy fraction and words/N is −0.64.
4. **Causal tests, on the same tokens:**
   - Removing the detector writes moved the final `<|im_end|>` log-probability at the early stops from −1.74 to −12.16. Natural ends moved by ±0.03.
   - The response grows with detector drive: +0.04 nats at drive ≤ 0.25, −4.8 at 0.25–0.75, −8.8 at 0.75–1.25, and −11.6 above 1.25.
   - Removing the answer-end neuron instead changed nothing: +0.0 points (SE 3.7).
5. **The same push has two side effects.** It causes `<|im_end|>` leaks inside the thinking, which end the generation with no answer. It also re-closes wandering math answers with a second `</think>` and a fresh boxed answer, which is useful.
6. **Why the earlier candidates failed.** They suppressed the detectors in the answer phase (PF5C5, among others). That removed the useful re-close and cost math: pooled 69/72 under start against 60/72. So the final edit keeps the detectors and cancels only their `<|im_end|>` route.

## Evidence (last generation pass, paired against the starting edit, n = 1)

The decision rule was written to notes.md before this pass.

**Instruction following: 160 new prompts never used for any choice.**

| Measure | Starting edit | Final edit | Paired difference |
|---|---|---|---|
| All instructions followed | 36.2% | 54.4% | +18.1 (SE 4.4) |
| Mean per instruction | – | – | +17.5 (SE 4.0) |
| Closed its thinking | 95.0% | 98.8% | – |
| Median answer length | 124 words | 174 words | – |

- The final edit has 10 passes that rely only on degenerate repeated tails, against 3 for start. Counting all of them as failures, the gain is +13.8 (SE 4.2).
- Gains by instruction:
  - at least N words: 21 → 48%
  - quotation wrapping: 39 → 69%
  - at least N sentences: 59 → 84%
  - paragraph count: 38 → 62%
  - postscript: 78 → 87%
  - end phrase: 50 → 70%
- Losses, all on small groups:
  - no commas: 69 → 54% (n = 13)
  - sections: 100 → 71% (n = 7)
  - title: 100 → 85% (n = 13)

**Math: 48 problems with known answers.**
- Correct 44 → 46 (1 pass→fail, 3 fail→pass).
- Every answer from the final edit closes its thinking and contains a `\boxed{}` answer (start: 45 of 48).
- Median 10 characters follow the last box for both models; the maximum is 12 for the final edit.
- No math answer from the final edit reached the token limit (start: 1).

**Code: 48 held-out MBPP problems not used to choose or check the edit.**
- Pass rate 27.1% → 20.8%, a difference of −6.2 (SE 4.6), with 1 fail→pass and 4 pass→fail.
- 43 of the 48 pairs first diverge inside the thinking at ordinary tokens, such as "NOT" against "not". That is drift from the tiny per-token change of the last MLP, and 3 of those 4 discordant pairs went pass→fail.
- Only one pair, mbpp/532, diverges in the answer. There the start edit ended after a repetitive note, while the final edit wrote a wrong second code block.
- Earlier evidence on removing this push in code answers: −0.8 (SE 2.5) on 256 held-out problems, measured with PF5C5.

## Why it should transfer

Drafting in the thinking and then copying is generic to prose answers. The repair acts only where the repetition detectors are strongly active, and only on the end-of-turn token. Code answers are not copies (copy fraction about 0.1) and their ends read 0.05–0.16, below the 0.3 threshold.

## What it might hurt

- **New failure mode.** In a degenerate answer loop, the remaining `</think>` push can re-close the answer and then repeat `</think>` until the token limit. This hit 3 of 160 instruction prompts, all of which also failed under start, and no math or code answers.
- **Longer answers.** Answer loops that the starting edit used to end now run to the token limit: 7 of 160 instruction answers were cut against 4.
- **Code.** A small loss can't be excluded: the −6.2 is not significant, and one loss is attributable to the mechanism.
- **Multiple choice.** This suite is untested. I expect mainly fewer empty answers from thinking leaks.
---

# Run B

# Final report: wrong function names (wrong_interface) on MBPP-style problems

**Submitted (/work/edited, copy in /work/final_best_5):** the starting edit, unchanged, plus three rank-1 edits that open the output gates of attention heads which copy the given function name from the prompt. They act only at the positions where the model writes a function name after `def`.

On 149 MBPP problems never used to choose the edit, wrong_interface falls from 24.4% to 11.5% of answers. The pass rate is +4.7 points against the starting edit (SE 1.9, two independent paired passes). On HumanEval-style prompts the estimate is −1.4 (SE 2.0), which is not significant. On APPS it is +2.0 (SE 1.3).

## What changed

All changes are in the attention `q_proj` matrices. In this model, rows h·512+256 … h·512+511 of `q_proj` produce the sigmoid output gate of head h. On every gate row of a chosen head I added the same vector, strength × d, where d is a 1,024-dim direction. The gate logit of that head therefore rises by strength × (d · normalized attention input) wherever the direction fires.

| tensor | rows (gate of heads) | strength | direction | changed entries |
|---|---|---|---|---|
| layers.19.self_attn.q_proj.weight | heads 0, 2, 5, 6, 7 | 2.0 | /work/e5m_dir.npy | 1,310,720 |
| layers.15.self_attn.q_proj.weight | head 7 (rows 3840–4095) | 3.0 | /work/e6s_dir.npy | 262,144 |
| layers.23.self_attn.q_proj.weight | heads 0, 1 | 3.0 | /work/e7m_dir.npy | 524,288 |

- In total that is 2,097,152 changed entries, but only three 1,024-dim directions and three strengths.
- The starting edit's down_proj tensors (layers 10, 13, 14; 6,144 entries) are byte-identical to /start.
- My edits touch none of the starting edit's tensors, so they can be merged directly with the other run's answer-ending work.

## Mechanism

- **Where renames happen.** 74% appear at the first `def` in the thinking. Once the first def is renamed, 85% of answers end as wrong_interface.
- **The model itself prefers the variant.** I teacher-forced the model's own prefixes on 94 divergence points. It preferred the variant name in 74% of them (median log-p: given name −1.94, variant −0.63). The presence penalty adds on top.
- **Copy heads with mostly closed gates.**
  - At the `def` position, layer-19 heads 0, 2, 5, 6 and 7 attend to the name in the prompt's example test. Their output supplied +6.7 logits for the given name over the variant where the model kept the name, against only +1.7 where it renamed. Their gates are mostly closed (sigmoid about 0.2–0.5).
  - Layer-23 heads 0 and 1 copy the prompt name at `def` and continue it inside the name.
  - Layer-15 head 7 is an induction head: from each piece of the name it attends to the next piece in the prompt (0.78–0.96).
- **Detectors.** Each direction is a linear discriminant fit only on set-A texts. It fires on positions where a name piece is about to be written (mean 1.0) and stays near 0 elsewhere (SD about 0.1).
- **Three side effects found and removed while building it:**
  - Opening the gates at the last name token turned case renames into extensions. This is why the earlier run's E4 failed: there the heads look back over the name's own pieces. The detectors now treat the last name token as a negative.
  - Opening at mid-word positions produced stutter names such as `are_EEqual` (8 of 256 answers). Refitting with mid-word negatives brought this to 0 of 1,024 answers since.
  - Opening layer-15 head 7 at the last name token copied the prompt's merged `(<literal>` token, giving `def f([a], n)`. Restricting E6 to word-start positions inside the name fixed this: 0 such def lines in passes 16 and 19.

## Evidence

**Teacher-forced (259 fresh divergence points from the model's own answers).** Probability of sampling the given name:

| position | start | edit |
|---|---|---|
| first name token | 0.18 | 0.40 |
| inside the name | 0.41 | 0.92 |
| end of the name (`(` against an extension) | 0.33 | 0.40 |

- The estimated rate of renamed first defs falls from about 0.29 to about 0.12.
- Fewer than 0.5% of other answer tokens change log-probability by more than 0.5. The largest changes are other def names pulled toward the given name.
- Strength followed a dose-response: layer-19 strength 2 was chosen over 4, because 4 brought about ten times more off-target changes and no better rename rate.

**Generation, paired against the starting edit:**

| check | pass rate start → edit | paired diff (SE) | wrong_interface |
|---|---|---|---|
| MBPP set B, pass 16 (128 problems × 2) | 21.9 → 26.6 | +4.7 (2.6) | 63 → 30 |
| MBPP set B, pass 19 (fresh seeds; 21 never-used problems + 107) | 21.9 → 26.6 | +4.7 (2.7) | 62 → 29 |
| **MBPP set B pooled (1,024 answers)** | | **+4.7 (1.9)** | **125 → 59 (24.4% → 11.5%)** |
| MBPP set A (choice set, previous version of the edit), pass 13 | 19.9 → 23.8 | +3.9 (3.0) | 68 → 33 |
| HE-style untyped (set B), pass 17 | 34.0 → 34.0 | +0.0 (2.9) | 2 → 3 |
| HE-style typed (set A), pass 20 | 31.6 → 28.9 | −2.7 (2.8) | 4 → 2 |
| **HE-style pooled** | | **−1.4 (2.0)** | |
| APPS (34 intro + 94 interview × 2), pass 18 | 3.9 → 5.9 | +2.0 (1.3) | n/a |

- The two set-B passes are independent samples: no identical texts and a median common prefix of 15 characters. That their totals match exactly is a coincidence. Pass 19 was graded with my own grader, which agreed with the harness on all 512 answers of pass 16.
- Rename kinds on set B: case 23 → 0, descriptive renames 57 → 29, extensions 34 → 22, no def 11 → 8.
- No old side effect came back: stutter names 0, `def f(<literal>` 0, EOS inside the thinking 2 against 2.
- Length and loops show no consistent shift. On APPS, cut +3.9 (SE 3.1) and length +874 tokens (SE 740) are not significant. The APPS loops are fence repetitions after the answer, the same failure in both models, and NameErrors are 6 against 6.
- **Caveat.** No pass/fail outcome from set B was used to choose anything. However, older set-B answer texts were part of a teacher-forced diagnostic set used to check the literal-copy side effect.

## Why it should carry over, and what it might hurt

**Why it should carry over.** The repaired mechanism is generic: weak copying of the name shown only in the example test. The detectors fire on "a function name is being written after `def`", whatever the problem, and were fit on set-A problems only. Both held-out set-B passes gave the same gain.

**What it might hurt:**
- **HumanEval-style prompts.** The pooled estimate is −1.4 (SE 2.0). Signatures are always kept there, and the extra syntax errors are ordinary slips, so I found no mechanism. Still, a small cost from the detectors' small values at other positions cannot be ruled out.
- **Helper functions.** Helper names may be pulled toward the prompt's name. This is rare on MBPP, and APPS NameErrors did not change.
- **Hidden bugs.** Answers whose names are fixed reveal bugs that were hidden before, mainly TypeErrors from dropping length parameters.

**Not addressed:**
- descriptive renames of terse names, about 6% of answers;
- extensions, about 4%;
- a test-only final block, about 1.5%;
- arity mismatches, about 9% of answers, worth at most about 1.8 points.