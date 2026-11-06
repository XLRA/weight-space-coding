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