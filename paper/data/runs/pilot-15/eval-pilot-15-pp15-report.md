# Held-out evaluation: pilot-15-pp15

Submission /opt/wsp/runs/pilot-15/submission (12 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5080.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 31.1% | 2.9% | 6.2% | 3,585 |
| mbpp+ | edited | 1512 | 32.5% | 2.6% | 5.4% | 3,544 |
| humaneval+ | edited | 656 | 28.1% | 3.4% | 7.9% | 3,680 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.08 points | 1.2 | 542 | 386 | 124 |
| mbpp+ | +16.34 points | 1.47 | 378 | 315 | 68 |
| humaneval+ | +2.29 points | 1.81 | 164 | 71 | 56 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 750, pass 675, runtime_error 343, wrong_interface 194, syntax_error 127, cut_at_max_tokens 62, timeout 11, no_code 6

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 42.7% | 1.5% | 1.1% | 8,248 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.20 points | 1.42 | 1000 | 112 | 90 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 545, pass 427, cut_at_max_tokens 15, no_answer 13

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 60.5% | 4.6% | 8.0% | 5,072 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +38.26 points | 1.84 | 541 | 457 | 43 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 655, not_followed 367, cut_at_max_tokens 47, no_answer 13; instruction-level accuracy 66.8%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.4% | 37.4% | 37.0% | 18,893 |
| easy | edited | 172 | 30.2% | 23.8% | 26.2% | 14,565 |
| medium | edited | 208 | 0.0% | 44.7% | 42.8% | 20,506 |
| hard | edited | 320 | 0.0% | 40.0% | 39.1% | 20,171 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.86 points | 1.57 | 175 | 42 | 1 |
| easy | +24.42 points | 5.49 | 43 | 42 | 0 |
| medium | -0.48 points | 0.48 | 52 | 0 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: cut_at_max_tokens 262, syntax_error 204, wrong_answer 156, pass 52, wrong_interface 12, runtime_error 12, timeout 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 55.0% | 12.8% | 9.2% | 15,342 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +48.15 points | 1.81 | 500 | 964 | 1 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1099, wrong_answer 382, no_answer 262, cut_at_max_tokens 257
