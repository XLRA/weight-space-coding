# Held-out evaluation: pilot-12-pp15

Submission /opt/wsp/runs/pilot-12/submission (6 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 27.7% | 2.8% | 5.7% | 3,487 |
| mbpp+ | edited | 1512 | 27.8% | 3.0% | 4.8% | 3,567 |
| humaneval+ | edited | 656 | 27.6% | 2.3% | 7.6% | 3,301 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +8.67 points | 1.07 | 542 | 327 | 139 |
| mbpp+ | +11.64 points | 1.31 | 378 | 257 | 81 |
| humaneval+ | +1.83 points | 1.72 | 164 | 70 | 58 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 693, pass 601, wrong_interface 409, runtime_error 297, syntax_error 92, cut_at_max_tokens 61, timeout 8, no_code 7

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 43.8% | 1.1% | 0.8% | 8,472 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +3.30 points | 1.49 | 1000 | 128 | 95 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 544, pass 438, cut_at_max_tokens 10, no_answer 8

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 59.4% | 4.1% | 8.0% | 5,159 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +37.15 points | 1.86 | 541 | 460 | 58 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 643, not_followed 384, cut_at_max_tokens 39, no_answer 16; instruction-level accuracy 67.8%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.6% | 25.0% | 24.3% | 15,981 |
| easy | edited | 172 | 30.8% | 15.7% | 16.9% | 12,589 |
| medium | edited | 208 | 0.0% | 31.2% | 29.3% | 17,207 |
| hard | edited | 320 | 0.0% | 25.9% | 25.0% | 17,008 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +6.00 points | 1.57 | 175 | 45 | 3 |
| easy | +25.00 points | 5.46 | 43 | 45 | 2 |
| medium | -0.48 points | 0.48 | 52 | 0 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 234, wrong_answer 208, cut_at_max_tokens 175, pass 53, runtime_error 17, wrong_interface 11, timeout 1, no_code 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 55.4% | 14.2% | 10.2% | 15,720 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +48.60 points | 1.79 | 500 | 973 | 1 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1108, wrong_answer 354, cut_at_max_tokens 283, no_answer 255
