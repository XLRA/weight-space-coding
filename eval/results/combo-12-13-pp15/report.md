# Held-out evaluation: combo-12-13-pp15

Submission /opt/wsp/runs/combo-12-13/submission (9 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 32.6% | 3.1% | 5.2% | 3,660 |
| mbpp+ | edited | 1512 | 33.9% | 3.2% | 4.8% | 3,732 |
| humaneval+ | edited | 656 | 29.6% | 2.7% | 6.1% | 3,494 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.56 points | 1.24 | 542 | 415 | 121 |
| mbpp+ | +17.79 points | 1.56 | 378 | 336 | 67 |
| humaneval+ | +3.81 points | 1.79 | 164 | 79 | 54 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 746, pass 707, runtime_error 328, wrong_interface 197, syntax_error 112, cut_at_max_tokens 67, no_code 6, timeout 5

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 42.9% | 1.3% | 0.9% | 8,324 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.40 points | 1.48 | 1000 | 122 | 98 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 552, pass 429, cut_at_max_tokens 13, no_answer 6

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 60.8% | 6.4% | 10.3% | 5,668 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +38.54 points | 1.89 | 541 | 460 | 43 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 658, not_followed 353, cut_at_max_tokens 63, no_answer 8; instruction-level accuracy 67.3%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.4% | 35.3% | 34.1% | 18,939 |
| easy | edited | 172 | 29.6% | 24.4% | 26.2% | 15,160 |
| medium | edited | 208 | 0.5% | 39.4% | 39.4% | 19,795 |
| hard | edited | 320 | 0.0% | 38.4% | 35.0% | 20,413 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.86 points | 1.55 | 175 | 43 | 2 |
| easy | +23.84 points | 5.45 | 43 | 42 | 1 |
| medium | +0.00 points | 0.69 | 52 | 1 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: cut_at_max_tokens 247, syntax_error 213, wrong_answer 165, pass 52, runtime_error 11, wrong_interface 9, no_code 2, timeout 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 54.1% | 14.9% | 11.2% | 15,725 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +47.35 points | 1.78 | 500 | 951 | 4 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1083, wrong_answer 369, cut_at_max_tokens 298, no_answer 250
