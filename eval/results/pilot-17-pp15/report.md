# Held-out evaluation: pilot-17-pp15

Submission /opt/wsp/runs/pilot-17/submission (10 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5080.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 31.6% | 2.6% | 6.1% | 3,674 |
| mbpp+ | edited | 1512 | 33.6% | 2.8% | 4.8% | 3,677 |
| humaneval+ | edited | 656 | 26.8% | 2.1% | 9.2% | 3,666 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.50 points | 1.25 | 542 | 409 | 138 |
| mbpp+ | +17.46 points | 1.53 | 378 | 341 | 77 |
| humaneval+ | +1.07 points | 1.85 | 164 | 68 | 61 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 771, pass 684, runtime_error 336, wrong_interface 189, syntax_error 115, cut_at_max_tokens 57, no_code 9, timeout 7

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 44.2% | 1.2% | 1.0% | 8,275 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +3.70 points | 1.52 | 1000 | 134 | 97 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 529, pass 442, no_answer 18, cut_at_max_tokens 11

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 61.0% | 2.6% | 6.0% | 4,690 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +38.72 points | 1.86 | 541 | 471 | 52 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 660, not_followed 366, no_answer 39, cut_at_max_tokens 17; instruction-level accuracy 67.9%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.4% | 2.9% | 7.9% | 12,148 |
| easy | edited | 172 | 29.1% | 2.9% | 5.8% | 10,508 |
| medium | edited | 208 | 1.0% | 1.0% | 7.2% | 11,146 |
| hard | edited | 320 | 0.0% | 4.1% | 9.4% | 13,682 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.86 points | 1.46 | 175 | 42 | 1 |
| easy | +23.26 points | 4.95 | 43 | 40 | 0 |
| medium | +0.48 points | 1.08 | 52 | 2 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 446, wrong_answer 158, pass 52, cut_at_max_tokens 20, wrong_interface 16, runtime_error 5, timeout 2, no_code 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 56.0% | 5.2% | 2.9% | 14,172 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +49.20 points | 1.82 | 500 | 987 | 3 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1120, wrong_answer 479, no_answer 297, cut_at_max_tokens 104
