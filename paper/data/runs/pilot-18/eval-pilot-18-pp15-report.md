# Held-out evaluation: pilot-18-pp15

Submission /opt/wsp/runs/pilot-18/submission (12 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5080.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 31.8% | 2.2% | 4.8% | 3,499 |
| mbpp+ | edited | 1512 | 33.2% | 2.2% | 4.2% | 3,473 |
| humaneval+ | edited | 656 | 28.5% | 2.1% | 6.0% | 3,559 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.73 points | 1.18 | 542 | 405 | 129 |
| mbpp+ | +17.06 points | 1.45 | 378 | 328 | 70 |
| humaneval+ | +2.74 points | 1.81 | 164 | 77 | 59 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 782, pass 689, runtime_error 350, wrong_interface 149, syntax_error 132, cut_at_max_tokens 48, no_code 12, timeout 6

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 43.1% | 0.8% | 0.6% | 8,226 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.60 points | 1.45 | 1000 | 118 | 92 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 536, pass 431, no_answer 26, cut_at_max_tokens 7

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 59.2% | 2.5% | 6.3% | 4,628 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +36.97 points | 1.94 | 541 | 461 | 61 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 641, not_followed 379, no_answer 42, cut_at_max_tokens 20; instruction-level accuracy 66.5%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.0% | 3.7% | 6.0% | 11,499 |
| easy | edited | 172 | 28.5% | 2.3% | 5.8% | 10,567 |
| medium | edited | 208 | 0.0% | 1.4% | 5.3% | 10,241 |
| hard | edited | 320 | 0.0% | 5.9% | 6.6% | 12,818 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.43 points | 1.49 | 175 | 40 | 2 |
| easy | +22.67 points | 5.25 | 43 | 40 | 1 |
| medium | -0.48 points | 0.48 | 52 | 0 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 410, wrong_answer 185, pass 49, cut_at_max_tokens 26, wrong_interface 16, runtime_error 10, timeout 3, no_code 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 55.5% | 5.2% | 2.4% | 13,815 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +48.65 points | 1.77 | 500 | 973 | 0 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1109, wrong_answer 484, no_answer 304, cut_at_max_tokens 103
