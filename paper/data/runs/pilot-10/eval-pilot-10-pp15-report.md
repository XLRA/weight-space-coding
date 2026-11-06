# Held-out evaluation: pilot-10-pp15

Submission /opt/wsp/runs/pilot-10/submission (3 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 27.6% | 2.1% | 5.2% | 3,358 |
| mbpp+ | edited | 1512 | 27.1% | 2.5% | 4.4% | 3,470 |
| humaneval+ | edited | 656 | 29.0% | 1.4% | 7.2% | 3,100 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +8.58 points | 1.06 | 542 | 325 | 139 |
| mbpp+ | +10.91 points | 1.3 | 378 | 250 | 85 |
| humaneval+ | +3.20 points | 1.75 | 164 | 75 | 54 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 632, pass 599, wrong_interface 398, runtime_error 323, syntax_error 111, no_code 51, cut_at_max_tokens 46, timeout 8

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 42.9% | 0.3% | 0.6% | 8,139 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.40 points | 1.43 | 1000 | 114 | 90 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 559, pass 429, no_answer 9, cut_at_max_tokens 3

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 56.6% | 0.2% | 5.6% | 3,666 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +34.29 points | 1.98 | 541 | 433 | 62 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 612, not_followed 426, no_answer 42, cut_at_max_tokens 2; instruction-level accuracy 64.3%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.6% | 14.7% | 24.0% | 13,674 |
| easy | edited | 172 | 29.1% | 7.6% | 19.2% | 10,906 |
| medium | edited | 208 | 1.0% | 19.7% | 28.9% | 14,610 |
| hard | edited | 320 | 0.3% | 15.3% | 23.4% | 14,554 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +6.00 points | 1.54 | 175 | 43 | 1 |
| easy | +23.26 points | 5.35 | 43 | 40 | 0 |
| medium | +0.48 points | 1.08 | 52 | 2 | 1 |
| hard | +0.31 points | 0.31 | 80 | 1 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 285, wrong_answer 185, cut_at_max_tokens 103, pass 53, no_code 41, wrong_interface 29, runtime_error 4

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 51.0% | 3.9% | 6.4% | 13,542 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +44.25 points | 1.76 | 500 | 891 | 6 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1021, no_answer 650, wrong_answer 251, cut_at_max_tokens 78
