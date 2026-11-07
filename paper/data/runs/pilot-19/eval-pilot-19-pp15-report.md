# Held-out evaluation: pilot-19-pp15

Submission /opt/wsp/runs/pilot-19/submission (13 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 33.0% | 2.5% | 5.6% | 3,701 |
| mbpp+ | edited | 1512 | 34.1% | 3.0% | 5.1% | 3,827 |
| humaneval+ | edited | 656 | 30.3% | 1.4% | 6.7% | 3,410 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.93 points | 1.19 | 542 | 431 | 129 |
| mbpp+ | +17.99 points | 1.48 | 378 | 345 | 73 |
| humaneval+ | +4.57 points | 1.8 | 164 | 86 | 56 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 752, pass 715, runtime_error 333, wrong_interface 163, syntax_error 136, cut_at_max_tokens 55, no_code 7, timeout 7

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 43.4% | 1.1% | 0.8% | 8,285 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.90 points | 1.44 | 1000 | 118 | 89 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 539, pass 434, no_answer 16, cut_at_max_tokens 11

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 56.3% | 2.2% | 7.0% | 4,629 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +34.01 points | 1.91 | 541 | 419 | 51 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 609, not_followed 408, no_answer 47, cut_at_max_tokens 18; instruction-level accuracy 65.3%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.3% | 2.7% | 8.0% | 12,226 |
| easy | edited | 172 | 29.1% | 0.0% | 6.4% | 9,976 |
| medium | edited | 208 | 0.5% | 2.4% | 7.7% | 11,334 |
| hard | edited | 320 | 0.0% | 4.4% | 9.1% | 14,016 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.71 points | 1.47 | 175 | 42 | 2 |
| easy | +23.26 points | 5.09 | 43 | 41 | 1 |
| medium | +0.00 points | 0.69 | 52 | 1 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 433, wrong_answer 170, pass 51, cut_at_max_tokens 19, runtime_error 13, wrong_interface 10, no_code 3, timeout 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 56.0% | 5.1% | 3.0% | 14,227 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +49.20 points | 1.84 | 500 | 986 | 2 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1120, wrong_answer 497, no_answer 283, cut_at_max_tokens 100
