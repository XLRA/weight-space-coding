# Held-out evaluation: pilot-20-pp15

Submission /opt/wsp/runs/pilot-20/submission (318 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 46.0% | 2.3% | 2.2% | 5,913 |
| mbpp+ | edited | 1512 | 47.4% | 1.9% | 1.8% | 5,275 |
| humaneval+ | edited | 656 | 42.8% | 3.0% | 3.0% | 7,382 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +26.98 points | 1.56 | 542 | 666 | 81 |
| mbpp+ | +31.28 points | 1.87 | 378 | 509 | 36 |
| humaneval+ | +17.07 points | 2.71 | 164 | 157 | 45 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: pass 998, wrong_answer 758, runtime_error 271, cut_at_max_tokens 49, wrong_interface 45, syntax_error 34, timeout 10, no_code 3

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 46.4% | 0.1% | 0.1% | 6,228 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.90 points | 1.52 | 1000 | 147 | 88 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 535, pass 464, cut_at_max_tokens 1

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 71.6% | 6.4% | 14.5% | 5,356 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +49.35 points | 1.83 | 541 | 558 | 24 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 775, not_followed 249, cut_at_max_tokens 58; instruction-level accuracy 77.5%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 13.6% | 19.0% | 22.3% | 13,228 |
| easy | edited | 172 | 48.8% | 4.1% | 5.2% | 6,606 |
| medium | edited | 208 | 1.9% | 19.7% | 24.5% | 14,187 |
| hard | edited | 320 | 2.2% | 26.6% | 30.0% | 16,163 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.00 points | 2.2 | 175 | 85 | 1 |
| easy | +43.02 points | 6.3 | 43 | 74 | 0 |
| medium | +1.44 points | 1.44 | 52 | 4 | 1 |
| hard | +2.19 points | 1.55 | 80 | 7 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: wrong_answer 261, runtime_error 150, cut_at_max_tokens 133, pass 95, timeout 44, syntax_error 17

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 65.8% | 16.2% | 6.1% | 13,894 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +59.00 points | 1.8 | 500 | 1182 | 2 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1316, wrong_answer 358, cut_at_max_tokens 322, no_answer 4
