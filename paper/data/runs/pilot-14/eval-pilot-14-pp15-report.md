# Held-out evaluation: pilot-14-pp15

Submission /opt/wsp/runs/pilot-14/submission (9 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 32.3% | 1.8% | 4.8% | 3,347 |
| mbpp+ | edited | 1512 | 34.9% | 1.9% | 4.0% | 3,353 |
| humaneval+ | edited | 656 | 26.4% | 1.5% | 6.7% | 3,334 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.28 points | 1.26 | 542 | 418 | 130 |
| mbpp+ | +18.78 points | 1.56 | 378 | 350 | 66 |
| humaneval+ | +0.61 points | 1.74 | 164 | 68 | 64 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 751, pass 701, runtime_error 333, wrong_interface 200, syntax_error 129, cut_at_max_tokens 39, no_code 10, timeout 5

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 45.2% | 0.3% | 0.2% | 7,984 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +4.70 points | 1.49 | 1000 | 135 | 88 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 525, pass 452, no_answer 20, cut_at_max_tokens 3

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 58.2% | 3.0% | 5.9% | 4,908 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +35.95 points | 1.87 | 541 | 440 | 51 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 630, not_followed 379, no_answer 44, cut_at_max_tokens 29; instruction-level accuracy 65.6%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 7.0% | 2.7% | 5.4% | 11,328 |
| easy | edited | 172 | 28.5% | 1.2% | 5.2% | 9,794 |
| medium | edited | 208 | 0.0% | 2.4% | 4.3% | 10,700 |
| hard | edited | 320 | 0.0% | 3.8% | 6.2% | 12,561 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.43 points | 1.42 | 175 | 39 | 1 |
| easy | +22.67 points | 4.91 | 43 | 39 | 0 |
| medium | -0.48 points | 0.48 | 52 | 0 | 1 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 389, wrong_answer 206, pass 49, wrong_interface 20, cut_at_max_tokens 19, runtime_error 13, timeout 2, no_code 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 56.9% | 4.5% | 2.3% | 13,567 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +50.10 points | 1.8 | 500 | 1004 | 2 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1138, wrong_answer 483, no_answer 289, cut_at_max_tokens 90
