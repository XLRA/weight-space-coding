# Held-out evaluation: T-pp15

Submission /opt/wsp/train/combined/T-step_100 (318 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 42.3% | 1.1% | 0.6% | 6,446 |
| mbpp+ | edited | 1512 | 45.4% | 0.6% | 0.3% | 5,531 |
| humaneval+ | edited | 656 | 35.2% | 2.3% | 1.2% | 8,556 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +23.29 points | 1.58 | 542 | 621 | 116 |
| mbpp+ | +29.30 points | 1.86 | 378 | 491 | 48 |
| humaneval+ | +9.45 points | 2.73 | 164 | 130 | 68 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: pass 918, wrong_answer 770, runtime_error 229, syntax_error 162, wrong_interface 34, cut_at_max_tokens 24, no_code 20, timeout 11

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 45.0% | 0.1% | 0.1% | 6,506 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +4.50 points | 1.47 | 1000 | 132 | 87 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 549, pass 450, no_answer 1

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 67.7% | 0.8% | 5.7% | 4,347 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +45.47 points | 1.87 | 541 | 533 | 41 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 733, not_followed 309, no_answer 34, cut_at_max_tokens 6; instruction-level accuracy 73.6%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.4% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.0% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 11.7% | 0.4% | 1.4% | 11,251 |
| easy | edited | 172 | 44.2% | 0.0% | 0.0% | 5,581 |
| medium | edited | 208 | 1.0% | 0.5% | 1.9% | 11,626 |
| hard | edited | 320 | 1.2% | 0.6% | 1.9% | 14,055 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +10.29 points | 1.91 | 175 | 73 | 1 |
| easy | +38.37 points | 5.72 | 43 | 67 | 1 |
| medium | +0.96 points | 0.67 | 52 | 2 | 0 |
| hard | +1.25 points | 0.98 | 80 | 4 | 0 |

- original: cut_at_max_tokens 687, pass 10, runtime_error 1, timeout 1, wrong_answer 1
- edited: wrong_answer 291, syntax_error 157, runtime_error 125, pass 82, timeout 36, no_code 4, cut_at_max_tokens 3, wrong_interface 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 63.9% | 10.9% | 0.3% | 13,842 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +57.10 points | 1.79 | 500 | 1143 | 1 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1278, wrong_answer 505, cut_at_max_tokens 207, no_answer 10
