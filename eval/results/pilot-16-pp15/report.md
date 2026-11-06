# Held-out evaluation: pilot-16-pp15

Submission /opt/wsp/runs/pilot-16/submission (9 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 31.4% | 1.8% | 4.9% | 3,342 |
| mbpp+ | edited | 1512 | 33.6% | 1.9% | 4.2% | 3,388 |
| humaneval+ | edited | 656 | 26.2% | 1.7% | 6.7% | 3,236 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.32 points | 1.23 | 542 | 402 | 135 |
| mbpp+ | +17.46 points | 1.53 | 378 | 336 | 72 |
| humaneval+ | +0.46 points | 1.72 | 164 | 66 | 63 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 788, pass 680, runtime_error 338, wrong_interface 184, syntax_error 116, cut_at_max_tokens 39, no_code 13, timeout 10

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 40.6% | 1.1% | 0.7% | 8,125 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.10 points | 1.4 | 1000 | 99 | 98 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 578, pass 406, cut_at_max_tokens 10, no_answer 6

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 60.3% | 2.6% | 6.2% | 4,776 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +37.99 points | 1.87 | 541 | 459 | 48 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 652, not_followed 387, cut_at_max_tokens 26, no_answer 17; instruction-level accuracy 67.5%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.4% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.0% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 6.4% | 3.9% | 9.6% | 11,764 |
| easy | edited | 172 | 25.6% | 2.3% | 8.7% | 10,464 |
| medium | edited | 208 | 0.0% | 2.9% | 12.0% | 10,914 |
| hard | edited | 320 | 0.3% | 5.3% | 8.4% | 13,016 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.00 points | 1.37 | 175 | 35 | 0 |
| easy | +19.77 points | 4.96 | 43 | 34 | 0 |
| medium | +0.00 points | 0.0 | 52 | 0 | 0 |
| hard | +0.31 points | 0.31 | 80 | 1 | 0 |

- original: cut_at_max_tokens 687, pass 10, runtime_error 1, timeout 1, wrong_answer 1
- edited: syntax_error 389, wrong_answer 192, pass 45, wrong_interface 33, cut_at_max_tokens 27, runtime_error 10, timeout 3, no_code 1

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 56.0% | 3.5% | 2.2% | 13,477 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +49.25 points | 1.8 | 500 | 988 | 3 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1121, wrong_answer 482, no_answer 328, cut_at_max_tokens 69
