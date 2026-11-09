# Held-out evaluation: R-O-pp15

Submission /opt/wsp/train/runs/R-O/round_2/checkpoint (318 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 32.5% | 1.8% | 1.9% | 1,518 |
| mbpp+ | edited | 1512 | 33.3% | 2.0% | 1.9% | 1,607 |
| humaneval+ | edited | 656 | 30.6% | 1.2% | 2.1% | 1,315 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.42 points | 1.3 | 542 | 428 | 137 |
| mbpp+ | +17.13 points | 1.56 | 378 | 334 | 75 |
| humaneval+ | +4.88 points | 2.2 | 164 | 94 | 62 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 792, pass 704, runtime_error 298, wrong_interface 275, syntax_error 56, cut_at_max_tokens 38, timeout 5

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 36.6% | 14.7% | 14.5% | 6,007 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | -3.90 points | 1.55 | 1000 | 101 | 140 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 509, pass 366, cut_at_max_tokens 124, no_answer 1

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 36.6% | 4.9% | 6.4% | 2,361 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +14.33 points | 1.93 | 541 | 263 | 108 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: not_followed 635, pass 396, cut_at_max_tokens 49, no_answer 2; instruction-level accuracy 47.8%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 9.4% | 20.3% | 22.1% | 7,894 |
| easy | edited | 172 | 36.0% | 7.0% | 9.3% | 3,559 |
| medium | edited | 208 | 1.4% | 19.7% | 20.2% | 7,590 |
| hard | edited | 320 | 0.3% | 27.8% | 30.3% | 10,422 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +7.86 points | 1.86 | 175 | 59 | 4 |
| easy | +30.23 points | 6.31 | 43 | 55 | 3 |
| medium | +0.96 points | 1.18 | 52 | 3 | 1 |
| hard | +0.31 points | 0.31 | 80 | 1 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: wrong_answer 317, cut_at_max_tokens 142, runtime_error 131, pass 66, timeout 19, syntax_error 18, wrong_interface 5, no_code 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 42.8% | 35.4% | 34.0% | 12,792 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +36.00 points | 1.66 | 500 | 732 | 12 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 856, cut_at_max_tokens 699, wrong_answer 410, no_answer 35
