# Held-out evaluation: G-O-pp15

Submission /opt/wsp/train/runs/G-O/checkpoints/step_100 (318 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 44.8% | 4.3% | 4.1% | 6,082 |
| mbpp+ | edited | 1512 | 46.8% | 3.2% | 2.9% | 5,270 |
| humaneval+ | edited | 656 | 40.2% | 7.0% | 6.7% | 7,952 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +25.74 points | 1.6 | 542 | 658 | 100 |
| mbpp+ | +30.62 points | 1.89 | 378 | 510 | 47 |
| humaneval+ | +14.48 points | 2.83 | 164 | 148 | 53 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: pass 971, wrong_answer 753, runtime_error 239, cut_at_max_tokens 94, syntax_error 53, wrong_interface 45, timeout 10, no_code 3

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 45.6% | 0.5% | 0.5% | 5,955 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.10 points | 1.51 | 1000 | 141 | 90 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 540, pass 456, cut_at_max_tokens 4

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 70.1% | 9.0% | 17.7% | 5,795 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +47.78 points | 1.8 | 541 | 540 | 23 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 758, not_followed 228, cut_at_max_tokens 96; instruction-level accuracy 73.4%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.4% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.0% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 14.6% | 32.7% | 36.0% | 15,696 |
| easy | edited | 172 | 50.0% | 7.0% | 11.1% | 6,879 |
| medium | edited | 208 | 6.7% | 36.1% | 40.4% | 17,204 |
| hard | edited | 320 | 0.6% | 44.4% | 46.6% | 19,456 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.14 points | 2.22 | 175 | 92 | 0 |
| easy | +44.19 points | 6.16 | 43 | 76 | 0 |
| medium | +6.73 points | 2.92 | 52 | 14 | 0 |
| hard | +0.62 points | 0.62 | 80 | 2 | 0 |

- original: cut_at_max_tokens 687, pass 10, runtime_error 1, timeout 1, wrong_answer 1
- edited: cut_at_max_tokens 229, wrong_answer 185, runtime_error 137, pass 102, timeout 40, syntax_error 7

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 65.0% | 16.8% | 8.3% | 13,817 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +58.20 points | 1.79 | 500 | 1165 | 1 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1300, wrong_answer 359, cut_at_max_tokens 335, no_answer 6
