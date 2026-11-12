# Held-out evaluation: G-E2-pp15

Submission /opt/wsp/train/runs/G-E2/checkpoints/step_50 (318 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 41.6% | 1.8% | 2.3% | 5,290 |
| mbpp+ | edited | 1512 | 44.6% | 1.9% | 1.6% | 5,090 |
| humaneval+ | edited | 656 | 34.8% | 1.5% | 3.7% | 5,754 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +22.60 points | 1.46 | 542 | 565 | 75 |
| mbpp+ | +28.51 points | 1.81 | 378 | 464 | 33 |
| humaneval+ | +8.99 points | 2.08 | 164 | 101 | 42 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: pass 903, wrong_answer 828, runtime_error 275, syntax_error 69, wrong_interface 43, cut_at_max_tokens 38, timeout 10, no_code 2

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 45.6% | 0.6% | 0.2% | 7,740 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.10 points | 1.45 | 1000 | 132 | 81 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 531, pass 456, no_answer 8, cut_at_max_tokens 5

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 62.3% | 1.7% | 6.4% | 4,715 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +40.02 points | 1.91 | 541 | 474 | 41 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 674, not_followed 344, no_answer 48, cut_at_max_tokens 16; instruction-level accuracy 69.5%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 10.4% | 0.3% | 4.3% | 9,340 |
| easy | edited | 172 | 38.4% | 0.6% | 6.4% | 6,296 |
| medium | edited | 208 | 1.9% | 0.0% | 3.9% | 8,957 |
| hard | edited | 320 | 0.9% | 0.3% | 3.4% | 11,224 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +8.86 points | 1.78 | 175 | 63 | 1 |
| easy | +32.56 points | 5.55 | 43 | 56 | 0 |
| medium | +1.44 points | 1.07 | 52 | 4 | 1 |
| hard | +0.94 points | 0.94 | 80 | 3 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 275, wrong_answer 247, pass 73, runtime_error 70, timeout 20, wrong_interface 8, no_code 5, cut_at_max_tokens 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 59.2% | 9.8% | 1.4% | 14,168 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +52.45 points | 1.82 | 500 | 1049 | 0 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1185, wrong_answer 461, cut_at_max_tokens 178, no_answer 176
