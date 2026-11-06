# Held-out evaluation: pilot-13-pp15

Submission /opt/wsp/runs/pilot-13/submission (6 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 1.5; the original model's answers are reused from pilot-10-pp15; generated on the RTX 5080.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 19.1% | 18.8% | 19.2% | 7,305 |
| mbpp+ | original | 1512 | 16.1% | 19.8% | 19.0% | 7,625 |
| humaneval+ | original | 656 | 25.8% | 16.6% | 19.8% | 6,566 |
| all | edited | 2168 | 31.5% | 1.4% | 6.2% | 3,327 |
| mbpp+ | edited | 1512 | 33.7% | 1.6% | 4.6% | 3,341 |
| humaneval+ | edited | 656 | 26.5% | 0.9% | 9.9% | 3,296 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +12.45 points | 1.21 | 542 | 410 | 140 |
| mbpp+ | +17.53 points | 1.48 | 378 | 335 | 70 |
| humaneval+ | +0.76 points | 1.83 | 164 | 75 | 70 |

- original: wrong_interface 622, wrong_answer 475, pass 413, cut_at_max_tokens 408, runtime_error 208, syntax_error 35, timeout 7
- edited: wrong_answer 764, pass 683, runtime_error 345, wrong_interface 186, syntax_error 97, no_code 58, cut_at_max_tokens 31, timeout 4

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 40.5% | 33.1% | 32.9% | 14,653 |
| all | edited | 1000 | 43.2% | 0.4% | 0.3% | 8,034 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +2.70 points | 1.48 | 1000 | 123 | 96 |

- original: pass 405, wrong_answer 327, cut_at_max_tokens 268
- edited: wrong_answer 560, pass 432, no_answer 5, cut_at_max_tokens 3

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 22.3% | 72.9% | 74.4% | 24,837 |
| all | edited | 1082 | 56.6% | 0.2% | 7.3% | 3,529 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +34.38 points | 1.98 | 541 | 440 | 68 |

- original: cut_at_max_tokens 785, pass 241, not_followed 56; instruction-level accuracy 19.8%
- edited: pass 613, not_followed 411, no_answer 56, cut_at_max_tokens 2; instruction-level accuracy 63.4%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 1.6% | 98.1% | 97.1% | 32,274 |
| easy | original | 172 | 5.8% | 93.6% | 92.4% | 31,043 |
| medium | original | 208 | 0.5% | 99.0% | 99.0% | 32,533 |
| hard | original | 320 | 0.0% | 100.0% | 98.4% | 32,768 |
| all | edited | 700 | 6.6% | 19.7% | 28.7% | 15,157 |
| easy | edited | 172 | 26.2% | 13.4% | 23.3% | 12,858 |
| medium | edited | 208 | 0.0% | 26.0% | 38.9% | 16,296 |
| hard | edited | 320 | 0.3% | 19.1% | 25.0% | 15,652 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +5.00 points | 1.31 | 175 | 36 | 1 |
| easy | +20.35 points | 4.58 | 43 | 35 | 0 |
| medium | -0.48 points | 0.48 | 52 | 0 | 1 |
| hard | +0.31 points | 0.31 | 80 | 1 | 0 |

- original: cut_at_max_tokens 687, pass 11, runtime_error 1, wrong_answer 1
- edited: syntax_error 250, wrong_answer 176, cut_at_max_tokens 138, pass 46, no_code 37, wrong_interface 35, runtime_error 16, timeout 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 6.8% | 93.8% | 90.6% | 31,054 |
| all | edited | 2000 | 51.0% | 4.2% | 6.3% | 13,513 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +44.15 points | 1.74 | 500 | 886 | 3 |

- original: cut_at_max_tokens 1861, pass 136, no_answer 2, wrong_answer 1
- edited: pass 1019, no_answer 624, wrong_answer 273, cut_at_max_tokens 84
