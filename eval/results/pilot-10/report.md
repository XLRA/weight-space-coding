# Held-out evaluation: pilot-10

Submission /opt/wsp/runs/pilot-10/submission (3 tensors); samples per problem code 4, mmlupro 1, ifeval 2, lcb 4, math500 4; presence penalty 0.0.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 25.8% | 7.4% | 13.0% | 4,792 |
| mbpp+ | edited | 1512 | 25.9% | 7.6% | 11.5% | 4,818 |
| humaneval+ | edited | 656 | 25.6% | 6.9% | 16.5% | 4,731 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +13.56 points | 1.17 | 542 | 390 | 96 |
| mbpp+ | +15.94 points | 1.47 | 378 | 303 | 62 |
| humaneval+ | +8.08 points | 1.82 | 164 | 87 | 34 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: wrong_answer 679, pass 560, wrong_interface 365, runtime_error 219, cut_at_max_tokens 160, syntax_error 141, no_code 40, timeout 4

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1000 | 28.8% | 84.4% | 83.9% | 28,189 |
| all | edited | 1000 | 39.9% | 4.6% | 7.0% | 6,904 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +11.10 points | 1.57 | 1000 | 184 | 73 |

- original: cut_at_max_tokens 668, pass 288, wrong_answer 44
- edited: wrong_answer 479, pass 399, no_answer 77, cut_at_max_tokens 45

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 1082 | 1.6% | 98.2% | 95.8% | 32,209 |
| all | edited | 1082 | 56.2% | 1.9% | 10.6% | 3,202 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +54.62 points | 1.83 | 541 | 591 | 0 |

- original: cut_at_max_tokens 1061, pass 17, not_followed 4; instruction-level accuracy 1.1%
- edited: pass 608, not_followed 444, cut_at_max_tokens 17, no_answer 13; instruction-level accuracy 64.7%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 700 | 0.9% | 99.1% | 98.9% | 32,523 |
| easy | original | 172 | 3.5% | 96.5% | 96.5% | 31,770 |
| medium | original | 208 | 0.0% | 100.0% | 100.0% | 32,768 |
| hard | original | 320 | 0.0% | 100.0% | 99.4% | 32,768 |
| all | edited | 700 | 5.3% | 66.1% | 69.4% | 24,615 |
| easy | edited | 172 | 21.5% | 48.3% | 57.6% | 19,313 |
| medium | edited | 208 | 0.0% | 73.6% | 75.0% | 26,368 |
| hard | edited | 320 | 0.0% | 70.9% | 72.2% | 26,326 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +4.43 points | 1.26 | 175 | 33 | 2 |
| easy | +18.02 points | 4.58 | 43 | 33 | 2 |
| medium | +0.00 points | 0.0 | 52 | 0 | 0 |
| hard | +0.00 points | 0.0 | 80 | 0 | 0 |

- original: cut_at_max_tokens 694, pass 6
- edited: cut_at_max_tokens 463, wrong_answer 81, syntax_error 80, pass 37, no_code 22, runtime_error 9, wrong_interface 6, timeout 2

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2000 | 2.5% | 99.3% | 97.5% | 32,571 |
| all | edited | 2000 | 46.3% | 25.4% | 29.1% | 15,417 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +43.75 points | 1.84 | 500 | 877 | 2 |

- original: cut_at_max_tokens 1949, pass 51
- edited: pass 926, cut_at_max_tokens 508, no_answer 383, wrong_answer 183
