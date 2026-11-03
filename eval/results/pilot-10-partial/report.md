# Held-out evaluation: pilot-10-partial

Submission /opt/wsp/runs/pilot-10/submission (3 tensors); samples per problem code 4, mmlupro 1; presence penalty 0.0.

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
