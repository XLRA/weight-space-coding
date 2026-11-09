# Held-out evaluation: pilot-14-code-pp0

Submission /opt/wsp/runs/pilot-14/submission (9 tensors); samples per problem code 4; presence penalty 0.0; the original model's answers are reused from pilot-10; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 28.5% | 5.8% | 11.6% | 4,585 |
| mbpp+ | edited | 1512 | 29.7% | 5.9% | 10.8% | 4,588 |
| humaneval+ | edited | 656 | 25.8% | 5.5% | 13.3% | 4,580 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +16.24 points | 1.27 | 542 | 451 | 99 |
| mbpp+ | +19.71 points | 1.59 | 378 | 355 | 57 |
| humaneval+ | +8.23 points | 1.87 | 164 | 96 | 42 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: wrong_answer 679, pass 618, runtime_error 280, syntax_error 232, wrong_interface 222, cut_at_max_tokens 125, no_code 8, timeout 4
