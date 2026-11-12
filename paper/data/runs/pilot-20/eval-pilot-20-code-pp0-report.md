# Held-out evaluation: pilot-20-code-pp0

Submission /opt/wsp/runs/pilot-20/submission (318 tensors); samples per problem code 4; presence penalty 0.0; the original model's answers are reused from pilot-10; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 44.4% | 7.5% | 7.7% | 7,243 |
| mbpp+ | edited | 1512 | 46.6% | 5.8% | 5.9% | 6,316 |
| humaneval+ | edited | 656 | 39.5% | 11.4% | 11.7% | 9,380 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +32.15 points | 1.58 | 542 | 761 | 64 |
| mbpp+ | +36.57 points | 1.98 | 378 | 585 | 32 |
| humaneval+ | +21.95 points | 2.35 | 164 | 176 | 32 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: pass 963, wrong_answer 727, runtime_error 240, cut_at_max_tokens 163, wrong_interface 42, syntax_error 27, timeout 6
