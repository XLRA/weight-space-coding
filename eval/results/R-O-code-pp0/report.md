# Held-out evaluation: R-O-code-pp0

Submission /opt/wsp/train/runs/R-O/round_2/checkpoint (318 tensors); samples per problem code 4; presence penalty 0.0; the original model's answers are reused from pilot-10; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 31.3% | 7.1% | 7.3% | 3,325 |
| mbpp+ | edited | 1512 | 32.8% | 7.3% | 6.6% | 3,411 |
| humaneval+ | edited | 656 | 27.7% | 6.7% | 8.8% | 3,126 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +19.00 points | 1.38 | 542 | 502 | 90 |
| mbpp+ | +22.82 points | 1.68 | 378 | 388 | 43 |
| humaneval+ | +10.21 points | 2.29 | 164 | 114 | 47 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: wrong_answer 731, pass 678, runtime_error 284, wrong_interface 276, cut_at_max_tokens 154, syntax_error 40, timeout 5
