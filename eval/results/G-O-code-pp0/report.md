# Held-out evaluation: G-O-code-pp0

Submission /opt/wsp/train/runs/G-O/checkpoints/step_100 (318 tensors); samples per problem code 4; presence penalty 0.0; the original model's answers are reused from pilot-10; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 43.5% | 14.9% | 14.7% | 8,664 |
| mbpp+ | edited | 1512 | 46.4% | 12.8% | 12.5% | 7,721 |
| humaneval+ | edited | 656 | 36.9% | 19.7% | 19.8% | 10,839 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +31.27 points | 1.62 | 542 | 742 | 64 |
| mbpp+ | +36.44 points | 2.03 | 378 | 585 | 34 |
| humaneval+ | +19.36 points | 2.4 | 164 | 157 | 30 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: pass 944, wrong_answer 608, cut_at_max_tokens 323, runtime_error 225, wrong_interface 32, syntax_error 32, timeout 4
