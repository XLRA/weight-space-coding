# Held-out evaluation: G-E2-code-pp0

Submission /opt/wsp/train/runs/G-E2/checkpoints/step_50 (318 tensors); samples per problem code 4; presence penalty 0.0; the original model's answers are reused from pilot-10; generated on the RTX 5090.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 2168 | 12.3% | 59.0% | 59.0% | 19,782 |
| mbpp+ | original | 1512 | 10.0% | 58.8% | 57.6% | 19,731 |
| humaneval+ | original | 656 | 17.5% | 59.3% | 62.2% | 19,900 |
| all | edited | 2168 | 38.6% | 2.5% | 3.5% | 5,672 |
| mbpp+ | edited | 1512 | 41.7% | 2.7% | 3.3% | 5,559 |
| humaneval+ | edited | 656 | 31.6% | 2.0% | 4.1% | 5,933 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +26.38 points | 1.52 | 542 | 637 | 65 |
| mbpp+ | +31.75 points | 1.89 | 378 | 510 | 30 |
| humaneval+ | +14.02 points | 2.22 | 164 | 127 | 35 |

- original: cut_at_max_tokens 1278, wrong_interface 342, pass 266, wrong_answer 186, runtime_error 85, syntax_error 10, timeout 1
- edited: pass 838, wrong_answer 835, runtime_error 257, syntax_error 133, cut_at_max_tokens 54, wrong_interface 42, no_code 5, timeout 4
