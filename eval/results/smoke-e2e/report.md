# Held-out evaluation: smoke-e2e

Submission /opt/wsp/runs/e2e-1/submission (1 tensors); samples per problem code 1; presence penalty 0; SMOKE TEST, first 5 problems per set only.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | edited | 10 | 20.0% | 50.0% | 60.0% | 16,956 |
| mbpp+ | edited | 5 | 0.0% | 60.0% | 60.0% | 19,939 |
| humaneval+ | edited | 5 | 40.0% | 40.0% | 60.0% | 13,973 |
| all | original | 10 | 20.0% | 50.0% | 60.0% | 17,304 |
| mbpp+ | original | 5 | 0.0% | 40.0% | 60.0% | 14,014 |
| humaneval+ | original | 5 | 40.0% | 60.0% | 60.0% | 20,595 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 0.0 | 10 | 0 | 0 |
| mbpp+ | +0.00 points | 0.0 | 5 | 0 | 0 |
| humaneval+ | +0.00 points | 0.0 | 5 | 0 | 0 |

- edited: cut_at_max_tokens 5, pass 2, wrong_interface 1, wrong_answer 1, syntax_error 1
- original: cut_at_max_tokens 5, pass 2, wrong_interface 1, wrong_answer 1, runtime_error 1
