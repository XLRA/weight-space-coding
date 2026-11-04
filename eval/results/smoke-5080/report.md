# Held-out evaluation: smoke-5080

Submission /opt/wsp/runs/pilot-10/submission (3 tensors); samples per problem code 1, ifeval 1; presence penalty 1.5; SMOKE TEST, first 3 problems per set only; generated on the RTX 5080.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | edited | 6 | 50.0% | 0.0% | 0.0% | 2,020 |
| mbpp+ | edited | 3 | 66.7% | 0.0% | 0.0% | 2,970 |
| humaneval+ | edited | 3 | 33.3% | 0.0% | 0.0% | 1,071 |

- edited: pass 3, no_code 1, wrong_answer 1, syntax_error 1

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | edited | 3 | 33.3% | 0.0% | 33.3% | 4,494 |

- edited: not_followed 1, no_answer 1, pass 1; instruction-level accuracy 40.0%
