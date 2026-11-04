# Held-out evaluation: smoke-reuse

Submission /opt/wsp/runs/pilot-10/submission (3 tensors); samples per problem mmlupro 1, ifeval 1; presence penalty 0.0; SMOKE TEST, first 5 problems per set only; the original model's answers are reused from smoke-suites.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 20.0% | 80.0% | 80.0% | 26,517 |
| all | edited | 5 | 20.0% | 0.0% | 0.0% | 7,170 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 31.62 | 5 | 1 | 1 |

- original: cut_at_max_tokens 3, wrong_answer 1, pass 1
- edited: wrong_answer 3, pass 1, no_answer 1

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 0.0% | 100.0% | 100.0% | 32,768 |
| all | edited | 5 | 80.0% | 0.0% | 0.0% | 2,276 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +80.00 points | 20.0 | 5 | 4 | 0 |

- original: cut_at_max_tokens 5; instruction-level accuracy 0.0%
- edited: pass 4, not_followed 1; instruction-level accuracy 87.5%
