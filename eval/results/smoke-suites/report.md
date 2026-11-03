# Held-out evaluation: smoke-suites

Submission /opt/wsp/runs/e2e-1/submission (1 tensors); samples per problem code 1, mmlupro 1, ifeval 1, lcb 1, math500 1; presence penalty 0.0; SMOKE TEST, first 5 problems per set only.

Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per problem over its samples, with its standard error over problems.

## MBPP+ and HumanEval+ (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 10 | 20.0% | 50.0% | 60.0% | 17,304 |
| mbpp+ | original | 5 | 0.0% | 40.0% | 60.0% | 14,014 |
| humaneval+ | original | 5 | 40.0% | 60.0% | 60.0% | 20,595 |
| all | edited | 10 | 20.0% | 50.0% | 60.0% | 16,956 |
| mbpp+ | edited | 5 | 0.0% | 60.0% | 60.0% | 19,939 |
| humaneval+ | edited | 5 | 40.0% | 40.0% | 60.0% | 13,973 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 0.0 | 10 | 0 | 0 |
| mbpp+ | +0.00 points | 0.0 | 5 | 0 | 0 |
| humaneval+ | +0.00 points | 0.0 | 5 | 0 | 0 |

- original: cut_at_max_tokens 5, pass 2, wrong_interface 1, wrong_answer 1, runtime_error 1
- edited: cut_at_max_tokens 5, pass 2, wrong_interface 1, wrong_answer 1, syntax_error 1

## MMLU-Pro, 1,000 questions (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 20.0% | 80.0% | 80.0% | 26,517 |
| all | edited | 5 | 0.0% | 100.0% | 100.0% | 32,768 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | -20.00 points | 20.0 | 5 | 0 | 1 |

- original: cut_at_max_tokens 3, wrong_answer 1, pass 1
- edited: cut_at_max_tokens 5

## IFEval (strict prompt-level accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 0.0% | 100.0% | 100.0% | 32,768 |
| all | edited | 5 | 0.0% | 100.0% | 100.0% | 32,768 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 0.0 | 5 | 0 | 0 |

- original: cut_at_max_tokens 5; instruction-level accuracy 0.0%
- edited: cut_at_max_tokens 5; instruction-level accuracy 0.0%

## LiveCodeBench v6 additions (pass@1)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 0.0% | 100.0% | 100.0% | 32,768 |
| easy | original | 2 | 0.0% | 100.0% | 100.0% | 32,768 |
| medium | original | 2 | 0.0% | 100.0% | 100.0% | 32,768 |
| hard | original | 1 | 0.0% | 100.0% | 100.0% | 32,768 |
| all | edited | 5 | 0.0% | 100.0% | 100.0% | 32,768 |
| easy | edited | 2 | 0.0% | 100.0% | 100.0% | 32,768 |
| medium | edited | 2 | 0.0% | 100.0% | 100.0% | 32,768 |
| hard | edited | 1 | 0.0% | 100.0% | 100.0% | 32,768 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 0.0 | 5 | 0 | 0 |
| easy | +0.00 points | 0.0 | 2 | 0 | 0 |
| medium | +0.00 points | 0.0 | 2 | 0 | 0 |
| hard | +0.00 points | None | 1 | 0 | 0 |

- original: cut_at_max_tokens 5
- edited: cut_at_max_tokens 5

## MATH-500 (accuracy)

| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |
|---|---|---|---|---|---|---|
| all | original | 5 | 0.0% | 100.0% | 100.0% | 32,768 |
| all | edited | 5 | 0.0% | 100.0% | 100.0% | 32,768 |

| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |
|---|---|---|---|---|---|
| all | +0.00 points | 0.0 | 5 | 0 | 0 |

- original: cut_at_max_tokens 5
- edited: cut_at_max_tokens 5
