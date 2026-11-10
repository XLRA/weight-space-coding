# Training phase: plan (draft, 2026-11-07)

**Status:** approved as recommended on 2026-11-07, about 05:00 UTC. Step 1 is done (results under **Infrastructure**). E was frozen at 16:30 UTC as `pilot-14`'s edit (below). R-O started at 09:05 UTC and R-E at 16:42 UTC. On the user's go-ahead, GRPO (G-E, then G-O) was queued at 17:51 UTC behind the RFT runs on the 5090 (`train/grpo-queue-2026-11-07.sh`). The RFT arms' held-out evaluations run between G-E and G-O (`train/eval-rft-arms.sh`, also on the user's go-ahead).
- **Since 2026-11-08,** G-O and G-E2 run on tests the problem statements don't show (Deviations).
- **G-O ended on 2026-11-10.** G-E2 should end on 2026-11-11, around 16:00–18:00 UTC.
- **After that,** the GRPO arms' held-out evaluations, T's, and the removal's check run by themselves (`train/eval-grpo-arms.sh`; Deviations, 2026-11-10).

**Changes:** any change to this recipe after a trained arm's held-out evaluation must be listed under **Deviations** at the end, with its date and reason.

**Names used below:**
- **O:** the unedited Qwen3.5-0.8B.
- **E:** the final weight-space edit: `pilot-19`'s if it holds up held out (no suite clearly worse than `pilot-14`, at least level overall), otherwise `pilot-14`'s.
  - **Frozen 2026-11-07 as `pilot-14`'s edit** (`runs/pilot-14/submission`, the same files as `runs/start-r4/submission`).
  - **Why:** paired against `pilot-14` (`scratch/paired.py pilot-19-pp15 pilot-14-pp15`), `pilot-19` was clearly worse on no suite. But over the five suites it was −0.74 points (SE 0.53), so it was not level overall. "At least level" is read as a difference of zero or more.
  - **Per suite:** code +0.65 (0.97), MMLU-Pro −1.80 (1.39), IFEval −1.94 (1.65), LiveCodeBench +0.29 (0.70), MATH-500 −0.90 (0.96).
- **ΔE:** E minus O. It is sparse: a few columns, rows and rank-1 terms in about a dozen tensors.

## Decisions for you

| # | Decision | Recommended | Alternative |
|---|---|---|---|
| 1 | Training method | **RL on test pass/fail (GRPO)** as the main arms, after a cheap rejection-sampling fine-tune (RFT) that also shakes the pipeline down | RFT only: faster and simpler, but weaker evidence about what training can do |
| 2 | Training data | **The agent's own practice set** (319 MBPP + 1,400 APPS problems), so training and the agent had the same problems | A larger open set of coding problems with tests, as a second stage, if the data-matched arms leave a clear question |
| 3 | GPU time | **About 4 days of the 5090, with llama-swap stopped while it trains.** Checkpoint checks run on the 5080 beside it. | Train only at night, and take about twice as long |
| 4 | Who designs the training | **A fixed recipe, written down and committed before any held-out evaluation, run by me** | An agent run in which Opus designs the training itself. That's a different experiment, best left for later. |

## What the phase answers

1. **Training against editing:** with the same practice problems, how does gradient training of O compare with the agent's edit E? This is held out, on the same five suites, and per GPU hour.
2. **Do they compose:** does training from E end higher, or get there in fewer steps, than training from O with the same recipe?
3. **Do the edits survive training:** after training from E, do the edited neurons and gates still do their jobs? And does removing ΔE from the trained model still cost what it cost O?
4. **Are edits portable:** does adding ΔE to the model trained from O help, with no further search?
5. **Cost:** GPU hours for training and evaluation, against the agent's runs. `pilot-10` to `pilot-18` ran for 40 hours in all, each run holding one GPU, and used 4.9M Opus output tokens (`paper/data/runs.csv`).

## Arms

| Arm | Starts from | Training | Answers | New held-out evaluations |
|---|---|---|---|---|
| O, E | — | none | the reference points | none; their answers exist |
| R-O, R-E | O, E | RFT, 2 rounds | pipeline check; cheap training against editing | 2 |
| G-O, G-E | O, E | GRPO, same steps and settings | questions 1–3 | 2 |
| T | G-O + ΔE | none | question 4 | 1, only if its panel isn't clearly worse than G-O's |

- **Replicates:** one seed per arm. If G-O and G-E end within two standard errors of each other on code, a second seed of both settles it (2–3 more days).

## The recipe

**Problems:**
- **Source:** the practice set with reference solutions that pass the grader, minus the 49 MBPP problems the overlap check dropped (see Data hygiene). Training: 218 MBPP and 1,200 APPS problems (`train/data/train.jsonl`).
- **Validation:** a fixed split, drawn before the overlap check as 64 MBPP and 200 APPS problems (APPS stratified by difficulty, seed 20261107). The check then removed 12 of its MBPP problems, so it holds 52 MBPP and 200 APPS (`train/data/val.jsonl`). It is never trained on and is used only to choose checkpoints.
- **Prompts:** the evaluation's own format: the MBPP-style function prompt, or the stdin/stdout prompt with thinking on.

**GRPO** (the main arms):
- **Batch:** 64 problems × 8 samples = 512 answers per step.
- **Reward:** 1 if the final code block passes all of the problem's tests in the grading sandbox, otherwise 0. That includes answers cut at the limit and answers with no code. There is no format or length bonus.
- **Sampling during training:** temperature 1.0, top_p 1.0, no presence penalty, at most 16,384 new tokens.
  - Sampling must be on-policy for the update to be valid, and the presence penalty would make it off-policy.
  - The evaluation keeps its fixed protocol (presence penalty 1.5). The checkpoint panels run at 1.5, so the gap is measured as training goes.
- **Update** (`train/grpo.py`):
  - group-normalised advantages, no KL term, and the clip-higher variant (0.2 / 0.28);
  - token-level averaging, and groups where all 8 answers score the same are skipped;
  - truncated importance weights, capped at 2, for the small differences between vLLM's and the trainer's log-probs;
  - **4 updates per step** (fixed 2026-11-07, before any run): each step's batch is split by problem into 4 mini-batches of 16 problems, with one optimizer step each. The old-policy log-probs are taken once, before the first. With a single update per batch the policy would never have moved since sampling, and the clipping would never act.
- **Trainable:**
  - All 24 transformer blocks are trained.
  - The tied embedding/output matrix (254M parameters) and the final norm stay frozen, the same tensors the agent was not allowed to edit. That also cuts optimizer memory by about a third.
  - The vision encoder (101M) and the multi-token-prediction head (21M) are unused.
- **Optimizer:** AdamW, learning rate 1e-6, 10 warm-up steps then constant, gradient clip 1.0, bf16 with fp32 master weights.
- **Length:** 150 steps for both arms, about 6 passes over the training problems. The smoke run (timeline step 3) may lower the step count or the 16k cap for both arms together, never for one.
  - **Set 2026-11-07, before either run:** the smoke run is G-E's own first 20 steps, since nothing in a step depends on the step count. If those steps show that the count or the cap must change, G-E restarts with the new settings and G-O gets the same. Otherwise both arms keep 150 steps at 16k.

**Choosing the checkpoint** (fixed in advance):
- **Checks:** every 25 steps, the 5080 runs the panel (1,000 questions, presence penalty 1.5) and the validation problems.
  - **How** (`train/run-ckpt.sh`, fixed 2026-11-07, before any run): the panel's questions once each and the validation problems 4 times each, at the evaluation's protocol (presence penalty 1.5, at most 32,768 tokens), with the GPU server's prompts and seeds. The panel is graded by the driver's panel grader, and the validation answers by the practice grader.
  - **Step 0:** each arm's starting model is checked the same way on the 5080, so every pairing uses the same GPU and code. `train/watch-ckpts.sh` checks each checkpoint as it is saved.
- **The pick:** the arm's checkpoint is the one with the best validation pass rate among those whose panel mean is no more than 2.5 points below the arm's starting model.
  - **Ties** go to the earlier checkpoint (fixed 2026-11-07, before any choice). If no checkpoint qualifies, the arm gets no held-out evaluation. `train/eval-rft-arms.sh` applies the rule and records the candidates in `train/evals/<arm>/choice.json`.
- **The held-out evaluation runs once per arm, on that checkpoint only.**

**RFT** (first, as a cheap arm and a shakedown):
- **Samples:** 8 per training problem, at the evaluation's sampling settings but capped at 16,384 new tokens.
- **What it keeps:** up to 2 distinct passing answers per problem.
- **Fine-tune:** 2 epochs on those answers, learning rate 1e-5, then sample again from the new model and repeat once.
- **Cost:** about 5 GPU-hours from E. From O it takes about 12, because most of O's answers run to the cap.
- **Details** (`train/rft.py`, fixed 2026-11-07, before any run):
  - **Restart point:** round 2 fine-tunes from the arm's start, on samples drawn from round 1's model, not from round 1's model itself (the ReST-EM recipe, Singh et al. 2023). Each round starting from the base guards against drift over rounds.
  - **Distinct answers:** two kept answers must have different final code blocks; candidates are taken in a seeded random order, so the answer's length plays no part.
  - **Batches:** 32 answers per optimizer step, the loss averaged over the batch's answer tokens (the prompt is not trained), 5 warm-up steps then cosine decay. The stop token ending each answer is trained too.
  - **The arm's checkpoints** are the ends of the two rounds.

## Measurement

- **Held-out:**
  - The five suites, with the protocol unchanged: temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 tokens.
  - Each arm is paired against O, whose answers are reused, and against its own starting model.
  - For the trained arms, the code suite is also run at presence penalty 0, since training sampled without it.
  - **E at presence penalty 0:** E has held-out results only at 1.5, so its code suite at 0 (`pilot-14-code-pp0`) runs with the RFT arms' evaluations, for pairing R-E and G-E with their start.
- **Panel:** at every checkpoint, as above, so each arm has a curve on all four kinds of question.
- **Do the edits survive training** (question 3)?
  - **Activation probes:** a fixed set of about 200 teacher-forced texts. For each edited neuron, record its activation where it is meant to fire and where it isn't, in E and in G-E. The neurons are the loop-exit columns in layers 10/13/14, layer-23 neurons 3109/2664/1965/2621, the layer-22 pair 2979/2055, and the name gates.
  - **Removal:** G-E − ΔE, scored by its panel. If the edits still matter after training, removing them costs about what it costs E.
  - **How far training moved each edited row:** in norm, and relative to the rest of its tensor.
- **Portability** (question 4): T = G-O + ΔE, with its panel first, then held-out.
- **Paper data:** `collect.py` gains `trainings.csv` (config, reward and length per step, checkpoints, GPU hours), the arms' evaluations, and lineage edges for training.
  - **Done 2026-11-07:** `trainings.csv` (one row per arm), `rft_rounds.csv`, `rft_steps.csv`, `grpo_steps.csv`, `checkpoints.csv` (every check, paired with the arm's start), the arms in `lineage.csv`, and the small records under `paper/data/train/`. The held-out evaluations of the arms land in `evals.csv` like the agent runs'.

## Infrastructure

1. **A training image, `wsp-train:1`** (built 2026-11-07, `train/Dockerfile`, versions in `train/freeze.txt`): `wsp-runtime:1` plus TRL 1.17.1, liger-kernel 0.8.5 and flash-linear-attention 0.6.0.
   - **Pinned at the base image's versions:** torch 2.13, transformers 5.19, vLLM 0.32, triton, numpy and the tokenizer stack. The evaluation and the harness keep using `wsp-runtime:1`.
   - **Why the extra kernels:** 18 of the model's 24 layers are linear attention (Gated DeltaNet). Without flash-linear-attention, transformers falls back to a slow PyTorch path. causal-conv1d was left out: its PyTorch path is a small convolution.
   - **The training loop is our own** (`train/grpo.py`, decided 2026-11-07), not TRL's GRPO. It reuses what the harness has already proven on this model and these versions: vLLM with the harness's weight-swap extension, the HF model built like `gpu_server.py`'s, and the evaluation's grader. TRL stays installed for reference.
2. **Log-probs in chunks:** the vocabulary has 248,320 entries. Full fp32 logits for a 16k-token answer would take 16 GB.
3. **Agreement check before any run:** on 50 sampled answers, the trainer's log-probs must match vLLM's to a small tolerance. It's done once per image and recorded.
   - **Done 2026-11-07 on the 5080** (`train/check_logprobs.py`, `train/checks/logprobs-5080.json`): over 50 answers (138,710 tokens), the mean absolute difference is 0.015, the 99th percentile 0.12 and the largest 0.31. The probability ratio is within 0.89–1.12 for 98% of tokens.
   - **Kernels:** the 18 linear-attention layers run on flash-linear-attention's kernels.
   - **Speed:** a forward and backward pass on the 5080 takes 0.5 s at 4k tokens, 1.1 s at 8k and 2.5 s at 16k, with a peak of 6.5 GB. The first input of 8k and 16k tokens compiles once, taking 7 s and 14 s; new lengths after that do not recompile.
4. **Grading:** the evaluation's sandbox (unprivileged, 2 GB memory, time limits), run as a pool on the server's 16 cores.
   - **Which grader:** the practice grader the agent used, `harness/grade.py`, which the panel's code suite also uses. The older copy in `scripts/` lacks its 2026-11-06 fixes: one BLAS thread per solution (otherwise a dozen solutions importing numpy at once fail with ImportError), and null bytes and deeply nested code graded instead of raising.
   - **Reading answers that reopen their reasoning** (found 2026-11-07 in start-E's check; the recipe is unchanged):
     - **The rules:** every code grader here takes the last code block after the first `</think>`: the practice grader, which is the reward, and the held-out code suites (MBPP+ and HumanEval+ through `harness/grade_run.py`, LiveCodeBench through `eval/grade_suites.py`). The held-out MMLU-Pro, MATH-500 and IFEval read the text after the last `</think>`.
     - **Corrected:** a first version of this note said the held-out code suites read after the last `</think>`. They read after the first, like the reward.
     - **Same on O:** the two readings extract the same code from every validation answer of start-O and R-O round 1, none of which has a second `</think>`.
     - **Not on E:** 608 of E's 965 finished answers reopen their reasoning after `</think>`, with draft code blocks, and the readings extract different code from 365.
     - **What it costs E** (`train/grader_rules.py`, `train/checks/grader-rules.json`): reading its code after the last `</think>` adds 9 passes, all on APPS. Validation goes from 10.0% to 10.9%, +0.89 (0.29) paired over the 252 problems. Of the 384 answers that fail to parse, 163 have no code after the last `</think>` and 111 are wrong answers.
     - **So:** the format costs E little, and the reward and the held-out code suites read it the same way.
5. **Memory on the 5090:**
   - The trainable weights (the 24 blocks, 498M parameters) with fp32 master copies, gradients and AdamW state take about 10 GB.
   - vLLM's weights and cache get about 10 GB.
   - Activations at one sequence per micro-batch, with checkpointing, fit in the remaining 10 GB.
6. **Weight push (GRPO) checked 2026-11-07** (`train/check_push.py`, `train/checks/push-5080.json`, passed). After each step, vLLM must hold the trainer's new weights. The check:
   - **Change:** it scaled the 24 MLP down-projections by 0.7 in the trainer and ran one step's cycle (vLLM asleep, a forward and backward pass, vLLM awake, push). vLLM's log-probs moved by 0.46 on average, as much as the trainer's, and stayed within 0.02 of the trainer's.
   - **Undo:** undoing the change brought vLLM back to within 0.003 of where it started.
7. **The GPU memory between RFT rounds** (fixed 2026-11-07): R-O's first round ended in an out-of-memory error as vLLM woke up for round 2 on the 5090.
   - **Cause:** the graph behind the last answer's loss kept every trainable parameter and its gradient alive after the trainer was deleted, about 5.8 GiB. vLLM needs 25.6 GiB to wake up at 0.85 of the 5090.
   - **Fix:** `rft.py` frees the graph after each answer and logs the free memory before waking vLLM. R-O resumes at round 2 from its saved round-1 checkpoint. The recipe is unchanged.
8. **vLLM's share of the 5090 in GRPO** (set 2026-11-07, before either run): 0.60, not `grpo.py`'s default 0.30. The recipe is unchanged.
   - **Why:** at 0.30, vLLM's cache holds about 480k tokens, too few for 256 answers of several thousand tokens at once, so sampling would queue and recompute.
   - **Room:** unlike RFT's, the GRPO trainer stays loaded between steps. At each wake it holds about 6 GiB (fp32 blocks 1.9, AdamW state 3.7, the bf16 embedding 0.5), its gradients are freed, and the weight push adds 0.9 GiB. With two CUDA contexts, vLLM's 19.1 GiB at 0.60 leaves about 4.5 GiB of the card's 31.8 GiB free.
   - **Fallback:** if a run ends out of GPU memory, its second attempt and every later run use 0.30. `grpo.py` records the free memory before each wake in `steps.jsonl`.
9. **Two safeguards in `grpo.py`** (2026-11-07, before either run; neither changes a healthy step):
   - **Non-finite gradients:** a mini-batch whose gradient norm is NaN or infinite is skipped rather than written into every weight, and counted as `updates_skipped`.
   - **Resuming:** a run resumes from its last checkpoint (every 25 steps). The steps logged after it are redone, so their rows move to `steps.redone.jsonl` and `steps.jsonl` keeps one row per step.

## Data hygiene

- **Overlap checks** (`train/prepare_data.py`, results in `train/data/overlap.json`): each practice problem is compared with MBPP+, HumanEval+, LiveCodeBench `test6` and the panel's MBPP tasks. APPS is from 2021 and the LiveCodeBench problems are from 2025-01 to 2025-04.
  - **The rule:** a problem is dropped if a quarter of its informative 13-grams are shared, if its task sentence's content words have a Jaccard similarity of 0.75 or more with a held-out task's, or if it has the same function name and a Jaccard of 0.5 or more.
  - **Why not plain 13-grams:** a first version that dropped any shared 13-gram caught 253 problems, nearly all on contest boilerplate ("given from standard input in the following format") and sample numbers.
  - **What it found:** MBPP repeats tasks across its splits, word for word (`change_date_format`, `harmonic_sum`) or as template variants (the fourth power against the fifth, 28 days against 31). Training could memorise one and reuse it on its held-out twin, which an edit chosen by forward passes could not. So variants are dropped too.
  - **Result:** 49 MBPP problems dropped and no APPS problem. They form 35 matching pairs with MBPP+ or HumanEval+ tasks and 23 with panel tasks; some problems match both.
- **Nothing held out goes in:** no held-out problem, answer or score enters training or the choice of checkpoint.
- **No solutions written by Claude:** the comparison is the model against itself, and Anthropic's terms restrict training models on Claude's outputs. Distilling from an open-weight teacher is out of scope for this phase.

## Timeline (5090 unless noted)

| When | Step | GPU |
|---|---|---|
| Now, while `pilot-19` runs | 1. Build `wsp-train:1`. Make the validation split and run the overlap checks. Run the agreement check, and 5 GRPO steps at 4k tokens. | 5080 |
| Day 1, after `pilot-19`'s result | 2. Freeze E and commit this plan to both repos. Run R-E and R-O and their evaluations. | about 22 h |
| Day 2 | 3. GRPO smoke run: 20 steps from E, to measure throughput and fix the step count and length cap. Then G-E. | about 26 h |
| Days 3–4 | 4. G-O. From O, early answers often run to the length cap, so its steps are slower until it learns to stop. | about 25–50 h |
| Day 5 | 5. Held-out evaluations of G-E, G-O and T, about 2.5 h each. Then the analysis, and both repos. | about 8 h |

- **Total:** about 80–110 GPU-hours on the 5090, so about 4 days with llama-swap stopped. Checkpoint panels and validation run on the 5080 throughout, at about 35 minutes per checkpoint.
- **Measured on day 1:**
  - **Checks:** start-O's check took 4.4 hours on the 5080, not 35 minutes. 76% of its validation answers ran to the 32k cap, and the 5080 generated 2,700 tokens a second. Checks of models that finish their answers should be much faster.
  - **R-O:** round 1 took 5.3 hours on the 5090: 152M tokens sampled at about 8,100 tokens a second (79% of answers ran to the 16k cap), then 3 minutes of fine-tuning.
  - **R-E:** its first 512 problems took 55–59% of R-O's time (832 s and 2,126 s against 1,421 s and 3,862 s for the two chunks of 256), because E's answers are shorter. That puts a round from E at about 3 hours.
  - **R-O's round-1 check** took 2.4 hours on the 5080, against 4.4 for start-O, because the trained model finishes its answers.
  - **start-E's check** took 1.1 hours. Only 4% of E's validation answers ran to the cap.
  - **The RFT arms' held-out evaluations** were moved to between G-E and G-O (2026-11-07). They take about 6 hours: the five suites at 1.5 for each arm (about 2.2 hours each, as `pilot-19`'s took), and the code suite at 0 for each arm and for E.

## Risks

- **G-O may barely learn.** O passes about 4.5% of practice problems and runs most answers to the length cap, so most of its groups get no reward signal at first.
  - That is itself a result: the edits as a warm start for RL. But it can burn days.
  - **Stop rule:** if G-O's validation pass rate hasn't moved by step 50, stop it and report it as is.
  - **Made precise on 2026-11-07, before the run:** once G-O's step-50 check is graded, G-O stops unless its validation pass rate is above start-O's by more than two standard errors. `grpo-queue-2026-11-07.sh` applies it.
- **Training without the presence penalty** may change behaviour under it. The checkpoint panels at 1.5 show this early, and the code suite at presence penalty 0 shows the other side.
- **Overfitting the practice set:** about 1,450 training problems, 6 passes each. The validation split and the panel watch for it.
- **Library fit:** the software stack is recent and the model is a hybrid architecture. The fallback loop above is the plan B, and step 1 finds out before the 5090 is needed.
- **The estimates are rough:** about 8,000 tokens a second on the 5090 at batch 512, from the evaluation logs. The smoke run replaces them with measurements.

## Deviations

- **2026-11-08 16:24 UTC, G-O held**, before any trained arm's held-out evaluation:
  - **What G-E learned:** its checks showed it printing the example output from APPS statements. 956 of the 1,200 APPS training problems and 156 of the 200 APPS validation problems are tested only on the examples their statement shows. So the reward and the validation pass rate both count such an answer as a pass.
  - **What it's worth:** on the 96 validation problems where copying can't pass, G-E moved +1.8 (2.5) at step 50, while its validation pass rate moved +51.9 (3.1) (`train/copy_check.py`).
  - **The hold:** G-O would train on the same reward, so `train/runs/G-O/stopped` holds it until the reward is fixed.
  - **G-E** runs to its end for the record. Its checks' validation pass rate counts the copying, so the checkpoint rule as written can't choose a G-E checkpoint.
- **2026-11-08 20:30 UTC, the reward fixed with tests the statements don't show; G-O and G-E2 queued.** This was done on the user's go-ahead, before any GRPO arm's held-out evaluation.
  - **The tests** (`train/make_tests.py` in the grading sandbox; `train/make-tests.sh` builds them):
    - **New inputs:** each APPS example input is mutated. Numbers are nudged or redrawn within the range their line or column uses. Letters and grid characters are swapped for others the input uses. Lines, tokens per line, counts and query types stay fixed.
    - **New tests:** an input becomes a test when 8 of the problem's APPS human solutions all exit cleanly and print the same output. Only solutions that pass the problem's examples are used. Each problem gets up to 10 new tests.
    - **No model involved:** the expected outputs come from human solutions. TACO's tests were considered and ruled out, since GPT-4 wrote their inputs.
  - **How often a new test is wrong:** the solutions beyond the first 8 are held out. 96.6% of them pass every new test of their problem: 4,606 of 4,768 on training problems, 802 of 833 on validation ones.
  - **Usable problems:**
    - **The rule** (fixed before any check was regraded): a problem needs at least 3 unseen tests, at least 2 different outputs among them, and an unseen output that no shown test has. Then printing an example's output, or any one fixed output, fails it.
    - **Count:** 999 of the 1,200 APPS training problems and 168 of the 200 validation ones are usable.
    - **MBPP** is unchanged and usable, since its prompt shows only 1 of each problem's 3 tests.
  - **Data:**
    - `train/data2/train.jsonl` is GRPO's training set: the 218 MBPP problems and the 999 usable APPS ones, with 11.7 tests per APPS problem on average.
    - `train/data2/val.jsonl` has all 252 validation problems with the new tests, each marked usable or not.
    - `train/data2/build.json` has the counts and md5s.
  - **Validation from now on:** `val2`, the pass rate on data2's tests over the 220 usable validation problems (52 MBPP, 168 APPS), from `train/ckpt_grade.py`.
    - The checkpoint rule and G-O's step-50 rule read `val2`.
    - Every earlier check was regraded on it by `train/regrade-val2.sh`.
  - **R-O:** its held-out evaluation started at 19:25 on round 2, which the old validation had chosen. On `val2`, round 2 is also the better checkpoint: 13.4% against 10.1%. So the corrected rule picks the same one.
  - **R-E** still gets no evaluation. The panel part of the rule is unchanged.
  - **G-E:** by the corrected rule its checkpoint would be step 25, at +1.7 (1.1) on `val2`. G-E2 replaces it as the GRPO arm from E, so G-E gets no held-out evaluation.
  - **The runs** (`train/grpo-queue-2026-11-08.sh`):
    - **What runs:** G-O, then G-E2 (G-E again, from E), each with `--data /train/data2/train.jsonl`. The recipe is otherwise unchanged.
    - **When:** they start once `grpo-queue-2026-11-07.sh` has run the RFT arms' held-out evaluations.
    - **Why G-O first:** it has not run at all, and T needs it.
    - **G-E** stays in the record as the run on the reward that could be gamed.
  - **The pilot** (50 examples-only problems, before the build):
    - 42 got new tests.
    - Of G-E's answers that passed the sample's 20 validation problems, 238 of 247 fail the new tests.
    - A first version also mutated counts and query types. 8% of held-out human solutions failed its tests, so counts and query types are now kept fixed.
- **2026-11-10 15:50 UTC, the GRPO arms' held-out evaluations, T and the removal set up.** This was before any GRPO arm's held-out evaluation, and before T's check had finished.
  - **G-O's checkpoint:** step 100, by the rule on `val2`.
    - **Its scores:** `val2` 19.7%, +14.9 (1.9) against start-O. Panel mean 57.0, +30.3 (1.6).
    - **The comparison:** every checkpoint's panel was far above start-O's, so the rule compared all six on `val2`.
    - **The record:** `train/evals/G-O/choice.json`, written by `train/arm_choice.py`. It applies the rule as `train/eval-rft-arms.sh` did, but on `val2`.
  - **Checks that failed:** a checkpoint whose check failed twice can't be judged, so the choice leaves it out.
  - **ΔE is bf16(E) − O.**
    - **Why:** E's submission holds fp32 values, but the evaluation and the trainer both round them to bf16 when they load them (`harness/gpu_server.py`, `train/common.py`). So bf16(E) is the edit the model ran with.
    - **What rounding drops:** 4,777 of the 2,115,584 entries E changes. All are in the three q_proj tensors, where the change was under half a bf16 step.
  - **T** = G-O/step_100 + ΔE (`train/combine.py`; the weights are in `train/combined/T-step_100`).
    - **Its entries:** each is the bf16 value nearest the exact sum, with ties going to the even value.
    - **The check:** a second computation in torch agrees with it. The 309 other tensors are byte-identical to the checkpoint's, and the 9 edited ones are within half a bf16 step.
  - **"Clearly worse":** the plan didn't define it.
    - **The definition:** T's panel mean is more than 2 standard errors below G-O/step_100's, paired over the panel's 1,000 questions (`arm_choice.py panel-ok`).
    - **When:** fixed before T's check on the 5080 started at 15:45 UTC.
    - **The evaluation it decides:** T's one held-out evaluation, the five suites at presence penalty 1.5 (`T-pp15`).
  - **The removal** (question 3):
    - **What runs:** G-E2's chosen checkpoint − ΔE, built the same way. It is checked on the 5080 and paired with that checkpoint (`train/evals/G-E2-minus-E/`).
    - **If no G-E2 checkpoint qualifies:** the removal uses G-E2's last checked checkpoint.
    - **The comparison:** E − ΔE is exactly O. So start-O against start-E, on the same panel, is what removing ΔE costs E.
  - **The order:** `train/eval-grpo-arms.sh`, which `train/queue/after-G-E2.sh` runs when G-E2 ends.
    - **On the 5090:** G-O's two evaluations (`G-O-pp15`, `G-O-code-pp0`), then G-E2's once its checks are done, then T's.
    - **On the 5080:** the removal's check, at the same time.
    - **The cost:** about 8–12 hours of the 5090. R-O's two evaluations took 2.7 hours, and the GRPO arms write longer answers.
- **2026-11-10 19:35 UTC, the other analyses of question 3 built: the activation probes and the row movements.** This was before G-E2's checkpoint is chosen, and before any GRPO arm's held-out evaluation.
  - **The probes** (`train/probes.py`, run per model by `train/run-probes.sh`; results in `train/analysis/probes/`):
    - **The texts:** 201 answers that O and E wrote at their starting checks, to validation problems and panel questions, drawn once with seed 20261110. The plan said about 200. The instruction-following answers fell short: only 21 of E's 244 repeat themselves at 40 or more positions after the thinking closed, against a quota of 40. They are re-tokenized with their prompts and teacher-forced. `index.json` has their md5 and counts; the texts themselves stay out of git, since they hold dataset problems.
    - **The units:** the ones E changes. These are the five loop detectors and their combined drive, the answer-end neuron, the layer-23 neurons 3109 and 2664, the pair 1965 + 2621, and the name gates (eight heads in layers 15, 19 and 23).
    - **Left out:** the plan also listed the layer-22 pair 2979/2055. That is `pilot-18`'s change, which E (`pilot-14`'s edit) doesn't have.
    - **Where each is meant to fire,** from the agent's own reports:
      - verbatim repetition, for the detectors and 3109;
      - the closing fence that ends the turn, for 1420;
      - the thinking closed at least twice, the last segment under 60 tokens, for 2664;
      - "answer is" or "Answer:" after the thinking closed in a math answer, for the pair;
      - a function name being written after def, for the gates.
      `probes.py`'s docstring defines each exactly.
    - **The measure:** for each unit, the AUROC between its labelled positions and the rest, and its mean activation there. On the labels each text counts once. That weighting was decided before any model was probed, after the build showed that 5 texts hold 65% of the degen and statement positions.
    - **The models:** every model with weights, not only E and G-E: O, E, T, every checkpoint of R-O, R-E, G-O, G-E and G-E2, and the removal. G-E2 is the arm from E, as decided on 2026-11-08.
  - **The row movements** (`train/row_moves.py`; `train/analysis/rows.md`), for each unit:
    - **kept:** how much of ΔE is still there along its own direction;
    - **moved:** training's change to the unit, against the edit's size;
    - **cos:** the cosine of that change with ΔE;
    - **vs_rest:** the change against the median row or column of the same tensor.
  - **When they run:**
    - **Now:** on the models that have weights so far, on the 5080 between G-E2's checks (`scratch/probes-2026-11-10.sh`). The row movements need no GPU.
    - **After G-E2 ends:** `train/eval-grpo-arms.sh` probes the rest (G-E2's later checkpoints and the removal) on the 5080 after the removal's check, then writes both summaries.
- **2026-11-10 20:40 UTC, the probes corrected after the sanity check on E.** This was before any other model's probe results were read.
  - **What the check found:** on E, three units scored at or below chance: neuron 2664 (AUROC 0.15), the pair (0.48), and the layer-15 gate (0.57 for its detector). Their labels or their measure didn't match how the agent built them, according to `pilot-13`'s and `pilot-14`'s notes (`explanations.md`):
    - **2664** ends the turn in a degenerate re-close cycle. In E's own answers, it fires at the last position of 17 of the 22 finished answers that end in that state, with activations of 2 to 16. Where it fires, the answer ends, so the re-closes left in the texts are where it didn't. The label had every position in the state.
    - **The layer-15 gate** was rebuilt by `pilot-13` to open only at word-start positions inside a name, where the next piece begins with "_" or a capital letter. The layer-19 and layer-23 gates open at the def position and at word-start positions, and were fitted against mid-word positions and the name's last token. The label had every position inside the name, def included.
    - **The four layer-23 neurons are switches:** silu of a thresholded readout. Silu dips below zero just under the threshold, so near-misses rank below positions where the neuron is off. That pulled the pair's AUROC to 0.48, though it fires at 27% of its positions and at 0.002% of the rest.
  - **The fix:**
    - **degen** is now the last position of a finished answer, kept whole, that ends in the degenerate state. The state's other positions are left out.
    - **name** is now the def and word-start positions (layers 19 and 23), and **name-inner** the word-start positions only (layer 15). Mid-word positions and the name's last token are controls.
    - **The switches' AUROC** (3109, 2664 and the pair) is now taken on the gate's pre-activation, which is monotone. Their mean activation stays, and the share of positions where each is on (activation above 0.5) is added.
    - **The texts are the same 201,** with the same seed and selection. Only the stored labels changed (md5 578ce0c4).
  - **E after the fix:** AUROC 0.99 for 2664, 1.00 for the pair and 0.88 for 3109; 0.99 and 0.94 for the layer-15 detector and gate; 0.99–1.00 for the other gates and detectors. The loop detectors are unchanged at 0.66–0.84 (their combined drive 0.88). They are the model's own neurons, and verbatim repetition is only a proxy for a loop.
  - **O and T-step_100 had been probed with the first version.** Their results were deleted unread, and every model is probed again.
