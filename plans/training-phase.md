# Training phase: plan (draft, 2026-11-07)

**Status:** approved as recommended on 2026-11-07, about 05:00 UTC. Step 1 is done (results under **Infrastructure**). E was frozen at 16:30 UTC as `pilot-14`'s edit (below). R-O started at 09:05 UTC and R-E at 16:42 UTC. On the user's go-ahead, GRPO (G-E, then G-O) was queued at 17:51 UTC behind the RFT runs on the 5090 (`train/grpo-queue-2026-11-07.sh`). The RFT arms' held-out evaluations run between G-E and G-O (`train/eval-rft-arms.sh`, also on the user's go-ahead).

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
   - **Where it differs from the held-out suites** (found 2026-11-07 in start-E's check; the recipe is unchanged):
     - **The rules:** the practice grader takes the last code block after the first `</think>`. The held-out suites read the text after the last `</think>`.
     - **Same on O:** the two extract the same code from every validation answer of start-O and R-O round 1, none of which has a second `</think>`.
     - **Not on E:** 608 of E's 965 finished answers reopen their reasoning after `</think>`, with draft code blocks, and the rules extract different code from 365. 133 answers that fail to parse under the practice rule parse under the held-out one, and 28 the other way.
     - **So:** the reward is the stricter grader on E's format, and what it rewards, one clean final answer, both graders accept.
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

None yet.
