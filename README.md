# Weight-space programming for code

An AI agent edits a small language model's weights directly, with no training, to make it better at coding.

This project recreates Experiment 8 of *Sparks of Weight-Space Programming* (Takehi et al., 2026), with coding in place of math. All credit for the method goes to the original paper.

## How it works

- **The agent** is Claude Opus 5.5, at maximum effort. **The model it edits** is Qwen3.5-0.8B with thinking on, at the revision the paper used (`2fc06364715b967f1860aea9cf38778875588b17`).
- **The agent edits weights from a Python kernel.** It reads the model's tensors, changes them and saves the result. There is no gradient training.
- **It tests its ideas with 10 to 20 forward passes per session.** A pass either has the model write up to 512 answers to practice problems and grades them, or returns the model's activations and token probabilities on given text.
- **It practises on 1,719 coding problems** from MBPP and APPS, kept apart from every benchmark.
- **After the first session, each one starts from an earlier edit,** with a short brief naming a failure pattern from the last evaluation (shares only, never individual problems).
- **Every edit is scored on five benchmarks the agent never sees,** paired problem by problem against the original model.

Every rule, check and number is in [RESULTS.md](RESULTS.md).

## Results

The agent's best edit came from session `pilot-14`. We call it **E**. Held out, against the original model:

| Benchmark | What it measures | Original | Edited (E) | Change (SE) |
|---|---|---|---|---|
| MBPP+ and HumanEval+ | writing Python functions | 19.0% | 32.3% | +13.3 (1.3) |
| LiveCodeBench | contest programming problems | 1.6% | 7.0% | +5.4 (1.4) |
| MMLU-Pro | multiple-choice knowledge | 40.5% | 45.2% | +4.7 (1.5) |
| IFEval | following instructions | 22.3% | 58.2% | +36.0 (1.9) |
| MATH-500 | math problems | 6.8% | 56.9% | +50.1 (1.8) |

The paper's math edit, for comparison, gained +5.2 on MMLU-Pro, +41.3 on IFEval and +40.2 on MATH-500. Sampling uses the model card's coding settings with the paper's presence penalty of 1.5, and allows up to 32,768 new tokens.

### How E was built

E adds up four sessions' changes: ten MLP neurons and three rank-1 attention changes, in six of the model's 24 layers.

1. **`pilot-10`** (10 passes) changed the output weights of six MLP neurons so that the model stops looping in its thinking. That alone took code from 19.0% to 27.6%, and MATH-500 from 6.8% to 51.0%.
2. **`pilot-12`** (20 passes) repurposed one neuron so that prose answers don't end too early. Held out: MATH-500 +4.4.
3. **`pilot-13`** (20 passes) added three rank-1 attention changes so that the model keeps the function name the problem asks for. Held out: MBPP+ +6.6.
4. **`pilot-14`** (20 passes) started from `pilot-12` and `pilot-13` added together, which kept both gains since they touch different weights. It added three neurons that stop answers from restarting their reasoning after the thinking closes. Held out: MATH-500 +2.8.

Five later sessions tried narrower fixes. None was clearly better than `pilot-14` overall, and one cost 4.6 points on MMLU-Pro.

## What we found

- **Weight-space programming carries over to code.** The first edit was chosen only on code problems, yet it lifted math and instruction following too.
- **Most of the gain is loop control, as in the paper.** The original model often loops in its thinking until it hits the token limit: on 19% of code answers and 94% of MATH-500's. E cuts that to under 5% on every benchmark.
- **The edits unlock what the model can already do.** Where looping wasn't the obstacle, as on LiveCodeBench's medium and hard problems, nothing improved.
- **The edits are small and readable.** Each is a few neurons or a rank-1 change with a mechanism the agent can state, and edits to different weights add up without interfering.
- **A narrow edit can cost a benchmark nobody is watching.** `pilot-16` fixed its target but lost 4.6 points on MMLU-Pro, which the agent hadn't checked on the edit it kept. Later sessions got the *panel*: a fixed set of 1,000 questions across code, math, multiple choice and instructions, apart from both the practice problems and the benchmarks.
- **The agent's own measurements overstate held-out gains,** most on benchmarks unlike its practice problems.

## Editing versus training

The second phase trains the model on the same practice problems, with a recipe fixed in advance ([the plan](plans/training-phase.md)). It asks whether training beats editing, and whether the two combine. There are two methods, each started from the original model (O) and from E:

- **RFT** (rejection-sampling fine-tuning): fine-tune on the model's own passing answers.
- **GRPO:** reinforcement learning, rewarded when the tests pass.

So far:

- **RFT from O matched E on code, but not elsewhere.** Held out, it scored 32.5% on MBPP+ and HumanEval+ (E 32.3%) and 9.4% on LiveCodeBench (E 7.0%), but 9 to 22 points below E on MMLU-Pro, IFEval and MATH-500.
- **RFT from E barely helped on code,** and cost 7.8 points on the panel.
- **GRPO from E gamed its reward.** Most APPS problems are tested only on the examples their statement shows, and within 50 steps the model learned to print the example output instead of solving the problem. We built new tests from APPS's own human-written solutions, with no other model involved. GRPO from O ran on them, and GRPO from E is rerunning on them.
- **GRPO from O beat both E and RFT on our own checks.** After 100 steps its pass rate on held-back practice problems was 19.7% (E 9.4%, RFT from O 13.4%), and it scored 57.0% on the panel (E 51.1%). Its held-out evaluation is next.
- **The edit still works on a trained model.** Adding E's weight change to GRPO-trained O cut the answers that hit the token limit from 21.8% to 2.4%, at no cost on the panel. Its pass rate didn't rise.

**Status (2026-11-10):** GRPO from E finishes on 2026-11-11. Then the held-out evaluations of both GRPO models, and of GRPO from O plus the edit, run.

## Repository

| Folder | What's in it |
|---|---|
| `harness/` | The agent's loop and prompt, the kernel, the forward-pass servers and the grader |
| `runs/` | Each session's brief, the agent's explanations and report, and its edit's manifest |
| `eval/` | The held-out evaluation, the panel, and each model's report in `eval/results/` |
| `train/` | The training phase: the trainers, the checkpoint checks and the new tests' builder |
| `plans/` | The training plan: its recipe, fixed in advance, and every deviation from it |
| `paper/` | Every table and agent transcript, collected in `paper/data/` for a write-up |
| `scripts/`, `baseline/` | Practice-set preparation and the original model's baseline |
| `scratch/` | One-off analyses |

**Not included:** model weights, datasets and edit tensors. All data is from public datasets: MBPP, APPS, EvalPlus (MBPP+ and HumanEval+), MMLU-Pro, IFEval, LiveCodeBench, MATH, MATH-500 and `argilla/ifeval-like-data`. `train/prepare_data.py` rebuilds the training and validation problems, and `train/make-tests.sh` the new tests.

**Software:** vLLM 0.32.0, transformers 5.19.0, torch 2.13.0 (CUDA 13.0) and Python 3.12, on an RTX 5090 and an RTX 5080 (and a cloud L4 for one session).

## Citation

This project recreates Experiment 8 of:

> Takehi et al. *Sparks of Weight-Space Programming.* 2026.
