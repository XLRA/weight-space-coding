# Editing after training: plan (2026-11-12)

**Status:** approved on 2026-11-12. `pilot-20` ran from 11:51 to 18:36 UTC, and its evaluations from 18:37 to 21:52 UTC. **Result: level with G-O** ("Result", below). Before the run, the harness tests passed (`runs/e2e-trained` and `e2e-trained-2`).

**Changes:** any change to this plan after `pilot-20` starts must be listed under **Deviations** at the end, with its date and reason.

**Names used below** (as in `plans/training-phase.md`):
- **O:** the unedited Qwen3.5-0.8B.
- **E:** the weight-space edit, `pilot-14`'s (`runs/pilot-14/submission`). **ΔE** is E minus O.
- **G-O:** GRPO from O, its step-100 checkpoint (`train/runs/G-O/checkpoints/step_100`), the best model so far.
- **T:** G-O plus ΔE.

## The question

Can the agent's weight editing improve on the best trained model?

- **What we know:** G-O beat E on every held-out suite but MMLU-Pro. Adding ΔE to G-O (T) ended its loops, but scored 0.6 to 2.9 points below G-O on every suite. Over the five suites T is −1.86 points (SE 0.57) against G-O.
- **Why T lost** (G-O-pp15 and T-pp15, `scratch/go_failures.py` and `go_failures2.py`): after E's loop exit closes the thinking, the trained model often opens it again and doesn't settle.
  - 43% of T's finished LiveCodeBench answers close the thinking two or more times, against 1.1% of G-O's, and 4% of those pass.
  - On MATH-500 the shares are 10.8% for T and 4.6% for G-O.
- **What `pilot-20` tests:** E was searched on O. Instead of carrying it over, an editing session starts from G-O and works on G-O's own failures. This is the training phase's fourth question (are edits portable?) from the other side.

## What G-O still gets wrong

From G-O-pp15 (shares only; the brief gives the agent these):
- **LiveCodeBench:**
  - 33% of the answers reach the 32k cap: 36% of the medium problems' and 44% of the hard ones'.
  - All of the cut answers are flagged as loops, and 96% never close the thinking.
  - None of G-O's passing answers is longer than 16k tokens (90th percentile 6,023).
- **IFEval:** 9% reach the cap, nearly all of them loops.
- **MATH-500:**
  - 17% reach the cap, about half of them loops and half long reasoning that doesn't repeat.
  - 12% of the right answers are longer than 16k tokens, so long reasoning there is often useful.
- **The rest:** MBPP+ and HumanEval+ 4%, MMLU-Pro under 1%.
- **Why training left the loops:** GRPO sampled at temperature 1.0 with no presence penalty, and cut answers at 16,384 tokens. It never saw the evaluation's sampling or its 32k cap.

## The run: `pilot-20`

- **Start:** G-O's step-100 checkpoint, all 318 decoder tensors (`driver.py --start-from`). In the session, `"model": "start"` and `"baseline": "start"` run G-O, and `"original"` still runs O.
- **Agent:** Opus 5.5 in Claude Code on the subscription, effort max, the conversation restarted past 400k tokens. The prompt is the standard one, plus the brief.
- **Budget:** 24 forward passes, as `pilot-19`. The panel is on (`--panel`), presence penalty 1.5, on the 5090.
- **Practice problems:** `practice/val2` (`train/held_back_practice.py`, `driver.py --practice`).
  - **What it holds:** the 220 usable validation problems (52 MBPP, 168 APPS), with data2's tests.
  - **Why only these:** G-O trained on the other practice problems, each about five times. On those, an edit's effect would be confounded with what training memorised.
  - **A caveat:** val2 also chose G-O's checkpoint, so G-O isn't entirely blind to it.
  - **Not included:** 233 more APPS problems that G-O never trained on, because their tests are weak (data2 couldn't make new ones, so many have only the statement's examples). Also left out are the 49 MBPP problems that the overlap check dropped.
  - **Enough?** A paired generation holds at most 256 answers per model, so 220 problems cover several disjoint passes.
- **The brief** (`harness/briefs/pilot-20.md`):
  - It says what the start is, how to undo a change, and G-O's held-out failure shares.
  - It says what T did, and sets the goal: end the loops so the thinking closes once and a complete answer follows. The answers must not reopen, long reasoning still making progress must not be cut, and answers that already finish must not change.
  - **This makes the evaluation a little less blind,** as with `pilot-11` to `pilot-19`. The brief gives failure patterns and rough shares, never problems or per-problem results.
- **E for reference:** `work/E/` holds E's nine tensors and `pilot-14`'s report, so the agent can study or reuse parts of it.
- **Harness changes** (2026-11-12; tested in `runs/e2e-trained-2` with `harness/tests/make_trained_start.py`, on the 5080 with the replay backend):
  - **`driver.py --practice DIR`** mounts another practice set at `/practice`. An `about.txt` in it replaces the prompt's description of the files.
  - **`kernel_prelude.py`:**
    - **`restore_start(name)`** puts the start's tensor back. `drop_edit` puts back O's, which with a trained start also removes training from that tensor.
    - **`changed()`** lists the tensors that differ from the start.
    - **The manifest** adds `changed_from_start`, and the restart message lists only the tensors that differ from the start.
    - **The whole-tensor scaling rule** also applies to the start's tensor.
  - **`eval/compare.py`** also prints the mean of the five suites' differences, with its SE.
- **Launch:** `scratch/launch-pilot-20.sh` checks the start against the files G-O-pp15 evaluated, checks the practice set and the brief, copies E into `work/E/`, and starts the driver and both hand-offs.

## Evaluation, fixed before the run

- **When:** only if the run ends with weights that differ from G-O's (`harness/after-run.sh`).
- **What:** one held-out evaluation, on the 5090, reusing O's answers:
  - `pilot-20-pp15`: the five suites at presence penalty 1.5 (`after-run.sh`);
  - `pilot-20-code-pp0`: the code suite at 0 (`harness/after-pilot-20.sh`).
- **Paired with G-O,** problem by problem (`eval/compare.py`). `after-pilot-20.sh` writes the results to `eval/results/pilot-20-pp15/vs-G-O.md` and `eval/results/pilot-20-code-pp0/vs-G-O.md`.
- **The main measure:** the mean of the five suites' paired differences against G-O at presence penalty 1.5. Its SE combines the suites' SEs as independent samples: √(ΣSE²)/5. This is the measure that froze E (`plans/training-phase.md`, Names).
- **Decision rule:**
  - **Better than G-O:** the mean is more than 2 SE above zero, and no suite is more than 2 SE below G-O.
  - **Worse:** the mean is more than 2 SE below zero.
  - **Level:** anything else.
- **Single suites:** each is reported with its SE, and counts as changed only beyond 2 SE.
- **Reported either way:**
  - per suite, the share of answers cut at 32k, the share that close the thinking two or more times, and the mean length;
  - the code suite at presence penalty 0, paired with G-O-code-pp0;
  - the session's cost: GPU hours, Opus output tokens and passes.
- **No reruns:** the evaluation runs once, and its result goes in the paper whatever it shows.

## What comes after

That is the user's call once the result is in. For example:
- **If `pilot-20` is better:** it becomes the best model. A further round could train from it (GRPO with the same recipe), or edit it again.
- **If it's level or worse:** the paper reports that, in this setup, editing didn't add to training. The session's notes may still show which mechanisms the trained model uses.

## Cost and time

- **The session:** about 4 to 7 hours of the 5090, with llama-swap stopped while the run holds the card. `pilot-19`'s 24 passes took 4.35 hours, and `pilot-14`'s 20 took 7.03.
  - The panel's G-O answers are generated in the agent's first panel pass, which takes about twice as long as later ones.
- **Opus:** on the subscription, not billed per token. `pilot-19` used 0.27M output tokens and `pilot-14` 0.96M. If a usage limit is reached, the driver waits for the reset.
- **The evaluation:** about 2 to 3 hours of the 5090 at presence penalty 1.5, and about 1 hour for the code suite at 0.
- **The 5080** stays free.

## Risks

- **The agent may find nothing that helps.** That is a result, and the plan reports it.
- **Selection on a small practice set:** the agent chooses on 220 problems and its own prompts. The held-out evaluation, run once, guards against that.
- **Context:** the start's 318 tensors make `edited()` long. The brief points the agent to `changed()`, and the restart message lists only the tensors it changed.
- **The panel's G-O answers** come from the same session and GPU as its edits, so the pairing is like `pilot-19`'s.

## Result (2026-11-12)

- **Level with G-O.** The mean of the five suites' paired differences is +0.68 (SE 0.53), 1.3 SE above zero, and no suite is more than 2 SE below G-O. G-O stays the best model.
  - **Per suite:** code +1.25 (0.97), MMLU-Pro +0.80 (1.35), IFEval +1.57 (1.45), LiveCodeBench −1.00 (1.12), MATH-500 +0.80 (0.91).
  - **Code at presence penalty 0:** +0.88 (0.89).
- **Against T:** +2.55 (0.54) over the five suites.
- **The edit:** 8 tensors in layers 12 and 13. It is a loop exit that a "thinking has closed" flag switches off in the answer.
- **The side measures:**
  - LiveCodeBench answers cut at the cap fell from 32.7% to 19.0%.
  - Finished answers that close the thinking two or more times stayed near G-O's: 0.7% on LiveCodeBench and 7.0% on MATH-500, against 1.1% and 4.6% (T: 43.3% and 10.8%).
- **Where the details are:** RESULTS.md, "Editing after training", has the rest of the side measures and the cost.

## Deviations

None.
