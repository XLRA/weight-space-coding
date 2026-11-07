You do not start from the unedited model. /work/edited already holds an earlier run's final edit, the starting edit (nine tensors), built in several steps by five earlier runs. /start holds a read-only copy of the same files and /start/report.md. The report explains each part, contains the earlier runs' final reports, and gives a held-out evaluation of the edit.

THE STARTING EDIT, in short (the report has the details):
- Loop exit and answer end (E2, E3): repetition-detector neurons in layers 10, 13 and 14 are wired to the end-of-thinking direction. Layer 13 neuron 1420 ends the turn after the first solution block closes.
- Layer 23 MLP neurons:
  - 3109, an end-of-turn brake for prose answers;
  - 2664, a stopper for degenerate re-close cycles;
  - 1965 and 2621, a pair that boxes answer statements after the thinking.
- Function-name gates: rank-1 edits of the output gates of attention heads make the model copy the given function name after "def". They sit in self_attn.q_proj of layer 19 heads 0, 2, 5, 6, 7, layer 15 head 7 and layer 23 heads 0, 1.

THE EVALUATION IS BROADER THAN "HOW IT WILL BE MEASURED" SAYS. Besides the code benchmarks, the same protocol and settings are used on four more held-out suites, each with its own user message. You never see these suites.
- Instruction-following prompts: a writing task with one or more verifiable instructions, checked by strict automatic checkers on the answer after the last </think>.
- Multiple-choice questions: ten options, many subjects, answered with "The answer is (X)" after the reasoning.
- Competition math, graded on the last \boxed{} after the last </think>.
- Competitive-programming problems: standard input and function form. They are graded like the code benchmarks, on the last ```python block after </think>.

THIS RUN CONSOLIDATES. Three runs each started from this starting edit and added one targeted change to it. Their final edits are your candidates:

| Folder | Target | What it changed, relative to the starting edit |
|---|---|---|
| /work/candidates/stopper/ | turns that end with no answer | One row of layers.23.mlp.gate_proj, row 2664 (1,024 weights): the degenerate-cycle stopper now fires at a segment boundary inside the cycle instead of right after a </think>, so a stopped turn is not left empty. |
| /work/candidates/copy/ | copying the request in prose answers | Three rank-1 terms in self_attn.q_proj: layer 15 head 7 (query and gate rows) and layer 11 head 1 (query rows). Where the answer reproduces the request, these two induction heads read the prompt instead of the thinking's corrupted drafts. |
| /work/candidates/identifiers/ | identifiers in code answers | Two neurons of the layer-22 MLP, 2979 and 2055: their rows of gate_proj and up_proj and their columns of down_proj (6,144 weights). A capped pair that fires on the last token of the prompt's function name in a `def` line and pushes toward `(`, so the first def no longer runs past the given name (`min_Num` → `min_Num_sum_even`). |
| /work/candidates/sum/ | all three | The starting edit plus the three changes added together. |

- **Folder contents.** Each folder holds the tensors that differ from the starting edit, as complete .npy tensors like the ones save_edit writes, and report.md, the run's final report (sum/ has no report).
- **Changes.** A change is the folder's tensor minus /start/<name>.npy, or minus load_original(name) where /start has no such tensor.
- **Installing.** To install a candidate, save its tensors with save_edit(name, torch.from_numpy(np.load(path))). Use drop_edit or the /start tensors to go back.

HELD-OUT RESULTS. The stopper and the copy change were evaluated on the held-out suites with this run's sampling settings, paired against the starting edit (standard errors in brackets):

| Suite | Starting edit | stopper − start | copy − start |
|---|---|---|---|
| Code benchmarks | 32.3% | −0.97 (0.77) | −0.78 (0.91) |
| Code, HumanEval-style prompts only | 26.4% | −0.15 (1.38) | +0.46 (1.54) |
| Multiple choice | 45.2% | −4.60 (1.30) | −1.00 (1.36) |
| Instruction following | 58.2% | +2.03 (1.43) | +2.77 (1.64) |
| Competitive programming | 7.0% | −0.57 (0.78) | +0.43 (0.74) |
| Competition math | 56.9% | −0.85 (0.91) | −0.90 (0.95) |

- **The stopper did what it was for.** Instruction-following answers with nothing after the last </think> fell from 44 to 17 of 1,082.
- **But it cost 4.6 points on multiple choice.**
  - The lost questions now get a wrong letter, not a missing letter or a cut: 106 went from the right letter to a wrong one, 59 the other way. Answers without a letter fell from 20 to 6 of 1,000, and cut answers rose from 3 to 10.
  - The losses are spread over all subjects. They are long answers, about 8,000 tokens.
  - Its report tested multiple choice only with a variant it then dropped (v2, with guard neuron 3222), not with the submitted version.
- **Math boxes.** With the stopper, math answers without a final box rose from 289 to 328 of 2,000, while cut answers fell from 90 to 69.
- **The copy change** gained 2.8 points on instruction following. The other suites moved by a point or less, within noise.
- **The identifier change** has not been evaluated held out. On its run's own practice problems (paired against the starting edit, 512 MBPP-style answers per model), the renames it targets fell from 20 to 7. Pass rates moved +1.2 (2.0) on MBPP-style prompts and −2.8 (2.6) on HumanEval-style prompts, both within noise.
- **No two candidates change the same tensor**, so the sum is each change in place, unaltered.

YOUR GOAL is the edit that does best on the evaluation as a whole: the five suites together, with none clearly worse than the starting edit.
- Decide which changes to keep, whole or in part, and at what strength.
- Where a change helps one suite and costs another, find out why if you can, and keep the part that helps. For example: what in the stopper's new gate changes multiple-choice answers, and does a narrower version keep the instruction-following gain without the loss? Otherwise drop the change.
- New mechanisms for new targets are not this run's task.

THE PANEL is your main measurement (forward pass (c) above). Its questions are of the same kinds as the held-out suites. The starting edit's panel answers were generated before this run, with the same settings and seeds: choice 44.8%, math 58.0%, instructions 79.2%, code 21.2%. So each panel pass generates only your edit and takes about 15 minutes.

A plan that fits the budget:
1. One panel pass for each candidate and for the sum.
2. Compare the candidates with each other question by question, with /work/panel_<k>.jsonl.
3. To understand a loss you want to repair, use your own generations and teacher-forced reads. For example: ten-option multiple-choice questions you write in the "The answer is (X)" format, hard math problems with known answers, prose prompts with verifiable instructions. Do not reproduce public benchmark items from memory.
4. A final panel pass of your final edit, and a paired generation on practice code problems not used before ("baseline": "start").

HOW MUCH THE PANEL CAN TELL. Two checks before this run measured it.
- **The same weights give the same answers.** Three panel generations of the starting edit, two of them in separate passes, gave byte-identical answers to all 1,000 questions. A second panel pass of an unchanged edit tells you nothing new.
- **Any change gives new answers.** With one weight moved by a single bf16 rounding step (1 of 3.7 million entries of a tensor nothing else touches), none of the 1,000 answers stayed the same; they parted after the first 2 to 4% of their text. That edit does nothing, yet the panel read: choice +1.6 (SE 2.7), math +1.6 (2.0), instructions −4.8 (2.6), code −3.2 (2.3), mean over the four −1.2 (1.2), with 148 questions flipped between passing and failing.
- **So read differences in standard errors.** The reported standard error is about right (all four null readings were within two of them). Only a difference beyond about two standard errors is evidence: about 5 points on one suite, about 2.5 on the mean over the four. Smaller differences, in either direction, are what a change that does nothing produces.
- Every choice made by comparing panel results makes the chosen edit's panel numbers look a little better than they will be held out. So choose details (strengths, thresholds, rows) on your own prompts and practice problems. Use the panel for a few decisions: which changes to keep, and whether a repair worked.

COMPARING WITH THE STARTING EDIT. "model": "start" runs the starting edit: always the files in /start, whatever /work/edited holds.
- In a generation with "model": "both", add "baseline": "start" to run the starting edit and your current edits on the same items and get the paired difference edited − start. Without it, the baseline is the unedited model, as described above.
- Teacher-forced passes take "model": "start" too.
- The panel compares with the starting edit by default.
- In the kernel, the starting edit's tensors are /start/<tensor name>.npy.

SETTINGS OF THIS RUN. Every generation uses presence penalty 1.5, and so does the evaluation. You have {{N_PASSES}} forward passes.

YOUR FINAL REPORT, in addition to what HOW THE SESSION RUNS asks, gives:
- every tensor of your final edit that differs from the starting edit, and for each part where it comes from (which candidate, at what strength) or what you changed and why;
- which changes you kept and which you dropped, with the evidence for each;
- the final panel result of your final edit against the starting edit, per suite;
- the paired code result on practice problems not used to choose;
- what you expect on each held-out suite.
