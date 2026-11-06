You do not start from the unedited model. /work/edited already holds an earlier run's final edit, built in several steps by four earlier runs (nine tensors in all). /start holds a read-only copy of the same files and /start/report.md. The report explains each part, contains the earlier runs' final reports, and gives a held-out evaluation of the edit. Read the report first. You may keep, change or remove any part of the starting edit.

THE STARTING EDIT, in short (the report has the details):
- Loop exit and answer end (E2, E3): repetition-detector neurons in layers 10, 13 and 14 are wired to the end-of-thinking direction. Layer 13 neuron 1420 ends the turn after the first solution block closes.
- Layer 23 MLP neurons:
  - 3109, an end-of-turn brake for prose answers;
  - 2664, a stopper for degenerate re-close cycles;
  - 1965 and 2621, a pair that boxes answer statements after the thinking.
- Function-name gates: rank-1 edits of the output gates of attention heads make the model copy the given function name after "def". They sit in self_attn.q_proj of layer 19 heads 0, 2, 5, 6, 7, layer 15 head 7 and layer 23 heads 0, 1.
  - About half of the wrong names had come from the presence penalty. The penalty lowers tokens already generated, so it pushed down name tokens the thinking had already used.

THE EVALUATION IS BROADER THAN "HOW IT WILL BE MEASURED" SAYS. Besides the code benchmarks, the same protocol and settings are used on four more held-out suites, each with its own user message. You never see these suites.
- Instruction-following prompts: a writing task with one or more verifiable instructions, checked by strict automatic checkers on the answer after the last </think>.
- Multiple-choice questions: ten options, many subjects, answered with "The answer is (X)" after the reasoning.
- Competition math, graded on the last \boxed{} after the last </think>.
- Competitive-programming problems: standard input and function form. They are graded like the code benchmarks, on the last ```python block after </think>.

YOUR TARGET: THE CODE ANSWER'S INTERFACE AND IDENTIFIERS. The code benchmarks are what "HOW IT WILL BE MEASURED" describes. Their prompts come in two forms:
- MBPP-style: the same form as the MBPP practice problems, a task and one example test.
- HumanEval-style: the function's signature and docstring in a code block.

The starting edit was evaluated on held-out problems you do not have, with this run's sampling settings: 2,168 answers, 1,512 of them MBPP-style. It passes 32.3%. Of the failures, about 300 fail on the interface or on a name, not on the logic.
- **Wrong function name: 200 answers**, nearly all MBPP-style (13% of those). The gates halved this (398 before).
  - 92 add or drop a prefix or suffix: smallest_num becomes find_smallest_num, subject_marks becomes subject_marks_sort.
  - 62 invent a different descriptive name: len_log becomes find_longest_word_length.
  - 28 define no function at all.
  - 18 change the case or the underscores: count_Set_Bits becomes count_set_bits, surfacearea_sphere becomes surface_area_sphere.
- **Wrong number of parameters: about 62 MBPP-style answers** fail with a TypeError because the function takes a different number of parameters than the example test passes.
- **A name used that was never defined: 45 answers** fail with a NameError.
  - A parameter is renamed in the body: def check(n), then num in the body.
  - A type is used in an annotation without being imported: from typing import Set, Tuple, then List[int].
- **Low priority:** 129 answers are syntax errors, many in test code appended after the function. Only 16 of them would pass if the block were cut after the function.

The likely mechanism is the one the function-name gates repaired, beyond the function name. With presence penalty 1.5, every token already generated in the thinking or the answer is lowered. So writing an identifier exactly as the prompt gives it, or exactly as the answer itself defined it a few lines up, is penalized token by token, and a near-synonym wins. This covers the example test's name and arguments, a parameter's name, and an imported name.

YOUR GOAL is to raise the code pass rate by keeping the answer's identifiers consistent with the prompt and with the answer itself:
- the function name and the number of parameters of the example test;
- parameter and variable names reused in the body;
- names used in annotations and imports.

Do this without lowering the other suites. Repairs that act on the copying situation inside code are in scope, as the gates do for names after "def". Raising tokens regardless of context is out of scope. The function-name gates must keep working; you may extend or change them.

TWO OTHER RUNS work at the same time:
- One works on turns that end with no answer, in the MLP tensors.
- One works on copying text from the prompt in prose answers, in the attention tensors. It may change the function-name gates, but it keeps them working.

Afterwards the three edits will be combined by adding each run's change (its final tensor minus the starting tensor). So:
- You may change MLP and attention tensors (mlp.*, self_attn.*, linear_attn.*). Do not change any norm or embedding.
- Keep your change small and specific. Prefer low-rank changes to a few heads or a few MLP neurons, active only inside code answers, where an identifier is being written.
- Do not undo or move a part of the starting edit unless your target needs it.
- Your final report lists every tensor you changed and, for each, the heads, neurons, rows or columns changed and the rank of the change.

HOW TO TEST IT.
- **Problems:** the MBPP practice problems have the MBPP-style form and their own tests. run_tests gives the status and, for runtime errors, the exception type.
  - Count wrong names (status wrong_interface), TypeError and NameError runtime errors, and passes.
  - For HumanEval-style prompts, write your own: a signature and docstring in a code block, with your own tests. Do not reproduce public benchmark items from memory.
- **Sets:** keep separate problem sets for choosing and for confirming. Make the confirmation sets large enough to see a change of a few points in the pass rate and in the interface failures.
- **Other checks:** check that a few prose prompts and math problems with known answers do not change for the worse. Your edit should be inactive outside code.

COMPARING WITH THE STARTING EDIT. "model": "start" runs the starting edit: always the files in /start, whatever /work/edited holds.
- In a generation with "model": "both", add "baseline": "start" to run the starting edit and your current edits on the same items and get the paired difference edited − start. Without it, the baseline is the unedited model, as described above.
- Teacher-forced passes take "model": "start" too.
- In the kernel, the starting edit's tensors are /start/<tensor name>.npy.

SETTINGS OF THIS RUN. Every generation uses presence penalty 1.5, and so does the evaluation.
- In a teacher-forced pass the logits are the model's own, without the penalty. To see what the penalty does at a position, lower by 1.5 every token that was already generated before it.
- This run's GPU is a cloud A100 reached through the server's tunnel, so each pass also spends some time moving files. Its time is paid by the hour, so prefer "baseline": "start" over the unedited model, whose answers often loop to the token limit.
- You have {{N_PASSES}} forward passes.

YOUR FINAL REPORT, in addition to what HOW THE SESSION RUNS asks, gives the paired difference of your final edit against the starting edit ("baseline": "start") on:
- your confirmation problems: pass rate, wrong names, TypeError and NameError runtime errors, and other failures;
- HumanEval-style prompts, if you wrote some;
- your prose and math checks;
- the list of changes described above.
