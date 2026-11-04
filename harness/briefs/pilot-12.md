You do not start from the unedited model. /work/edited already holds an earlier run's final edit (three mlp.down_proj tensors of layers 10, 13 and 14; 6,144 changed weights), and /start holds a read-only copy of the same files and /start/report.md, that run's final report. Read the report first. You may keep, change or remove any part of the starting edit.

THE STARTING EDIT, in short (the report has the details):
- E2, loop exit: five repetition-detector neurons (layer 10 neurons 59 and 120, layer 13 neurons 16 and 7, layer 14 neuron 239) are wired to the model's end-of-thinking direction, at a strength chosen by extrapolation (2.5) rather than measured.
- E3, answer end: layer 13 neuron 1420 is wired to an answer-complete direction, so the turn ends after the first solution block closes.
- Together they cut answers that loop to the token limit from most to few, and raise the pass rate clearly.

THE EVALUATION IS BROADER THAN "HOW IT WILL BE MEASURED" SAYS. Besides the code benchmarks, the same protocol and settings are used on four more held-out suites, each with its own user message:
- instruction-following prompts: a writing task with one or more verifiable instructions (for example a minimum or maximum number of words, sentences or paragraphs, wrapping the whole response in double quotation marks, ending with a postscript that starts with P.S., all lowercase, no commas, a given number of highlighted sections or bullet points, JSON output). The answer after </think> is checked by strict automatic checkers; a prompt counts only if every instruction is followed. No </think> means an empty answer.
- multiple-choice questions (ten options, many subjects), answered with "The answer is (X)" after the reasoning;
- competition math, with the final answer in \boxed{} after </think>;
- competitive-programming problems (standard input and function form), graded like the code benchmarks.
You never see these suites. The starting edit gains a lot on instruction following and math, because the unedited model rarely closes its thinking there.

WHAT AN EARLIER RUN FOUND ABOUT CODE. A run before this one started from the same edit, studied the code failures for 10 passes and kept the edit unchanged. Its measurements: answers that stop inside an unclosed block on stdin problems are mostly degenerate loops in the answer on problems the model does not solve (only about 2% of failing answers contain any passing code); wrong function names are worth at most about 5 points of MBPP-style problems, and an attention-gate edit that made the model copy the function name measured -3.9 points (SE 2.9) and was dropped; the remaining wrong answers are algorithmic. It also found that in answer loops the loop exit pushes toward <|im_end|>, because </think> is already penalized by the presence penalty. So code is not this run's target; keep its gains.

WHAT IS LEFT, AND YOUR TARGET. An evaluation of the starting edit on the held-out suites, with this run's sampling settings, found that it ends prose answers too early:
- On instruction-following prompts that both models answered, the starting edit followed fewer instructions than the unedited model (about 74% against 83%). The largest losses are wrapping the whole response in quotation marks (the closing quotation mark is missing), the postscript (missing), and the number of words, sentences and paragraphs (too few).
- For "at least N words" instructions, the starting edit's answers reach a median of about two thirds of N; the unedited model's answers, when it answers, reach about N.
- Most of these answers end cleanly at the end of a sentence, so the turn ends early rather than being cut. A few end in a repeated list or sentence.
- Two mechanisms are plausible, and you should find out which: E3's answer-complete neuron firing at the end of prose sentences or paragraphs, and E2's detectors firing on repetitive prose and pushing toward <|im_end|>.

YOUR GOAL is to stop the edit from ending prose answers early, while keeping what it does for code and math: the loop exit, and ending the turn once a code answer or a boxed answer is complete. Repairs that act on the specific situation are in scope, for example ending the turn only after a closed code block or a \boxed{} answer, or not letting the loop exit end the turn in the answer. Raising or lowering a token regardless of context is still out of scope.

HOW TO TEST IT. Write your own instruction-following prompts with "prompts" or "prompts_file", in the style above. Write new prompts: do not reproduce prompts of public benchmarks from memory. Check the answers in the kernel with your own checkers (the text after the last </think>). Keep separate prompt sets for choosing and for confirming. Also write a few math problems with known answers in the form "<problem>\nPlease reason step by step, and put your final answer within \boxed{}.", to check that answers still end with a boxed answer. Check the code practice problems with "baseline": "start" so that the code gains do not fall.

COMPARING WITH THE STARTING EDIT. "model": "start" runs the starting edit (always the files in /start, whatever /work/edited holds). In a generation with "model": "both", add "baseline": "start" to run the starting edit and your current edits on the same items and get the paired difference edited - start; without it the baseline is the unedited model, as described above. Teacher-forced passes take "model": "start" too. In the kernel, the starting edit's tensors are /start/<tensor name>.npy.

SETTINGS OF THIS RUN. Every generation uses presence penalty 1.5, and so does the evaluation. You have {{N_PASSES}} forward passes.

YOUR FINAL REPORT, in addition to what HOW THE SESSION RUNS asks: the paired difference of your final edit against the starting edit ("baseline": "start") on your confirmation prompts (instructions followed) and on code practice problems not used to choose the edit, and what you checked about math answers.
