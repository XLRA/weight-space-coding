You do not start from the unedited model. /work/edited already holds the combination of two earlier runs' final edits, both built on a third run's edit (nine tensors in all), and /start holds a read-only copy of the same files and /start/report.md, which explains the combination and contains both runs' final reports. Read the report first. You may keep, change or remove any part of the starting edit, but this run's target is the one below, and another run is changing the starting edit's answer endings at the same time.

THE STARTING EDIT, in short (the report has the details):
- E2, loop exit: five repetition-detector neurons (layer 10 neurons 59 and 120, layer 13 neurons 16 and 7, layer 14 neuron 239) are wired to the model's end-of-thinking direction, at strength 2.5.
- E3, answer end: layer 13 neuron 1420 is wired to an answer-complete direction, so the turn ends after the first solution block closes.
- Neuron 3109, end-of-turn brake: layer 23 MLP neuron 3109 is repurposed to cancel the detectors' push on <|im_end|> where they fire strongly, so prose answers that copy a draft from the thinking are no longer cut short.
- Function-name gates: rank-1 edits of the output gates of attention heads (self_attn.q_proj of layer 19 heads 0, 2, 5, 6, 7, layer 15 head 7 and layer 23 heads 0, 1) make the model copy the given function name after "def".

THE EVALUATION IS BROADER THAN "HOW IT WILL BE MEASURED" SAYS. Besides the code benchmarks, the same protocol and settings are used on four more held-out suites, each with its own user message:
- instruction-following prompts: a writing task with one or more verifiable instructions. The answer after the last </think> is checked by strict automatic checkers; a prompt counts only if every instruction is followed.
- multiple-choice questions (ten options, many subjects), answered with "The answer is (X)" after the reasoning;
- competition math, graded on the last \boxed{} after the last </think>;
- competitive-programming problems (standard input and function form), graded like the code benchmarks.
You never see these suites.

YOUR TARGET: INSTRUCTIONS THAT CONSTRAIN THE WHOLE ANSWER. An evaluation of the starting edit on held-out instruction-following prompts you do not have, with this run's sampling settings, found that it follows about 61% of the prompts fully and 67% of the single instructions. The instruction types followed least, with the approximate share of those instructions followed:
- repeat the request word for word before answering: about 40%;
- a given letter appearing at least, or fewer than, N times: about 35% (this needs counting, and may be out of reach);
- at least or at most N words: about 55%; number of paragraphs: about 50%; number of sentences: about 70%;
- no commas anywhere in the response: about 65%;
- the whole response in lowercase: about 65%;
- words in all capital letters at least, or fewer than, N times: about 50%;
- a keyword at least N times: about 65%; a number of highlighted sections: about 70%.
Two of these got worse when neuron 3109 was added: all lowercase from about 77% to 67%, and the capital-word limits from about 72% to 50%. Its answers are longer, and each extra sentence is another chance to break a rule that applies to the whole answer.

YOUR GOAL is to raise the share of these instructions that the model follows, without lowering the others. Choose the types where you find a clear mechanism; you do not need to cover them all. One kind of mechanism to look for: the prompt states a constraint ("no commas", "all lowercase"), and the answer drifts away from it as it grows. Repairs that act on the specific situation are in scope, for example lowering commas only when the prompt forbids them, or keeping lowercase only when the prompt asks for it. Raising or lowering a token regardless of context is out of scope.

DO NOT CHANGE THE STARTING EDIT'S TENSORS: the mlp tensors of layers 10, 13, 14 and 23, and the q_proj tensors of layers 15, 19 and 23. Another run is working on them, on answers that restart after </think>. Put your edits in other tensors, so that the two runs' edits can be combined afterwards.

HOW TO TEST IT.
- Write your own instruction-following prompts with "prompts" or "prompts_file", in the style above. Write new prompts: do not reproduce prompts of public benchmarks from memory.
- Check the answers in the kernel with your own checkers, on the text after the last </think>.
- Keep separate prompt sets for choosing and for confirming, and make the confirmation sets large enough to see a change of a few points for the types you target.
- Check that the code practice problems do not get worse ("baseline": "start"), and that a few math problems with known answers still end with a correct boxed answer.

COMPARING WITH THE STARTING EDIT. "model": "start" runs the starting edit (always the files in /start, whatever /work/edited holds). In a generation with "model": "both", add "baseline": "start" to run the starting edit and your current edits on the same items and get the paired difference edited - start; without it the baseline is the unedited model, as described above. Teacher-forced passes take "model": "start" too. In the kernel, the starting edit's tensors are /start/<tensor name>.npy.

SETTINGS OF THIS RUN. Every generation uses presence penalty 1.5, and so does the evaluation. This run's GPU is smaller than the other run's: generations of the unedited model, which often loop to the token limit, take several times longer than those of the starting edit, so prefer "baseline": "start". You have {{N_PASSES}} forward passes.

YOUR FINAL REPORT, in addition to what HOW THE SESSION RUNS asks, gives the paired difference of your final edit against the starting edit ("baseline": "start") on:
- your confirmation prompts, per instruction type and in all (prompts and instructions followed);
- code practice problems not used to choose the edit;
- your math check.
