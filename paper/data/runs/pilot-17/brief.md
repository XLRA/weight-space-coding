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
- Instruction-following prompts: a writing task with one or more verifiable instructions. The answer after the last </think> is checked by strict automatic checkers, and a prompt counts only if every instruction is followed.
- Multiple-choice questions: ten options, many subjects, answered with "The answer is (X)" after the reasoning.
- Competition math, graded on the last \boxed{} after the last </think>.
- Competitive-programming problems: standard input and function form. They are graded like the code benchmarks, on the last ```python block after </think>.

YOUR TARGET: COPYING TEXT FROM THE PROMPT. The starting edit was evaluated on held-out instruction-following prompts you do not have, with this run's sampling settings.
- **Overall:** it fully follows about 58% of the prompts.
- **Copy-type instructions:** in about half of the failed prompts, at least one failed instruction requires reproducing text given in the prompt. Approximate shares followed:
  - "First repeat the request word for word without change, then give your answer": about 42%, the same for the two edits before it.
  - Include the given keywords: about 65%.
  - Use a given word at least N times: about 60%.
  - End with an exact given phrase: about 90%.
  - A postscript starting with a given marker such as "P.S.": about 77%.
  - Wrap the response in double quotation marks: about 74%.
- **How the "repeat the request" answers fail:**
  - About half copy the request and then change a word partway: "called" becomes "call", "her father" becomes "his father", "you have seen" becomes "you have been told".
  - About a quarter do not start with the request at all.
  - About one in seven copy it only after other text.
  - In 42 of the 48 failures, the start of the request had already been written in the thinking.
- **The likely mechanism** is the same one the function-name gates repaired. With presence penalty 1.5, every token already generated in the thinking or the answer is lowered. So an exact copy of text the model has already written is penalized word by word, and a near-synonym wins.
- **A run on another GPU found the same.** It found that the failed copies are corrupted copies, pointing to the name-copying heads of layers 15, 19 and 23. Its own instruction edit in another place is not part of your starting edit.

YOUR GOAL is to raise the share of copy-type instructions that the model follows, without lowering the other instructions, code or math. One kind of mechanism to look for: the heads that read the prompt's text when the answer reproduces it. Make them win over the penalty where the answer is copying given text: the request, a quoted phrase, a required keyword or marker.

Repairs that act on the copying situation are in scope, as the function-name gates do for names after "def". Raising tokens regardless of context is out of scope.

The function-name gates must keep working. Check that function names after "def" are still copied, using the MBPP practice problems.

ANOTHER RUN works at the same time on turns that end with no answer, in the MLP tensors. To let the two runs' edits be combined afterwards:
- Change only attention tensors: self_attn.q_proj, k_proj, v_proj and o_proj of the full-attention layers 3, 7, 11, 15, 19 and 23, and the linear_attn.* tensors of the other layers.
- Do not change any mlp tensor, norm or embedding.
- The q_proj tensors of layers 15, 19 and 23 already hold the function-name gates. You may change them further, but keep the gates.

HOW TO TEST IT.
- **Prompts:** write your own instruction-following prompts with "prompts" or "prompts_file", in the styles above.
  - Examples: repeat the request word for word, then answer; include keywords X and Y; use the word Z at least N times; end with an exact phrase; add a postscript; quote the whole response.
  - Write new prompts. Do not reproduce prompts of public benchmarks from memory.
- **Checkers:** check the answers in the kernel with your own checkers, on the text after the last </think>.
- **Sets:** keep separate prompt sets for choosing and for confirming. Make the confirmation sets large enough to see a change of a few points for the types you target.
- **Other checks:** check that the code practice problems do not fall (function names included), and that a few math problems with known answers still end with a correct boxed answer.

COMPARING WITH THE STARTING EDIT. "model": "start" runs the starting edit: always the files in /start, whatever /work/edited holds.
- In a generation with "model": "both", add "baseline": "start" to run the starting edit and your current edits on the same items and get the paired difference edited − start. Without it, the baseline is the unedited model, as described above.
- Teacher-forced passes take "model": "start" too.
- In the kernel, the starting edit's tensors are /start/<tensor name>.npy.

SETTINGS OF THIS RUN. Every generation uses presence penalty 1.5, and so does the evaluation.
- In a teacher-forced pass the logits are the model's own, without the penalty. To see what the penalty does at a position, lower by 1.5 every token that was already generated before it.
- This run's GPU is smaller than the other run's. Generations of the unedited model often loop to the token limit and take several times longer than those of the starting edit, so prefer "baseline": "start".
- You have {{N_PASSES}} forward passes.

YOUR FINAL REPORT, in addition to what HOW THE SESSION RUNS asks, gives the paired difference of your final edit against the starting edit ("baseline": "start") on:
- your confirmation prompts, per instruction type and in all (prompts and instructions followed);
- code practice problems not used to choose the edit, including wrong function names;
- your math check.
