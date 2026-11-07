"""Writes e2e_replies.json: scripted agent replies for an end-to-end driver test.

Run the driver with: --backend replay --replay tests/e2e_replies.json --passes 2 --handoff-every 7
--kernel-seconds 20. Expected: turns 3-5 are sent back (no explanation, no block, unknown id),
turn 7 is pass 1 (request 1) and is followed by a handoff that keeps the kernel, turn 9 hits the
kernel time limit, turn 11 is pass 2 (the last), turn 12 is refused, turn 14 ends the run.
"""
import json
import os

F = "`" * 3
EXPLAIN = "I am running this step to check one part of the harness, and a failure here would mean a bug to fix. "


def py(why, code):
    return f"{EXPLAIN}{why}\n\n{F}python\n{code.strip()}\n{F}\n"


def fwd(why, req):
    return f"{EXPLAIN}{why}\n\n{F}forward\n{json.dumps(req, indent=1)}\n{F}\n"


REPLIES = [
    py("Check the predefined names and run_tests.", """
# hypothesis: the prelude is loaded and run_tests accepts a reference solution
import json
probs = [json.loads(l) for l in open("/practice/mbpp_practice.jsonl")]
print(len(NAMES), len(probs), edited())
print(run_tests(probs[0]["reference"], probs[0]["id"]), run_tests("def nothing(): pass", probs[0]["id"]))
marker = 12345
open("/work/notes.md", "w").write("# notes\\n- e2e test run\\n")
"""),
    py("A line of three backticks inside a string must not end the block.", f"""
# hypothesis: the driver keeps the whole block
s = '''
{F}
'''
print(repr(s), marker)
"""),
    f"{F}python\nprint('this has no explanation and must not run')\n{F}\n",
    "I am still thinking about what to do next, so this reply has no block at all.",
    fwd("An unknown problem id must be refused without using a pass.",
        {"type": "generate", "model": "both", "problems": ["mbpp/does-not-exist"]}),
    py("Make a mild edit and write the problem list for the next pass.", """
# hypothesis: zeroing 64 input columns of one down_proj is a mild edit
w = load_original("model.language_model.layers.15.mlp.down_proj.weight").clone(); w[:, :64] = 0
print(save_edit("model.language_model.layers.15.mlp.down_proj.weight", w))
json.dump([p["id"] for p in probs[:6]], open("/work/probs.json", "w"))
"""),
    fwd("A small paired generation with activations, through a problems file.",
        {"type": "generate", "model": "both", "problems_file": "/work/probs.json",
         "prompts": ["Write a Python function that returns the square of a number."], "n": 2, "max_tokens": 1024,
         "activations": {"layers": [0, 12, 24], "mlp": [15]}}),
    py("After the handoff the kernel should still hold its variables; then read the graded answers.", """
# hypothesis: variables survive a routine handoff, and the graded files and activation index are complete
print("marker" in dir(), marker)
rows = [json.loads(l) for l in open("/work/gen_1_edited.jsonl")]
print(len(rows), sorted(set(r["status"] for r in rows)), sorted(rows[0]))
print(json.load(open("/work/gen_1_acts_edited/index.json"))["samples"][0])
graded = [r for r in rows if r["status"] != "ungraded"]
pr = {p["id"]: p["problem"] for p in probs}
render = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
texts = [render(pr[r["id"]]) + r["text"] for r in graded[:2]]
json.dump(texts, open("/work/texts.json", "w")); print([len(tok.encode(t)) for t in texts])
"""),
    py("This sleeps past the 20-second test limit, so the kernel must be restarted.", """
# hypothesis: the driver enforces the time limit
import time; time.sleep(60)
"""),
    py("After the restart the variables are gone but the edit file is still there.", """
# hypothesis: a restart loses variables, keeps /work
print("marker" in dir(), edited())
"""),
    fwd("A teacher-forced read of two answers with every kind of output; this is the last pass.",
        {"type": "teacher_forced", "model": "edited", "texts_file": "/work/texts.json",
         "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 24], "mlp_act": [15],
                    "attn_layers": [3]}}),
    fwd("The budget is used, so this must be refused.",
        {"type": "teacher_forced", "model": "original", "texts": ["def f():\n    return 1\n"]}),
    py("Read the teacher-forced arrays and keep a copy of the edit.", """
# hypothesis: the pass wrote every array it reported
import numpy as np, shutil, os
for f in sorted(os.listdir("/work/fwd_2")): print(f, np.load("/work/fwd_2/" + f).shape if f.endswith(".npy") else "")
shutil.copytree("/work/edited", "/work/best_1", dirs_exist_ok=True); print(edited())
"""),
    "Final report (replay): I zeroed 64 input columns of layers.15.mlp.down_proj as a harness test. "
    "This is not a real edit.",
]

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_replies.json")
    json.dump(REPLIES, open(out, "w"), indent=1)
    print(f"{len(REPLIES)} replies -> {out}")
