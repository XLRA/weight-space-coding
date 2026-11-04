"""Writes start_replies.json: scripted replies for a driver test of --start-from, --brief and
--presence-penalty.

Run the driver with: --backend replay --replay tests/start_replies.json --passes 3 --kernel-seconds 120
--start-from /opt/wsp/runs/pilot-10/submission --brief briefs/pilot-11.md --presence-penalty 1.5.
Expected: turn 1 finds the starting edit in /work/edited and /start; turn 2 is refused ("baseline"
without "both"); turn 4 is pass 1, a paired generation edited - start; turns 5-6 are passes 2-3,
teacher-forced reads of one text with the start and original weights; turn 7 finds that they
differ; turn 8 ends the run.
"""
import json
import os

from make_e2e import fwd, py

TEXT = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"

REPLIES = [
    py("The starting edit must be in /work/edited and, read-only, in /start, with the earlier report.", """
# hypothesis: --start-from copied pilot-10's submission to both places
import json, os, numpy as np
print(edited(), sorted(os.listdir("/start")))
n = edited()[0]
print(bool((load(n).numpy() == np.load(f"/start/{n}.npy")).all()), os.access("/start", os.W_OK))
print(open("/start/report.md").read()[:80])
"""),
    fwd("A baseline without a paired generation must be refused without using a pass.",
        {"type": "generate", "model": "edited", "baseline": "start", "problems": ["mbpp/2"]}),
    py("Drop one tensor of the starting edit, so edited and start differ, and pick four problems.", """
# hypothesis: drop_edit leaves /start alone
probs = [json.loads(l) for l in open("/practice/mbpp_practice.jsonl")]
drop_edit("model.language_model.layers.14.mlp.down_proj.weight")
json.dump([p["id"] for p in probs[:4]], open("/work/p4.json", "w"))
print(edited(), len(os.listdir("/start")))
"""),
    fwd("A paired generation against the starting edit.",
        {"type": "generate", "model": "both", "baseline": "start", "problems_file": "/work/p4.json", "n": 1,
         "max_tokens": 1024}),
    fwd("Read one text with the starting edit.", {"type": "teacher_forced", "model": "start", "texts": [TEXT]}),
    fwd("Read the same text with the original weights; this is the last pass.",
        {"type": "teacher_forced", "model": "original", "texts": [TEXT]}),
    py("The two reads must differ, which shows the start weights were swapped in, and pass 1 wrote both files.", """
# hypothesis: "start" runs different weights from "original"
a, b = np.load("/work/fwd_2/token_logprobs.npy"), np.load("/work/fwd_3/token_logprobs.npy")
print("max abs logprob difference", float(np.nanmax(np.abs(a - b))))
print(sorted(f for f in os.listdir("/work") if f.startswith("gen_1")))
"""),
    "Final report (replay): a harness test of --start-from, --brief and --presence-penalty. No real edit.",
]

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_replies.json")
    json.dump(REPLIES, open(out, "w"), indent=1)
    print(f"{len(REPLIES)} replies -> {out}")
