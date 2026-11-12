"""Writes trained_start_replies.json: scripted replies for a driver test of a start that changes every tensor (a
trained model), with another practice set (--practice), the edit helpers for such a start and the panel.

    python3 tests/make_trained_start.py        (on the server: it picks a practice id that practice/val2 lacks)

Run the driver with (e2e-trained-2/work/E/ first holding runs/pilot-14/submission's .npy files and its report):
WSP_PANEL=/opt/wsp/panel/test/panel.jsonl WSP_PANEL_MAX_TOKENS=3000 driver.py e2e-trained-2 --backend replay
--replay tests/trained_start_replies.json --passes 4 --kernel-seconds 300 --handoff-every 4 --panel --gpu 5080
--start-from /opt/wsp/train/runs/G-O/checkpoints/step_100 --practice /opt/wsp/practice/val2
--brief briefs/pilot-20.md --presence-penalty 1.5.
Expected: turn 1 finds the 318 trained tensors in /work/edited and /start, the 220 practice problems and E, and
changed() empty; turn 2 finds a scaling of a trained tensor refused, restore_start undoing an edit, drop_edit putting
back the untrained tensor, and ends with one tensor edited and one dropped; turn 3 is refused (a problem of the full
practice set that this one lacks); turn 4 is pass 1, a paired generation edited - start, and the conversation then
restarts (its first message lists only the edited and the dropped tensor); turns 5-6 are passes 2-3, teacher-forced
reads with the trained and the untrained weights; turn 7 finds that they differ; turn 8 is pass 4, a panel of math and
code with the start's answers generated, and the conversation restarts again; turn 9 checks the panel's outcomes;
turn 10 ends the run.
"""
import json
import os

from make_e2e import fwd, py

TEXT = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"


def missing_id():
    """A problem of the full practice set that practice/val2 lacks (on the server), or a fixed guess elsewhere."""
    try:
        val2 = {json.loads(line)["id"] for line in open("/opt/wsp/practice/val2/mbpp_practice.jsonl")}
        return next(json.loads(line)["id"] for line in open("/opt/wsp/practice/final/mbpp_practice.jsonl")
                    if json.loads(line)["id"] not in val2)
    except OSError:
        return "mbpp/2"


REPLIES = [
    py("The trained model must be the start, in /work/edited and /start; the practice set must be the 220 held-back "
       "problems; E must be in /work/E; and nothing differs from the start yet.", """
# hypothesis: --start-from copied all 318 trained tensors, --practice mounted practice/val2
import json, os, time, numpy as np
e = sorted(f for f in os.listdir("/work/edited") if f.endswith(".npy"))
s = sorted(f for f in os.listdir("/start") if f.endswith(".npy"))
print(len(e), len(s), e == s, len(NAMES), len(edited()))
n = "model.language_model.layers.14.mlp.down_proj.weight"
print("load is the start:", bool((load(n).numpy() == np.load(f"/start/{n}.npy")).all()),
      "and differs from the original:", bool((load(n) != load_original(n)).any()))
print("start manifest:", json.load(open("/work/edited/manifest.json")))
print({fn: sum(1 for _ in open(f"/practice/{fn}")) for fn in sorted(os.listdir("/practice"))})
print(sorted(os.listdir("/work/E")))
t = time.time(); c = changed(); print("changed():", c, f"{time.time() - t:.1f} s for the first manifest")
"""),
    py("The edit helpers with a trained start: scaling refused, restore_start, drop_edit; then leave one edit and one "
       "dropped tensor, and pick four problems.", """
# hypothesis: restore_start goes back to /start and drop_edit to the original; the manifest is cached
n = "model.language_model.layers.14.mlp.down_proj.weight"
w = load(n)
try:
    print(save_edit(n, w * 1.5))
except PermissionError as err:
    print("refused as expected:", err)
w2 = w.clone(); w2[:, 239] = 0
t = time.time(); print(save_edit(n, w2), f"{time.time() - t:.1f} s"); print("changed:", changed())
print(restore_start(n), "changed:", changed(), "equal to start:", bool((load(n).numpy() == np.load(f"/start/{n}.npy")).all()))
m = "model.language_model.layers.10.mlp.down_proj.weight"
drop_edit(m); print("after drop_edit:", changed(), "equal to original:", bool((load(m) == load_original(m)).all()))
print(restore_start(m), changed())
w2 = w.clone(); w2[:, 239] *= 0.5
print(save_edit(n, w2))
k = "model.language_model.layers.0.input_layernorm.weight"
drop_edit(k)
print("changed:", changed(), "edited:", len(edited()))
man = json.load(open("/work/edited/manifest.json")); print(len(man), man[n])
probs = [json.loads(l) for f in ("mbpp_practice.jsonl", "apps_practice.jsonl") for l in list(open(f"/practice/{f}"))[:2]]
json.dump([p["id"] for p in probs], open("/work/p4.json", "w")); print([p["id"] for p in probs])
"""),
    fwd("A problem of the full practice set that this run's practice set lacks must be refused without using a pass.",
        {"type": "generate", "model": "edited", "problems": [missing_id()]}),
    fwd("A paired generation against the trained start on four held-back problems.",
        {"type": "generate", "model": "both", "baseline": "start", "problems_file": "/work/p4.json", "n": 1,
         "max_tokens": 1024}),
    fwd("Read one text with the trained start.", {"type": "teacher_forced", "model": "start", "texts": [TEXT]}),
    fwd("Read the same text with the untrained weights.",
        {"type": "teacher_forced", "model": "original", "texts": [TEXT]}),
    py("The two reads must differ, which shows the trained weights were swapped in, and pass 1 wrote both files.", """
# hypothesis: "start" runs the trained weights, "original" the untrained ones
a, b = np.load("/work/fwd_2/token_logprobs.npy"), np.load("/work/fwd_3/token_logprobs.npy")
print("max abs logprob difference", float(np.nanmax(np.abs(a - b))))
print(sorted(f for f in os.listdir("/work") if f.startswith("gen_1")))
for f in ("gen_1_start.jsonl", "gen_1_edited.jsonl"):
    rows = [json.loads(l) for l in open(f"/work/{f}")]
    print(f, [(r["id"], r["n_tokens"], r["status"]) for r in rows])
"""),
    fwd("Math and code from the panel; the trained start's answers are generated in this pass, the last one.",
        {"type": "panel", "suites": ["math", "code"]}),
    py("The panel's outcomes must be in /work without their texts.", """
# hypothesis: the panel wrote outcomes for both suites against the start (the file is named by the request, 4)
rows = [json.loads(l) for l in open("/work/panel_4.jsonl")]
print(len(rows), sorted({r["q"].split("/")[0] for r in rows}), rows[0])
"""),
    "Final report (replay): a harness test of a trained start (all 318 tensors), --practice and the panel. "
    "The edit is a test edit, not a real one.",
]

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_start_replies.json")
    json.dump(REPLIES, open(out, "w"), indent=1)
    print(f"{len(REPLIES)} replies -> {out}")
