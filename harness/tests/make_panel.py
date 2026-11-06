"""Writes panel_replies.json: scripted replies for a driver test of --panel, and a small test panel.

    python3 tests/make_panel.py        (on the server: also writes /opt/wsp/panel/test/panel.jsonl, 4 per suite)

Run the driver with: WSP_PANEL=/opt/wsp/panel/test/panel.jsonl WSP_PANEL_MAX_TOKENS=3000 driver.py panel-test
--backend replay --replay tests/panel_replies.json --passes 3 --kernel-seconds 120 --panel --gpu 5080
--start-from /opt/wsp/runs/pilot-14/submission --presence-penalty 1.5.
Expected: turn 1 finds the panel text in the prompt's place and no panel data in the kernel, and drops one tensor of
the starting edit; turns 2-3 are refused (an unknown suite, an unknown baseline); turn 4 is pass 1, a panel of all
four suites that generates the start answers too ("panel/both"); turn 5 finds no panel answers in /work; turn 6 is
pass 2, two suites with the start answers reused ("panel/edited"); turn 7 is pass 3, math against the original
weights ("panel/both" again); turn 8 ends the run.
"""
import json
import os
from collections import defaultdict

from make_e2e import fwd, py

REPLIES = [
    py("The prompt must describe the panel, and the kernel must not see the panel's questions.", """
# hypothesis: the panel's data stays outside the kernel's mounts
import os, glob
print(os.path.exists("/opt/wsp/panel"), [p for d in ("/work", "/start", "/practice", "/harness", "/model", "/tmp")
                                         for p in glob.glob(f"{d}/**/panel*.jsonl", recursive=True)])
print(sorted(os.listdir("/work")))
n = edited()[0]
print(drop_edit(n), len(edited()))
"""),
    fwd("An unknown suite must be refused without using a pass.", {"type": "panel", "suites": ["nope"]}),
    fwd("An unknown baseline must be refused without using a pass.", {"type": "panel", "baseline": "edited"}),
    fwd("The whole panel; the start answers are generated in this pass.", {"type": "panel"}),
    py("The panel's answers must not be in /work.", """
# hypothesis: the driver moved request 1's answers out of /work and wrote outcomes without text
import json
print(sorted(f for f in os.listdir("/work") if f.startswith("gen_")))
rows = [json.loads(l) for l in open("/work/panel_1.jsonl")]
print(len(rows), rows[0], rows[-1]["q"])
"""),
    fwd("Two suites; the start answers are reused.", {"type": "panel", "suites": ["choice", "code"]}),
    fwd("Math against the unedited weights.", {"type": "panel", "suites": ["math"], "baseline": "original"}),
]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "panel_replies.json")
    json.dump(REPLIES, open(out, "w"), indent=1)
    print(f"{len(REPLIES)} replies -> {out}")
    if os.path.exists("/opt/wsp/panel/final/panel.jsonl"):
        by = defaultdict(list)
        for line in open("/opt/wsp/panel/final/panel.jsonl"):
            p = json.loads(line)
            by[p["suite"]].append(p)
        os.makedirs("/opt/wsp/panel/test", exist_ok=True)
        with open("/opt/wsp/panel/test/panel.jsonl", "w") as f:
            for rows in by.values():
                for p in rows[:4]:
                    f.write(json.dumps(p) + "\n")
        print("test panel -> /opt/wsp/panel/test/panel.jsonl")
