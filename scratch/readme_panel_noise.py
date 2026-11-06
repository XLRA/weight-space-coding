"""Adds the panel's two noise checks to /opt/wsp/README.md (backup: /opt/wsp/backup/README.md.before-panel-noise)."""
import shutil

P = "/opt/wsp/README.md"
shutil.copy(P, "/opt/wsp/backup/README.md.before-panel-noise")
s = open(P, encoding="utf-8").read()
anchor = "    - that nothing is left in `/work`.\n"
add = """  - **Noise** (2026-11-06; both on the 5090 with the replay backend, from `start-r4`, presence penalty 1.5; `paper/data/panels.csv`):
    - **`runs/panel-null`, the starting edit against itself:** three generations, two of them in separate passes, were byte-identical on all 1,000 questions. vLLM on one GPU with the same batch and seeds is deterministic, so repeating a panel pass on an unchanged edit adds nothing.
    - **`runs/panel-ulp`, one weight moved by one bf16 step** (`layers.5.mlp.down_proj`, entry 2,716,541 of 3,670,016, 0.0050354 → 0.0050659; `scratch/panel_ulp_replies.json`, `scratch/run-panel-ulp.sh`): none of the 1,000 answers stayed the same. They parted from the originals after the first 2–4% of their text (median per suite).
      - **Panel:** choice +1.6 (SE 2.7), math +1.6 (2.0), instructions −4.8 (2.6), code −3.2 (2.3), mean −1.2 (1.2); 148 questions flipped.
      - **What it means:** the paired standard error is about right, but the shared seeds do not keep two models' answers alike. Pairing removes the questions' difficulty, not the sampling noise. Only differences beyond about two standard errors are evidence: about 5 points on one suite, 2.5 on the mean. The same holds for paired practice passes and for the held-out evaluation.
      - **Prompt fix:** `prompt.txt` and `prompt_panel.txt` used to say the two models' answers "largely share their random numbers". Both now say what was measured (backups `*.before-seed-note` in `/opt/wsp/backup`).
"""
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + add)
open(P, "w", encoding="utf-8").write(s)
print("README updated")
