"""Adds the panel to /opt/wsp/README.md (backup: README.md.before-panel)."""
import shutil

P = "/opt/wsp/README.md"
shutil.copy(P, P + ".before-panel")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep("""| `briefs/` | Run-specific text added to the prompt with `--brief` (`pilot-11.md`). |
""", """| `briefs/` | Run-specific text added to the prompt with `--brief` (`pilot-11.md`). |
| `grade_panel.py`, `prompt_panel.txt` | The panel's grader and the agent's description of it, inserted into `prompt.txt` as forward pass (c) (see **Panel** below). |
""")

rep("""- **Brief:** `--brief FILE` adds the file's text to the prompt under `THIS RUN.` It is copied to `runs/<run>/brief.md` and sent again at every handoff.
""", """- **Brief:** `--brief FILE` adds the file's text to the prompt under `THIS RUN.` It is copied to `runs/<run>/brief.md` and sent again at every handoff.
- **Panel** (`--panel`, since 2026-11-06): `{"type": "panel"}` is a third kind of forward pass, a fixed check of the edit beyond code.
  - **Why:** round 4's agents measured only code on practice problems. `pilot-16` lost 4.6 points on MMLU-Pro without seeing it.
  - **Questions:** `/opt/wsp/panel/final/panel.jsonl`, 1,000 in all, built by `eval/prepare_panel.py`. They come from the held-out suites' sources but are not in the evaluation, and none is in the practice set:
    - 250 `choice`: MMLU-Pro test questions outside the evaluation's 1,000.
    - 250 `math`: MATH test problems outside MATH-500, matched by text (all 500 were found).
    - 250 `instructions`: argilla/ifeval-like-data, filtered subset (Qwen2.5-72B prompts in IFEval's style). Each has 1–3 constraints, a task, and no unfilled template slot. They are drawn so that constraint types and counts come closer to IFEval's mix: 1.9 constraints per prompt (IFEval: 1.5), 23 types.
    - 250 `code`: MBPP test tasks that MBPP+ left out (276), graded on MBPP's own asserts.
    - Every reference answer passes the grader, except 9 instruction prompts, which are left out.
  - **Protocol:** one answer per question with the run's sampling, at most 32,768 tokens. Graded by `harness/grade_panel.py` in `wsp-grade:1`, with the evaluation's own readings (`eval/suites.py` and the IFEval checkers) and the code grader.
  - **Baseline:** "start" by default (or "original"). Its answers are generated in the first panel pass ("model": "both") and cached in `runs/<run>/panel/base_<baseline>.jsonl`. Later passes generate only the edited model.
  - **What the agent sees:**
    - per suite, the score and the paired difference edited − baseline with its SE, flip counts, cut, no-answer and loop shares, and mean length;
    - the mean difference over the suites;
    - `/work/panel_<k>.jsonl`: per question, a stable label (`choice/17`), the subject, level or constraint types, and both models' status, length and cut. No text.
  - **What it doesn't see:** the questions or the answers.
    - The questions are in `/opt/wsp/panel`, which no agent container mounts.
    - The driver moves each panel pass's answers from `/work` to `runs/<run>/panel/` as soon as the pass ends.
  - **Test:** `tests/make_panel.py` writes `tests/panel_replies.json` and a 16-question test panel. The command is in its docstring (5080, `WSP_PANEL`, `WSP_PANEL_MAX_TOKENS=3000`). It passed on 2026-11-06. It checks:
    - the two refusals;
    - a first pass that also generates the start answers;
    - a pass that reuses them;
    - a pass against the original weights;
    - that nothing is left in `/work`.
""")
open(P, "w", encoding="utf-8").write(s)
print("README updated")
