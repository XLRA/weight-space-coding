"""Grade one panel generation for the driver: python3 grade_panel.py /work/gen_<k>[_<model>].raw.jsonl > graded.jsonl

The panel (README, "Panel") is a fixed set of questions beyond the practice problems, none of them in the
held-out evaluation: suites choice (MMLU-Pro), math (MATH), instructions (IFEval-style prompts) and code (MBPP
test tasks). Runs in the grading container
(wsp-grade:1: no network, read-only, uid 65534) with /panel/panel.jsonl (the questions and their grading data),
/harness (grade.py) and /evalbin (the evaluation's suites.py and IFEval checkers) on PYTHONPATH. Each answer is
graded as the evaluation grades its suite: code as the practice problems and MBPP+ (the last ```python block after
</think>, against the task's asserts); choice, math and instructions with the evaluation's MMLU-Pro, MATH-500 and
IFEval readings (suites.py).

Prints one row per answer: id, sample, suite, source, difficulty, n_tokens, finish_reason, loop, status (pass,
cut_at_max_tokens, no_answer, wrong_answer, not_followed, or a code failure category), exc (code), inst_ok and
inst_total (instructions).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import grade
import suites as S

GRADERS = {"choice": S.grade_choice, "math": S.grade_math, "instructions": S.grade_ifeval}
PROBS = {}


def one(r):
    p = PROBS[r["id"]]
    out = {"id": r["id"], "sample": r["sample"], "suite": p["suite"], "source": p["source"],
           "difficulty": p["difficulty"], "n_tokens": r["n_tokens"], "finish_reason": r["finish_reason"],
           "loop": grade.looped(r["text"])}
    if p["suite"] == "code":
        g = grade.grade(p, r)
        out["status"] = g["status"]
        if "exc" in g:
            out["exc"] = g["exc"]
    else:
        g = GRADERS[p["suite"]](p, r)
        out.update({k: v for k, v in g.items() if k != "pred"})
    return out


def load():
    PROBS.update({p["id"]: p for p in map(json.loads, open("/panel/panel.jsonl"))})


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    # Processes, not threads: math-verify's timeouts use signal.alarm (each worker's main thread), and the
    # IFEval checkers are not thread-safe (langdetect loads its profiles lazily into a global).
    with ProcessPoolExecutor(12, initializer=load) as pool:
        for o in pool.map(one, rows, chunksize=8):
            print(json.dumps(o))


if __name__ == "__main__":
    main()
