#!/usr/bin/env python3
"""Build the panel: the questions of the agent's cross-suite check (driver.py --panel; README, "Panel").

    /opt/wsp/venv/bin/python /opt/wsp/eval/prepare_panel.py [--per-suite 250]     (host, as root)

Writes /opt/wsp/panel/final/panel.jsonl (one question per line: id, suite, source, difficulty, problem = the user
message, and its suite's grading data) and stats.json. Every question comes from the source of a held-out suite but
is not in the evaluation, and none is in the practice set:
  choice        MMLU-Pro test questions outside the evaluation's fixed 1,000 (eval/raw/mmlupro_test.parquet); the
                evaluation's prompt and answer reading.
  math          MATH test problems (EleutherAI/hendrycks_math) that are not in MATH-500 (same problem text); the gold
                answer is the reference solution's last \\boxed{}; prompt and reading as for MATH-500.
  instructions  argilla/ifeval-like-data, subset "filtered" (prompts written by Qwen2.5-72B in IFEval's style, kept
                where its response passes lm-eval's IFEval checkers): prompts with 1 to 3 constraints that also hold a
                task (a question or a request) and no unfilled template slot, drawn so that the constraint types come
                closer to IFEval's mix. Graded like IFEval (strict, prompt level).
  code          MBPP test tasks that MBPP+ left out (MBPP+ is in the evaluation; the practice set has no test-split
                task); the prompt of MBPP+ and the practice problems, graded on MBPP's own asserts.
A question is kept only if a reference answer passes the panel's grader (harness/grade_panel.py in wsp-grade:1): the
right letter, the gold answer boxed, argilla's response, MBPP's reference code. Then --per-suite questions are drawn
per suite with seed 0.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suites as S  # noqa: E402

WSP = "/opt/wsp"
EVAL, HARNESS, OUT = f"{WSP}/eval", f"{WSP}/harness", f"{WSP}/panel"
RAW = f"{OUT}/raw"
HF = "https://huggingface.co/datasets"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus"]
F = "`" * 3
INSTRUCTION = ("Write a complete, self-contained Python 3 solution. "
               f"Put your final code in a single {F}python block at the end of your answer.")
SUITES = ["choice", "math", "instructions", "code"]
# A prompt holds a task: a question, or a request at the start of a sentence.
TASK = re.compile(r"\?|(?:^|[.!:\n]\s*)(?:Write|Describe|Explain|Provide|Create|Generate|List|Give|Tell|Compose|"
                  r"Summarize|Draft|Suggest|Share|Help|Outline|Discuss|Plan|Can you|Could you|Please)\b")
# Template slots the generator left unfilled, e.g. "the word [keyword] should appear".
SLOT = re.compile(r"\[(?:keywords?|num_\w+|relation|letter|end_phrase|postscript_marker|first_word|section_spliter|"
                  r"forbidden_words|n)\]", re.I)


def fetch():
    os.makedirs(f"{RAW}/math", exist_ok=True)
    files = {f"{RAW}/ifeval_like_filtered.parquet":
             f"{HF}/argilla/ifeval-like-data/resolve/main/filtered/train-00000-of-00001.parquet"}
    files.update({f"{RAW}/math/{s}.test.parquet": f"{HF}/EleutherAI/hendrycks_math/resolve/main/{s}/test-00000-of-00001.parquet"
                  for s in MATH_SUBJECTS})
    for path, url in files.items():
        if not os.path.exists(path):
            subprocess.run(["curl", "-sfL", "-o", path + ".tmp", url], check=True)
            os.replace(path + ".tmp", path)


def norm(t):
    return " ".join(t.split())


def choice_rows():
    held = S.mmlupro_ids(f"{EVAL}/raw")
    for _, r in pd.read_parquet(f"{EVAL}/raw/mmlupro_test.parquet").iterrows():
        if f"mmlupro_{r['question_id']}" in held:
            continue
        row = {"question": r["question"], "options": list(r["options"]), "category": r["category"]}
        yield ({"id": f"panel/choice/{r['question_id']}", "suite": "choice", "source": "mmlupro",
                "difficulty": r["category"], "problem": S.mmlu_prompt(row), "answer": r["answer"]},
               f"</think>\n\nThe answer is ({r['answer']}).")


def math_rows():
    held = {norm(json.loads(l)["problem"]) for l in open(f"{EVAL}/raw/math500_test.jsonl")}
    seen = 0
    for s in MATH_SUBJECTS:
        for i, r in enumerate(pd.read_parquet(f"{RAW}/math/{s}.test.parquet").itertuples()):
            if norm(r.problem) in held:
                seen += 1
                continue
            gold = S.last_boxed(r.solution)
            if gold:
                yield ({"id": f"panel/math/{s}/{i}", "suite": "math", "source": "math", "difficulty": r.level.lower(),
                        "problem": f"{r.problem}\n{S.MATH_INSTRUCTION}", "answer": gold, "subject": s},
                       f"</think>\n\nThe final answer is $\\boxed{{{gold}}}$.")
    print(f"math: {seen} test problems are in MATH-500 and left out")


def instruction_rows():
    held = {norm(json.loads(l)["prompt"]) for l in open(f"{EVAL}/raw/ifeval_input_data.jsonl")}
    for r in pd.read_parquet(f"{RAW}/ifeval_like_filtered.parquet").itertuples():
        ids = list(r.instruction_id_list)
        if not 1 <= len(ids) <= 3 or norm(r.prompt) in held or not TASK.search(r.prompt) or SLOT.search(r.prompt):
            continue
        kwargs = json.loads(r.kwargs)
        if not isinstance(kwargs, list) or len(kwargs) != len(ids):
            continue
        yield ({"id": f"panel/instructions/{r.key}", "suite": "instructions", "source": "ifeval-like",
                "difficulty": f"{len(ids)} constraint{'s' if len(ids) > 1 else ''}", "problem": r.prompt,
                "prompt": r.prompt, "instruction_id_list": ids, "kwargs": kwargs},
               f"</think>\n\n{r.response}")


def code_rows():
    plus = set(int(x) for x in pd.read_parquet(f"{EVAL}/raw/mbppplus.parquet")["task_id"])
    for r in pd.read_parquet(f"{WSP}/practice/raw/mbpp_test.parquet").itertuples():
        if int(r.task_id) in plus:
            continue
        tests = list(r.test_list)
        m = re.match(r"\s*assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", tests[0])
        yield ({"id": f"panel/code/mbpp-{r.task_id}", "suite": "code", "source": "mbpp-test", "difficulty": "basic",
                "problem": f"{r.text}\nYour code should pass this test:\n{tests[0]}\n\n{INSTRUCTION}",
                "test_format": "function", "entry_point": m.group(1) if m else None, "tests": tests,
                "test_setup": r.test_setup_code or ""},
               f"</think>\n\n{F}python\n{r.code}\n{F}\n")


def check(cands):
    """Grades every candidate's reference answer with the panel's grader; returns {id: status}."""
    os.makedirs(f"{OUT}/check", exist_ok=True)
    with open(f"{OUT}/candidates.jsonl", "w") as f:
        for p, _ in cands:
            f.write(json.dumps(p) + "\n")
    with open(f"{OUT}/check/gen_reference.raw.jsonl", "w") as f:
        for p, text in cands:
            f.write(json.dumps({"id": p["id"], "sample": 0, "text": text, "n_tokens": 0, "finish_reason": "stop"}) + "\n")
    r = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g,mode=1777",
         "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "16g",
         "--pids-limit", "512", "--cpus", "12", "-w", "/tmp", "-e", "PYTHONPATH=/harness:/evalbin",
         "-v", f"{OUT}/candidates.jsonl:/panel/panel.jsonl:ro", "-v", f"{HARNESS}:/harness:ro",
         "-v", f"{EVAL}/suites.py:/evalbin/suites.py:ro", "-v", f"{EVAL}/ifeval:/evalbin/ifeval:ro",
         "-v", f"{OUT}/check:/work:ro", "wsp-grade:1", "python3", "-W", "ignore", "/harness/grade_panel.py",
         "/work/gen_reference.raw.jsonl"], capture_output=True, text=True, timeout=3 * 3600)
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    if r.returncode or len(rows) != len(cands):
        raise SystemExit(f"the grader returned {len(rows)} of {len(cands)} rows (exit {r.returncode}): {r.stderr[-2000:]}")
    return {g["id"]: g["status"] for g in rows}


def draw_instructions(rows, n, rng):
    """Weighted draw without replacement (Efraimidis-Spirakis): a prompt's weight is its constraint types' largest
    ratio of IFEval's share to the candidates' share, so rarer types are drawn more often, times the same ratio for
    its number of constraints (IFEval has 1.5 per prompt on average, the candidates more)."""
    lists = [json.loads(l)["instruction_id_list"] for l in open(f"{EVAL}/raw/ifeval_input_data.jsonl")]
    ev, ev_n = Counter(t for x in lists for t in x), Counter(min(len(x), 3) for x in lists)
    cand = Counter(t for p in rows for t in p["instruction_id_list"])
    cand_n = Counter(len(p["instruction_id_list"]) for p in rows)
    ne, nc = sum(ev.values()), sum(cand.values())

    def weight(p):
        ids = p["instruction_id_list"]
        types = max((ev.get(t, 0.5) / ne) / (cand[t] / nc) for t in ids)
        return types * (ev_n[len(ids)] / len(lists)) / (cand_n[len(ids)] / len(rows))

    keyed = sorted(rows, key=lambda p: rng.random() ** (1 / weight(p)), reverse=True)
    return keyed[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-suite", type=int, default=250)
    args = ap.parse_args()
    fetch()
    cands = list(choice_rows()) + list(math_rows()) + list(instruction_rows()) + list(code_rows())
    print("candidates:", dict(Counter(p["suite"] for p, _ in cands)))
    status = check(cands)
    ok = [p for p, _ in cands if status[p["id"]] == "pass"]
    bad = Counter((p["suite"], status[p["id"]]) for p, _ in cands if status[p["id"]] != "pass")
    print("reference passes:", dict(Counter(p["suite"] for p in ok)), "fails:", dict(bad))
    rng = random.Random(0)
    final = []
    for suite in SUITES:
        rows = sorted((p for p in ok if p["suite"] == suite), key=lambda p: p["id"])
        if len(rows) < args.per_suite:
            raise SystemExit(f"{suite}: only {len(rows)} questions pass their reference check")
        final += draw_instructions(rows, args.per_suite, rng) if suite == "instructions" else \
            rng.sample(rows, args.per_suite)
    os.makedirs(f"{OUT}/final", exist_ok=True)
    with open(f"{OUT}/final/panel.jsonl", "w") as f:
        for p in final:
            f.write(json.dumps(p) + "\n")
    stats = {"per_suite": args.per_suite, "candidates": dict(Counter(p["suite"] for p, _ in cands)),
             "reference_pass": dict(Counter(p["suite"] for p in ok)),
             "reference_fail": {f"{s}/{st}": v for (s, st), v in bad.items()},
             "difficulty": {s: dict(Counter(p["difficulty"] for p in final if p["suite"] == s)) for s in SUITES},
             "constraint_types": dict(Counter(t for p in final if p["suite"] == "instructions"
                                              for t in p["instruction_id_list"]).most_common())}
    json.dump(stats, open(f"{OUT}/final/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
