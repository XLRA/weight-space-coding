"""Build the practice set the editing agent sees.

Practice = MBPP train/validation/prompt splits (function problems, assert tests)
         + APPS train stdin/stdout problems (no fn_name), tests capped at 10.
Held out for evaluation: the MBPP test split and every MBPP+ task (MBPP+ draws from all splits).

  python prep_practice.py             writes /opt/wsp/practice/candidates.jsonl
  (grade the reference solutions with grade.py in the sandbox -> candidates_refgraded.jsonl)
  python prep_practice.py --finalize  keeps the problems whose reference passes and writes
                                      /opt/wsp/practice/final/{mbpp,apps}_practice.jsonl + stats.json
"""
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

RAW = "/opt/wsp/practice/raw"
OUT = "/opt/wsp/practice"
INSTRUCTION = ("Write a complete, self-contained Python 3 solution. "
               "Put your final code in a single ```python block at the end of your answer.")
MAX_TESTS = 10
MAX_IO_BYTES = 200_000


def mbpp_rows(held_out):
    for split in ["train", "validation", "prompt"]:
        df = pd.read_parquet(f"{RAW}/mbpp_{split}.parquet")
        for r in df.itertuples():
            if int(r.task_id) in held_out:
                continue
            tests = list(r.test_list)
            m = re.match(r"\s*assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*([A-Za-z_]\w*)\s*\(", tests[0])
            entry = m.group(1) if m else None
            problem = f"{r.text}\nYour code should pass this test:\n{tests[0]}\n\n{INSTRUCTION}"
            yield {
                "id": f"mbpp/{r.task_id}", "source": "mbpp", "split": split, "difficulty": "basic",
                "problem": problem, "test_format": "function", "entry_point": entry,
                "tests": tests, "test_setup": r.test_setup_code or "", "reference": r.code,
            }


def apps_rows():
    with open(f"{RAW}/apps_train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("starter_code"):
                continue
            try:
                io = json.loads(r["input_output"]) if r["input_output"] else None
            except json.JSONDecodeError:
                continue
            if not io or io.get("fn_name") or not io.get("inputs"):
                continue
            # Some stdio problems store each input/output as a list of lines.
            join = lambda x: "\n".join(x) + "\n" if isinstance(x, list) and all(isinstance(y, str) for y in x) else x
            ins = [join(x) for x in io["inputs"][:MAX_TESTS]]
            outs = [join(x) for x in io["outputs"][:MAX_TESTS]]
            if not all(isinstance(x, str) for x in ins + outs):
                continue
            if sum(len(x) for x in ins + outs) > MAX_IO_BYTES:
                continue
            try:
                sols = json.loads(r["solutions"]) if r["solutions"] else []
            except json.JSONDecodeError:
                sols = []
            problem = (f"{r['question'].strip()}\n\nRead the input from standard input and write the answer "
                       f"to standard output.\n\n{INSTRUCTION}")
            yield {
                "id": f"apps/{r['id']}", "source": "apps", "split": "train", "difficulty": r["difficulty"],
                "problem": problem, "test_format": "stdio", "entry_point": None,
                "tests": [{"input": i, "output": o} for i, o in zip(ins, outs)], "test_setup": "",
                "reference": sols[0] if sols else None,
            }


def build():
    held_out = set(int(x) for x in pd.read_parquet("/opt/wsp/eval/raw/mbppplus.parquet")["task_id"])
    rows = list(mbpp_rows(held_out)) + list(apps_rows())
    with open(f"{OUT}/candidates.jsonl", "w") as f:
        for r in rows:
            if r["reference"]:
                f.write(json.dumps(r) + "\n")
    with open(f"{OUT}/candidates_refgen.jsonl", "w") as f:
        for r in rows:
            if r["reference"]:
                text = "<think></think>\n```python\n" + r["reference"] + "\n```"
                f.write(json.dumps({"id": r["id"], "text": text, "n_tokens": 0, "finish_reason": "stop"}) + "\n")
    print(Counter(r["source"] + "/" + r["difficulty"] for r in rows if r["reference"]))


def finalize():
    ok = {g["id"] for g in map(json.loads, open(f"{OUT}/candidates_refgraded.jsonl")) if g["status"] == "pass"}
    rows = [r for r in map(json.loads, open(f"{OUT}/candidates.jsonl")) if r["id"] in ok]
    os.makedirs(f"{OUT}/final", exist_ok=True)
    stats = {}
    for src in ["mbpp", "apps"]:
        sel = [r for r in rows if r["source"] == src]
        with open(f"{OUT}/final/{src}_practice.jsonl", "w") as f:
            for r in sel:
                f.write(json.dumps(r) + "\n")
        stats[src] = dict(Counter(r["difficulty"] for r in sel))
    json.dump(stats, open(f"{OUT}/final/stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    finalize() if "--finalize" in sys.argv else build()
