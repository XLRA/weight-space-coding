"""Headroom in MATH-500 'no_answer' answers of an evaluation: python3 math_oracle.py <results dir>."""
import collections
import json
import re
import sys

sys.path.insert(0, "/eval")
from suites import last_boxed, math_correct  # noqa: E402

res = sys.argv[1]
gold = {}
for l in open("/eval/final/suites/math500.jsonl"):
    p = json.loads(l)
    gold[p["id"]] = p.get("answer") or p.get("gold")
raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_math500_edited.raw.jsonl"))}
graded = [json.loads(l) for l in open(f"{res}/gen_math500_edited.jsonl")]
orig = {(r["id"], r["sample"]): r["status"] for r in map(json.loads, open(f"{res}/gen_math500_original.jsonl"))}

c = collections.Counter()
lvl = collections.defaultdict(collections.Counter)
nthink = collections.Counter()
for g in graded:
    k = (g["id"], g["sample"])
    lvl[g["difficulty"]][g["status"]] += 1
    if g["status"] != "no_answer":
        continue
    t = raw[k]["text"]
    n = t.count("</think>")
    nthink[min(n, 5)] += 1
    c["no_answer"] += 1
    any_box = last_boxed(t)
    if any_box is not None:
        c["has a \\boxed somewhere"] += 1
        if math_correct(any_box, gold[g["id"]]):
            c["last \\boxed anywhere is correct"] += 1
    # every boxed in the text
    boxes = [last_boxed(t[:m.end() + 400]) for m in re.finditer(r"\\boxed", t)]
    if any(b is not None and math_correct(b, gold[g["id"]]) for b in boxes[-6:]):
        c["one of last 6 \\boxed correct"] += 1
    if n >= 2:
        before = t[:t.rfind("</think>")]
        b2 = last_boxed(before.split("</think>", 1)[-1]) if "</think>" in before else None
        if b2 is not None:
            c["boxed between first and last </think>"] += 1
            if math_correct(b2, gold[g["id"]]):
                c["  ... and correct"] += 1
    if orig.get(k) == "pass":
        c["original passed same sample"] += 1
    tail = t[-300:].replace("\n", " ")
    if c["no_answer"] <= 4:
        print("TAIL:", tail[-300:], "\n")
print(dict(c))
print("</think> count in no_answer:", dict(sorted(nthink.items())))
for k in sorted(lvl):
    print(k, dict(lvl[k]))
