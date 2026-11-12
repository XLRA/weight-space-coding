"""Writes practice/val2/: the practice set for an editing run that starts from a model trained on
train/data2/train.jsonl (G-O, G-E2). It holds the 220 usable problems of train/data2/val.jsonl, which that
training never used.

    /opt/wsp/venv/bin/python train/held_back_practice.py

The rows have practice/final's fields, with data2's tests. An APPS problem keeps its own tests and gains up to 10
more, made by changing the inputs of its examples and kept where 8 human solutions agree on the output
(train/make_tests.py). These 220 problems (val2) also chose G-O's and G-E2's checkpoints
(plans/training-phase.md), so the trained models are not entirely blind to them.
about.txt replaces the prompt's description of the two files (harness/driver.py --practice).
"""
import json
import os
from collections import Counter

WSP = "/opt/wsp"
OUT = f"{WSP}/practice/val2"
KEYS = ["id", "source", "split", "difficulty", "problem", "test_format", "entry_point", "tests", "test_setup",
        "reference"]

final = {}
for fn in ("mbpp_practice.jsonl", "apps_practice.jsonl"):
    for line in open(f"{WSP}/practice/final/{fn}"):
        p = json.loads(line)
        final[p["id"]] = p
train = {json.loads(line)["id"] for line in open(f"{WSP}/train/data2/train.jsonl")}
use = [r for r in map(json.loads, open(f"{WSP}/train/data2/val.jsonl")) if r["usable"]]
assert len(use) == 220, len(use)
assert not {r["id"] for r in use} & train, "a validation problem is in the training data"
for r in use:  # the practice set's problems, with data2's tests
    p = final[r["id"]]
    assert all(r[k] == p[k] for k in KEYS if k != "tests"), r["id"]
    assert len(r["tests"]) - r["tests_added"] == len(p["tests"]) and all(t in r["tests"] for t in p["tests"]), r["id"]

os.makedirs(OUT, exist_ok=True)
by = {s: [r for r in use if r["source"] == s] for s in ("mbpp", "apps")}
for s, rs in by.items():
    with open(f"{OUT}/{s}_practice.jsonl", "w") as f:
        f.writelines(json.dumps({k: r[k] for k in KEYS}) + "\n" for r in rs)
json.dump({s: dict(Counter(r["difficulty"] for r in rs)) for s, rs in by.items()}, open(f"{OUT}/stats.json", "w"),
          indent=1)


def listing(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


mbpp, apps = by["mbpp"], by["apps"]
splits = [s for s in ("train", "validation", "prompt", "test") if any(r["split"] == s for r in mbpp)]
assert {r["split"] for r in mbpp} == set(splits)
level = Counter(r["difficulty"] for r in apps)
own = max(len(r["tests"]) - r["tests_added"] for r in apps)
gained = sum(r["tests_added"] > 0 for r in apps)
count = Counter(len(r["tests"]) for r in apps)
about = (f"/practice/mbpp_practice.jsonl ({len(mbpp)} short function-writing problems from the MBPP "
         f"{listing(splits)} split{'s' if len(splits) > 1 else ''}; each problem statement shows one of its tests), "
         f"/practice/apps_practice.jsonl ({len(apps)} APPS train problems whose programs read standard input: "
         f"{level['introductory']} introductory, {level['interview']} interview, {level['competition']} competition; "
         f"each has its own tests, at most {own}, and {gained} of them have up to 10 more, made by changing the "
         f"inputs of the problem's examples and kept where 8 human solutions agree on the output; most have "
         f"{count.most_common(1)[0][0]} tests in all, at most {max(count)}).")
open(f"{OUT}/about.txt", "w").write(about + "\n")
print(json.load(open(f"{OUT}/stats.json")))
print(about)
