"""IFEval failures of an evaluation by instruction type, and what the answers look like: python3 ifeval_oracle.py <results dir>."""
import collections
import json
import re
import statistics
import sys

sys.path.insert(0, "/eval")
from suites import after_think, ifeval_follow  # noqa: E402

res = sys.argv[1]
probs = {p["id"]: p for p in map(json.loads, open("/eval/final/suites/ifeval.jsonl"))}
raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_ifeval_edited.raw.jsonl"))}
graded = [json.loads(l) for l in open(f"{res}/gen_ifeval_edited.jsonl")]

per = collections.defaultdict(lambda: [0, 0])
shape = collections.Counter()
lens = {"pass": [], "not_followed": []}
nthink = {"pass": collections.Counter(), "not_followed": collections.Counter()}
reasoning_words = re.compile(r"\b(Wait|Let me|Let's|I need to|I should|the user (wants|asked))\b")
for g in graded:
    k = (g["id"], g["sample"])
    t = raw[k]["text"]
    st = g["status"]
    if st in lens:
        resp = after_think(t, empty_if_open=True)
        lens[st].append(len(resp.split()))
        nthink[st][min(t.count("</think>"), 4)] += 1
        if st == "not_followed":
            shape["not_followed"] += 1
            if t.count("</think>") > 1:
                shape[">1 </think>"] += 1
            if len(reasoning_words.findall(resp)) >= 3:
                shape["answer has >=3 reasoning phrases"] += 1
            # would the first clean segment pass? the text between the first and second </think>
            parts = t.split("</think>")
            if len(parts) > 2:
                first = parts[1].strip()
                if first and all(ifeval_follow(probs[g["id"]], first)):
                    shape["text after FIRST </think> (up to the next) passes"] += 1
    if st in ("pass", "not_followed"):
        f = ifeval_follow(probs[g["id"]], after_think(t, empty_if_open=True))
        for iid, ok in zip(probs[g["id"]]["instruction_id_list"], f):
            per[iid][0] += ok
            per[iid][1] += 1
print(dict(shape))
for st in lens:
    print(st, "answer words median", statistics.median(lens[st]), "p90", sorted(lens[st])[int(.9 * len(lens[st]))],
          "</think> counts", dict(sorted(nthink[st].items())))
print("\ninstruction type: followed / total (worst first, total >= 8)")
for iid, (ok, n) in sorted(per.items(), key=lambda x: x[1][0] / x[1][1]):
    if n >= 8:
        print(f"  {iid:<55} {ok:>4}/{n:<4} {100 * ok / n:5.1f}%  failures {n - ok}")
