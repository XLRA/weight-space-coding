"""Endings of edited IFEval answers that miss quotation / postscript / at-least-words instructions."""
import collections
import json
import sys

sys.path.insert(0, "/eval")
from suites import after_think, ifeval_follow  # noqa: E402

res = sys.argv[1]
probs = {p["id"]: p for p in map(json.loads, open("/eval/final/suites/ifeval.jsonl"))}
raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_ifeval_edited.raw.jsonl"))}
shown = collections.Counter()
kinds = collections.Counter()
for k, r in sorted(raw.items()):
    p = probs[k[0]]
    resp = after_think(r["text"], empty_if_open=True)
    if not resp:
        continue
    f = ifeval_follow(p, resp)
    for iid, ok in zip(p["instruction_id_list"], f):
        if ok or iid not in ("startend:quotation", "detectable_content:postscript", "length_constraints:number_words"):
            continue
        end = resp.rstrip()[-1]
        kinds[(iid, "ends with sentence punctuation" if end in ".!?\")*" else "ends otherwise")] += 1
        if shown[iid] < 3:
            shown[iid] += 1
            print(f"== {iid} {k} words={len(resp.split())} thinking_tokens~{len(r['text'].split('</think>')[0]) // 4}")
            print("   END:", repr(resp[-220:]))
for x, n in sorted(kinds.items()):
    print(x, n)
