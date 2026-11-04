"""Are IFEval answers cut short? python3 ifeval_length.py <results dir>"""
import collections
import json
import statistics
import sys

sys.path.insert(0, "/eval")
from suites import after_think, ifeval_follow  # noqa: E402

res = sys.argv[1]
probs = {p["id"]: p for p in map(json.loads, open("/eval/final/suites/ifeval.jsonl"))}
rows = {}
for m in ("original", "edited"):
    raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_ifeval_{m}.raw.jsonl"))}
    rows[m] = raw

c = collections.Counter()
ratios = collections.defaultdict(list)
both = collections.defaultdict(lambda: [0, 0, 0])  # per instruction: n with both answered, orig ok, edited ok
for k, e in rows["edited"].items():
    p = probs[k[0]]
    resp_e = after_think(e["text"], empty_if_open=True)
    o = rows["original"][k]
    resp_o = after_think(o["text"], empty_if_open=True) if o.get("finish_reason") != "length" else ""
    for iid, kw in zip(p["instruction_id_list"], p["kwargs"]):
        if iid == "length_constraints:number_words" and kw.get("relation") == "at least":
            n = len(resp_e.split())
            ratios["edited"].append(n / kw["num_words"])
            if resp_o:
                ratios["original"].append(len(resp_o.split()) / kw["num_words"])
    if resp_o and resp_e:
        fe, fo = ifeval_follow(p, resp_e), ifeval_follow(p, resp_o)
        for iid, a, b in zip(p["instruction_id_list"], fo, fe):
            both[iid][0] += 1
            both[iid][1] += a
            both[iid][2] += b
    # how does the edited answer end?
    last = resp_e.rstrip()[-1:] if resp_e.strip() else ""
    c["ends with . ! ? \" ) * or `" if last in ".!?\")*`" else "ends otherwise"] += 1
    c[f"finish {e.get('finish_reason')}"] += 1
print(dict(c))
for m, r in ratios.items():
    r = sorted(r)
    print(f"{m}: 'at least N words' answers, words/N median {statistics.median(r):.2f}, share below 1: "
          f"{sum(x < 1 for x in r)}/{len(r)}")
print("\nper instruction, items where both models answered: n, original ok, edited ok")
tot = [0, 0, 0]
for iid, (n, a, b) in sorted(both.items(), key=lambda x: x[1][2] - x[1][1]):
    tot = [tot[0] + n, tot[1] + a, tot[2] + b]
    if n >= 6:
        print(f"  {iid:<55} {n:>4} {a:>4} {b:>4}  {b - a:+d}")
print("total", tot)
