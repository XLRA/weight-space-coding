"""MMLU-Pro at presence penalty 1.5: how the edited model's thinking ends, and accuracy by exit type."""
import collections
import json
import sys

res = sys.argv[1]


def load(m):
    raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_mmlupro_{m}.raw.jsonl"))}
    gr = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{res}/gen_mmlupro_{m}.jsonl"))}
    return raw, gr


def exit_kind(text):
    if "</think>" not in text:
        return "no exit"
    think = text.split("</think>")[0]
    tail = think[-200:].strip()
    if len(tail) >= 40 and think.count(tail[-120:]) >= 2:
        return "loop exit"
    return "natural exit"


ro, go = load("original")
re_, ge = load("edited")
tab = collections.defaultdict(collections.Counter)
think_len = collections.defaultdict(list)
for k, g in ge.items():
    kind = exit_kind(re_[k]["text"])
    n = re_[k]["text"].count("</think>")
    o = go[k]["status"]
    e = g["status"]
    tab[kind]["n"] += 1
    tab[kind]["edited pass"] += e == "pass"
    tab[kind]["original pass"] += o == "pass"
    tab[kind]["original cut"] += o == "cut_at_max_tokens"
    tab[kind]["pass->fail"] += o == "pass" and e != "pass"
    tab[kind]["fail->pass"] += o != "pass" and e == "pass"
    tab[kind][">1 </think>"] += n > 1
    if o == "pass" and e != "pass":
        think_len["pass->fail edited"].append(len(re_[k]["text"].split("</think>")[0]))
        think_len["pass->fail original"].append(len(ro[k]["text"].split("</think>")[0]))
for kind, c in tab.items():
    print(kind, dict(c))
for k, v in think_len.items():
    v = sorted(v)
    print(k, "thinking chars median", v[len(v) // 2], "p25", v[len(v) // 4], "p75", v[3 * len(v) // 4])
# pass->fail where the edited thinking is much shorter than the original's
pf = [(len(re_[k]["text"].split("</think>")[0]), len(ro[k]["text"].split("</think>")[0]), exit_kind(re_[k]["text"]))
      for k in ge if go[k]["status"] == "pass" and ge[k]["status"] != "pass"]
print("pass->fail with edited thinking < half the original's:", sum(a < b / 2 for a, b, _ in pf), "of", len(pf),
      "; of those loop exits:", sum(a < b / 2 and x == "loop exit" for a, b, x in pf))
