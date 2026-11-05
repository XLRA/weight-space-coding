"""IFEval instruction types: follow rate per type for the unedited model, pilot-10 and the combined edit, and
how many failed instructions each type accounts for. Run in wsp-grade:1 with /opt/wsp/eval at /eval:
python3 ifeval_types.py <results dir A> <results dir B> (raw answers are read from each; originals from A)."""
import collections
import json
import sys

sys.path.insert(0, "/eval")
from suites import after_think, ifeval_follow  # noqa: E402

probs = {p["id"]: p for p in map(json.loads, open("/eval/final/suites/ifeval.jsonl"))}


def per_type(path):
    ok, tot = collections.Counter(), collections.Counter()
    shapes = collections.Counter()
    for r in map(json.loads, open(path)):
        p = probs[r["id"]]
        resp = after_think(r["text"], empty_if_open=True)
        f = ifeval_follow(p, resp)
        for iid, good in zip(p["instruction_id_list"], f):
            ok[iid] += bool(good)
            tot[iid] += 1
        if not all(f):
            shapes["failed answers"] += 1
            shapes[">1 </think>" if r["text"].count("</think>") > 1 else "1 </think>" if "</think>" in r["text"] else "no </think>"] += 1
    return ok, tot, shapes


A, B = sys.argv[1], sys.argv[2]
cols = [("unedited", f"{A}/gen_ifeval_original.raw.jsonl"), ("pilot-10", f"{A}/gen_ifeval_edited.raw.jsonl"),
        ("combo", f"{B}/gen_ifeval_edited.raw.jsonl")]
res = {name: per_type(path) for name, path in cols}
ok_c, tot_c, _ = res["combo"]
fails = {iid: tot_c[iid] - ok_c[iid] for iid in tot_c}
print(f"{'instruction':<46}{'n':>5}" + "".join(f"{n:>10}" for n, _ in cols) + f"{'combo fails':>13}")
for iid in sorted(fails, key=lambda i: -fails[i])[:25]:
    print(f"{iid:<46}{tot_c[iid]:>5}" + "".join(f"{100 * res[n][0][iid] / res[n][1][iid]:>9.0f}%" for n, _ in cols)
          + f"{fails[iid]:>13}")
print(f"total failed instructions (combo): {sum(fails.values())} of {sum(tot_c.values())}")
for n, _ in cols:
    print(n, dict(res[n][2]))
