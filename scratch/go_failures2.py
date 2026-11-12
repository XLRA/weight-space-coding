"""Runtime-error messages of G-O, and how often each model reopens its thinking (two or more </think>)."""
import json
import re
from collections import Counter

R = "/opt/wsp/eval/results"
SUITES = {"code": "gen_{}", "lcb": "gen_lcb_{}", "math500": "gen_math500_{}", "ifeval": "gen_ifeval_{}"}


def rows(run, suite):
    stem = SUITES[suite].format("edited")
    graded = [json.loads(l) for l in open(f"{R}/{run}/{stem}.jsonl")]
    raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{R}/{run}/{stem}.raw.jsonl"))}
    return [(g, raw[(g["id"], g["sample"])]) for g in graded]


for suite in ("code", "lcb"):
    exc = Counter()
    for g, _ in rows("G-O-pp15", suite):
        if g["status"] == "runtime_error":
            e = re.sub(r"'[^']{0,40}'", "'…'", str(g.get("exc", "?")))
            exc[e[:90]] += 1
    print(f"== G-O {suite} runtime errors, most common messages:")
    for k, v in exc.most_common(12):
        print(f"   {v:4d}  {k}")

print("\n== answers that close the thinking 2+ times (share of finished answers), and pass rate among them")
for run in ("pilot-14-pp15", "G-O-pp15", "T-pp15", "G-E2-pp15"):
    out = []
    for suite in SUITES:
        rs = [(g, r) for g, r in rows(run, suite) if r.get("finish_reason") != "length"]
        re2 = [(g, r) for g, r in rs if r["text"].count("</think>") >= 2]
        ok = sum(g["status"] == "pass" for g, _ in re2)
        out.append(f"{suite} {100 * len(re2) / max(1, len(rs)):.1f}% (pass {100 * ok / max(1, len(re2)):.0f}%)")
    print(f"   {run:15} " + "; ".join(out))
