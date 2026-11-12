"""pilot-20's side measures, as plans/edit-after-training.md promised, and the session's cost.

Per suite, for G-O, pilot-20 and T (G-O plus E's change): the share of answers cut at 32,768
tokens, the share of finished answers that close the thinking two or more times (with the pass
rate among them, as in scratch/go_failures2.py), and the mean length in tokens.
Then the session's forward passes by kind, and the GPU time they took (from the request to the
agent's next turn, in runs/pilot-20/driver.log).

Run on the server: python3 scratch/pilot20_measures.py
"""
import json
import re
from collections import Counter
from datetime import datetime

R = "/opt/wsp/eval/results"
SUITES = {"code": "gen_{}", "mmlupro": "gen_mmlupro_{}", "ifeval": "gen_ifeval_{}", "lcb": "gen_lcb_{}",
          "math500": "gen_math500_{}"}


def rows(run, suite):
    stem = SUITES[suite].format("edited")
    raw = {}
    for line in open(f"{R}/{run}/{stem}.raw.jsonl"):
        r = json.loads(line)
        raw[(r["id"], r["sample"])] = (r["text"].count("</think>"), r.get("finish_reason"))
    return [(g, *raw[(g["id"], g["sample"])]) for g in map(json.loads, open(f"{R}/{run}/{stem}.jsonl"))]


print("| Suite | Model | Cut at 32k | Finished answers closing the thinking 2+ times (pass rate among them) | Mean length |")
print("|---|---|---|---|---|")
for suite in SUITES:
    for run, name in (("G-O-pp15", "G-O"), ("pilot-20-pp15", "pilot-20"), ("T-pp15", "T")):
        rs = rows(run, suite)
        cut = [g for g, _, fr in rs if fr == "length"]
        assert len(cut) == sum(g.get("cut", g["status"] == "cut_at_max_tokens") for g, _, _ in rs)
        fin = [(g, c) for g, c, fr in rs if fr != "length"]
        re2 = [g for g, c in fin if c >= 2]
        ok = sum(g["status"] == "pass" for g in re2)
        mean = sum(g["n_tokens"] for g, _, _ in rs) / len(rs)
        print(f"| {suite} | {name} | {100 * len(cut) / len(rs):.1f}% | {100 * len(re2) / max(1, len(fin)):.1f}% "
              f"({100 * ok / max(1, len(re2)):.0f}%) | {mean:,.0f} |")

print("\nThe session's forward passes:")
t = lambda line: datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
log = open("/opt/wsp/runs/pilot-20/driver.log").read().splitlines()
kinds, busy, pending = Counter(), 0.0, None
for line in log:
    m = re.match(r"\S+ \S+ request \d+: (\S+)", line)
    if m:
        kinds[m.group(1)] += 1
        pending = t(line)
    elif pending and re.match(r"\S+ \S+ turn \d+: forward", line):
        busy += (t(line) - pending).total_seconds()
        pending = None
print("   ", dict(kinds), f"total {sum(kinds.values())}; GPU time {busy / 3600:.2f} h")
print("    session", log[0][:19], "to", [l for l in log if "run finished" in l][0][:19])
st = json.load(open("/opt/wsp/runs/pilot-20/state.json"))
print("    turns", st["turn"], "conversations", st["conv"] + 1, "code runs", st["code_runs"],
      "Opus output tokens", st["tokens"]["output_tokens"], "limit waits", st["limit_waits"])
