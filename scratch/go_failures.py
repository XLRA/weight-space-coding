"""Where G-O fails on the held-out suites (shares only), and how T's answers differ on LiveCodeBench."""
import json
import os
from collections import Counter, defaultdict

R = "/opt/wsp/eval/results"
SUITES = {"code": "gen_{}", "lcb": "gen_lcb_{}", "math500": "gen_math500_{}", "ifeval": "gen_ifeval_{}",
          "mmlupro": "gen_mmlupro_{}"}


def load(run, suite, model="edited"):
    stem = SUITES[suite].format(model)
    graded = [json.loads(l) for l in open(f"{R}/{run}/{stem}.jsonl")]
    raw = {(r["id"], r["sample"]): r for r in map(json.loads, open(f"{R}/{run}/{stem}.raw.jsonl"))}
    out = []
    for g in graded:
        r = raw[(g["id"], g["sample"])]
        g.setdefault("cut", r.get("finish_reason") == "length")
        g.setdefault("loop", bool(g.get("looped", False)))
        g.setdefault("n_tokens", r.get("n_tokens", 0))
        g.setdefault("difficulty", r.get("difficulty", "?"))
        out.append((g, r))
    return out


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


for suite in SUITES:
    rows = load("G-O-pp15", suite)
    n = len(rows)
    st = Counter(g["status"] for g, _ in rows)
    cut = [(g, r) for g, r in rows if g["cut"]]
    fin_fail = [(g, r) for g, r in rows if not g["cut"] and g["status"] != "pass"]
    closes = lambda r: r["text"].count("</think>")  # noqa: E731
    print(f"\n== {suite}: {n} answers; " + ", ".join(f"{k} {pct(v, n)}" for k, v in st.most_common()))
    print(f"   cut {pct(len(cut), n)}: loop-flagged {pct(sum(g['loop'] for g, _ in cut), len(cut))}, "
          f"thinking never closed {pct(sum(closes(r) == 0 for _, r in cut), len(cut))}, "
          f"closed 2+ times {pct(sum(closes(r) >= 2 for _, r in cut), len(cut))}")
    print(f"   finished but failed {pct(len(fin_fail), n)}: closed 2+ times "
          f"{pct(sum(closes(r) >= 2 for _, r in fin_fail), len(fin_fail))}, loop-flagged "
          f"{pct(sum(g['loop'] for g, _ in fin_fail), len(fin_fail))}, mean tokens "
          f"{sum(g['n_tokens'] for g, _ in fin_fail) / max(1, len(fin_fail)):,.0f}")
    passes = [(g, r) for g, r in rows if g["status"] == "pass"]
    if passes:
        toks = sorted(g["n_tokens"] for g, _ in passes)
        print(f"   passes: median tokens {toks[len(toks) // 2]:,}, 90th pct {toks[int(0.9 * len(toks))]:,}, "
              f"over 16k {pct(sum(t > 16384 for t in toks), len(toks))}, over 24k {pct(sum(t > 24576 for t in toks), len(toks))}")
    exc = Counter(str(g.get("exc", "?")).split(":")[0][:40] for g, _ in rows if g["status"] == "runtime_error")
    if exc:
        print("   runtime errors: " + ", ".join(f"{k} {v}" for k, v in exc.most_common(8)))
    if suite == "lcb":
        by = defaultdict(Counter)
        for g, _ in rows:
            by[g["difficulty"]][g["status"]] += 1
        for d, c in sorted(by.items()):
            tot = sum(c.values())
            print(f"   {d}: " + ", ".join(f"{k} {pct(v, tot)}" for k, v in c.most_common()))

# LiveCodeBench: problems G-O solves more often than T, and what T's answers did there
go = load("G-O-pp15", "lcb")
t = load("T-pp15", "lcb")
ps = defaultdict(lambda: [0, 0])
for g, _ in go:
    ps[g["id"]][0] += g["status"] == "pass"
for g, _ in t:
    ps[g["id"]][1] += g["status"] == "pass"
better = {k for k, (a, b) in ps.items() if a > b}
worse = {k for k, (a, b) in ps.items() if a < b}
print(f"\nLCB problems G-O solves more often than T: {len(better)}; fewer: {len(worse)}")
tst = Counter(g["status"] for g, _ in t if g["id"] in better)
gst = Counter(g["status"] for g, _ in go if g["id"] in better)
print("   on those, G-O's answers:", dict(gst.most_common()), " T's answers:", dict(tst.most_common()))
gt = [g["n_tokens"] for g, _ in go if g["id"] in better and g["status"] == "pass"]
tt = [g["n_tokens"] for g, _ in t if g["id"] in better]
print(f"   G-O's passing answers there: mean {sum(gt) / max(1, len(gt)):,.0f} tokens; T's answers: mean {sum(tt) / max(1, len(tt)):,.0f}")
tc = [r for g, r in t if g["id"] in better]
print(f"   T's answers there closing the thinking 2+ times: {pct(sum(r['text'].count('</think>') >= 2 for r in tc), len(tc))}")
