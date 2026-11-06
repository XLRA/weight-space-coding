"""Where pilot-14's edit fails, against its starting edit: python failures_p14.py (venv python)."""
import collections, json, random, sys
sys.path[:0] = ["/opt/wsp/eval", "/opt/wsp/harness"]
R = "/opt/wsp/eval/results"
random.seed(2)
def rows(run, name): return [json.loads(l) for l in open(f"{R}/{run}/{name}")]
def key(r): return (r["id"], r["sample"])
def shape(t):
    n = t.count("</think>")
    after = t.rsplit("</think>", 1)[1] if n else None
    return n, (len(after.strip()) if after is not None else -1)
for suite, raw, graded in (("ifeval", "gen_ifeval_edited.raw.jsonl", "gen_ifeval_edited.jsonl"),
                           ("mmlupro", "gen_mmlupro_edited.raw.jsonl", "gen_mmlupro_edited.jsonl"),
                           ("math500", "gen_math500_edited.raw.jsonl", "gen_math500_edited.jsonl")):
    new_raw = {key(r): r for r in rows("pilot-14-pp15", raw)}
    new = {key(r): r for r in rows("pilot-14-pp15", graded)}
    old = {key(r): r for r in rows("combo-12-13-pp15", graded)}
    na = [k for k, r in new.items() if r["status"] == "no_answer"]
    print(f"\n=== {suite}: no_answer {sum(r['status']=='no_answer' for r in old.values())} -> {len(na)}")
    print("  starting edit's status for the same item/sample:", dict(collections.Counter(old[k]["status"] for k in na)))
    sh = collections.Counter()
    toks = []
    for k in na:
        n, a = shape(new_raw[k]["text"])
        sh[f"{n} </think>" if n < 4 else "4+ </think>"] += 1
        sh["empty after last close" if a == 0 else "text after last close" if a > 0 else "never closed"] += 1
        toks.append(new[k]["n_tokens"])
    print("  shapes:", dict(sh))
    toks.sort(); print("  tokens: median", toks[len(toks)//2], "min", toks[0], "max", toks[-1])
    for k in random.sample(na, min(3, len(na))):
        t = new_raw[k]["text"]; n, a = shape(t)
        print(f"  -- {k} ({n} closes, {new[k]['n_tokens']} tokens) ends: ...{t[-260:]!r}")
    flips = collections.Counter()
    for k in new:
        if k in old and (old[k]["status"] == "pass") != (new[k]["status"] == "pass"):
            flips[("fail->pass" if new[k]["status"] == "pass" else "pass->fail") + ": " + (old[k]["status"] if new[k]["status"] == "pass" else new[k]["status"])] += 1
    print("  flips by sample (old status for fail->pass, new status for pass->fail):", dict(flips.most_common(8)))
# HumanEval+ flips
new = {key(r): r for r in rows("pilot-14-pp15", "gen_edited.jsonl")}
old = {key(r): r for r in rows("combo-12-13-pp15", "gen_edited.jsonl")}
for src in ("humaneval", "mbpp"):
    c = collections.Counter()
    for k, r in new.items():
        if not r["source"].startswith(src): continue
        o = old[k]
        if o["status"] == "pass" and r["status"] != "pass": c["pass->" + r["status"]] += 1
        if o["status"] != "pass" and r["status"] == "pass": c[o["status"] + "->pass"] += 1
    print(f"\n=== {src} flips:", dict(c.most_common()))
