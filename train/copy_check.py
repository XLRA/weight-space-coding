"""How much of the checks' validation pass rate comes from printing the statement's example output (2026-11-08).
It reads train/data and train/evals and runs no model code, so it runs on the host:

    /opt/wsp/venv/bin/python /opt/wsp/train/copy_check.py      (prints a table; writes train/checks/copying.json)

G-E (GRPO from E) learned to answer an APPS problem by printing the example output that its statement shows. Most
APPS problems here are tested only on those examples, so such an answer passes. For each check this gives:
- its validation pass rate as graded, over all 252 problems;
- the pass rate over the problems where printing the examples cannot pass: the 52 MBPP problems (3 tests, of which
  the prompt shows 1) and the APPS problems with a test whose input or output the statement does not show (compared
  with whitespace collapsed). These are the check's own grades, since every problem was graded on all its tests;
- that rate's paired difference against the arm's start over those problems (each problem's pass rate over its 4
  answers), with its standard error;
- how many of its APPS passes never read the input: no input(), stdin, open(0), fileinput or readline in the code.
  From G-E's step 100 most copying answers read the input first and print the example anyway, so this count misses
  them there. The pass rate over the problems above does not depend on how an answer copies.
It also counts the APPS and MBPP problems in train/data/train.jsonl and val.jsonl that are tested only on the
statement's examples.
"""
import gzip
import json
import math
import os
import re
import sys

T = os.environ.get("WSP_TRAIN", "/opt/wsp/train")
sys.path.insert(0, "/opt/wsp/harness")
import grade  # noqa: E402  (harness/grade.py: extract is how the reward reads an answer's code)

CHECKS = [("start-O", None), ("R-O/round_1", "start-O"), ("R-O/round_2", "start-O"),
          ("start-E", None), ("R-E/round_1", "start-E"), ("R-E/round_2", "start-E")]
READS = re.compile(r"\binput\s*\(|stdin|open\(\s*0|fileinput|readline")


def norm(s):
    return " ".join(s.split())


def shown(p, t):
    """Whether a test is one of the statement's own examples."""
    s = norm(p["problem"])
    if p["test_format"] == "function":
        return norm(t) in s
    return norm(t["input"]) in s and norm(t["output"]) in s


def examples_only(p):
    return all(shown(p, t) for t in p["tests"])


def paired(a, b):
    """Mean and SE (points) of a - b over paired values in [0, 1]."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return round(100 * m, 2), round(100 * sd / math.sqrt(n), 2)


def by_problem(rows, ids):
    """{problem id: pass rate over its answers}, for the problems in ids."""
    acc = {}
    for r in rows:
        if r["id"] in ids:
            acc.setdefault(r["id"], []).append(r["status"] == "pass")
    return {k: sum(v) / len(v) for k, v in acc.items()}


def rows(name):
    return [json.loads(line) for line in open(f"{T}/evals/{name}/val.jsonl")]


def main():
    val = {p["id"]: p for p in map(json.loads, open(f"{T}/data/val.jsonl"))}
    train = [json.loads(line) for line in open(f"{T}/data/train.jsonl")]
    keep = {i for i, p in val.items() if not examples_only(p)}
    out = {"data": {}, "kept_problems": sorted(keep), "checks": {}}
    for split, ps in (("train", train), ("val", list(val.values()))):
        for src in ("apps", "mbpp"):
            s = [p for p in ps if p["source"] == src]
            out["data"][f"{split} {src}"] = {"problems": len(s), "tested_only_on_examples": sum(map(examples_only, s))}
    steps = sorted((d for d in os.listdir(f"{T}/evals/G-E") if d.startswith("step_")), key=lambda d: int(d[5:]))
    checks = CHECKS + [(f"G-E/{d}", "start-E") for d in steps if os.path.exists(f"{T}/evals/G-E/{d}/val.jsonl")]
    print(f"kept: {len(keep)} validation problems ({sum(val[i]['source'] == 'mbpp' for i in keep)} MBPP, "
          f"{sum(val[i]['source'] == 'apps' for i in keep)} APPS); data: {out['data']}\n")
    print(f"{'check':<13} {'all':>6} {'kept':>6} {'MBPP':>6} {'APPS':>6}  kept vs start (SE)   APPS passes, reading no input")
    for name, base in checks:
        rs = rows(name)
        raw = {(g["id"], int(g["sample"])): g["text"]
               for g in map(json.loads, gzip.open(f"{T}/evals/{name}/val.raw.jsonl.gz", "rt"))}

        def rate(f):
            s = [r for r in rs if f(r)]
            return round(100 * sum(r["status"] == "pass" for r in s) / len(s), 2)

        c = {"all": rate(lambda r: True), "kept": rate(lambda r: r["id"] in keep),
             "kept_mbpp": rate(lambda r: r["id"] in keep and r["source"] == "mbpp"),
             "kept_apps": rate(lambda r: r["id"] in keep and r["source"] == "apps")}
        ap = [r for r in rs if r["source"] == "apps" and r["status"] == "pass"]
        c["apps_passes"] = len(ap)
        c["apps_passes_reading_no_input"] = sum(
            1 for r in ap if not READS.search(grade.extract(raw[(r["id"], int(r["sample"]))]) or ""))
        vs = ""
        if base:
            a, b = by_problem(rs, keep), by_problem(rows(base), keep)
            ids = sorted(a)
            m, se = paired([a[i] for i in ids], [b[i] for i in ids])
            c["kept_vs_start"] = {"base": base, "diff": m, "se": se}
            vs = f"{m:+.1f} ({se:.1f})"
        out["checks"][name] = c
        print(f"{name:<13} {c['all']:5.1f}% {c['kept']:5.1f}% {c['kept_mbpp']:5.1f}% {c['kept_apps']:5.1f}%  {vs:<19}  "
              f"{c['apps_passes_reading_no_input']} of {c['apps_passes']}")
    with open(f"{T}/checks/copying.json.tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(f"{T}/checks/copying.json.tmp", f"{T}/checks/copying.json")


if __name__ == "__main__":
    main()
