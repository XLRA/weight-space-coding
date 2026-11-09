"""Pairs two held-out evaluations' models with each other, problem by problem.

    python3 eval/compare.py <results/A>[:original] <results/B>[:original]

report.md pairs each model with the original. This pairs any two: for each suite and subset, A's pass rate on a problem
(averaged over its samples) minus B's, averaged over the problems, with its standard error over problems, as report.md
computes it. A folder stands for its edited model; `:original` picks its original model's answers instead, so
`compare.py results/X results/X:original` reproduces X's report. Both must have the same problems and samples.
"""
import json
import math
import os
import statistics
import sys
from collections import defaultdict

SUITES = [("code", "gen_{}.jsonl"), ("mmlupro", "gen_mmlupro_{}.jsonl"), ("ifeval", "gen_ifeval_{}.jsonl"),
          ("lcb", "gen_lcb_{}.jsonl"), ("math500", "gen_math500_{}.jsonl")]


def subset(suite, r):
    if suite == "code":
        return r["source"]
    if suite == "lcb":
        return r["difficulty"]
    return None


def load(spec, suite, pattern):
    folder, _, model = spec.partition(":")
    path = os.path.join(folder, pattern.format(model or "edited"))
    if not os.path.exists(path):
        return None
    passes, subsets = defaultdict(list), {}
    for line in open(path):
        r = json.loads(line)
        passes[r["id"]].append(r["status"] == "pass")
        subsets[r["id"]] = subset(suite, r)
    return {k: (sum(v) / len(v), len(v)) for k, v in passes.items()}, subsets


def main():
    a_spec, b_spec = sys.argv[1:3]
    print(f"A = {a_spec}\nB = {b_spec}\n")
    print("| Suite | Set | A | B | A - B | SE | Problems | A better | B better |")
    print("|---|---|---|---|---|---|---|---|---|")
    for suite, pattern in SUITES:
        a, b = load(a_spec, suite, pattern), load(b_spec, suite, pattern)
        if a is None or b is None:
            continue
        (a_rate, subsets), (b_rate, _) = a, b
        if set(a_rate) != set(b_rate) or any(a_rate[k][1] != b_rate[k][1] for k in a_rate):
            raise SystemExit(f"{suite}: the two evaluations have different problems or samples")
        groups = [("all", sorted(a_rate))]
        groups += [(s, sorted(k for k in a_rate if subsets[k] == s))
                   for s in sorted({v for v in subsets.values() if v is not None})]
        for name, ids in groups:
            diffs = [a_rate[k][0] - b_rate[k][0] for k in ids]
            mean = 100 * statistics.mean(diffs)
            se = 100 * statistics.stdev(diffs) / math.sqrt(len(diffs))
            print(f"| {suite} | {name} | {100 * statistics.mean(a_rate[k][0] for k in ids):.1f}% "
                  f"| {100 * statistics.mean(b_rate[k][0] for k in ids):.1f}% | {mean:+.2f} | {se:.2f} | {len(ids)} "
                  f"| {sum(d > 0 for d in diffs)} | {sum(d < 0 for d in diffs)} |")


if __name__ == "__main__":
    main()
