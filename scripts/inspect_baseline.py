"""Read-only look at the baseline: are the cuts loops, and what are the interface and self-test failures?"""
import ast
import json
import re

P = {p["id"]: p for p in map(json.loads, open("/opt/wsp/baseline/problems.jsonl"))}
G = {g["id"]: g for g in map(json.loads, open("/opt/wsp/baseline/gen.jsonl"))}
R = [json.loads(l) for l in open("/opt/wsp/baseline/graded.jsonl")]
FENCE = re.compile(r"```[ \t]*(?:python3?|py|Python)?[ \t]*\n(.*?)```", re.S)


def asc(s):
    return s.encode("ascii", "replace").decode().replace("\n", " | ")


def code_of(rid):
    return FENCE.findall(G[rid]["text"].split("</think>", 1)[1])[-1]


cut = [r for r in R if r["status"] == "cut_at_max_tokens"]
closed = sum("</think>" in G[r["id"]]["text"] for r in cut)
print(f"cut answers: {len(cut)}; closed </think> before the cut: {closed}; loop-flagged: {sum(r['loop'] for r in cut)}")
nc = [r for r in R if r["status"] != "cut_at_max_tokens"]
print(f"not cut: {len(nc)}; loop-flagged: {sum(r['loop'] for r in nc)}; pass: {sum(r['status'] == 'pass' for r in nc)}; "
      f"mean tokens: {sum(r['n_tokens'] for r in nc) / len(nc):.0f}")

print("\n== tails of 4 cut answers (last 220 chars)")
for r in cut[::140][:4]:
    print("-", r["id"], ":", asc(G[r["id"]]["text"][-220:]))

print("\n== wrong_interface examples")
for r in [r for r in R if r["status"] == "wrong_interface"][:6]:
    tree = ast.parse(code_of(r["id"]))
    defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    print("-", r["id"], "expects", P[r["id"]]["entry_point"], "| defines", defs[:5])

print("\n== self-test AssertionError examples (code tail)")
for r in [r for r in R if r.get("exc") == "AssertionError"][:2]:
    print("-", r["id"], ":", asc(code_of(r["id"])[-260:]))
