"""Where the combined edit still fails: python failures_combo.py <results dir> (run with the venv python)."""
import ast
import collections
import json
import random
import re
import sys

sys.path[:0] = ["/opt/wsp/eval", "/opt/wsp/harness"]
import grade  # noqa: E402
import suites as S  # noqa: E402

res = sys.argv[1]
random.seed(1)


def rows(name):
    return [json.loads(l) for l in open(f"{res}/{name}")]


def tail(t, n=220):
    return t[-n:].replace("\n", "\\n")


# ---------------- MATH-500 ----------------
raw = {(r["id"], r["sample"]): r for r in rows("gen_math500_edited.raw.jsonl")}
g = rows("gen_math500_edited.jsonl")
print("MATH-500 statuses by level:")
lv = collections.defaultdict(collections.Counter)
for r in g:
    lv[r["difficulty"]][r["status"]] += 1
for k in sorted(lv):
    print(f"  {k}: {dict(lv[k])}")
for st in ("cut_at_max_tokens", "no_answer", "wrong_answer"):
    sel = [r for r in g if r["status"] == st]
    phase = collections.Counter()
    boxes = collections.Counter()
    loop = sum(r["loop"] for r in sel)
    ans_len = []
    for r in sel:
        t = raw[(r["id"], r["sample"])]["text"]
        n = t.count("</think>")
        phase["no </think>" if n == 0 else "1 </think>" if n == 1 else "2+ </think>"] += 1
        if n:
            a = t.split("</think>", 1)[1]
            ans_len.append(len(a))
            boxes[min(a.count("\\boxed"), 3)] += 1
    ans_len.sort()
    med = ans_len[len(ans_len) // 2] if ans_len else 0
    print(f"\n{st}: {len(sel)} answers, loop-flagged {loop}; {dict(phase)}; boxed count in answer part {dict(sorted(boxes.items()))}; "
          f"median chars after the first </think> {med:,}")
    for r in random.sample(sel, min(4, len(sel))):
        t = raw[(r["id"], r["sample"])]["text"]
        print(f"  [{r['difficulty']}, {r['n_tokens']} tok, </think> x{t.count('</think>')}] ...{tail(t)}")

# per-problem: do failures sit on problems the edit solves in other samples?
by = collections.defaultdict(list)
for r in g:
    by[r["id"]].append(r["status"])
mixed = sum(1 for v in by.values() if "pass" in v and any(s in ("cut_at_max_tokens", "no_answer") for s in v))
never = sum(1 for v in by.values() if "pass" not in v)
fails_on_mixed = sum(sum(s in ("cut_at_max_tokens", "no_answer") for s in v) for v in by.values() if "pass" in v)
print(f"\nproblems with a pass in some sample and a cut/no-answer in another: {mixed} "
      f"({fails_on_mixed} cut/no-answer answers on them); problems never solved: {never} of {len(by)}")

# ---------------- LiveCodeBench ----------------
raw = {(r["id"], r["sample"]): r for r in rows("gen_lcb_edited.raw.jsonl")}
g = rows("gen_lcb_edited.jsonl")
print("\nLCB statuses by difficulty:")
lv = collections.defaultdict(collections.Counter)
for r in g:
    lv[r["difficulty"]][r["status"]] += 1
for k in sorted(lv):
    print(f"  {k}: {dict(lv[k])}")
kinds = collections.Counter()
examples = collections.defaultdict(list)
fences = collections.Counter()
for r in g:
    if r["status"] != "syntax_error":
        continue
    t = raw[(r["id"], r["sample"])]["text"]
    a = t.split("</think>", 1)[1]
    fences[min(a.count("```"), 6)] += 1
    code = grade.extract(t)
    try:
        ast.parse(code)
        kinds["code alone parses; fails only with the prelude"] += 1
        examples["prelude"].append((r, code))
        continue
    except SyntaxError as e:
        msg = re.sub(r"\(.*?\)|'.*?'|line \d+", "", e.msg).strip()
        kinds[msg] += 1
        examples[msg].append((r, code, e.lineno, e.text))
print(f"\nLCB syntax errors: {sum(kinds.values())}; ``` count in the answer part: {dict(sorted(fences.items()))}")
for k, n in kinds.most_common(10):
    print(f"  {n:4d}  {k}")
for k, n in kinds.most_common(4):
    for ex in examples[k][:2]:
        if k == "prelude":
            r, code = ex
            print(f"\n  [{k}] {r['id']}: code starts {code[:200]!r}")
        else:
            r, code, ln, txt = ex
            lines = code.splitlines()
            ctx = "\\n".join(lines[max(0, (ln or 1) - 3):(ln or 1) + 1])
            print(f"\n  [{k}] {r['id']} ({len(lines)} lines, error at {ln}): ...{ctx[:400]}")
print("\nLCB_PRELUDE:", repr(S.LCB_PRELUDE[:300]))

# ---------------- code suite ----------------
g = rows("gen_edited.jsonl")
c = collections.Counter(r["status"] for r in g)
exc = collections.Counter(r.get("exc") for r in g if r["status"] == "runtime_error")
print(f"\ncode statuses: {dict(c)}\nruntime errors by exception: {dict(exc.most_common(8))}")
kinds = collections.Counter()
for r in g:
    if r["status"] != "syntax_error":
        continue
    try:
        ast.parse(r["code"])
        kinds["parses alone"] += 1
    except SyntaxError as e:
        kinds[re.sub(r"\(.*?\)|'.*?'|line \d+", "", e.msg).strip()] += 1
    except TypeError:
        kinds["no code"] += 1
print(f"code syntax errors: {dict(kinds.most_common(6))}")
