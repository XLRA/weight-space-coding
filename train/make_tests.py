"""Make tests for the APPS practice problems that the statement does not show (2026-11-08).

Most APPS problems in train/data are tested only on their statement's examples, so printing the example output
passes (G-E learned to). This makes new tests from APPS's own human solutions, with no model involved:
- each human solution (up to SOLUTIONS of them) runs on the problem's tests, and the ones that pass are kept;
- the example inputs are mutated: integers nudged or redrawn within the range their line (or column) uses, letters
  and grid characters swapped within the characters the input uses, with the lines and the number of tokens on each
  line unchanged. Counts (a number equal to the next line's length, or to how many lines follow) and query types
  stay fixed (plan());
- a mutated input becomes a test when the first AGREE kept solutions all exit cleanly on it and print the same
  output (compared with whitespace collapsed, as the grader does). That output is the test's expected output.
The other kept solutions are held out: the share of them that pass every new test measures how often a new test
is wrong. Runs in the grading container (wsp-grade:1: no network, read-only, uid 65534):

    python3 /train/make_tests.py /work/in.jsonl > out.jsonl

in.jsonl has one problem per line: {"id", "tests": [{"input", "output"}], "solutions": [code, ...],
"answers": [{"check", "code"}]}, where answers are saved answers that pass the old tests; each gets run on the
new ones. Each output line has the problem's new tests and its counts.
"""
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/harness")
from grade import ENV, limits  # noqa: E402  (the grader's process limits: 2 GiB, 64 processes, uid 65534)

SOLUTIONS, AGREE, WANT, TRIES, TIMEOUT = 16, int(os.environ.get("AGREE", "8")), 10, 60, 4
TOK = re.compile(r"(\s+)")
INT = re.compile(r"[+-]?\d+")
FLOAT = re.compile(r"[+-]?\d+\.\d+")


def run(code, stdin, tmp):
    """stdout of the code on stdin, or None if it fails, times out or prints nothing."""
    path = os.path.join(tmp, "s.py")
    with open(path, "w") as f:
        f.write(code)
    try:
        r = subprocess.run([sys.executable, path], input=stdin, capture_output=True, text=True,
                           timeout=TIMEOUT, cwd=tmp, preexec_fn=limits, env=ENV)
    except (subprocess.TimeoutExpired, UnicodeDecodeError):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def passes(code, tests, tmp):
    for t in tests:
        out = run(code, t["input"], tmp)
        if out is None or out.split() != t["output"].split():
            return False
    return True


def new_int(v, span, rng):
    """A new value for an integer token. span is the range its line (or its column, in a run of lines of the same
    shape) uses in the example, so a value that must not exceed n or a position stays inside what the example shows."""
    lo = 1 if v >= 1 else (0 if v == 0 else -max(10, 2 * abs(v)))
    if span and span[0] < span[1]:
        a, b = max(lo, span[0]), span[1]
        s = rng.random()
        w = rng.randint(a, b) if s < 0.6 else v + rng.choice([-2, -1, 1, 2]) if s < 0.9 else rng.choice([a, b])
        return min(b, max(a, w))
    w = v + rng.choice([-2, -1, 1, 2]) if rng.random() < 0.7 else rng.randint(lo, max(lo, 2 * abs(v)))
    return max(lo, w)


def plan(text):
    """Which integer tokens of an input stay fixed, and each other integer's span (see new_int).
    Fixed: a single number on the first line (a test count, or n); a number that equals the next line's token count,
    or the length of the next line's only token, or how many lines of the next line's shape follow; and the first
    column of a run of 3+ lines of the same shape when it takes at most 3 values (a query type)."""
    toks = [l.split() for l in text.split("\n")]
    binary = any(len(t) >= 2 and set(t) <= {"0", "1"} and (t[0] == "0" or len(t) >= 4) for tl in toks for t in tl)

    def is_int(t):
        return bool(INT.fullmatch(t)) and not (len(t) > 1 and t.lstrip("+-")[0] == "0") and \
            not (binary and len(t) >= 2 and set(t) <= {"0", "1"})

    n, runs, run_of = len(toks), [], []
    for i, tl in enumerate(toks):
        if i and tl and len(tl) == len(toks[i - 1]):
            runs[-1][1] = i
        else:
            runs.append([i, i])
        run_of.append(len(runs) - 1)
    fixed, span = set(), {}
    for a, b in runs:
        width = len(toks[a])
        cols = [[int(toks[i][j]) for i in range(a, b + 1) if is_int(toks[i][j])] for j in range(width)]
        if b - a >= 2 and width >= 2 and cols[0] and len(cols[0]) == b - a + 1 and len(set(cols[0])) <= 3:
            fixed |= {(i, 0) for i in range(a, b + 1)}
        for i in range(a, b + 1):
            row = [int(t) for t in toks[i] if is_int(t)]
            for j, t in enumerate(toks[i]):
                if not is_int(t):
                    continue
                vals = cols[j] if b > a else row
                if len(vals) >= 2:
                    span[(i, j)] = (min(vals), max(vals))
    for i, tl in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < n else []
        follow = runs[run_of[i + 1]][1] - i if nxt else 0
        for j, t in enumerate(tl):
            if is_int(t):
                v = int(t)
                if (i == 0 and len(tl) == 1) or (nxt and (len(nxt) == v or (len(nxt) == 1 and len(nxt[0]) == v))) \
                        or follow == v:
                    fixed.add((i, j))
    return is_int, fixed, span


def classes(text):
    """The characters an input uses, by kind, so a mutated letter stays the kind of letter the input has."""
    seen = {"lower": set(), "upper": set(), "other": set()}
    for ch in text:
        if ch.islower():
            seen["lower"].add(ch)
        elif ch.isupper():
            seen["upper"].add(ch)
        elif not ch.isspace() and not ch.isdigit() and ch not in "+-.":
            seen["other"].add(ch)
    full = {"lower": "abcdefghijklmnopqrstuvwxyz", "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    return {k: sorted(v) if len(v) <= 4 or k == "other" else list(full[k]) for k, v in seen.items()}


def new_chars(tok, cls, rate, rng):
    out = []
    for ch in tok:
        kind = "lower" if ch.islower() else "upper" if ch.isupper() else "digit" if ch.isdigit() else "other"
        pool = list("0123456789") if kind == "digit" else cls.get(kind, [])
        if kind == "digit" and set(tok) <= {"0", "1"}:
            pool = ["0", "1"]
        out.append(rng.choice(pool) if pool and rng.random() < rate else ch)
    return "".join(out)


def mutate(text, cls, rng):
    rate = rng.choice([0.1, 0.25, 0.5])
    is_int, fixed, span = plan(text)
    lines = text.split("\n")
    for _ in range(3):  # until something changes
        out_lines = []
        for i, line in enumerate(lines):
            new, j = [], 0
            for p in TOK.split(line):
                if not p or p.isspace():
                    new.append(p)
                    continue
                if (i, j) in fixed or rng.random() >= rate:
                    new.append(p)
                elif is_int(p):
                    new.append(str(new_int(int(p), span.get((i, j)), rng)))
                elif FLOAT.fullmatch(p):
                    d = len(p.split(".")[1])
                    new.append(f"{float(p) * rng.uniform(0.5, 1.5):.{d}f}")
                else:
                    new.append(new_chars(p, cls, max(rate, 0.25), rng))
                j += 1
            out_lines.append("".join(new))
        out = "\n".join(out_lines)
        if out != text:
            return out
    return None


def make(prob):
    rng = random.Random(prob["id"])
    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o777)
        kept = [s for s in prob["solutions"][:SOLUTIONS] if passes(s, prob["tests"], tmp)]
        res = {"id": prob["id"], "solutions_tried": min(len(prob["solutions"]), SOLUTIONS), "solutions_kept": len(kept),
               "tests": [], "tries": 0}
        if len(kept) >= 2:
            judges, held = kept[:AGREE], kept[AGREE:]
            seen = {t["input"] for t in prob["tests"]}
            cls = classes("".join(t["input"] for t in prob["tests"]))
            while len(res["tests"]) < WANT and res["tries"] < TRIES:
                res["tries"] += 1
                src = rng.choice(prob["tests"])["input"]
                inp = mutate(src, cls, rng)
                if inp is None or inp in seen:
                    continue
                seen.add(inp)
                outs = []
                for s in judges:
                    o = run(s, inp, tmp)
                    if o is None or (outs and o.split() != outs[0].split()):
                        outs = None
                        break
                    outs.append(o)
                if outs:
                    res["tests"].append({"input": inp, "output": outs[0]})
            res["judges"] = len(judges)
            shown = {" ".join(t["output"].split()) for t in prob["tests"]}
            res["differs_from_examples"] = sum(" ".join(t["output"].split()) not in shown for t in res["tests"])
            res["distinct_outputs"] = len({" ".join(t["output"].split()) for t in res["tests"]})
            if res["tests"]:
                res["held_out"] = len(held)
                res["held_out_pass"] = sum(passes(s, res["tests"], tmp) for s in held)
                res["answers"] = {}
                for a in prob.get("answers", []):
                    c = res["answers"].setdefault(a["check"], [0, 0])
                    c[0] += 1
                    c[1] += passes(a["code"], res["tests"], tmp)
    return res


def main():
    probs = [json.loads(l) for l in open(sys.argv[1])]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as ex:
        for r in ex.map(make, probs):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
