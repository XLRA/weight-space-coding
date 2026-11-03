"""Grade generated answers by running their code against the problems' tests.

Runs in a container with no network, as an unprivileged user, on a read-only filesystem.
Usage: python3 grade.py problems.jsonl gen.jsonl > graded.jsonl   (summary on stderr)

Categories, assigned in this order:
  cut_at_max_tokens  the answer hit the token limit
  no_code            no ```python block after </think>
  syntax_error       the extracted code does not parse
  wrong_interface    function problem whose entry point is not defined
  timeout / runtime_error / wrong_answer / pass   from running the tests
Every answer also gets a loop flag: its last 200 characters occur 3+ times in its last 20,000.

For function problems the code is sanitized first, as EvalPlus does: only imports, function and
class definitions and plain assignments are kept, so the model's own example calls and asserts
are not run. Stdio programs run as written.
"""
import ast
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

FENCE = re.compile(r"```[ \t]*(?:python3?|py|Python)?[ \t]*\n(.*?)```", re.S)
FUNC_TIMEOUT, STDIO_TIMEOUT = 10, 6

FUNC_RUNNER = r'''
import json, sys, traceback
src, setup, tests = json.load(open(sys.argv[1]))
g = {"__name__": "__main__"}
try:
    exec(compile(src, "solution.py", "exec"), g)
    exec(setup, g)
except BaseException as e:
    print(json.dumps({"status": "runtime_error", "exc": type(e).__name__})); sys.exit()
for t in tests:
    try:
        exec(t, g)
    except AssertionError:
        print(json.dumps({"status": "wrong_answer"})); sys.exit()
    except BaseException as e:
        print(json.dumps({"status": "runtime_error", "exc": type(e).__name__})); sys.exit()
print(json.dumps({"status": "pass"}))
'''


KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.setsid()
    if os.getuid() == 0:  # inside the kernel container: run the code as nobody
        os.setgid(65534)
        os.setuid(65534)


def sanitize(tree):
    tree.body = [n for n in tree.body if isinstance(n, KEEP)]
    return ast.unparse(tree)


def looped(text):
    tail = text[-200:]
    return len(tail) == 200 and text[-20000:].count(tail) >= 3


def extract(text):
    if "</think>" not in text:
        return None
    blocks = FENCE.findall(text.split("</think>", 1)[1])
    return blocks[-1] if blocks else None


def defined_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def run_function(code, prob, tmp):
    path = os.path.join(tmp, "case.json")
    with open(path, "w") as f:
        json.dump([code, prob["test_setup"], prob["tests"]], f)
    try:
        r = subprocess.run([sys.executable, "-c", FUNC_RUNNER, path], capture_output=True, text=True,
                           timeout=FUNC_TIMEOUT, cwd=tmp, preexec_fn=limits)
        last = r.stdout.strip().splitlines()[-1:] or ['{"status": "runtime_error", "exc": "NoResult"}']
        return json.loads(last[0])
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except (json.JSONDecodeError, IndexError):
        return {"status": "runtime_error", "exc": "BadOutput"}


def run_stdio(code, prob, tmp):
    path = os.path.join(tmp, "solution.py")
    with open(path, "w") as f:
        f.write(code)
    for t in prob["tests"]:
        try:
            r = subprocess.run([sys.executable, path], input=t["input"], capture_output=True, text=True,
                               timeout=STDIO_TIMEOUT, cwd=tmp, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if r.returncode != 0:
            exc = (r.stderr.strip().splitlines() or ["Error"])[-1].split(":")[0][:40]
            return {"status": "runtime_error", "exc": exc}
        if r.stdout.split() != t["output"].split():
            return {"status": "wrong_answer"}
    return {"status": "pass"}


def grade_code(prob, code):
    """Run one piece of code against a problem's tests: {"status": ..., "exc"?: ...}."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"status": "syntax_error"}
    if prob["test_format"] == "function":
        if prob["entry_point"] and prob["entry_point"] not in defined_names(tree):
            return {"status": "wrong_interface"}
        code = sanitize(tree)
    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o777)
        return (run_function if prob["test_format"] == "function" else run_stdio)(code, prob, tmp)


def grade(prob, gen):
    out = {"id": gen["id"], "source": prob["source"], "difficulty": prob["difficulty"],
           "n_tokens": gen["n_tokens"], "loop": looped(gen["text"])}
    if gen["finish_reason"] == "length":
        return {**out, "status": "cut_at_max_tokens"}
    code = extract(gen["text"])
    if code is None:
        return {**out, "status": "no_code"}
    return {**out, **grade_code(prob, code)}


def main():
    probs = {p["id"]: p for p in map(json.loads, open(sys.argv[1]))}
    gens = [json.loads(l) for l in open(sys.argv[2])]
    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(lambda g: grade(probs[g["id"]], g), gens))
    for g, r in zip(gens, results):
        if "sample" in g:
            r["sample"] = g["sample"]
        print(json.dumps(r))

    def summarize(rows, label):
        n = len(rows)
        c = Counter(r["status"] for r in rows)
        loops = sum(r["loop"] for r in rows)
        mean_tok = sum(r["n_tokens"] for r in rows) / max(n, 1)
        print(f"\n== {label}: n={n}  pass={100 * c['pass'] / n:.1f}%  loop-flagged={100 * loops / n:.1f}%  "
              f"mean tokens={mean_tok:.0f}", file=sys.stderr)
        for k, v in c.most_common():
            print(f"   {k:<18} {v:>4}  {100 * v / n:5.1f}%", file=sys.stderr)

    summarize(results, "all")
    for key in sorted({(r["source"], r["difficulty"]) for r in results}):
        summarize([r for r in results if (r["source"], r["difficulty"]) == key], "/".join(key))
    exc = Counter(r.get("exc") for r in results if r["status"] == "runtime_error")
    print(f"\nruntime error types: {dict(exc.most_common(8))}", file=sys.stderr)


if __name__ == "__main__":
    main()
