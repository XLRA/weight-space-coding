"""Grade one generation file of a held-out suite: python3 grade_suites.py <suite> /work/gen_<suite>_<model>.raw.jsonl

Runs in the grading container (wsp-grade:1: no network, read-only, uid 65534) with /suites = final/suites,
/harness (grade.py) and /evalbin (this file, suites.py, ifeval/) on PYTHONPATH. Prints one graded row per
answer: id, sample, source, difficulty, n_tokens, cut, loop, status, and pred / inst_ok, inst_total / exc.
"""
import ast
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache

import grade
import suites as S


@lru_cache(maxsize=8)
def lcb_tests(path):
    return json.load(open(f"/suites/{path}"))


def run_function(code, setup, tests, timeout):
    """grade.run_function with a time limit for the whole script (LiveCodeBench: per test, plus slack)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o777)
        path = os.path.join(tmp, "case.json")
        with open(path, "w") as f:
            json.dump([code, setup, tests], f)
        try:
            r = subprocess.run([sys.executable, "-c", grade.FUNC_RUNNER, path], capture_output=True, text=True,
                               timeout=timeout, cwd=tmp, preexec_fn=grade.limits, env=grade.ENV)
            last = r.stdout.strip().splitlines()[-1:] or ['{"status": "runtime_error", "exc": "NoResult"}']
            return json.loads(last[0])
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except (json.JSONDecodeError, IndexError):
            return {"status": "runtime_error", "exc": "BadOutput"}


def grade_lcb(prob, row):
    if row["finish_reason"] == "length":
        return {"status": "cut_at_max_tokens"}
    code = grade.extract(row["text"])
    if code is None:
        return {"status": "no_code"}
    code = S.LCB_PRELUDE + code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"status": "syntax_error"}
    t = lcb_tests(prob["tests_file"])
    if prob["test_format"] == "stdio":
        with tempfile.TemporaryDirectory() as tmp:
            os.chmod(tmp, 0o777)
            return grade.run_stdio(code + "\n", {"tests": t["cases"]}, tmp)
    if "Solution" not in grade.defined_names(tree):
        return {"status": "wrong_interface"}
    tests = S.lcb_function_tests(prob["func_name"], t["cases"])
    return run_function(grade.sanitize(tree), "import json\nimport sys\nsys.setrecursionlimit(50000)\n", tests,
                        S.LCB_TEST_TIMEOUT * len(tests) + 5)


GRADERS = {"lcb": grade_lcb, "ifeval": S.grade_ifeval, "mmlupro": S.grade_choice, "math500": S.grade_math}
SUITE, PROBS = None, {}


def one(r):
    p = PROBS[r["id"]]
    g = GRADERS[SUITE](p, r)
    return {"id": r["id"], "sample": r["sample"], "source": p["source"], "difficulty": p["difficulty"],
            "n_tokens": r["n_tokens"], "cut": r["finish_reason"] == "length", "loop": grade.looped(r["text"]), **g}


def main():
    global SUITE, PROBS
    SUITE, path = sys.argv[1], sys.argv[2]
    PROBS = {p["id"]: p for p in map(json.loads, open(f"/suites/{SUITE}.jsonl"))}
    rows = [json.loads(l) for l in open(path)]
    if SUITE == "lcb":  # threads only wait on the solutions' processes
        rows.sort(key=lambda r: r["id"])  # answers to one problem share its cached tests
        pool = ThreadPoolExecutor(12)
    else:
        # math-verify's timeouts use signal.alarm, which needs each worker's main thread; and the IFEval
        # checkers are not thread-safe (langdetect loads its language profiles lazily into a global).
        pool = ProcessPoolExecutor(12)
    with pool:
        for o in pool.map(one, rows, chunksize=8):
            print(json.dumps(o))


if __name__ == "__main__":
    main()
