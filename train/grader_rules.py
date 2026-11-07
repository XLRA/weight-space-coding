"""What an answer format costs under the code graders' reading (2026-11-07, on the user's go-ahead). Run in
wsp-train:1 with no network, like ckpt_grade.py's grading:

    python3 /train/grader_rules.py <check name>...     (prints a line per check; writes /train/checks/grader-rules.json)

Every code grader here takes the last ```python block after the FIRST </think> (harness/grade.py, extract): the
practice grader, which is the training reward and grades the checks' validation answers, and the held-out code
suites (MBPP+ and HumanEval+ through harness/grade_run.py, LiveCodeBench through eval/grade_suites.py). The held-out
MMLU-Pro, MATH-500 and IFEval read the text after the LAST </think> (eval/suites.py, after_think).

For each check's validation answers, the answers whose code the two readings extract differently are graded both
ways, with the same tests and sandbox: as the checks read them, and with the code read after the last </think>. All
other answers keep the status in the check's val.jsonl, since their code is the same either way. The regrade of the
first reading is compared with val.jsonl, as a test that the grading reproduces.
"""
import gzip
import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/train")  # grade is harness/grade.py, on PYTHONPATH (/harness)
import common  # noqa: E402
import grade  # noqa: E402

THREADS = int(os.environ.get("THREADS", "6"))


def last_reading(text):
    """The text cut so that grade.extract reads the code after the last </think>."""
    return "</think>" + text.split("</think>")[-1] if "</think>" in text else text


def pct(k, n):
    return round(100 * k / n, 2)


def paired(a, b):
    """Mean and SE (points) of a - b over paired values in [0, 1] (as in ckpt_grade.py)."""
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return round(100 * m, 2), round(100 * sd / math.sqrt(len(d)), 2)


def one(name, probs):
    d = f"/train/evals/{name}"
    raw = [json.loads(line) for line in gzip.open(f"{d}/val.raw.jsonl.gz", "rt")]
    checked = [json.loads(line) for line in open(f"{d}/val.jsonl")]
    assert len(raw) == len(checked) and all(r["id"] == c["id"] and r["sample"] == c["sample"]
                                            for r, c in zip(raw, checked))
    todo = [i for i, r in enumerate(raw) if r["finish_reason"] != "length"
            and grade.extract(r["text"]) != grade.extract(last_reading(r["text"]))]

    def both(i):
        r, p = raw[i], probs[raw[i]["id"]]
        return grade.grade(p, r)["status"], grade.grade(p, {**r, "text": last_reading(r["text"])})["status"]

    with ThreadPoolExecutor(THREADS) as ex:
        regraded = dict(zip(todo, ex.map(both, todo)))
    first = [c["status"] for c in checked]
    last = [regraded[i][1] if i in regraded else s for i, s in enumerate(first)]
    n = len(raw)
    out = {"answers": n, "finished": sum(r["finish_reason"] != "length" for r in raw),
           "more_than_one_think_end": sum(r["text"].count("</think>") > 1 for r in raw),
           "read_differently": len(todo),
           "regrade_reproduces_check": sum(regraded[i][0] == first[i] for i in todo),
           "pass": {"first_think_end": pct(first.count("pass"), n), "last_think_end": pct(last.count("pass"), n),
                    "either": pct(sum(a == "pass" or b == "pass" for a, b in zip(first, last)), n)},
           "by_source": {}, "status": {"first_think_end": dict(Counter(first)), "last_think_end": dict(Counter(last))},
           "changes": {f"{first[i]} -> {regraded[i][1]}": 0 for i in todo}}
    for i in todo:
        out["changes"][f"{first[i]} -> {regraded[i][1]}"] += 1
    out["changes"] = dict(sorted(out["changes"].items(), key=lambda kv: -kv[1]))
    for src in sorted({r["source"] for r in checked}):
        idx = [i for i, c in enumerate(checked) if c["source"] == src]
        out["by_source"][src] = {"first_think_end": pct(sum(first[i] == "pass" for i in idx), len(idx)),
                                 "last_think_end": pct(sum(last[i] == "pass" for i in idx), len(idx))}
    # paired over problems: each problem's pass rate over its answers, last reading minus first
    ids = sorted({c["id"] for c in checked})
    rate = lambda st, pid: (sum(st[i] == "pass" for i, c in enumerate(checked) if c["id"] == pid)
                            / sum(c["id"] == pid for c in checked))
    m, se = paired([rate(last, p) for p in ids], [rate(first, p) for p in ids])
    out["last_minus_first"] = {"diff": m, "se": se, "problems": len(ids)}
    return out


def main():
    probs = {p["id"]: p for p in common.load_problems("/train/data/val.jsonl")}
    path = "/train/checks/grader-rules.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name in sys.argv[1:]:
        o = one(name, probs)
        res[name] = o
        print(f"{name}: {o['read_differently']} of {o['finished']} finished answers read differently "
              f"(regrade reproduces the check on {o['regrade_reproduces_check']}); pass {o['pass']['first_think_end']}% "
              f"as checked, {o['pass']['last_think_end']}% after the last </think>, {o['pass']['either']}% either; "
              f"paired {o['last_minus_first']['diff']:+} ({o['last_minus_first']['se']})", flush=True)
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
