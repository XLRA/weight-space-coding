"""Grading and summary for one checkpoint check (plans/training-phase.md, "Choosing the checkpoint"). Run by
train/run-ckpt.sh in wsp-train:1 with no GPU and no network, after the panel has been graded.

    python3 /train/ckpt_grade.py <name> [<base name>]

1. Grades /train/evals/<name>/val.raw.jsonl.gz with the practice grader (harness/grade.py, the training reward's):
   val.jsonl, one row per answer without its text.
2. Summarises the panel (panel.jsonl, graded by harness/grade_panel.py) and the validation answers, and, given a base
   check (the arm's starting model), pairs them with it over the same questions: summary.json.
   - Panel: the pass rate per suite and their mean; paired differences per suite with standard errors over questions,
     and the mean's difference with SE sqrt(sum of the suites' SE^2) / 4.
   - Validation: the pass rate over all answers and by source; the paired difference over problems (each problem's
     pass rate over its 4 answers), with its SE.
   - panel_within_2_5: whether the panel mean is no more than 2.5 points below the base (the choice rule).
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

SUITES = ("choice", "math", "instructions", "code")


def grade_val(d):
    probs = {p["id"]: p for p in common.load_problems("/train/data/val.jsonl")}
    raw = [json.loads(line) for line in gzip.open(f"{d}/val.raw.jsonl.gz", "rt")]
    with ThreadPoolExecutor(12) as ex:
        graded = list(ex.map(lambda r: grade.grade(probs[r["id"]], r), raw))
    with open(f"{d}/val.jsonl.tmp", "w") as f:
        for r, g in zip(raw, graded):
            f.write(json.dumps({"id": r["id"], "sample": r["sample"], "source": g["source"],
                                "difficulty": g["difficulty"], "n_tokens": r["n_tokens"],
                                "finish_reason": r["finish_reason"], "loop": g["loop"], "status": g["status"]}) + "\n")
    os.replace(f"{d}/val.jsonl.tmp", f"{d}/val.jsonl")


def rows(path):
    return [json.loads(line) for line in open(path)]


def pct(v):
    return round(100 * v, 2)


def paired(a, b):
    """Mean and SE (points) of a - b over paired values in [0, 1]."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    return round(100 * m, 2), round(100 * sd / math.sqrt(n), 2)


def by_problem(val):
    """{problem id: pass rate over its answers}."""
    acc = {}
    for r in val:
        acc.setdefault(r["id"], []).append(r["status"] == "pass")
    return {k: sum(v) / len(v) for k, v in acc.items()}


def summarize(d, base):
    panel, val = rows(f"{d}/panel.jsonl"), rows(f"{d}/val.jsonl")
    out = {"name": os.path.relpath(d, "/train/evals"), "panel": {}, "val": {}}
    for s in SUITES:
        e = [r for r in panel if r["suite"] == s]
        out["panel"][s] = {"n": len(e), "pass": pct(sum(r["status"] == "pass" for r in e) / len(e)),
                           "cut": pct(sum(r["finish_reason"] == "length" for r in e) / len(e)),
                           "mean_tokens": round(sum(r["n_tokens"] for r in e) / len(e))}
    out["panel_mean"] = round(sum(out["panel"][s]["pass"] for s in SUITES) / len(SUITES), 2)
    src = Counter(r["source"] for r in val)
    out["val"] = {"answers": len(val), "pass": pct(sum(r["status"] == "pass" for r in val) / len(val)),
                  "by_source": {k: pct(sum(r["status"] == "pass" for r in val if r["source"] == k) / n)
                                for k, n in src.items()},
                  "problems_with_a_pass": sum(v > 0 for v in by_problem(val).values()),
                  "cut": pct(sum(r["finish_reason"] == "length" for r in val) / len(val)),
                  "mean_tokens": round(sum(r["n_tokens"] for r in val) / len(val)),
                  "status": dict(Counter(r["status"] for r in val))}
    if base:
        bp = {r["id"]: r for r in rows(f"{base}/panel.jsonl")}
        vs, ses = {}, []
        for s in SUITES:
            e = [r for r in panel if r["suite"] == s]
            m, se = paired([r["status"] == "pass" for r in e], [bp[r["id"]]["status"] == "pass" for r in e])
            vs[s] = {"diff": m, "se": se}
            ses.append(se)
        mean_diff = round(sum(vs[s]["diff"] for s in SUITES) / len(SUITES), 2)
        mean_se = round(math.sqrt(sum(x * x for x in ses)) / len(SUITES), 2)
        ve, vb = by_problem(val), by_problem(rows(f"{base}/val.jsonl"))
        ids = sorted(ve)
        vm, vse = paired([ve[i] for i in ids], [vb[i] for i in ids])
        out["vs_base"] = {"base": os.path.relpath(base, "/train/evals"), "panel": vs,
                          "panel_mean": {"diff": mean_diff, "se": mean_se},
                          "val": {"diff": vm, "se": vse}, "panel_within_2_5": mean_diff >= -2.5}
    json.dump(out, open(f"{d}/summary.json", "w"), indent=1)
    return out


def main():
    d = f"/train/evals/{sys.argv[1]}"
    base = f"/train/evals/{sys.argv[2]}" if len(sys.argv) > 2 else None
    if not os.path.exists(f"{d}/val.jsonl"):
        grade_val(d)
    out = summarize(d, base)
    p = ", ".join(f"{s} {out['panel'][s]['pass']}" for s in SUITES)
    line = f"{out['name']}: panel mean {out['panel_mean']} ({p}); validation {out['val']['pass']}"
    if base:
        v = out["vs_base"]
        line += (f"; against {v['base']}: panel mean {v['panel_mean']['diff']:+} (SE {v['panel_mean']['se']}), "
                 f"validation {v['val']['diff']:+} (SE {v['val']['se']})")
    print(line, flush=True)


if __name__ == "__main__":
    main()
