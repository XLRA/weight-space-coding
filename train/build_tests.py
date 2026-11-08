"""Build train/data2: the practice set with tests the APPS statements do not show (2026-11-08, plans/training-phase.md,
"Deviations"). Runs on the host; train/make-tests.sh runs all three steps:

    /opt/wsp/venv/bin/python /opt/wsp/train/build_tests.py prep       # train/tests/work/in.jsonl
    (train/make_tests.py in the grading container)                    # train/tests/work/out.jsonl
    /opt/wsp/venv/bin/python /opt/wsp/train/build_tests.py assemble   # train/data2/{train,val}.jsonl, build.json

prep: every APPS problem in train/data (1,200 train, 200 val) with its tests and up to 16 of the human solutions
APPS ships for it (practice/raw/apps_train.jsonl, duplicates removed, in APPS's order).

assemble: each APPS problem keeps its tests and gets make_tests.py's new ones after them. A test is unseen when the
statement does not show both its input and its output (collapsing whitespace, as train/copy_check.py does). A problem
is usable when it has at least 3 unseen tests, with at least 2 different outputs among them, and an unseen output
that no shown test has: printing an example's output, or any one fixed output, then fails. (Rule fixed before any
check was regraded on the new tests.) MBPP problems are unchanged and usable: their prompt shows 1 of 3 tests.
- data2/train.jsonl: the usable training problems, GRPO's training set;
- data2/val.jsonl: all 252 validation problems with the new tests, each marked "usable". Honest validation is the
  pass rate over the usable ones (train/regrade_val.py, ckpt_grade.py);
- data2/build.json: the counts, the rule, and the md5s of the inputs and of make_tests.py.
"""
import hashlib
import json
import os
import sys
from collections import Counter

T = "/opt/wsp/train"
RAW = "/opt/wsp/practice/raw/apps_train.jsonl"
WORK = f"{T}/tests/work"
OUT = f"{T}/data2"
MIN_UNSEEN, MIN_OUTPUTS = 3, 2


def norm(s):
    return " ".join(s.split())


def shown(p, t):
    s = norm(p["problem"])
    return norm(t["input"]) in s and norm(t["output"]) in s


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def rows(path):
    return [json.loads(l) for l in open(path)]


def prep():
    os.makedirs(WORK, exist_ok=True)
    raw = {r["id"]: r for r in rows(RAW)}
    n = 0
    with open(f"{WORK}/in.jsonl", "w") as f:
        for split in ("train", "val"):
            for p in rows(f"{T}/data/{split}.jsonl"):
                if p["source"] != "apps":
                    continue
                sols = list(dict.fromkeys(json.loads(raw[int(p["id"].split("/")[1])]["solutions"] or "[]")))
                f.write(json.dumps({"id": p["id"], "split": split, "tests": p["tests"], "solutions": sols[:16]}) + "\n")
                n += 1
    print(f"{n} APPS problems -> {WORK}/in.jsonl")


def usable(p):
    unseen = [t for t in p["tests"] if not shown(p, t)]
    outs = {norm(t["output"]) for t in unseen}
    seen_outs = {norm(t["output"]) for t in p["tests"] if shown(p, t)}
    return len(unseen) >= MIN_UNSEEN and len(outs) >= MIN_OUTPUTS and bool(outs - seen_outs)


def assemble():
    made = {r["id"]: r for r in rows(f"{WORK}/out.jsonl")}
    os.makedirs(OUT, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        probs, c = [], Counter()
        for p in rows(f"{T}/data/{split}.jsonl"):
            c[p["source"]] += 1
            if p["source"] == "apps":
                m = made[p["id"]]
                p = {**p, "tests": p["tests"] + m["tests"], "tests_added": len(m["tests"])}
                p["usable"] = usable(p)
                c["apps_with_new_tests"] += bool(m["tests"])
                c["apps_usable"] += p["usable"]
                c["new_tests"] += len(m["tests"])
                c["held_out_solutions"] += m.get("held_out", 0) if m["tests"] else 0
                c["held_out_solutions_passing"] += m.get("held_out_pass", 0) if m["tests"] else 0
                c["problems_with_under_2_solutions_kept"] += m["solutions_kept"] < 2
            else:
                p = {**p, "tests_added": 0, "usable": True}
            probs.append(p)
        keep = [p for p in probs if p["usable"]] if split == "train" else probs
        with open(f"{OUT}/{split}.jsonl.tmp", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in keep)
        os.replace(f"{OUT}/{split}.jsonl.tmp", f"{OUT}/{split}.jsonl")
        stats[split] = {**dict(c), "written": len(keep), "usable": sum(p["usable"] for p in probs)}
    build = {"rule": f"usable: >= {MIN_UNSEEN} unseen tests, >= {MIN_OUTPUTS} different outputs among them, "
                     "and an unseen output that no shown test has",
             "stats": stats,
             "md5": {"make_tests.py": md5(f"{T}/make_tests.py"), "data/train.jsonl": md5(f"{T}/data/train.jsonl"),
                     "data/val.jsonl": md5(f"{T}/data/val.jsonl"), "practice/raw/apps_train.jsonl": md5(RAW),
                     "tests/work/out.jsonl": md5(f"{WORK}/out.jsonl")}}
    json.dump(build, open(f"{OUT}/build.json", "w"), indent=1)
    print(json.dumps(build, indent=1))


if __name__ == "__main__":
    {"prep": prep, "assemble": assemble}[sys.argv[1]]()
