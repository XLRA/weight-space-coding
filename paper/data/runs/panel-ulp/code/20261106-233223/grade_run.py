"""Grade one generation pass for the driver: python3 grade_run.py /work/gen_<k>.raw.jsonl > graded.jsonl

Runs in the grading container (no network, read-only, uid 65534). Each output row is the raw row
plus loop, code (the extracted ```python block, or null), and for practice problems source,
difficulty, status and exc (exception type of a runtime error). Answers to the agent's own
prompts ("prompt/<i>") have no tests and get status "ungraded". $WSP_FUNC_TIMEOUT overrides the
time limit of a function problem (the held-out evaluation's test scripts are longer).
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import grade

grade.FUNC_TIMEOUT = int(os.environ.get("WSP_FUNC_TIMEOUT", grade.FUNC_TIMEOUT))


def main():
    probs = {}
    for fn in sorted(glob.glob("/practice/*.jsonl")):
        for line in open(fn):
            p = json.loads(line)
            probs[p["id"]] = p
    rows = [json.loads(l) for l in open(sys.argv[1])]

    def one(r):
        out = dict(r, loop=grade.looped(r["text"]), code=grade.extract(r["text"]))
        p = probs.get(r["id"])
        if p is None:
            return dict(out, status="ungraded")
        g = grade.grade(p, r)
        out.update(source=p["source"], difficulty=p["difficulty"], status=g["status"])
        if "exc" in g:
            out["exc"] = g["exc"]
        return out

    with ThreadPoolExecutor(12) as ex:
        for o in ex.map(one, rows):
            print(json.dumps(o))


if __name__ == "__main__":
    main()
