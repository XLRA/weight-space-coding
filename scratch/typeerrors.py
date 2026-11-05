"""The messages behind the code suite's TypeErrors (combined edit). Run in wsp-runtime:1, no network:
python3 typeerrors.py <problems.jsonl> <gen_edited.jsonl>. Each answer runs in a child process with a timeout."""
import collections
import json
import multiprocessing as mp
import re
import sys

probs = {p["id"]: p for p in map(json.loads, open(sys.argv[1]))}
rows = [r for r in map(json.loads, open(sys.argv[2])) if r.get("exc") == "TypeError"]


def first_error(args):
    code, setup, tests, q = args
    try:
        g = {}
        exec(setup or "", g)
        exec(code, g)
        for t in tests if isinstance(tests, list) else [tests]:
            try:
                exec(t if isinstance(t, str) else json.dumps(t), g)
            except TypeError as e:
                q.put(f"TypeError: {e}")
                return
            except Exception:
                continue
        q.put("no TypeError reproduced")
    except TypeError as e:
        q.put(f"TypeError at definition: {e}")
    except Exception as e:  # noqa: BLE001
        q.put(f"{type(e).__name__} before tests")


def run(r):
    p = probs[r["id"]]
    q = mp.Queue()
    pr = mp.Process(target=first_error, args=((r["code"], p.get("test_setup"), p["tests"], q),))
    pr.start()
    pr.join(10)
    if pr.is_alive():
        pr.kill()
        return "timeout"
    return q.get() if not q.empty() else "no result"


p0 = probs[rows[0]["id"]]
print("tests field:", type(p0["tests"]).__name__, repr(p0["tests"])[:300])
kinds = collections.Counter()
examples = {}
for r in rows:
    msg = run(r)
    k = re.sub(r"'[^']*'", "'X'", msg)
    k = re.sub(r"\d+", "N", k)
    kinds[k] += 1
    examples.setdefault(k, (r["id"], msg))
for k, n in kinds.most_common(15):
    print(f"{n:4d}  {k[:110]}   e.g. {examples[k][0]}")
