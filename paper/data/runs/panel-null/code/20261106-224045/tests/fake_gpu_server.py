"""Stand-in for gpu_server.py with no GPU, for tests of the Colab relay and worker (colab_local_worker.py).

Same protocol: writes /requests/server.ready, then answers each /requests/fwd_<k>.json with fwd_<k>.result.json.
A generation writes gen_<k>[_<variant>].raw.jsonl rows that hold a wrong solution (n rows per problem, prompt and item);
a teacher-forced request writes small tokens.npy and token_logprobs.npy, with meta.json, in /work/fwd_<k>/. FAKE_SECONDS (default 20) is each request's time.
"""
import glob
import json
import os
import time

import numpy as np

REQ, WORK = "/requests", "/work"
WAIT = float(os.environ.get("FAKE_SECONDS", "20"))
TEXT = "<think>\nA stand-in answer.\n</think>\n\n```python\ndef solution(*args):\n    return None\n```"


def generation(k, req):
    keys = list(req.get("problems", [])) + [f"prompt/{i}" for i in range(len(req.get("prompts", [])))]
    keys += [key for key, _ in req.get("items", [])]  # the driver's own questions (the panel)
    variants = [req.get("baseline", "original"), "edited"] if req["model"] == "both" else [req["model"]]
    result = {"outputs": {}, "seconds": {}}
    for v in variants:
        time.sleep(WAIT / len(variants))
        path = f"{WORK}/gen_{k}{f'_{v}' if len(variants) > 1 else ''}.raw.jsonl"
        with open(path, "w") as f:
            for key in keys:
                for s in range(int(req.get("n", 1))):
                    row = {"id": key, "sample": s, "text": TEXT, "n_tokens": 20, "finish_reason": "stop"}
                    if key.startswith("prompt/"):
                        row["prompt"] = req["prompts"][int(key.split("/")[1])]
                    f.write(json.dumps(row) + "\n")
        result["outputs"][v] = path
        result["seconds"][v] = WAIT / len(variants)
    return result


def teacher_forced(k, req):
    time.sleep(WAIT)
    os.makedirs(f"{WORK}/fwd_{k}", exist_ok=True)
    folder = f"{WORK}/fwd_{k}"
    np.save(f"{folder}/tokens.npy", np.arange(12, dtype=np.int32).reshape(3, 4))
    np.save(f"{folder}/token_logprobs.npy", np.full((3, 4), -1.5, np.float32))
    meta = {"model": req.get("model", "original"), "lengths": [4, 4, 4],
            "files": {"tokens": "tokens.npy", "token_logprobs": "token_logprobs.npy"},
            "hidden_index": [], "mlp_layer": [], "attn_layer": [], "note": "fake"}
    json.dump(meta, open(f"{folder}/meta.json", "w"))
    return {"folder": folder, "fake": True}


def main():
    time.sleep(3)
    open(f"{REQ}/server.ready", "w").write(str(time.time()))
    print("fake gpu server ready", flush=True)
    while True:
        for path in sorted(glob.glob(f"{REQ}/fwd_*.json")):
            if path.endswith(".result.json") or os.path.exists(path[:-5] + ".result.json"):
                continue
            k = int(os.path.basename(path)[4:-5])
            t0 = time.time()
            req = json.load(open(path))
            try:
                res = {"generate": generation, "teacher_forced": teacher_forced}[req["type"]](k, req)
                res["ok"] = True
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            res["wall_seconds"] = round(time.time() - t0, 1)
            with open(path[:-5] + ".result.json.tmp", "w") as f:
                json.dump(res, f)
            os.replace(path[:-5] + ".result.json.tmp", path[:-5] + ".result.json")
            print(f"fwd_{k}: ok={res['ok']}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
