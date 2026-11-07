"""Generation for one checkpoint check (plans/training-phase.md, "Choosing the checkpoint"). Run by train/run-ckpt.sh.

    python3 /train/ckpt_eval.py <name> </ckpt | original> <the folder's host path, for the record>

In wsp-train:1 (no network), vLLM loads the checkpoint with the harness's own swap (wsp_ext.wsp_set_weights: each
.npy cast to bf16) and answers exactly as the harness's GPU server does (harness/gpu_server.py: the problem as the
user turn with thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, at most 32,768 new tokens,
seed sha256(id#sample)), in one batch:
- the panel's 1,000 questions, once each;
- the 252 validation problems (train/data/val.jsonl), 4 times each.
Writes /train/evals/<name>/panel.raw.jsonl and val.raw.jsonl.gz (rows as gpu_server.py writes them) and gen.json.
The answers are graded afterwards, without the GPU (run-ckpt.sh).
"""
import glob
import gzip
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, "/train")
import common  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

NAME, FOLDER, HOST_PATH = sys.argv[1:4]
OUT = f"/train/evals/{NAME}"
PANEL = "/panel/panel.jsonl"
VAL = "/train/data/val.jsonl"
MAX_NEW = 32768  # harness/driver.py PANEL_MAX_TOKENS, and the evaluation's limit
VAL_SAMPLES = 4
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)


def seed_for(key, sample):  # harness/gpu_server.py
    return int(hashlib.sha256(f"{key}#{sample}".encode()).hexdigest()[:8], 16)


def main():
    os.makedirs(OUT, exist_ok=True)
    panel = [json.loads(line) for line in open(PANEL)]
    val = common.load_problems(VAL)
    items = [(p["id"], p["problem"], 1) for p in panel] + [(p["id"], p["problem"], VAL_SAMPLES) for p in val]
    llm = LLM(model=common.MODEL, language_model_only=True, max_model_len=MAX_NEW + 4096,
              gpu_memory_utilization=float(os.environ.get("WSP_GPU_MEMORY", "0.65")), max_num_seqs=256,
              enable_prefix_caching=False, seed=0, worker_extension_cls="wsp_ext.WspWorkerExtension")
    n_tensors = 0
    if FOLDER != "original":
        pairs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(f"{FOLDER}/*.npy"))]
        assert pairs, f"no .npy files in {HOST_PATH}"
        n_tensors = llm.collective_rpc("wsp_set_weights", args=(pairs,))[0]
    prompts, params, meta = [], [], []
    for key, msg, n in items:
        text = common.render(msg)
        for s in range(n):
            prompts.append(text)
            params.append(SamplingParams(max_tokens=MAX_NEW, seed=seed_for(key, s), **SAMPLING))
            meta.append((key, s))
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.time() - t0
    rows = []
    for (key, s), o in zip(meta, outs):
        c = o.outputs[0]
        rows.append({"id": key, "sample": s, "text": c.text, "n_tokens": len(c.token_ids),
                     "finish_reason": c.finish_reason})
    with open(f"{OUT}/panel.raw.jsonl.tmp", "w") as f:
        for r in rows[:len(panel)]:
            f.write(json.dumps(r) + "\n")
    with gzip.open(f"{OUT}/val.raw.jsonl.gz.tmp", "wt") as f:
        for r in rows[len(panel):]:
            f.write(json.dumps(r) + "\n")
    os.replace(f"{OUT}/val.raw.jsonl.gz.tmp", f"{OUT}/val.raw.jsonl.gz")
    os.replace(f"{OUT}/panel.raw.jsonl.tmp", f"{OUT}/panel.raw.jsonl")
    tokens = sum(r["n_tokens"] for r in rows)
    json.dump({"name": NAME, "weights": HOST_PATH, "tensors_loaded": n_tensors, "panel_questions": len(panel),
               "val_problems": len(val), "val_samples": VAL_SAMPLES, "sampling": SAMPLING, "max_new": MAX_NEW,
               "answers": len(rows), "tokens": tokens, "generate_seconds": round(seconds),
               "tokens_per_second": round(tokens / seconds), "gpu_memory": os.environ.get("WSP_GPU_MEMORY", "0.65"),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
              open(f"{OUT}/gen.json", "w"), indent=1)
    print(f"{NAME}: {len(rows)} answers, {tokens} tokens in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
