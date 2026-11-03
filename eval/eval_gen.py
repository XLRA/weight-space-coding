"""Held-out generation for evaluate.py (GPU container, no network).

    python3 eval_gen.py <models, e.g. original,edited> '<{"suite": samples, ...} as JSON>' [first N problems per set]

Suites run in the order given, each for every model before the next suite starts, so a stopped
evaluation leaves whole suites with both models. "code" reads /eval/problems.jsonl (MBPP+ and
HumanEval+) and writes /out/gen_<model>.raw.jsonl; another suite reads /eval/suites/<suite>.jsonl
and writes /out/gen_<suite>_<model>.raw.jsonl. The editing runs' protocol: chat template with
thinking, temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence penalty $WSP_PRESENCE (0 by
default, the coding setting the agent works with), at most 32,768 new tokens, seed fixed by problem
and sample. The edited model is the original checkpoint with /submission/<name>.npy swapped in, cast
to bfloat16. Answers are appended chunk by chunk, so a restart skips what is done.
"""
import glob
import hashlib
import json
import os
import sys
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MAX_NEW, CHUNK = 32768, 512
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                presence_penalty=float(os.environ.get("WSP_PRESENCE", "0")))


def seed_for(key, sample):
    return int(hashlib.sha256(f"{key}#{sample}".encode()).hexdigest()[:8], 16)


def load(suite, limit):
    path = "/eval/problems.jsonl" if suite == "code" else f"/eval/suites/{suite}.jsonl"
    probs, per_set = [], {}
    for p in map(json.loads, open(path)):
        per_set[p["source"]] = per_set.get(p["source"], 0) + 1
        if not limit or per_set[p["source"]] <= limit:
            probs.append({"id": p["id"], "problem": p["problem"]})
    return probs


def main():
    models, spec = sys.argv[1].split(","), json.loads(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tok = AutoTokenizer.from_pretrained("/model")
    llm = LLM(model="/model", language_model_only=True, max_model_len=MAX_NEW + 4096, gpu_memory_utilization=0.9,
              max_num_seqs=256, enable_prefix_caching=False, seed=0, worker_extension_cls="wsp_ext.WspWorkerExtension")
    edits = sorted(glob.glob("/submission/*.npy"))
    print(f"suites {spec}; {len(edits)} edited tensors; sampling {SAMPLING}", flush=True)
    loaded = None
    for suite, n in spec.items():
        probs = load(suite, limit)
        for model in models:
            if edits and model != loaded:
                llm.collective_rpc("wsp_set_weights", args=([(os.path.basename(p)[:-4], p if model == "edited" else None)
                                                             for p in edits],))
                llm.reset_prefix_cache()
                loaded = model
            out = f"/out/gen_{model}.raw.jsonl" if suite == "code" else f"/out/gen_{suite}_{model}.raw.jsonl"
            done = set()
            if os.path.exists(out):
                done = {(r["id"], r["sample"]) for r in map(json.loads, open(out))}
            todo = [(p, s) for p in probs for s in range(n) if (p["id"], s) not in done]
            print(f"{suite} / {model}: {len(probs)} problems x {n}; {len(done)} done, {len(todo)} to go", flush=True)
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i:i + CHUNK]
                prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"]}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True) for p, _ in chunk]
                params = [SamplingParams(max_tokens=MAX_NEW, seed=seed_for(p["id"], s), **SAMPLING) for p, s in chunk]
                t0 = time.time()
                outs = llm.generate(prompts, params, use_tqdm=False)
                with open(out, "a") as f:
                    for (p, s), o in zip(chunk, outs):
                        c = o.outputs[0]
                        f.write(json.dumps({"id": p["id"], "sample": s, "text": c.text, "n_tokens": len(c.token_ids),
                                            "finish_reason": c.finish_reason}) + "\n")
                ntok = sum(len(o.outputs[0].token_ids) for o in outs)
                print(f"{suite} / {model}: {i + len(chunk)}/{len(todo)} in {time.time() - t0:.0f} s "
                      f"({ntok / (time.time() - t0):.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
