"""Part 0 baseline: the unedited model answers every problem in /work/problems.jsonl once.

Runs inside the vLLM container with the model at /model and the baseline folder at /work.
Protocol: chat template with thinking enabled; the model card's sampling for precise coding
(temperature 0.6, top_p 0.95, top_k 20, min_p 0, presence penalty 0); at most 32,768 new tokens;
a fixed seed per problem. Results are appended to /work/gen.jsonl chunk by chunk, so a restart
skips problems already answered.
"""
import hashlib
import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "/model"
MAX_NEW = 32768
CHUNK = 258
OUT = "/work/gen.jsonl"


def seed_for(pid):
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)


def main():
    probs = [json.loads(l) for l in open("/work/problems.jsonl")]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT)}
    todo = [p for p in probs if p["id"] not in done]
    print(f"{len(probs)} problems, {len(done)} done, {len(todo)} to go", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    render = lambda p: tok.apply_chat_template(
        [{"role": "user", "content": p["problem"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True)
    with open("/work/eval_prompt_format.txt", "w") as f:
        f.write(render(probs[0]))

    common = dict(model=MODEL, max_model_len=MAX_NEW + 4096, gpu_memory_utilization=0.90,
                  max_num_seqs=256, seed=0)
    try:
        llm = LLM(language_model_only=True, **common)
    except TypeError:
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **common)

    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
                                 max_tokens=MAX_NEW, seed=seed_for(p["id"])) for p in chunk]
        t0 = time.time()
        outs = llm.generate([render(p) for p in chunk], params, use_tqdm=False)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        with open(OUT, "a") as f:
            for p, o in zip(chunk, outs):
                c = o.outputs[0]
                f.write(json.dumps({"id": p["id"], "text": c.text, "n_tokens": len(c.token_ids),
                                    "finish_reason": c.finish_reason}) + "\n")
        print(f"chunk {i // CHUNK + 1}: {len(chunk)} answers, {ntok} tokens in {dt:.0f}s "
              f"({ntok / dt:.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
