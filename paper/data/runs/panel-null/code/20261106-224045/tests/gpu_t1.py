"""GPU server self-test (host side). Usage: gpu_t1.py <run dir>

Checks that HF and vLLM agree on log-probs, that edits reach both copies of the model,
that restoring the original really restores it, and that generation + activations work.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

RUN = sys.argv[1]
REQ = f"{RUN}/requests"
TEXTS = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
    "print([fibonacci(i) for i in range(10)])\n",
    "Write a function to find the maximum of three numbers.\n\ndef max_of_three(a, b, c):\n    return max(a, b, c)\n",
]
k = 0


def submit(req):
    global k
    k += 1
    with open(f"{REQ}/fwd_{k}.json", "w") as f:
        json.dump(req, f)
    res_path = f"{REQ}/fwd_{k}.result.json"
    t0 = time.time()
    while not os.path.exists(res_path):
        time.sleep(1)
    res = json.load(open(res_path))
    print(f"fwd_{k} {req['type']}/{req.get('model')}: ok={res['ok']} {time.time() - t0:.0f}s"
          + ("" if res["ok"] else "\n" + res["error"] + "\n" + res.get("trace", "")[-1500:]))
    if not res["ok"]:
        sys.exit(1)
    return k, res


def kernel(code):
    r = subprocess.run(["/opt/wsp/venv/bin/python", "/opt/wsp/harness/kernel_client.py", f"{RUN}/ctl/kernel.sock", "300"],
                       input=code, capture_output=True, text=True)
    print("kernel:", r.stdout.strip()[-600:], r.stderr.strip()[-300:])


def hf_lp(kk):
    a = np.load(f"{RUN}/work/fwd_{kk}/token_logprobs.npy")
    meta = json.load(open(f"{RUN}/work/fwd_{kk}/meta.json"))
    return [a[i, 1:n] for i, n in enumerate(meta["lengths"])]


def vl_lp(res):
    return [np.array(x[1:], dtype=np.float32) for x in res["logprobs"]]


def diff(a, b):
    return float(np.mean([np.abs(x - y).mean() for x, y in zip(a, b)]))


kernel("print(edited())")
_, v_orig = submit({"type": "debug_vllm_logprobs", "model": "original", "texts": TEXTS})
k_tf, _ = submit({"type": "teacher_forced", "model": "original", "texts": TEXTS,
                  "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 16, 24], "mlp_act": [15],
                             "attn_layers": [15]}})
h_orig = hf_lp(k_tf)
print("HF vs vLLM, original weights: mean |dlogprob| =", round(diff(h_orig, vl_lp(v_orig)), 4))
for name in ["tokens", "token_logprobs", "topk_ids", "hidden_states", "mlp_act", "attn"]:
    a = np.load(f"{RUN}/work/fwd_{k_tf}/{name}.npy")
    print(f"   {name}: {a.shape} {a.dtype}")
attn = np.load(f"{RUN}/work/fwd_{k_tf}/attn.npy")
n0 = json.load(open(f"{RUN}/work/fwd_{k_tf}/meta.json"))["lengths"][0]
print("   attention rows sum to 1:", bool(np.allclose(attn[0, 0, :, :n0, :n0].astype(np.float32).sum(-1), 1, atol=2e-2)))

kernel("""
w = load_original("model.language_model.layers.15.mlp.down_proj.weight").clone(); w[:, :512] = 0
print(save_edit("model.language_model.layers.15.mlp.down_proj.weight", w))
q = load_original("model.language_model.layers.3.self_attn.q_proj.weight").clone(); q[:512] *= -1
print(save_edit("model.language_model.layers.3.self_attn.q_proj.weight", q))
z = load_original("model.language_model.layers.0.linear_attn.in_proj_qkv.weight").clone(); z[:256] = 0
print(save_edit("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", z))
""")
_, v_ed = submit({"type": "debug_vllm_logprobs", "model": "edited", "texts": TEXTS})
k_tf2, _ = submit({"type": "teacher_forced", "model": "edited", "texts": TEXTS, "return": {"token_logprobs": True}})
h_ed = hf_lp(k_tf2)
print("HF vs vLLM, edited weights:   mean |dlogprob| =", round(diff(h_ed, vl_lp(v_ed)), 4))
print("edited vs original (vLLM):    mean |dlogprob| =", round(diff(vl_lp(v_ed), vl_lp(v_orig)), 4))
_, v_back = submit({"type": "debug_vllm_logprobs", "model": "original", "texts": TEXTS})
k_tf3, _ = submit({"type": "teacher_forced", "model": "original", "texts": TEXTS, "return": {"token_logprobs": True}})
print("original restored (vLLM):     mean |dlogprob| =", round(diff(vl_lp(v_back), vl_lp(v_orig)), 5))
print("original restored (HF):       mean |dlogprob| =", round(diff(hf_lp(k_tf3), h_orig), 5))

probs = [json.loads(l)["id"] for l in open("/opt/wsp/practice/final/mbpp_practice.jsonl")][:6]
k_g, res = submit({"type": "generate", "model": "both", "problems": probs, "prompts": ["Say hello in Python."],
                   "n": 2, "max_tokens": 1024, "activations": {"layers": [0, 12, 24], "mlp": [15]}})
print(json.dumps(res)[:600])
for v in ["original", "edited"]:
    rows = [json.loads(l) for l in open(f"{RUN}/work/gen_{k_g}_{v}.raw.jsonl")]
    idx = json.load(open(f"{RUN}/work/gen_{k_g}_acts_{v}/index.json"))
    a = np.load(f"{RUN}/work/gen_{k_g}_acts_{v}/resid_mean.npy")
    print(f"   {v}: {len(rows)} answers, finish={[r['finish_reason'] for r in rows][:6]}, resid_mean {a.shape}, "
          f"samples in index {len(idx['samples'])}")
o = [json.loads(l) for l in open(f"{RUN}/work/gen_{k_g}_original.raw.jsonl")]
e = [json.loads(l) for l in open(f"{RUN}/work/gen_{k_g}_edited.raw.jsonl")]
print("   same seed, original vs edited first 80 chars equal:", sum(x["text"][:80] == y["text"][:80] for x, y in zip(o, e)), "of", len(o))
print("   answer excerpt:", o[0]["text"][:200].encode("ascii", "replace").decode().replace("\n", " | "))
kernel("for n in edited(): drop_edit(n)\nprint(edited())")
