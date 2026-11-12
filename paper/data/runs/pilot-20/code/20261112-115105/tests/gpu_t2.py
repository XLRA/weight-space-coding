"""Follow-up to gpu_t1: is the HF copy deterministic, and do HF and vLLM agree under a mild edit?"""
import json
import sys

sys.argv = sys.argv[:2]
exec(open("/opt/wsp/harness/tests/gpu_t1.py").read().split("kernel(\"print(edited())\")")[0])  # helpers only
k = max(int(f[4:-12]) for f in __import__("os").listdir(REQ) if f.endswith(".result.json"))

a, _ = submit({"type": "teacher_forced", "model": "original", "texts": TEXTS, "return": {"token_logprobs": True}})
b, _ = submit({"type": "teacher_forced", "model": "original", "texts": TEXTS, "return": {"token_logprobs": True}})
print("HF original, two identical runs: mean |dlogprob| =", round(diff(hf_lp(a), hf_lp(b)), 6))

kernel("""
w = load_original("model.language_model.layers.15.mlp.down_proj.weight").clone(); w[:, :64] = 0
print(save_edit("model.language_model.layers.15.mlp.down_proj.weight", w))
""")
_, v_o = submit({"type": "debug_vllm_logprobs", "model": "original", "texts": TEXTS})
_, v_e = submit({"type": "debug_vllm_logprobs", "model": "edited", "texts": TEXTS})
c, _ = submit({"type": "teacher_forced", "model": "edited", "texts": TEXTS, "return": {"token_logprobs": True}})
print("mild edit, edited vs original (vLLM): ", round(diff(vl_lp(v_e), vl_lp(v_o)), 4))
print("mild edit, HF vs vLLM (edited):       ", round(diff(hf_lp(c), vl_lp(v_e)), 4))
probs = [json.loads(l)["id"] for l in open("/opt/wsp/practice/final/mbpp_practice.jsonl")][:8]
g, res = submit({"type": "generate", "model": "both", "problems": probs, "n": 1, "max_tokens": 512})
o = [json.loads(l) for l in open(f"{RUN}/work/gen_{g}_original.raw.jsonl")]
e = [json.loads(l) for l in open(f"{RUN}/work/gen_{g}_edited.raw.jsonl")]
print("mild edit, same seed: first 60 chars equal in", sum(x["text"][:60] == y["text"][:60] for x, y in zip(o, e)), "of", len(o))
kernel("for n in edited(): drop_edit(n)\nprint(edited())")
