"""Forward-pass server for one editing run (GPU container, no network).

The driver validates each ```forward request and writes it to /requests/fwd_<k>.json; this server
runs it and writes /requests/fwd_<k>.result.json ({"ok": true, ...} or {"ok": false, "error": ...}).
Grading of generations is done afterwards by the driver, in the no-GPU sandbox.

Generation uses vLLM. Edited weights are swapped into the running engine in place (wsp_ext.py),
so "original", "edited" and "both" share one engine. Teacher-forced reads and generation
activations use a Hugging Face copy of the model kept in the same weight state.

Hidden-state index h: 0 = embeddings, h = l + 1 = output of decoder layer l (tensor names
model.language_model.layers.l.*), so h runs 0..24. MLP activations are the inputs of
layers.l.mlp.down_proj, indexed by decoder layer l = 0..23.
"""
import glob
import hashlib
import json
import os
import time
import traceback

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from vllm import LLM, SamplingParams

torch.set_grad_enabled(False)
MODEL, WORK, REQ = "/model", "/work", "/requests"
MAX_NEW, ACT_TOKENS, CHUNK = 32768, 8192, 1024
PREFIX = "model.language_model."
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)

tok = AutoTokenizer.from_pretrained(MODEL)
INDEX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
LM_NAMES = sorted(n for n in INDEX if n.startswith(PREFIX))
_files = {}


def original(name):
    fn = f"{MODEL}/{INDEX[name]}"
    if fn not in _files:
        _files[fn] = safe_open(fn, "pt")
    return _files[fn].get_tensor(name)


def current_edits():
    """{name: (path, fingerprint)} for the edit files in /work/edited right now."""
    out = {}
    for path in glob.glob(f"{WORK}/edited/*.npy"):
        st = os.stat(path)
        out[os.path.basename(path)[:-4]] = (path, (st.st_mtime_ns, st.st_size))
    return out


class WeightState:
    """Tracks which tensors of one model copy hold edits, and switches between variants."""

    def __init__(self, apply):
        self.apply, self.loaded = apply, {}  # name -> fingerprint of the edit currently in place

    def set(self, variant):
        target = current_edits() if variant == "edited" else {}
        pairs = []
        for name in set(self.loaded) | set(target):
            if name in target and self.loaded.get(name) == target[name][1]:
                continue
            pairs.append((name, target[name][0] if name in target else None))
        if pairs:
            self.apply(pairs)
        self.loaded = {n: fp for n, (_, fp) in target.items()}
        return len(pairs)


# Set by init(), which main() calls: building the engine at import time would re-run in vLLM's
# spawned worker processes.
llm = vllm_state = hf = hf_state = None
_eager = None
PROBLEMS = {}


def init():
    global llm, vllm_state, hf, hf_state
    llm = LLM(model=MODEL, language_model_only=True, max_model_len=MAX_NEW + 4096, gpu_memory_utilization=0.62,
              max_num_seqs=256, enable_prefix_caching=False, seed=0,
              worker_extension_cls="wsp_ext.WspWorkerExtension")

    def vllm_apply(pairs):
        llm.collective_rpc("wsp_set_weights", args=(pairs,))
        llm.reset_prefix_cache()

    vllm_state = WeightState(vllm_apply)
    hf, hf_state = build_hf("sdpa")
    for fn in sorted(glob.glob("/practice/*.jsonl")):
        for line in open(fn):
            p = json.loads(line)
            PROBLEMS[p["id"]] = p


# ---------- Hugging Face copy (teacher-forced reads, activations) ----------

def build_hf(attn_impl):
    cfg = AutoConfig.from_pretrained(MODEL).text_config
    cfg._attn_implementation = attn_impl
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Qwen3_5ForCausalLM(cfg)
    torch.set_default_dtype(torch.float32)
    sd = {"model." + n[len(PREFIX):]: original(n) for n in LM_NAMES}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    model.lm_head.weight = model.model.embed_tokens.weight  # tied
    model.eval().requires_grad_(False)
    params = dict(model.named_parameters())

    def apply(pairs):
        for name, path in pairs:
            t = original(name) if path is None else torch.from_numpy(np.load(path, allow_pickle=False))
            params["model." + name[len(PREFIX):]].data.copy_(t.to(torch.bfloat16))

    return model, WeightState(apply)


def eager_model():
    global _eager
    if _eager is None:
        _eager = build_hf("eager")
    return _eager


def layer_out(out):
    return out[0] if isinstance(out, tuple) else out


def hf_forward(model, ids, hidden=(), mlp=(), attn=()):
    """One text through the HF model. Returns final hidden states and the requested activations (GPU)."""
    acts, hooks = {}, []
    if 0 in hidden:
        hooks.append(model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: acts.__setitem__(("h", 0), o[0])))
    for h in hidden:
        if h >= 1:
            hooks.append(model.model.layers[h - 1].register_forward_hook(
                lambda m, i, o, h=h: acts.__setitem__(("h", h), layer_out(o)[0])))
    for l in mlp:
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, i, l=l: acts.__setitem__(("m", l), i[0][0])))
    try:
        with torch.no_grad():
            out = model.model(input_ids=ids[None], use_cache=False, output_attentions=bool(attn))
    finally:
        for hk in hooks:
            hk.remove()
    if attn:
        # Linear-attention layers have no attention matrix; the tuple may hold only full-attention layers.
        full = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
        per_layer = len(out.attentions) == model.config.num_hidden_layers
        for l in attn:
            acts[("a", l)] = out.attentions[l if per_layer else full.index(l)][0]
    return out.last_hidden_state[0], acts


def logprobs(model, final, ids, topk):
    """log p(token_t | tokens < t) for t >= 1 (position 0 is nan), and the top-k next-token
    distribution after each position."""
    n = len(ids)
    tlp = torch.full((n,), float("nan"))
    tk_ids = torch.zeros((n, topk), dtype=torch.int32) if topk else None
    tk_lp = torch.zeros((n, topk)) if topk else None
    W = model.lm_head.weight
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        lp = (final[s:e] @ W.T).float().log_softmax(-1)
        nxt = ids[s + 1:e + 1]
        if len(nxt):
            tlp[s + 1:s + 1 + len(nxt)] = lp[:len(nxt)].gather(1, nxt[:, None])[:, 0].cpu()
        if topk:
            v, i = lp.topk(topk, dim=-1)
            tk_ids[s:e], tk_lp[s:e] = i.int().cpu(), v.cpu()
    return tlp, tk_ids, tk_lp


# ---------- requests ----------

def render_user(msg):
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def seed_for(key, sample):
    return int(hashlib.sha256(f"{key}#{sample}".encode()).hexdigest()[:8], 16)


def generation(k, req):
    items = [(pid, PROBLEMS[pid]["problem"]) for pid in req.get("problems", [])]
    items += [(f"prompt/{i}", msg) for i, msg in enumerate(req.get("prompts", []))]
    n, max_tokens = int(req.get("n", 1)), int(req.get("max_tokens", MAX_NEW))
    variants = ["original", "edited"] if req["model"] == "both" else [req["model"]]
    result = {"outputs": {}, "seconds": {}}
    for v in variants:
        t0 = time.time()
        vllm_state.set(v)
        prompts, params, meta = [], [], []
        for key, msg in items:
            text = render_user(msg)
            for s in range(n):
                prompts.append(text)
                params.append(SamplingParams(max_tokens=max_tokens, seed=seed_for(key, s), **SAMPLING))
                meta.append((key, s, msg if key.startswith("prompt/") else None))
        outs = llm.generate(prompts, params, use_tqdm=False)
        suffix = f"_{v}" if len(variants) > 1 else ""
        path = f"{WORK}/gen_{k}{suffix}.raw.jsonl"
        with open(path, "w") as f:
            for (key, s, msg), o in zip(meta, outs):
                c = o.outputs[0]
                row = {"id": key, "sample": s, "text": c.text, "n_tokens": len(c.token_ids),
                       "finish_reason": c.finish_reason}
                if msg is not None:
                    row["prompt"] = msg
                f.write(json.dumps(row) + "\n")
        result["outputs"][v] = path
        if req.get("activations"):
            result.setdefault("activations", {})[v] = activations(k, suffix, v, outs, meta, req["activations"])
        result["seconds"][v] = round(time.time() - t0, 1)
    return result


def activations(k, suffix, variant, outs, meta, spec):
    hidden = sorted(set(int(h) for h in spec.get("layers", [])))
    mlp = sorted(set(int(l) for l in spec.get("mlp", []) or []))
    hf_state.set(variant)
    folder = f"{WORK}/gen_{k}_acts{suffix}"
    os.makedirs(folder, exist_ok=True)
    N, H, M = len(outs), len(hidden), len(mlp)
    arrs = {"resid_last": np.zeros((N, H, 1024), np.float16), "resid_mean": np.zeros((N, H, 1024), np.float16),
            "mlp_last": np.zeros((N, M, 3584), np.float16), "mlp_mean": np.zeros((N, M, 3584), np.float16)}
    for r, o in enumerate(outs):
        p_ids, a_ids = list(o.prompt_token_ids), list(o.outputs[0].token_ids)[:ACT_TOKENS]
        ids = torch.tensor(p_ids + a_ids, device="cuda")
        _, acts = hf_forward(hf, ids, hidden, mlp)
        last, ans = len(p_ids) - 1, slice(len(p_ids), len(p_ids) + max(len(a_ids), 1))
        for j, h in enumerate(hidden):
            x = acts[("h", h)]
            arrs["resid_last"][r, j] = x[last].float().cpu().numpy()
            arrs["resid_mean"][r, j] = x[ans].float().mean(0).cpu().numpy()
        for j, l in enumerate(mlp):
            x = acts[("m", l)]
            arrs["mlp_last"][r, j] = x[last].float().cpu().numpy()
            arrs["mlp_mean"][r, j] = x[ans].float().mean(0).cpu().numpy()
    files = {}
    for name, a in arrs.items():
        if a.shape[1]:
            np.save(f"{folder}/{name}.npy", a)
            files[name] = f"{name}.npy"
    index = {"hidden_index": hidden, "mlp_layer": mlp, "files": files,
             "note": "rows follow 'samples'; *_last = last prompt token, *_mean = mean over the first 8,192 "
                     "answer tokens; hidden index h: 0 = embeddings, h = l+1 = output of layer l",
             "samples": [{"row": r, "id": key, "sample": s} for r, (key, s, _) in enumerate(meta)]}
    json.dump(index, open(f"{folder}/index.json", "w"))
    return folder


def teacher_forced(k, req):
    ret = req.get("return", {})
    hidden = sorted(set(int(h) for h in ret.get("hidden_states", [])))
    mlp = sorted(set(int(l) for l in ret.get("mlp_act", [])))
    attn = sorted(set(int(l) for l in ret.get("attn_layers", [])))
    topk = int(ret.get("topk", 0))
    if "chats" in req:
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking", False)))
                 for c in req["chats"]]
    else:
        texts = list(req["texts"])
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    lens = [len(x) for x in ids]
    if not texts or len(texts) > 1024 or max(lens) > 32768 or sum(lens) > 524288:
        raise ValueError(f"limits: {len(texts)} texts (max 1,024), longest {max(lens, default=0):,} tokens "
                         f"(max 32,768), total {sum(lens):,} tokens (max 524,288)")
    if attn and (len(texts) > 2 or max(lens) > 2048):
        raise ValueError("attn_layers: at most 2 texts of at most 2,048 tokens")
    N, L = len(texts), max(lens)
    size = N * L * (4 + 4 + topk * 8 + 2 * 1024 * len(hidden) + 2 * 3584 * len(mlp)) + N * len(attn) * 8 * L * L * 2
    if size > 6e9:
        raise ValueError(f"the result would be {size / 1e9:.1f} GB (max 6 GB): ask for fewer texts, layers or top-k")
    model = hf
    if attn:
        model, state = eager_model()
        state.set(req["model"])
    hf_state.set(req["model"])
    folder = f"{WORK}/fwd_{k}"
    os.makedirs(folder, exist_ok=True)
    out = {"tokens": np.full((N, L), -1, np.int32)}
    if ret.get("token_logprobs", True):
        out["token_logprobs"] = np.full((N, L), np.nan, np.float32)
    if topk:
        out["topk_ids"] = np.zeros((N, L, topk), np.int32)
        out["topk_logprobs"] = np.zeros((N, L, topk), np.float32)
    if hidden:
        out["hidden_states"] = np.zeros((N, len(hidden), L, 1024), np.float16)
    if mlp:
        out["mlp_act"] = np.zeros((N, len(mlp), L, 3584), np.float16)
    if attn:
        heads = model.config.num_attention_heads
        out["attn"] = np.zeros((N, len(attn), heads, L, L), np.float16)
    for i, x in enumerate(ids):
        t = torch.tensor(x, device="cuda")
        final, acts = hf_forward(model, t, hidden, mlp, attn)
        n = len(x)
        out["tokens"][i, :n] = x
        if "token_logprobs" in out or topk:
            tlp, tki, tkl = logprobs(model, final, t, topk)
            if "token_logprobs" in out:
                out["token_logprobs"][i, :n] = tlp.numpy()
            if topk:
                out["topk_ids"][i, :n], out["topk_logprobs"][i, :n] = tki.numpy(), tkl.numpy()
        for j, h in enumerate(hidden):
            out["hidden_states"][i, j, :n] = acts[("h", h)].float().cpu().numpy()
        for j, l in enumerate(mlp):
            out["mlp_act"][i, j, :n] = acts[("m", l)].float().cpu().numpy()
        for j, l in enumerate(attn):
            out["attn"][i, j, :, :n, :n] = acts[("a", l)].float().cpu().numpy()
    for name, a in out.items():
        np.save(f"{folder}/{name}.npy", a)
    meta = {"model": req["model"], "lengths": lens, "files": {n: f"{n}.npy" for n in out},
            "hidden_index": hidden, "mlp_layer": mlp, "attn_layer": attn,
            "note": "right-padded to the longest text (tokens = -1, logprobs = nan); token_logprobs[t] = "
                    "log p(token t | tokens < t); topk at position t = next-token distribution after token t; "
                    "hidden index h: 0 = embeddings, h = l+1 = output of layer l"}
    json.dump(meta, open(f"{folder}/meta.json", "w"))
    return {"folder": folder}


def debug_vllm_logprobs(k, req):
    """Harness self-test only (the driver never forwards this type): vLLM's own prompt log-probs,
    to check that weight swaps reach the engine and match the HF copy."""
    vllm_state.set(req["model"])
    ids = [tok.encode(t, add_special_tokens=False) for t in req["texts"]]
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], SamplingParams(max_tokens=1, prompt_logprobs=0),
                        use_tqdm=False)
    lps = [[None] + [o.prompt_logprobs[t][x[t]].logprob for t in range(1, len(x))] for o, x in zip(outs, ids)]
    return {"logprobs": lps}


HANDLERS = {"generate": generation, "teacher_forced": teacher_forced, "debug_vllm_logprobs": debug_vllm_logprobs}


def main():
    init()
    open(f"{REQ}/server.ready", "w").write(str(time.time()))
    print("gpu server ready", flush=True)
    while True:
        for path in sorted(glob.glob(f"{REQ}/fwd_*.json")):
            if path.endswith(".result.json"):
                continue
            res_path = path[:-5] + ".result.json"
            if os.path.exists(res_path):
                continue
            k = int(os.path.basename(path)[4:-5])
            t0 = time.time()
            try:
                req = json.load(open(path))
                res = HANDLERS[req["type"]](k, req)
                res["ok"] = True
            except Exception as e:  # noqa: BLE001 - every failure goes back to the driver
                res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:3000], "trace": traceback.format_exc()[-4000:]}
            res["wall_seconds"] = round(time.time() - t0, 1)
            torch.cuda.empty_cache()
            with open(res_path + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(res_path + ".tmp", res_path)
            print(f"fwd_{k}: ok={res['ok']} in {res['wall_seconds']}s", flush=True)
        open(f"{REQ}/server.alive", "w").write(str(time.time()))
        time.sleep(2)


if __name__ == "__main__":
    main()
