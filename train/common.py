"""Shared pieces of the training phase (plans/training-phase.md): problems, prompts, the trainable model, log-probs.

Runs inside wsp-train:1 with the model at /model (read-only) and train/ at /train. The model is built the way
harness/gpu_server.py builds its HF copy (Qwen3_5ForCausalLM from the text config, sdpa for the six full-attention
layers; the 18 linear-attention layers use flash-linear-attention's kernels, which wsp-train:1 adds), with two
differences for training:
- the 24 blocks are kept in fp32 as master weights and run under bf16 autocast: an AdamW step at learning rate 1e-6
  is far below bf16's resolution for weights of this size, so bf16 parameters would not move (the frozen embedding
  stays in bf16);
- only the blocks are trainable. The tied embedding/output matrix and the final norm are frozen, the tensors the
  agent was not allowed to edit.
"""
import glob
import json
import os
import random

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

MODEL = "/model"
PREFIX = "model.language_model."
INDEX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
LM_NAMES = sorted(n for n in INDEX if n.startswith(PREFIX))
tok = AutoTokenizer.from_pretrained(MODEL)
_files = {}


def load_problems(path, n=None, seed=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if n is not None:
        rows = random.Random(seed).sample(rows, n)
    return rows


def render(problem_text):
    """The evaluation's prompt: the problem as the user turn, thinking on (harness/gpu_server.py render_user)."""
    return tok.apply_chat_template([{"role": "user", "content": problem_text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def prompt_ids(problem_text):
    return tok(render(problem_text), add_special_tokens=False).input_ids


def original(name):
    fn = f"{MODEL}/{INDEX[name]}"
    if fn not in _files:
        _files[fn] = safe_open(fn, "pt")
    return _files[fn].get_tensor(name)


def start_weights(start_dir=None):
    """{checkpoint name: tensor}: the original language-model weights, with a submission's .npy files over them."""
    w = {n: original(n) for n in LM_NAMES}
    if start_dir:
        for path in glob.glob(f"{start_dir}/*.npy"):
            name = os.path.basename(path)[:-4]
            assert name in w, name
            w[name] = torch.from_numpy(np.load(path, allow_pickle=False)).to(torch.bfloat16)
    return w


def hf_name(name):
    return "model." + name[len(PREFIX):]


def ckpt_name(name):
    return PREFIX + name[len("model."):]


def build_model(start_dir=None, trainable=True):
    """The HF model on the GPU: blocks in fp32 (trainable if asked), embedding and final norm frozen."""
    cfg = AutoConfig.from_pretrained(MODEL).text_config
    cfg._attn_implementation = "sdpa"
    with torch.device("cuda"):
        model = Qwen3_5ForCausalLM(cfg)
    sd = {hf_name(n): t.float() for n, t in start_weights(start_dir).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
    # The frozen tied embedding/output matrix stays in bf16, as vLLM holds it: the residual stream then starts in bf16
    # as in vLLM, it takes 0.5 GB instead of 1 GB, and autocast does not re-cast it for every chunk of logits.
    model.model.embed_tokens.to(torch.bfloat16)
    model.lm_head.weight = model.model.embed_tokens.weight  # tied
    for n, p in model.named_parameters():
        p.requires_grad_(trainable and n.startswith("model.layers."))
    return model


def block_state(model):
    """{checkpoint name: bf16 tensor} of the 24 blocks, as vLLM and the evaluation load them."""
    return {ckpt_name(n): p.detach().to(torch.bfloat16) for n, p in model.named_parameters()
            if n.startswith("model.layers.")}


def _chunk_logprobs(h, lm_w, tgt):
    logits = torch.nn.functional.linear(h, lm_w).float()
    return torch.log_softmax(logits, -1).gather(-1, tgt[:, None])[:, 0]


def token_logprobs(model, ids, start, chunk=1024):
    """log p(ids[t] | ids[:t]) for t >= start, as a 1-D tensor.

    The logits are made 1,024 positions at a time and recomputed in the backward pass, so a 16k-token answer never
    holds its full 16k x 248,320 fp32 logits (16 GB)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        hs, tgt = h[start - 1:-1], ids[start:]
        lm_w = model.lm_head.weight
        parts = []
        for s in range(0, len(tgt), chunk):
            if torch.is_grad_enabled():
                parts.append(checkpoint(_chunk_logprobs, hs[s:s + chunk], lm_w, tgt[s:s + chunk], use_reentrant=False))
            else:
                parts.append(_chunk_logprobs(hs[s:s + chunk], lm_w, tgt[s:s + chunk]))
    return torch.cat(parts)
