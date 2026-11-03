"""One-off: what tensors the checkpoint has and what the vLLM image provides (run inside the image)."""
import importlib.util
import json
from collections import Counter

idx = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
names = sorted(idx)
print("total tensors:", len(names))
print(Counter(".".join(n.split(".")[:2]) for n in names))
lm = [n for n in names if n.startswith("model.language_model.")]
print("language_model tensors:", len(lm))
for n in lm:
    if ".layers.0." in n or ".layers.3." in n or ".layers." not in n:
        print("  ", n)
print("other:", [n for n in names if not n.startswith(("model.language_model.", "model.visual."))][:20])
cfg = json.load(open("/model/config.json"))
print(json.dumps({k: v for k, v in cfg.items() if k != "vision_config"})[:2500])

import torch, transformers, vllm
print("transformers", transformers.__version__, "torch", torch.__version__, "vllm", vllm.__version__)
for m in ["fla", "causal_conv1d", "flash_attn", "safetensors", "pandas", "scipy", "sklearn", "sympy", "networkx", "zmq"]:
    print(" ", m, bool(importlib.util.find_spec(m)))
from transformers.models import qwen3_5
print([x for x in dir(qwen3_5) if "Qwen" in x])
print("LLM.collective_rpc:", hasattr(vllm.LLM, "collective_rpc"), "LLM.reset_prefix_cache:", hasattr(vllm.LLM, "reset_prefix_cache"))
from vllm.engine.arg_utils import EngineArgs
print("worker_extension_cls:", "worker_extension_cls" in EngineArgs.__dataclass_fields__,
      "language_model_only:", "language_model_only" in EngineArgs.__dataclass_fields__)
