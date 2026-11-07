"""Agreement check before any training run (plans/training-phase.md, "Infrastructure", item 3).

    train/run-check.sh <5080|5090> [start submission]

1. vLLM samples one answer each for 50 training problems (temperature 1.0, top_p 1.0, at most 3,072 new tokens) and
   returns each sampled token's log-prob.
2. The training model (fp32 master weights, bf16 autocast, flash-linear-attention) scores the same tokens.
3. The two sets of log-probs are compared: these differences are what the importance weights in the GRPO update
   correct, so they must be small.
4. One forward and backward pass with gradient checkpointing is timed at 4,096, 8,192 and 16,384 tokens, twice each
   (the first includes the kernels' compilation for that length), with peak memory.
5. Which implementation each linear-attention kernel function uses (flash-linear-attention, or transformers' slow
   PyTorch reference) is recorded, with transformers' fallback warnings.

Writes /train/checks/logprobs-<gpu>.json.
"""
import gc
import glob
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, "/train")
GPU, START = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)

warnings = []


class Catch(logging.Handler):
    def emit(self, record):
        warnings.append(record.getMessage())


logging.getLogger("transformers").addHandler(Catch())

import common  # noqa: E402  (after the handler, so import-time warnings are caught)
from vllm import LLM, SamplingParams  # noqa: E402


def kernels():
    """Which implementation each of the linear-attention layers' kernel functions resolved to: transformers' decorator
    (use_kernel_func_from_hub_with_fallback) keeps it in its closure as `implementation`; a function from
    transformers itself is the slow reference path."""
    import inspect
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    out = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        f = getattr(m, name, None)
        impl = inspect.getclosurevars(f).nonlocals.get("implementation") if f else None
        out[name] = f"{impl.__module__}.{impl.__name__}" if impl else "?"
    return out


def main():
    probs = common.load_problems("/train/data/train.jsonl", n=50, seed=7)
    pids = [common.prompt_ids(p["problem"]) for p in probs]
    llm = LLM(model=common.MODEL, language_model_only=True, max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("WSP_GPU_MEMORY", "0.40")), max_num_seqs=64,
              enable_prefix_caching=False, seed=0, enable_sleep_mode=True,
              worker_extension_cls="wsp_ext.WspWorkerExtension")
    if START:  # the harness's own swap: each .npy of the submission, cast to bf16 (harness/wsp_ext.py)
        pairs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(f"{START}/*.npy"))]
        llm.collective_rpc("wsp_set_weights", args=(pairs,))
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": x} for x in pids],
                        [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=3072, logprobs=0, seed=i)
                         for i in range(len(pids))])
    gen_s = time.time() - t0
    samples = []
    for x, o in zip(pids, outs):
        c = o.outputs[0]
        lp = [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]
        samples.append((x, list(c.token_ids), lp, c.finish_reason))
    llm.sleep(level=1)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    model = common.build_model(START, trainable=True)
    model.eval()
    diffs, seq_sums = [], []
    with torch.no_grad():
        for x, comp, lp, _ in samples:
            ids = torch.tensor(x + comp, device="cuda")
            hf = common.token_logprobs(model, ids, len(x)).cpu()
            d = hf - torch.tensor(lp)
            diffs.append(d)
            seq_sums.append(float(d.sum()))
    d = torch.cat(diffs)
    q = torch.quantile(d.abs(), torch.tensor([0.5, 0.9, 0.99, 0.999]))
    agreement = {
        "tokens": len(d), "sequences": len(samples),
        "mean_abs": float(d.abs().mean()), "mean_signed": float(d.mean()),
        "abs_p50_p90_p99_p999": [float(v) for v in q], "abs_max": float(d.abs().max()),
        "share_abs_over_0.1": float((d.abs() > 0.1).float().mean()),
        "share_abs_over_1": float((d.abs() > 1).float().mean()),
        "per_sequence_sum_abs_median": float(torch.tensor(seq_sums).abs().median()),
        "token_ratio_exp_diff_p01_p99": [float(v) for v in torch.quantile(d.exp(), torch.tensor([0.01, 0.99]))],
        "mean_new_tokens": sum(len(s[1]) for s in samples) / len(samples),
        "cut": sum(s[3] == "length" for s in samples), "vllm_generate_seconds": round(gen_s, 1),
    }

    model.train()
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    timing = {}
    base = samples[0][0] + samples[0][1]
    warm = torch.tensor((base * 4)[:1024], device="cuda")  # the kernels' first-call compilation stays out of the timing
    (-common.token_logprobs(model, warm, 256).mean()).backward()
    model.zero_grad(set_to_none=True)
    # Twice at each length (the first pass at a new length may include the kernels' compilation), then once at a
    # length 37 tokens shorter: if that is slow again, the kernels recompile for every exact length.
    for n in (4096, 8192, 16384):
        runs = []
        for m in (n, n, n - 37):
            ids = torch.tensor((base * (m // len(base) + 1))[:m], device="cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.time()
            loss = -common.token_logprobs(model, ids, 256).mean()
            loss.backward()
            torch.cuda.synchronize()
            runs.append((round(time.time() - t, 2), round(torch.cuda.max_memory_allocated() / 2**30, 2)))
            model.zero_grad(set_to_none=True)
        timing[n] = {"seconds": runs[1][0], "first_seconds": runs[0][0], "new_length_seconds": runs[2][0],
                     "peak_gb": max(r[1] for r in runs)}
    fallback = [w for w in warnings if "falling back" in w.lower()]
    out = {"gpu": GPU, "start": START, "torch": torch.__version__, "agreement": agreement,
           "fwd_bwd": timing, "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
           "kernels": kernels(), "linear_attention_fallback_warnings": fallback}
    os.makedirs("/train/checks", exist_ok=True)
    json.dump(out, open(f"/train/checks/logprobs-{GPU}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
