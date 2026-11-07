"""Weight-push check before any GRPO run: does vLLM really hold the trainer's weights after each step?

    train/run-check.sh <5080|5090> push [start submission]

grpo.py keeps vLLM and the trainer on one GPU. After each update vLLM wakes and the trainer's blocks are pushed into
it (save to /dev/shm, train_ext.train_load_file). A push that silently failed would leave vLLM sampling from an old
model while the trainer moves on, and at learning rate 1e-6 the importance weights would hide it for a long time. This
check makes a change too large to miss and takes it through the same sequence as a training step:
1. vLLM starts from the original weights; the trainer loads the start and pushes it. vLLM's log-probs of 16
   teacher-forced texts (training problems with their reference solutions) are compared with the trainer's: the
   baseline agreement.
2. The trainer's 24 MLP down-projections are scaled by 0.7. Then, as in a step: vLLM sleeps, the trainer runs a forward
   and backward pass, the cache is emptied, vLLM wakes and the blocks are pushed. vLLM must agree with the changed
   trainer about as well as in 1, and differ clearly from its own log-probs in 1.
3. The change is undone and pushed the same way: vLLM must be back at its log-probs of 1.
Writes /train/checks/push-<gpu>.json.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/train")
import common  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

GPU, START = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
SHM = f"/dev/shm/wsp-check-push-{GPU}.safetensors"


def push(llm, model):  # as grpo.push
    save_file(common.block_state(model), SHM)
    llm.collective_rpc("train_load_file", args=(SHM,))
    os.remove(SHM)
    torch.cuda.empty_cache()


def vllm_lp(llm, seqs):
    outs = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=0),
                        use_tqdm=False)
    return [torch.tensor([o.prompt_logprobs[t][s[t]].logprob for t in range(1, len(s))]) for s, o in zip(seqs, outs)]


def trainer_lp(model, seqs):
    model.eval()
    with torch.no_grad():
        out = [common.token_logprobs(model, torch.tensor(s, device="cuda"), 1).float().cpu() for s in seqs]
    model.train()
    return out


def gap(a, b):
    d = torch.cat([x - y for x, y in zip(a, b)]).abs()
    return {"mean_abs": round(float(d.mean()), 5), "p99": round(float(torch.quantile(d, 0.99)), 4),
            "max": round(float(d.max()), 4)}


def step_cycle(llm, model, seq):
    """What a GRPO step does between sampling and the next sampling."""
    llm.sleep(level=1)
    loss = -common.token_logprobs(model, torch.tensor(seq, device="cuda"), 1).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    llm.wake_up()
    push(llm, model)


def main():
    probs = common.load_problems("/train/data/train.jsonl", n=16, seed=11)
    seqs = [(common.prompt_ids(p["problem"]) + common.tok(p["reference"], add_special_tokens=False).input_ids)[:3000]
            for p in probs]
    llm = LLM(model=common.MODEL, language_model_only=True, max_model_len=4096,
              gpu_memory_utilization=float(os.environ.get("WSP_GPU_MEMORY", "0.40")), max_num_seqs=64,
              enable_prefix_caching=False, seed=0, enable_sleep_mode=True,
              worker_extension_cls="train_ext.TrainWorkerExtension")
    model = common.build_model(START, trainable=True)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    push(llm, model)
    v0, t0 = vllm_lp(llm, seqs), trainer_lp(model, seqs)
    before = gap(v0, t0)

    down = [p for n, p in model.named_parameters() if n.endswith("mlp.down_proj.weight")]
    with torch.no_grad():
        for p in down:
            p.mul_(0.7)
    step_cycle(llm, model, seqs[0])
    v1, t1 = vllm_lp(llm, seqs), trainer_lp(model, seqs)
    after, moved, trainer_moved = gap(v1, t1), gap(v1, v0), gap(t1, t0)

    with torch.no_grad():
        for p in down:
            p.div_(0.7)
    step_cycle(llm, model, seqs[0])
    v2, t2 = vllm_lp(llm, seqs), trainer_lp(model, seqs)
    restored, back = gap(v2, t2), gap(v2, v0)

    tol = 2 * before["mean_abs"] + 0.01
    ok = {"agrees_after_the_change": after["mean_abs"] <= tol,
          "the_change_reached_vllm": moved["mean_abs"] >= 10 * before["mean_abs"],
          "agrees_after_undoing_it": restored["mean_abs"] <= tol,
          "vllm_back_where_it_was": back["mean_abs"] <= tol}
    out = {"gpu": GPU, "start": START, "texts": len(seqs), "tokens": sum(len(s) - 1 for s in seqs),
           "down_proj_tensors_scaled": len(down), "vllm_vs_trainer_before": before,
           "vllm_vs_trainer_after_change": after, "vllm_moved_by_the_change": moved,
           "trainer_moved_by_the_change": trainer_moved, "vllm_vs_trainer_after_undo": restored,
           "vllm_after_undo_vs_before": back, "tolerance": round(tol, 5), "tests": ok, "pass": all(ok.values())}
    os.makedirs("/train/checks", exist_ok=True)
    json.dump(out, open(f"/train/checks/push-{GPU}.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
