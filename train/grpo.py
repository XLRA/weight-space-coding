"""GRPO on test pass/fail (plans/training-phase.md, "The recipe").

    train/run-train.sh <gpu> grpo <arm> [--start <submission>] [options]   (in wsp-train:1, no network, holding the GPU's lock)

One process holds both copies of the model on one GPU:
- vLLM samples (sleep mode frees its memory while the trainer works);
- the trainer (common.build_model: fp32 master weights for the 24 blocks, bf16 autocast) takes the updates and pushes
  its bf16 blocks back into vLLM after each step, through /dev/shm.

Each step:
1. Sample: the next --prompts training problems, --group answers each (temperature 1.0, top_p 1.0, no presence
   penalty, at most --max-new tokens), with seeds fixed by (seed, step, problem, sample).
2. Grade: the practice grader (harness/grade.py, which the panel's code suite also uses) in a pool, each test as user nobody with 2 GB of memory, in a
   container with no network. Reward 1 if the answer passes all its tests, else 0 (cut, no code and errors are 0).
3. Advantages: per problem, (r - mean) / std over its group. Groups whose answers all score the same are skipped.
4. Update: the batch is split into --minibatches parts (by problem). Before the first, the trainer's own log-probs
   of every kept answer are taken as the old policy. For each part, the loss is the clipped surrogate
   min(rho A, clip(rho, 1 - clip_low, 1 + clip_high) A), with rho = exp(logp - old), times a truncated importance
   weight min(exp(old - logp_vllm), tis_cap) that corrects for the sampler's small numerical differences, summed
   over tokens and divided by the part's token count (token-level averaging). No KL term. Gradient clip --grad-clip,
   AdamW, learning rate --lr after --warmup steps of linear warm-up.
5. Push the new weights to vLLM.

Writes /train/runs/<arm>/: config.json, steps.jsonl (one line per step), samples/step_<k>.jsonl.gz (every answer
with its reward and grade), checkpoints/step_<k>/ (every block tensor as .npy, the evaluation's submission format)
and resume.pt (the latest trainer state). Running the same command again resumes from resume.pt.
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, "/train")  # grade is harness/grade.py, on PYTHONPATH (/harness)
import common  # noqa: E402
import grade  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--start", default=None, help="a submission folder (/start); none: the original model")
ap.add_argument("--data", default="/train/data/train.jsonl")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--prompts", type=int, default=64)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--max-new", type=int, default=16384)
ap.add_argument("--minibatches", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--clip-low", type=float, default=0.2)
ap.add_argument("--clip-high", type=float, default=0.28)
ap.add_argument("--tis-cap", type=float, default=2.0)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--ckpt-every", type=int, default=25)
ap.add_argument("--gpu-memory", type=float, default=0.30)
args = ap.parse_args()

OUT = f"/train/runs/{args.arm}"
SHM = f"/dev/shm/wsp-train-{args.arm}.safetensors"


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.gmtime()) + msg, flush=True)


def batches(problems):
    """The training problems in a fresh seeded order each epoch, args.prompts at a time, forever."""
    epoch = 0
    while True:
        order = problems[:]
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for i in range(0, len(order) - args.prompts + 1, args.prompts):
            yield epoch, order[i:i + args.prompts]
        epoch += 1


def push(llm, model):
    save_file(common.block_state(model), SHM)
    llm.collective_rpc("train_load_file", args=(SHM,))
    os.remove(SHM)
    torch.cuda.empty_cache()  # the bf16 copies made for the file go back to the device, for vLLM's KV cache


def save_checkpoint(model, opt, step):
    d = f"{OUT}/checkpoints/step_{step}"
    os.makedirs(d, exist_ok=True)
    for name, t in common.block_state(model).items():
        np.save(f"{d}/{name}.npy", t.float().cpu().numpy())
    json.dump({"arm": args.arm, "step": step, "start": args.start, "tensors": "all 24 blocks, bf16 values as fp32"},
              open(f"{d}/manifest.json", "w"), indent=1)
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"step": step, "model": trainable, "opt": opt.state_dict()}, f"{OUT}/resume.pt.tmp")
    os.replace(f"{OUT}/resume.pt.tmp", f"{OUT}/resume.pt")
    log(f"checkpoint step {step}")


def grade_all(items):
    with ThreadPoolExecutor(12) as ex:
        return list(ex.map(lambda x: grade.grade(x[0], x[1]), items))


def main():
    os.makedirs(f"{OUT}/samples", exist_ok=True)
    cfg = {**vars(args), "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    problems = common.load_problems(args.data)
    llm = LLM(model=common.MODEL, language_model_only=True, max_model_len=args.max_new + 4096,
              gpu_memory_utilization=args.gpu_memory, max_num_seqs=256, enable_prefix_caching=False, seed=0,
              enable_sleep_mode=True, worker_extension_cls="train_ext.TrainWorkerExtension")
    model = common.build_model(args.start, trainable=True)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    step0 = 0
    if os.path.exists(f"{OUT}/resume.pt"):
        st = torch.load(f"{OUT}/resume.pt", map_location="cuda")
        named = dict(model.named_parameters())
        for n, t in st["model"].items():
            named[n].data.copy_(t)
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
        log(f"resumed at step {step0}")
    else:
        json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
    push(llm, model)  # vLLM now holds exactly the trainer's weights (the start edit, or the resumed state)
    feed = batches(problems)
    for _ in range(step0):  # the same problem order as an unbroken run
        next(feed)

    for step in range(step0 + 1, args.steps + 1):
        t0 = time.time()
        epoch, batch = next(feed)
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        pids = [common.prompt_ids(p["problem"]) for p in batch]
        sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, logprobs=0,
                              seed=(args.seed * 1_000_003 + step * 1009 + i * 31 + s) % 2**31)
               for i in range(len(batch)) for s in range(args.group)]
        outs = llm.generate([{"prompt_token_ids": x} for x in pids for _ in range(args.group)], sps, use_tqdm=False)
        llm.sleep(level=1)
        t_gen = time.time() - t0

        answers = []
        for k, o in enumerate(outs):
            i, c = k // args.group, o.outputs[0]
            answers.append({"i": i, "prompt": pids[i], "ids": list(c.token_ids), "text": c.text,
                            "finish": c.finish_reason,
                            "vlp": [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)]})
        graded = grade_all([(batch[a["i"]], {"id": batch[a["i"]]["id"], "n_tokens": len(a["ids"]), "text": a["text"],
                                             "finish_reason": a["finish"]}) for a in answers])
        for a, gr in zip(answers, graded):
            a["reward"] = 1.0 if gr["status"] == "pass" else 0.0
            a["status"] = gr["status"]
        t_grade = time.time() - t0 - t_gen

        kept_groups = []
        for i in range(len(batch)):
            grp = answers[i * args.group:(i + 1) * args.group]
            r = torch.tensor([a["reward"] for a in grp])
            if r.std() > 0:
                adv = (r - r.mean()) / (r.std() + 1e-6)
                for a, v in zip(grp, adv.tolist()):
                    a["adv"] = v
                kept_groups.append(i)
        kept = [a for a in answers if "adv" in a and len(a["ids"]) > 0]

        stats = {"loss": [], "clip_frac": [], "tis_mean": [], "mismatch_abs": [], "grad_norm": []}
        if kept:
            model.eval()
            with torch.no_grad():
                for a in kept:
                    ids = torch.tensor(a["prompt"] + a["ids"], device="cuda")
                    a["old"] = common.token_logprobs(model, ids, len(a["prompt"])).detach()
            model.train()
            parts = [kept_groups[j::args.minibatches] for j in range(args.minibatches)]
            for part in parts:
                seqs = [a for a in kept if a["i"] in set(part)]
                if not seqs:
                    continue
                n_tok = sum(len(a["ids"]) for a in seqs)
                opt.zero_grad(set_to_none=True)
                part_loss, clipped, tis_sum = 0.0, 0, 0.0
                for a in seqs:
                    ids = torch.tensor(a["prompt"] + a["ids"], device="cuda")
                    logp = common.token_logprobs(model, ids, len(a["prompt"]))
                    old, vlp = a["old"], torch.tensor(a["vlp"], device="cuda")
                    rho = torch.exp(logp - old)
                    adv = a["adv"]
                    surr = torch.minimum(rho * adv, torch.clamp(rho, 1 - args.clip_low, 1 + args.clip_high) * adv)
                    tis = torch.clamp(torch.exp(old - vlp), max=args.tis_cap)
                    loss = -(surr * tis).sum() / n_tok
                    loss.backward()
                    part_loss += float(loss)
                    clipped += int(((rho < 1 - args.clip_low) | (rho > 1 + args.clip_high)).sum())
                    tis_sum += float(tis.sum())
                    stats["mismatch_abs"].append(float((old - vlp).abs().mean()))
                gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                stats["loss"].append(part_loss)
                stats["clip_frac"].append(clipped / n_tok)
                stats["tis_mean"].append(tis_sum / n_tok)
                stats["grad_norm"].append(float(gn))
            opt.zero_grad(set_to_none=True)
            for a in kept:
                del a["old"]
        t_train = time.time() - t0 - t_gen - t_grade
        torch.cuda.empty_cache()  # release the trainer's cached activations and logits before vLLM maps its memory back
        llm.wake_up()
        push(llm, model)
        t_all = time.time() - t0

        n = len(answers)
        lens = [len(a["ids"]) for a in answers]
        status = {}
        for a in answers:
            status[a["status"]] = status.get(a["status"], 0) + 1
        mean = lambda v: sum(v) / len(v) if v else None
        row = {"step": step, "epoch": epoch, "lr": lr, "problems": len(batch), "answers": n,
               "pass_rate": sum(a["reward"] for a in answers) / n,
               "problems_with_a_pass": sum(any(a["reward"] for a in answers[i * args.group:(i + 1) * args.group])
                                           for i in range(len(batch))),
               "groups_kept": len(kept_groups), "answers_trained": len(kept), "tokens_trained": sum(len(a["ids"]) for a in kept),
               "cut_share": status.get("cut_at_max_tokens", 0) / n, "status": status,
               "mean_tokens": mean(lens), "max_tokens": max(lens),
               "loss": mean(stats["loss"]), "clip_frac": mean(stats["clip_frac"]), "tis_mean": mean(stats["tis_mean"]),
               "mismatch_abs": mean(stats["mismatch_abs"]), "grad_norm": mean(stats["grad_norm"]),
               "seconds": {"generate": round(t_gen), "grade": round(t_grade), "train": round(t_train),
                           "total": round(t_all)},
               "time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
        with open(f"{OUT}/steps.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        with gzip.open(f"{OUT}/samples/step_{step}.jsonl.gz", "wt") as f:
            for a in answers:
                f.write(json.dumps({"id": batch[a["i"]]["id"], "reward": a["reward"], "status": a["status"],
                                    "adv": a.get("adv"), "tokens": len(a["ids"]), "finish": a["finish"],
                                    "text": a["text"]}) + "\n")
        log(f"step {step}: pass {row['pass_rate']:.3f}, kept {len(kept_groups)}/{len(batch)} groups, "
            f"cut {row['cut_share']:.2f}, mean {row['mean_tokens']:.0f} tokens, loss {row['loss']}, "
            f"gen {t_gen:.0f}s grade {t_grade:.0f}s train {t_train:.0f}s total {t_all:.0f}s")
        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(model, opt, step)
    log("done")


if __name__ == "__main__":
    main()
