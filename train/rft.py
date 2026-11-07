"""Rejection-sampling fine-tune, RFT (plans/training-phase.md, "RFT").

    train/run-train.sh <gpu> rft <arm> [--start <submission folder on the host>] [options]

Each round r (1 .. --rounds):
1. Sample: --samples answers per training problem from the latest model (the arm's start in round 1, round r-1's
   model after that), at the evaluation's sampling settings (temperature 0.6, top_p 0.95, top_k 20, presence penalty
   1.5) but with at most --max-new new tokens. Seeds are fixed by (seed, round, problem, sample).
2. Grade: the practice grader (harness/grade.py, which the panel's code suite also uses) in a pool, while the next chunk is sampled.
3. Keep: per problem, up to --keep passing answers whose final code blocks (grade.extract) differ, taken in a seeded
   random order.
4. Fine-tune from the arm's start, not from round r-1's model (ReST-EM, Singh et al. 2023: each round restarts from
   the base, on samples from the latest model): --epochs passes over the kept answers in a seeded order, --batch
   answers per optimizer step, next-token cross-entropy on the answer tokens only, averaged over the batch's tokens.
   AdamW at --lr, linear warm-up for --warmup steps then cosine decay towards 0, gradient clip --grad-clip, fp32
   master weights for the 24 blocks under bf16 autocast (common.build_model).
5. Save round_<r>/checkpoint/ (every block tensor as .npy, the evaluation's submission format). The next round's
   sampling loads it into vLLM with the harness's own swap (wsp_ext.wsp_set_weights).

vLLM and the trainer take turns on the GPU: vLLM sleeps (weights to host memory, cache freed) while the trainer works,
and the trainer is deleted before vLLM wakes, so sampling gets most of the GPU.

Writes /train/runs/<arm>/: config.json, train.log, and per round round_<r>/: samples/chunk_<k>.jsonl.gz (every answer
with its grade; token ids only for passing answers), kept.jsonl.gz (the fine-tuning set), summary.json, sft.jsonl (one
line per optimizer step) and checkpoint/. Running the same command again skips every finished chunk, selection and
fine-tune.
"""
import argparse
import gc
import glob
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
from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arm")
ap.add_argument("--start", default=None, help="a submission folder (/start); none: the original model")
ap.add_argument("--data", default="/train/data/train.jsonl")
ap.add_argument("--problems", type=int, default=None, help="only this many training problems (a seeded draw), for tests")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--samples", type=int, default=8)
ap.add_argument("--keep", type=int, default=2)
ap.add_argument("--max-new", type=int, default=16384)
ap.add_argument("--chunk", type=int, default=256, help="problems per sampling chunk")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--grad-clip", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--gpu-memory", type=float, default=0.85)
args = ap.parse_args()

OUT = f"/train/runs/{args.arm}"
# The evaluation's sampling (harness/gpu_server.py SAMPLING, at the protocol's presence penalty).
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5)
IM_END = common.tok.convert_tokens_to_ids("<|im_end|>")
STOP_IDS = {IM_END, common.tok.eos_token_id}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.gmtime()) + msg, flush=True)


class Sampler:
    """vLLM, tracking which weights it holds and whether it sleeps."""

    def __init__(self, max_model_len):
        self.llm = LLM(model=common.MODEL, language_model_only=True, max_model_len=max_model_len,
                       gpu_memory_utilization=args.gpu_memory, max_num_seqs=256, enable_prefix_caching=False,
                       seed=0, enable_sleep_mode=True, worker_extension_cls="wsp_ext.WspWorkerExtension")
        self.held, self.asleep = None, False  # None: the original model

    def hold(self, folder):
        """Awake, holding the weights of a submission folder (None: the original)."""
        if self.asleep:
            if torch.cuda.is_initialized():
                free, total = torch.cuda.mem_get_info()
                log(f"waking vLLM: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
            self.llm.wake_up()
            self.asleep = False
        if folder != self.held:
            pairs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(f"{folder}/*.npy"))]
            n = self.llm.collective_rpc("wsp_set_weights", args=(pairs,))
            log(f"vLLM now holds {folder} ({n[0]} tensors)")
            self.held = folder

    def sleep(self):
        if not self.asleep:
            self.llm.sleep(level=1)
            self.asleep = True
        gc.collect()
        torch.cuda.empty_cache()


def sample_round(sampler, problems, r):
    """Every training problem, --samples times, graded; one file per chunk."""
    rd = f"{OUT}/round_{r}/samples"
    os.makedirs(rd, exist_ok=True)
    pool = ThreadPoolExecutor(12)
    pending = []

    def flush(wait):
        for item in list(pending):
            path, rows, futs = item
            if not (wait or all(f.done() for f in futs)):
                continue
            with gzip.open(path + ".tmp", "wt") as fh:
                for row, f in zip(rows, futs):
                    g = f.result()
                    row["status"], row["loop"] = g["status"], g["loop"]
                    if row["status"] != "pass":
                        del row["ids"]
                    fh.write(json.dumps(row) + "\n")
            os.replace(path + ".tmp", path)
            pending.remove(item)

    for k in range(0, len(problems), args.chunk):
        path = f"{rd}/chunk_{k // args.chunk}.jsonl.gz"
        if os.path.exists(path):
            continue
        chunk = problems[k:k + args.chunk]
        t0 = time.time()
        pids = [common.prompt_ids(p["problem"]) for p in chunk]
        sps = [SamplingParams(**SAMPLING, max_tokens=args.max_new,
                              seed=(args.seed * 1_000_003 + r * 100_003 + (k + i) * 31 + s) % 2**31)
               for i in range(len(chunk)) for s in range(args.samples)]
        outs = sampler.llm.generate([{"prompt_token_ids": x} for x in pids for _ in range(args.samples)], sps,
                                    use_tqdm=False)
        rows, futs = [], []
        for j, o in enumerate(outs):
            p, c = chunk[j // args.samples], o.outputs[0]
            rows.append({"id": p["id"], "sample": j % args.samples, "tokens": len(c.token_ids),
                         "finish": c.finish_reason, "ids": list(c.token_ids), "text": c.text})
            futs.append(pool.submit(grade.grade, p, {"id": p["id"], "n_tokens": len(c.token_ids), "text": c.text,
                                                     "finish_reason": c.finish_reason}))
        pending.append((path, rows, futs))
        n_tok, dt = sum(x["tokens"] for x in rows), time.time() - t0
        log(f"round {r}: sampled problems {k + 1}-{k + len(chunk)} of {len(problems)}: {n_tok} tokens in {dt:.0f}s "
            f"({n_tok / dt:.0f} tok/s)")
        flush(False)
    flush(True)
    pool.shutdown()


def select(problems, r):
    """Up to --keep passing answers per problem with different final code; writes kept.jsonl.gz and summary.json."""
    rows = []
    for path in sorted(glob.glob(f"{OUT}/round_{r}/samples/chunk_*.jsonl.gz")):
        rows += [json.loads(line) for line in gzip.open(path, "rt")]
    assert len(rows) == len(problems) * args.samples, (len(rows), len(problems))
    passing = {}
    for row in rows:
        if row["status"] == "pass":
            passing.setdefault(row["id"], []).append(row)
    rng = random.Random(args.seed * 7919 + r)
    kept = []
    for p in problems:
        cand = sorted(passing.get(p["id"], []), key=lambda x: x["sample"])
        rng.shuffle(cand)
        seen = set()
        for row in cand:
            code = grade.extract(row["text"]).strip()
            if code not in seen:
                seen.add(code)
                kept.append(row)
            if len(seen) == args.keep:
                break
    with gzip.open(f"{OUT}/round_{r}/kept.jsonl.gz", "wt") as fh:
        for row in kept:
            fh.write(json.dumps(row) + "\n")

    status, by_source = {}, {}
    for row in rows:
        status[row["status"]] = status.get(row["status"], 0) + 1
    src = {p["id"]: p["source"] for p in problems}
    for row in rows:
        s = by_source.setdefault(src[row["id"]], {"answers": 0, "pass": 0})
        s["answers"] += 1
        s["pass"] += row["status"] == "pass"
    summary = {"round": r, "sampled_from": "start" if r == 1 else f"round_{r - 1}", "problems": len(problems),
               "answers": len(rows), "pass_rate": status.get("pass", 0) / len(rows),
               "problems_with_a_pass": len(passing), "kept_answers": len(kept),
               "cut_share": status.get("cut_at_max_tokens", 0) / len(rows), "status": status,
               "pass_rate_by_source": {k: v["pass"] / v["answers"] for k, v in by_source.items()},
               "mean_tokens": sum(x["tokens"] for x in rows) / len(rows),
               "mean_tokens_kept": sum(x["tokens"] for x in kept) / max(1, len(kept)),
               "loop_share": sum(bool(x["loop"]) for x in rows) / len(rows)}
    json.dump(summary, open(f"{OUT}/round_{r}/summary.json", "w"), indent=1)
    log(f"round {r}: pass rate {summary['pass_rate']:.3f}, {len(passing)} of {len(problems)} problems with a pass, "
        f"{len(kept)} answers kept, cut {summary['cut_share']:.3f}, mean {summary['mean_tokens']:.0f} tokens")
    return kept


def lr_at(step, total):
    if step <= args.warmup:
        return args.lr * step / args.warmup
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / (total - args.warmup + 1)))


def finetune(by_id, kept, r):
    """Fine-tune from the arm's start on the kept answers; writes round_<r>/sft.jsonl and round_<r>/checkpoint/."""
    rd = f"{OUT}/round_{r}"
    model = common.build_model(args.start, trainable=True)
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    seqs, added_stop = [], 0
    for row in kept:
        ids = row["ids"]
        if ids[-1] not in STOP_IDS:  # a passing answer ended on its own; train the stop token too
            ids, added_stop = ids + [IM_END], added_stop + 1
        seqs.append((common.prompt_ids(by_id[row["id"]]["problem"]), ids))
    log(f"round {r}: fine-tuning on {len(seqs)} answers, {sum(len(s[1]) for s in seqs)} tokens "
        f"(stop token appended to {added_stop})")
    order = []
    for e in range(args.epochs):
        o = list(range(len(seqs)))
        random.Random(args.seed * 1000 + r * 10 + e).shuffle(o)
        order += o
    batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
    with open(f"{rd}/sft.jsonl", "w") as fh:
        for step, b in enumerate(batches, 1):
            t0 = time.time()
            lr = lr_at(step, len(batches))
            for g in opt.param_groups:
                g["lr"] = lr
            n_tok = sum(len(seqs[i][1]) for i in b)
            opt.zero_grad(set_to_none=True)
            nll = 0.0
            for i in b:
                prompt, ids = seqs[i]
                lp = common.token_logprobs(model, torch.tensor(prompt + ids, device="cuda"), len(prompt))
                loss = -lp.sum() / n_tok
                loss.backward()
                nll += float(loss)
                # The graph behind lp and loss holds every trainable parameter and its gradient. Kept past the last
                # step, it held about 5.8 GiB after the model was deleted, and vLLM could not wake up on the 5090.
                del lp, loss
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            row = {"step": step, "of": len(batches), "epoch": (step - 1) * args.batch // len(seqs) + 1, "lr": lr,
                   "answers": len(b), "tokens": n_tok, "nll_per_token": nll, "grad_norm": float(gn),
                   "seconds": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if step == 1 or step % 10 == 0 or step == len(batches):
                log(f"round {r}: step {step}/{len(batches)}, nll {nll:.4f}, grad norm {float(gn):.3f}, lr {lr:.2e}, "
                    f"{n_tok} tokens in {row['seconds']}s")
    d = f"{rd}/checkpoint"
    os.makedirs(d + ".tmp", exist_ok=True)
    for name, t in common.block_state(model).items():
        np.save(f"{d}.tmp/{name}.npy", t.float().cpu().numpy())
    json.dump({"arm": args.arm, "round": r, "start": args.start, "answers": len(seqs),
               "tensors": "all 24 blocks, bf16 values as fp32"}, open(f"{d}.tmp/manifest.json", "w"), indent=1)
    os.replace(d + ".tmp", d)
    log(f"round {r}: checkpoint saved")
    opt.zero_grad(set_to_none=True)
    del model, opt, params
    gc.collect()
    torch.cuda.empty_cache()
    log(f"round {r}: trainer freed, {torch.cuda.memory_reserved() / 2**30:.2f} GiB still reserved by PyTorch")


def main():
    os.makedirs(OUT, exist_ok=True)
    problems = common.load_problems(args.data)
    if args.problems:
        problems = random.Random(args.seed).sample(problems, args.problems)
    by_id = {p["id"]: p for p in problems}
    if not os.path.exists(f"{OUT}/config.json"):
        json.dump({**vars(args), "sampling": SAMPLING, "torch": torch.__version__,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())},
                  open(f"{OUT}/config.json", "w"), indent=1)
    max_prompt = max(len(common.prompt_ids(p["problem"])) for p in problems)
    sampler = Sampler(args.max_new + max_prompt + 16)
    latest = args.start  # the folder whose weights the next round samples from
    for r in range(1, args.rounds + 1):
        rd = f"{OUT}/round_{r}"
        if os.path.exists(f"{rd}/checkpoint/manifest.json"):
            log(f"round {r}: already done")
            latest = f"{rd}/checkpoint"
            continue
        if os.path.exists(f"{rd}/summary.json"):
            kept = [json.loads(line) for line in gzip.open(f"{rd}/kept.jsonl.gz", "rt")]
        else:
            sampler.hold(latest)
            sample_round(sampler, problems, r)
            kept = select(problems, r)
        if not kept:
            log(f"round {r}: no passing answers to train on; stopping")
            break
        sampler.sleep()
        finetune(by_id, kept, r)
        latest = f"{rd}/checkpoint"
    log("done")


if __name__ == "__main__":
    main()
