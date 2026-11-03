"""The GPUs that agent runs (driver.py --gpu) and evaluations (evaluate.py run --gpu) can use on the server.

5090 (32 GB, nvidia-smi index 1): shared with llama-swap, which a job stops while it holds the card and starts
again afterwards. 5080 (16 GB, index 0): shared with other services, which keep running; a job there
asks vLLM for a smaller share, so that about 6-7 GB stay free for them (measured 2026-11-04: a run's GPU
server at 0.35 peaked at 7.6 GB in all). One job per GPU at a time: each GPU has its own lock file under
/opt/wsp/runs/.

run_memory is vLLM's gpu_memory_utilization in a run's GPU server (which also holds a Hugging Face copy of
the model, about 3.7 GB); eval_memory is the same in an evaluation (vLLM only).
"""
GPUS = {
    "5090": dict(device="nvidia.com/gpu=GPU-<uuid>", index=1, lock=".gpu.lock",
                 llama_swap=True, run_memory=0.62, eval_memory=0.90),
    "5080": dict(device="nvidia.com/gpu=GPU-<uuid>", index=0, lock=".gpu-5080.lock",
                 llama_swap=False, run_memory=0.45, eval_memory=0.60),
}
