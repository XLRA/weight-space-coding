"""The GPUs that agent runs (driver.py --gpu) and evaluations (evaluate.py run --gpu) can use on the server.

5090 (32 GB, nvidia-smi index 1): shared with llama-swap, which a job stops while it holds the card and starts
again afterwards. 5080 (16 GB, index 0): shared with other services, which keep running; a job there
asks vLLM for a smaller share, so that about 6-7 GB stay free for them (measured 2026-11-04: a run's GPU
server at 0.35 peaked at 7.6 GB in all). One job per GPU at a time: each GPU has its own lock file under
/opt/wsp/runs/.

run_memory is vLLM's gpu_memory_utilization in a run's GPU server (which also holds a Hugging Face copy of
the model, about 3.7 GB); eval_memory is the same in an evaluation (vLLM only).

colab (remote): a Google Colab notebook on an A100 40 GB, for agent runs only (not evaluations). The run's GPU
container is then colab_relay.py, an HTTP server on the Docker network of the server's Cloudflare tunnel
(relay_network, as relay_alias:relay_port), which colab_worker.py in the notebook polls over HTTPS through the
tunnel. start_timeout is how long the driver waits for the notebook to connect, units_per_hour the A100's price in
Colab compute units (README, "Colab runner"), and units_by_gpu the price of each GPU Colab may connect instead, by a
part of its name, as Colab's resources panel shows it (the L4 when no A100 was free, 2026-11-06).
"""
GPUS = {
    "5090": dict(device="nvidia.com/gpu=GPU-<uuid>", index=1, lock=".gpu.lock",
                 llama_swap=True, run_memory=0.62, eval_memory=0.90),
    "5080": dict(device="nvidia.com/gpu=GPU-<uuid>", index=0, lock=".gpu-5080.lock",
                 llama_swap=False, run_memory=0.45, eval_memory=0.60),
    "colab": dict(remote=True, device=None, index=None, lock=".gpu-colab.lock", llama_swap=False, run_memory=0.55,
                  eval_memory=None, start_timeout=12 * 3600, units_per_hour=5.4, units_by_gpu={"A100": 5.4, "L4": 1.94},
                  relay_network="compose_default", relay_alias="wsp-colab-relay", relay_port=8790),
}
LOCAL = sorted(n for n, g in GPUS.items() if not g.get("remote"))
