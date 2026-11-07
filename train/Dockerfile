# wsp-train:1, the training phase's image (plans/training-phase.md, "Infrastructure").
# wsp-runtime:1 (torch 2.13, transformers 5.19, vLLM 0.32) plus the training libraries.
#   docker build -t wsp-train:1 -f /opt/wsp/train/Dockerfile /opt/wsp/train
# - Pinned: the packages that decide the model's numerics and its sampling stay at the base image's versions (torch,
#   transformers, vLLM, triton, numpy, tokenizers, safetensors, flashinfer, xformers), through a constraints file
#   made from the base image. Other base packages may move. That is safe because the evaluation and the harness keep
#   using wsp-runtime:1; this image is only for training.
# - flash-linear-attention gives transformers the fused Gated DeltaNet kernels for Qwen3.5's 18 linear-attention
#   layers; without it they fall back to a reference PyTorch path that transformers itself warns is much slower.
# - The resolved versions are written to /opt/train-freeze.txt and copied to train/freeze.txt after the build.
FROM wsp-runtime:1
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1
RUN pip list --format=freeze | grep -i -E "^(torch|torchvision|torchaudio|transformers|vllm|triton|numpy|tokenizers|safetensors|flashinfer[a-z_-]*|xformers)==" > /opt/base-constraints.txt \
 && cat /opt/base-constraints.txt \
 && pip install -c /opt/base-constraints.txt trl liger-kernel flash-linear-attention \
 && pip list --format=freeze > /opt/train-freeze.txt \
 && python3 -c "import trl, fla, liger_kernel, torch, transformers, vllm; print('trl', trl.__version__, 'fla', fla.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__)"
