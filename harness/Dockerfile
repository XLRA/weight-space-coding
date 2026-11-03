# Runtime for the editing kernel (CPU) and the forward-pass server (GPU).
# The vLLM image already has torch, transformers, safetensors, tokenizers, numpy, sympy and networkx.
FROM vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
RUN pip install --no-cache-dir pandas scipy scikit-learn && pip freeze > /opt/wsp-runtime-freeze.txt
ENTRYPOINT []
