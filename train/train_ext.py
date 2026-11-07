"""vLLM worker extension for training: the harness's swap (wsp_set_weights) plus loading a whole safetensors file.

Loaded with LLM(worker_extension_cls="train_ext.TrainWorkerExtension"), with /harness and /train on PYTHONPATH.
"""
import torch
from safetensors.torch import load_file

from wsp_ext import WspWorkerExtension


class TrainWorkerExtension(WspWorkerExtension):
    def train_load_file(self, path):
        """Load every tensor in a safetensors file (checkpoint names, bf16) into the running model."""
        weights = load_file(path)
        model = self.model_runner.get_model() if hasattr(self.model_runner, "get_model") else self.model_runner.model
        model.load_weights(weights=iter(weights.items()))
        torch.cuda.synchronize()
        return len(weights)
