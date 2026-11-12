"""vLLM worker extension: swap individual weight tensors in place, from the checkpoint or an edit file.

Loaded into vLLM's worker process via LLM(worker_extension_cls="wsp_ext.WspWorkerExtension").
Tensors are cast to bfloat16 (the model's dtype), as the evaluation does.
"""
import json

import numpy as np
import torch
from safetensors import safe_open

MODEL = "/model"


class WspWorkerExtension:
    def wsp_set_weights(self, pairs):
        """pairs: list of (checkpoint name, path to a .npy edit or None for the original)."""
        if not hasattr(self, "_wsp_index"):
            self._wsp_index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
            self._wsp_files = {}
        weights = []
        for name, path in pairs:
            if path is None:
                fn = f"{MODEL}/{self._wsp_index[name]}"
                if fn not in self._wsp_files:
                    self._wsp_files[fn] = safe_open(fn, "pt")
                t = self._wsp_files[fn].get_tensor(name)
            else:
                t = torch.from_numpy(np.load(path, allow_pickle=False))
            weights.append((name, t.to(torch.bfloat16)))
        model = self.model_runner.get_model() if hasattr(self.model_runner, "get_model") else self.model_runner.model
        model.load_weights(weights=iter(weights))
        torch.cuda.synchronize()
        return len(weights)
