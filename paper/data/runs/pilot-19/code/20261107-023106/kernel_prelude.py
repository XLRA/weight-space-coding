"""Names defined in the editing kernel before the agent's first code block.

NAMES, load(name), load_original(name), save_edit(name, tensor), drop_edit(name), edited(),
tok (the tokenizer) and run_tests(code, problem_id). Gradients are disabled.
"""
import json as _json
import os as _os
import sys as _sys

import numpy as _np
import torch as _torch
from safetensors import safe_open as _safe_open
from transformers import AutoTokenizer as _AutoTokenizer

_torch.set_grad_enabled(False)
_torch.set_num_threads(int(_os.environ.get("OMP_NUM_THREADS", "8")))

_MODEL = "/model"
_EDIT_DIR = "/work/edited"
_os.makedirs(_EDIT_DIR, exist_ok=True)
_INDEX = _json.load(open(f"{_MODEL}/model.safetensors.index.json"))["weight_map"]
NAMES = sorted(n for n in _INDEX if n.startswith("model.language_model."))
_REFUSED = {
    "model.language_model.norm.weight": "the final norm is out of scope",
    "model.language_model.embed_tokens.weight": "the input embedding / output layer (tied) is out of scope",
}
_files = {}


def _file(name):
    path = f"{_MODEL}/{_INDEX[name]}"
    if path not in _files:
        _files[path] = _safe_open(path, "pt")
    return _files[path]


def _check_name(name):
    if name not in NAMES:
        raise KeyError(f"{name!r} is not a language-model tensor; see NAMES")


def load_original(name):
    """The unedited tensor, as float32."""
    _check_name(name)
    return _file(name).get_tensor(name).float()


def load(name):
    """Your current edit of the tensor if there is one, else the original (float32)."""
    _check_name(name)
    path = f"{_EDIT_DIR}/{name}.npy"
    if _os.path.exists(path):
        return _torch.from_numpy(_np.load(path, allow_pickle=False))
    return load_original(name)


def _write_manifest():
    entries = {}
    for fn in sorted(_os.listdir(_EDIT_DIR)):
        if fn.endswith(".npy"):
            name = fn[:-4]
            new = _np.load(f"{_EDIT_DIR}/{fn}", allow_pickle=False)
            old = load_original(name).numpy()
            entries[name] = {"shape": list(new.shape), "changed_entries": int((new != old).sum())}
    with open(f"{_EDIT_DIR}/manifest.json", "w") as f:
        _json.dump(entries, f, indent=1)


def save_edit(name, tensor):
    """Replace a tensor's values (same shape). Refused for out-of-scope tensors and whole-tensor scalings."""
    _check_name(name)
    if name in _REFUSED:
        raise PermissionError(f"save_edit refused: {_REFUSED[name]}")
    new = _torch.as_tensor(tensor).detach().to("cpu", _torch.float32)
    old = load_original(name)
    if tuple(new.shape) != tuple(old.shape):
        raise ValueError(f"shape {tuple(new.shape)} does not match {tuple(old.shape)}")
    if not _torch.isfinite(new).all():
        raise ValueError("the tensor has non-finite values")
    if _torch.equal(new, old):
        drop_edit(name)
        return f"{name}: identical to the original, so no edit is stored"
    nz = old != 0
    if nz.any() and _torch.equal(new[~nz], old[~nz]):
        ratio = new[nz] / old[nz]
        if (ratio - ratio.mean()).abs().max() <= 1e-6 * max(ratio.mean().abs().item(), 1e-12):
            raise PermissionError("save_edit refused: the edit only multiplies the whole tensor by a constant")
    _np.save(f"{_EDIT_DIR}/{name}.npy", new.numpy())
    _write_manifest()
    changed = int((new != old).sum())
    return f"{name}: saved ({changed:,} of {old.numel():,} entries differ from the original)"


def drop_edit(name):
    """Remove your edit of a tensor (it goes back to the original)."""
    path = f"{_EDIT_DIR}/{name}.npy"
    if _os.path.exists(path):
        _os.remove(path)
    _write_manifest()


def edited():
    """Names of the tensors you have edited."""
    return sorted(fn[:-4] for fn in _os.listdir(_EDIT_DIR) if fn.endswith(".npy"))


tok = _AutoTokenizer.from_pretrained(_MODEL)

_sys.path.insert(0, "/harness")
import grade as _grade  # noqa: E402

_problems = None


def run_tests(code, problem_id):
    """Run a piece of code (yours or one the model wrote) against a practice problem's tests.

    This runs the code, not the model. Returns {"status": pass | wrong_answer | runtime_error |
    timeout | syntax_error | wrong_interface, "exc": exception type for runtime errors}.
    """
    global _problems
    if _problems is None:
        _problems = {}
        for fn in sorted(_os.listdir("/practice")):
            if fn.endswith(".jsonl"):
                for line in open(f"/practice/{fn}"):
                    p = _json.loads(line)
                    _problems[p["id"]] = p
    return _grade.grade_code(_problems[problem_id], code)
