"""Names defined in the editing kernel before the agent's first code block.

NAMES, load(name), load_original(name), save_edit(name, tensor), drop_edit(name), edited(),
restore_start(name), changed(), tok (the tokenizer) and run_tests(code, problem_id). Gradients are disabled.
"""
import json as _json
import os as _os
import shutil as _shutil
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
_START_DIR = "/start"  # the run's starting edit (driver.py --start-from), read-only; empty without one
_HAS_START = _os.path.isdir(_START_DIR) and any(fn.endswith(".npy") for fn in _os.listdir(_START_DIR))
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


def _start(name):
    """The starting edit's tensor (float32), or None if the start does not change it."""
    path = f"{_START_DIR}/{name}.npy"
    return _torch.from_numpy(_np.load(path, allow_pickle=False)) if _os.path.exists(path) else None


_manifest_cache = {}  # file name -> ((mtime, size), entry); save_edit, drop_edit and restore_start drop theirs


def _write_manifest():
    entries = {}
    for fn in sorted(_os.listdir(_EDIT_DIR)):
        if fn.endswith(".npy"):
            name = fn[:-4]
            st = _os.stat(f"{_EDIT_DIR}/{fn}")
            key = (st.st_mtime_ns, st.st_size)
            if fn not in _manifest_cache or _manifest_cache[fn][0] != key:
                new = _np.load(f"{_EDIT_DIR}/{fn}", allow_pickle=False)
                old = load_original(name).numpy()
                entry = {"shape": list(new.shape), "changed_entries": int((new != old).sum())}
                if _HAS_START:
                    start = _start(name)
                    entry["changed_from_start"] = int((new != (old if start is None else start.numpy())).sum())
                _manifest_cache[fn] = (key, entry)
            entries[name] = _manifest_cache[fn][1]
    with open(f"{_EDIT_DIR}/manifest.json", "w") as f:
        _json.dump(entries, f, indent=1)


def _scaled(new, base):
    """Whether new is base multiplied by one constant (its zeros kept)."""
    nz = base != 0
    if nz.any() and _torch.equal(new[~nz], base[~nz]):
        ratio = new[nz] / base[nz]
        return bool((ratio - ratio.mean()).abs().max() <= 1e-6 * max(ratio.mean().abs().item(), 1e-12))
    return False


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
    if _scaled(new, old):
        raise PermissionError("save_edit refused: the edit only multiplies the whole tensor by a constant")
    start = _start(name)
    if start is not None and not _torch.equal(new, start) and _scaled(new, start):
        raise PermissionError("save_edit refused: the edit only multiplies the starting edit's whole tensor by a "
                              "constant")
    _np.save(f"{_EDIT_DIR}/{name}.npy", new.numpy())
    _manifest_cache.pop(f"{name}.npy", None)
    _write_manifest()
    changed = int((new != old).sum())
    return f"{name}: saved ({changed:,} of {old.numel():,} entries differ from the original)"


def drop_edit(name):
    """Remove your edit of a tensor (it goes back to the original, also where the run's start had changed it;
    restore_start(name) goes back to the start)."""
    path = f"{_EDIT_DIR}/{name}.npy"
    if _os.path.exists(path):
        _os.remove(path)
    _manifest_cache.pop(f"{name}.npy", None)
    _write_manifest()


def restore_start(name):
    """Put back the starting edit's tensor (/start/<name>.npy), undoing your changes to it; the original if the
    start does not change it."""
    _check_name(name)
    if not _os.path.exists(f"{_START_DIR}/{name}.npy"):
        drop_edit(name)
        return f"{name}: back to the original, which the start does not change"
    _shutil.copyfile(f"{_START_DIR}/{name}.npy", f"{_EDIT_DIR}/{name}.npy")
    _manifest_cache.pop(f"{name}.npy", None)
    _write_manifest()
    return f"{name}: back to the starting edit's values"


def edited():
    """Names of the tensors you have edited (in a run with a start, the start's tensors too)."""
    return sorted(fn[:-4] for fn in _os.listdir(_EDIT_DIR) if fn.endswith(".npy"))


def changed():
    """Names of the tensors whose current values differ from the run's start (from the original if it has none),
    including the start's tensors that drop_edit put back to the original."""
    _write_manifest()
    key = "changed_from_start" if _HAS_START else "changed_entries"
    entries = _json.load(open(f"{_EDIT_DIR}/manifest.json"))
    dropped = [fn[:-4] for fn in _os.listdir(_START_DIR) if fn.endswith(".npy") and fn[:-4] not in entries] \
        if _HAS_START else []
    return sorted([n for n, e in entries.items() if e[key]] + dropped)


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
