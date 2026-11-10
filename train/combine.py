"""Adds the edit's delta ΔE to a trained checkpoint, or takes it away (plans/training-phase.md, "Measurement";
Deviations, 2026-11-10). On the host, numpy only:
    /opt/wsp/venv/bin/python train/combine.py <out folder> <checkpoint folder> <+|-> [<edit folder>]
e.g. combine.py /opt/wsp/train/combined/T-step_100 /opt/wsp/train/runs/G-O/checkpoints/step_100 +
- T = G-O's chosen checkpoint + ΔE (portability, question 4).
- G-E2's chosen checkpoint − ΔE (removal, question 3).
ΔE is E as the model runs it, minus O. The evaluation and the trainer both load a submission's fp32 values rounded to
bf16 (harness/gpu_server.py; train/common.py, start_weights), so ΔE = bf16(E) − O on each tensor E's submission holds
(pilot-14's 9), with O read from the model's safetensors. The output is the checkpoint with those tensors replaced by
the bf16 value nearest to checkpoint ± ΔE (ties to even; computed exactly in float64 and rounded once), stored as fp32
like the checkpoints. Every other tensor is copied unchanged, and manifest.json says how it was made.
Checks: E's fp32 values change exactly the entries its manifest counts, rounding E to bf16 only takes entries out of
ΔE, and the checkpoint's values are bf16 values.
"""
import json
import os
import shutil
import sys

import numpy as np

MODEL = "/opt/wsp/models/qwen3.5-0.8b"
EDIT = "/opt/wsp/runs/pilot-14/submission"
TINY = 2.0 ** -126  # below this bf16 is subnormal and keeps fewer bits than bf16() below; no weight is that small


class Original:
    """The model's safetensors, read with numpy (as paper/collect.py reads them)."""

    def __init__(self):
        idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
        self.files = {k: f"{MODEL}/{v}" for k, v in idx.items()}
        self.headers = {}

    def get(self, name):
        path = self.files[name]
        if path not in self.headers:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.headers[path] = (8 + n, json.loads(f.read(n)))
        base, head = self.headers[path]
        meta = head[name]
        if meta["dtype"] != "BF16":
            raise SystemExit(f"{name} is {meta['dtype']}, not BF16")
        a, b = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + a)
            raw = f.read(b - a)
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def bf16(x):
    """float64 values -> the nearest bf16 values (ties to even, like torch's conversion), as float64.
    bf16 keeps 8 significant bits: x = m * 2**e with 0.5 <= |m| < 1, and m * 2**8 is rounded to an integer."""
    tiny = (x != 0) & (np.abs(x) < TINY)
    if tiny.any():
        raise SystemExit(f"{int(tiny.sum())} values below 2**-126, where bf16 is subnormal")
    m, e = np.frexp(x)
    return np.ldexp(np.round(np.ldexp(m, 8)), e - 8)


def is_bf16(x32):
    return bool(((x32.view(np.uint32) & 0xFFFF) == 0).all())


def main():
    out, ckpt, sign = sys.argv[1:4]
    edit = sys.argv[4] if len(sys.argv) > 4 else EDIT
    if sign not in ("+", "-"):
        raise SystemExit("the sign is + or -")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    orig = Original()
    edit_manifest = json.load(open(f"{edit}/manifest.json"))
    names = sorted(edit_manifest)
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in sorted(os.listdir(ckpt)):
        if f.endswith(".npy") and f[:-4] not in names:
            shutil.copy2(f"{ckpt}/{f}", f"{tmp}/{f}")
    stats = {}
    for n in names:
        o = orig.get(n).astype(np.float64)
        e32 = np.load(f"{edit}/{n}.npy", allow_pickle=False)
        c32 = np.load(f"{ckpt}/{n}.npy", allow_pickle=False)
        if not (e32.shape == o.shape == c32.shape and e32.dtype == c32.dtype == np.float32):
            raise SystemExit(f"{n}: {e32.shape} {e32.dtype}, {o.shape}, {c32.shape} {c32.dtype}")
        if not is_bf16(c32):
            raise SystemExit(f"{n}: the checkpoint's values are not bf16 values")
        e = e32.astype(np.float64)
        edit_changed = int((e != o).sum())
        if edit_changed != edit_manifest[n]["changed_entries"]:
            raise SystemExit(f"{n}: E changes {edit_changed} entries, its manifest says {edit_manifest[n]['changed_entries']}")
        d = bf16(e) - o  # exact: both are bf16 values
        if ((d != 0) & (e == o)).any():
            raise SystemExit(f"{n}: rounding E to bf16 changed an entry E leaves alone")
        c = c32.astype(np.float64)
        t = bf16(c + d if sign == "+" else c - d)  # c ± d is exact in float64, so this rounds once
        t32 = t.astype(np.float32)
        if not (t32.astype(np.float64) == t).all():
            raise SystemExit(f"{n}: a bf16 value didn't fit fp32")
        np.save(f"{tmp}/{n}.npy", t32)
        stats[n] = {"edit_changed_entries": edit_changed, "delta_changed_entries": int((d != 0).sum()),
                    "edit_entries_not_bf16": int((bf16(e) != e).sum()), "delta_norm": float(np.linalg.norm(d)),
                    "delta_max_abs": float(np.abs(d).max()), "checkpoint_norm": float(np.linalg.norm(c)),
                    "entries_differing_from_checkpoint": int((t32 != c32).sum())}
        s = stats[n]
        print(f"{n}: E changes {edit_changed} entries, {s['delta_changed_entries']} after rounding to bf16; "
              f"ΔE norm {s['delta_norm']:.4f}; {s['entries_differing_from_checkpoint']} entries differ from the checkpoint",
              flush=True)
    copied = len([f for f in os.listdir(tmp) if f.endswith(".npy")]) - len(names)
    manifest = {"made_by": "train/combine.py", "checkpoint": ckpt, "sign": sign, "edit": edit, "model": MODEL,
                "what": f"checkpoint {sign} ΔE, ΔE = bf16(E) - O on E's {len(names)} tensors; each entry is the bf16 "
                        "value nearest the exact result (ties to even)",
                "checkpoint_manifest": json.load(open(f"{ckpt}/manifest.json")),
                "tensors": "all 24 blocks, bf16 values as fp32", "copied_unchanged": copied, "delta_tensors": stats}
    with open(f"{tmp}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    os.rename(tmp, out)
    print(f"{out}: {copied} tensors copied, {len(names)} combined")


if __name__ == "__main__":
    main()
