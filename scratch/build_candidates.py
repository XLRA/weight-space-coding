"""Builds a consolidation run's candidate folders (round 5, README "Round 5").

    docker run --rm -v /opt/wsp/models/qwen3.5-0.8b:/model:ro -v <start submission>:/start:ro -v /opt/wsp/runs:/runs:ro \
        -v /opt/wsp/scratch:/scratch:ro -v <run>/work/candidates:/out wsp-runtime:1 \
        python3 /scratch/build_candidates.py stopper=/runs/pilot-16/submission copy=/runs/pilot-17/submission ...

For each label=submission: /out/<label>/ gets the submission's tensors that differ from the starting edit (or from the
original where the starting edit has no such tensor), and report.md (the run's final_answer.md). /out/sum/ gets the
starting edit plus all the changes added (each change = candidate - start, in float32). Prints, per tensor, the rows
each candidate changes and the rows two candidates both change.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
from safetensors import safe_open

MODEL, START, OUT = "/model", "/start", "/out"
INDEX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]


def original(name):
    with safe_open(f"{MODEL}/{INDEX[name]}", framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def base(name):
    p = f"{START}/{name}.npy"
    return np.load(p).astype(np.float32) if os.path.exists(p) else original(name)


def main():
    total, rows = {}, {}
    for spec in sys.argv[1:]:
        label, sub = spec.split("=", 1)
        os.makedirs(f"{OUT}/{label}", exist_ok=True)
        for p in sorted(glob.glob(f"{sub}/*.npy")):
            name = os.path.basename(p)[:-4]
            b = base(name)
            d = np.load(p).astype(np.float32) - b
            if not np.any(d):
                continue
            shutil.copy(p, f"{OUT}/{label}/{name}.npy")
            total.setdefault(name, b.copy())
            total[name] += d
            r = set(np.nonzero(np.any(d != 0, axis=1))[0].tolist())
            rows.setdefault(name, {})[label] = r
            print(f"{label}: {name.replace('model.language_model.', '')}: {np.count_nonzero(d):,} entries in {len(r)} "
                  f"rows, max |change| {np.abs(d).max():.4g}, change norm / tensor norm "
                  f"{np.linalg.norm(d) / np.linalg.norm(b):.4f}")
        report = os.path.join(os.path.dirname(sub.rstrip("/")), "final_answer.md")
        if os.path.exists(report):
            shutil.copy(report, f"{OUT}/{label}/report.md")
    os.makedirs(f"{OUT}/sum", exist_ok=True)
    for name, t in total.items():
        np.save(f"{OUT}/sum/{name}.npy", t)
    for name, by in rows.items():
        labels = sorted(by)
        for i, a in enumerate(labels):
            for c in labels[i + 1:]:
                both = by[a] & by[c]
                print(f"overlap {a} & {c} in {name.replace('model.language_model.', '')}: {len(both)} rows"
                      + (f" ({sorted(both)[:8]}...)" if both else ""))
    print(f"sum: {len(total)} tensors")


if __name__ == "__main__":
    main()
