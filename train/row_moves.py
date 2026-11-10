"""How far training moved the weights E's edit changed (plans/training-phase.md, "Do the edits survive training?").
On the host, numpy only:
    /opt/wsp/venv/bin/python /opt/wsp/train/row_moves.py
Reads every checkpoint of every arm (train/runs/<arm>/checkpoints/step_N and round_N/checkpoint) and the weights
train/combine.py made (train/combined/<name>), and writes train/analysis/rows.csv and rows.md.

A unit is one place E changed: the down_proj column of each loop detector and of the answer-end neuron; the gate_proj
row, up_proj row and down_proj column of each repurposed layer-23 neuron; and the 256 gate rows of each name-gate
head (train/probes.py lists what each does). With O the original, ΔE = bf16(E) − O on the unit (train/combine.py),
S the arm's start (O or E) and W the model's weights:
- kept = <W − O, ΔE> / |ΔE|²: how much of the edit is there along its own direction. 1 at E; for an arm from O, how
  far training alone moved the unit toward the edit.
- moved = |W − S| / |ΔE|: training's change to the unit, against the edit's size.
- cos = cos(W − S, ΔE): negative when training pushed back against the edit.
- vs_rest = the unit's rows' (or columns') mean change |W − S|, over the median change of all the tensor's rows (or
  columns): how training treated these units against the rest of their tensor.
The groups in rows.md pool their units' vectors.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from combine import EDIT, Original, bf16  # noqa: E402

TRAIN = os.environ.get("WSP_TRAIN", "/opt/wsp/train")
OUT = f"{TRAIN}/analysis"
P = "model.language_model.layers."

UNITS = []  # (unit, tensor, axis: 0 rows / 1 columns, indices)
for layer, cols in ((10, [59, 120]), (13, [7, 16]), (14, [239]), (13, [1420])):
    UNITS += [(f"L{layer}.{c} down", f"{P}{layer}.mlp.down_proj.weight", 1, [c]) for c in cols]
for n in (3109, 2664, 1965, 2621):
    UNITS += [(f"L23.{n} gate", f"{P}23.mlp.gate_proj.weight", 0, [n]),
              (f"L23.{n} up", f"{P}23.mlp.up_proj.weight", 0, [n]),
              (f"L23.{n} down", f"{P}23.mlp.down_proj.weight", 1, [n])]
for layer, heads in ((15, [7]), (19, [0, 2, 5, 6, 7]), (23, [0, 1])):
    UNITS += [(f"L{layer}.h{h} gate rows", f"{P}{layer}.self_attn.q_proj.weight", 0,
               list(range(h * 512 + 256, h * 512 + 512))) for h in heads]
GROUPS = {"loop detectors (5 columns)": ["L10.59 down", "L10.120 down", "L13.7 down", "L13.16 down", "L14.239 down"],
          "answer end (L13 1420)": ["L13.1420 down"],
          "brake (L23 3109)": ["L23.3109 gate", "L23.3109 up", "L23.3109 down"],
          "cycle stop (L23 2664)": ["L23.2664 gate", "L23.2664 up", "L23.2664 down"],
          "box pair (L23 1965, 2621)": [f"L23.{n} {p}" for n in (1965, 2621) for p in ("gate", "up", "down")],
          "name gate L15 (1 head)": ["L15.h7 gate rows"],
          "name gates L19 (5 heads)": [f"L19.h{h} gate rows" for h in (0, 2, 5, 6, 7)],
          "name gates L23 (2 heads)": ["L23.h0 gate rows", "L23.h1 gate rows"]}
ARM = re.compile(r"^[RG]-[OE]\d*$")


def models():
    """(name, folder, start) for every checkpoint and every combined model; start None for combined ones."""
    out = []
    for arm in sorted(os.listdir(f"{TRAIN}/runs")):
        if not ARM.match(arm):
            continue
        start = arm.split("-")[1].rstrip("0123456789")
        steps = glob.glob(f"{TRAIN}/runs/{arm}/checkpoints/step_*")
        for c in sorted(steps, key=lambda p: int(p.rsplit("_", 1)[1])):
            out.append((f"{arm}/{os.path.basename(c)}", c, start))
        for c in sorted(glob.glob(f"{TRAIN}/runs/{arm}/round_*/checkpoint")):
            out.append((f"{arm}/{c.split('/')[-2]}", c, start))
    for c in sorted(glob.glob(f"{TRAIN}/combined/*")):
        if os.path.isdir(c) and not c.endswith(".tmp"):
            out.append((os.path.basename(c), c, None))
    return out


def take(W, axis, idx):
    return W[idx, :] if axis == 0 else W[:, idx].T


def main():
    orig = Original()
    O = {t: orig.get(t).astype(np.float64) for t in {u[1] for u in UNITS}}
    E = {t: bf16(np.load(f"{EDIT}/{t}.npy").astype(np.float64)) for t in O}
    rows, pooled = [], {}
    for name, folder, start in [("E", None, "O")] + models():
        W = E if folder is None else {t: np.load(f"{folder}/{t}.npy").astype(np.float64) for t in O}
        S = None if start is None else (O if start == "O" else E)
        acc = {}
        for unit, t, axis, idx in UNITS:
            dE = take(E[t] - O[t], axis, idx)
            dM = take(W[t] - O[t], axis, idx)
            r = {"model": name, "start": start or "", "unit": unit, "kept": (dM * dE).sum() / (dE * dE).sum(),
                 "moved": np.nan, "cos": np.nan, "vs_rest": np.nan}
            if S is not None:
                D = W[t] - S[t]
                dS = take(D, axis, idx)
                norms = np.linalg.norm(D, axis=1 - axis)
                med = np.median(norms)
                r["moved"] = np.linalg.norm(dS) / np.linalg.norm(dE)
                r["cos"] = (dS * dE).sum() / (np.linalg.norm(dS) * np.linalg.norm(dE)) if dS.any() else np.nan
                r["vs_rest"] = norms[idx].mean() / med if med > 0 else np.nan
                acc[unit] = ((dM * dE).sum(), (dE * dE).sum(), (dS * dS).sum(), (dS * dE).sum())
            else:
                acc[unit] = ((dM * dE).sum(), (dE * dE).sum(), np.nan, np.nan)
            rows.append(r)
        for g, units in GROUPS.items():
            me, ee, ss, se = (sum(acc[u][i] for u in units) for i in range(4))
            pooled[(name, g)] = {"kept": me / ee, "moved": np.sqrt(ss / ee), "cos": se / np.sqrt(ss * ee) if ss else np.nan}
        print(f"{name}: " + ", ".join(f"{g.split(' (')[0]} {pooled[(name, g)]['kept']:.2f}" for g in GROUPS), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/rows.csv", "w") as f:
        f.write("model,start,unit,kept,moved,cos,vs_rest\n")
        for r in rows:
            f.write(f"{r['model']},{r['start']},{r['unit']},{r['kept']:.4f},{r['moved']:.4f},{r['cos']:.4f},"
                    f"{r['vs_rest']:.3f}\n")
    names = list(dict.fromkeys(r["model"] for r in rows))
    lines = ["# How far training moved the weights E's edit changed (train/row_moves.py)", ""]
    for key, what in (("kept", "kept = <W − O, ΔE> / |ΔE|²: the share of the edit still there along its direction"),
                      ("moved", "moved = |W − start| / |ΔE|: training's change against the edit's size"),
                      ("cos", "cos(W − start, ΔE): negative when training pushed back against the edit")):
        lines += [f"{what}:", "", "| Group | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        fmt = {"kept": "{:.4f}", "moved": "{:.3%}", "cos": "{:+.2f}"}[key]
        for g in GROUPS:
            lines.append(f"| {g} | " + " | ".join("" if np.isnan(pooled[(n, g)][key]) else fmt.format(pooled[(n, g)][key])
                                                for n in names) + " |")
        lines.append("")
    lines += ["vs_rest, per unit: the unit's change over the median change of its tensor's rows or columns:", "",
              "| Unit | " + " | ".join(n for n in names if n != "E") + " |", "|---|" + "---|" * (len(names) - 1)]
    for unit, *_ in UNITS:
        vals = {r["model"]: r["vs_rest"] for r in rows if r["unit"] == unit}
        lines.append(f"| {unit} | " + " | ".join("" if np.isnan(vals[n]) else f"{vals[n]:.2f}" for n in names if n != "E") + " |")
    open(f"{OUT}/rows.md", "w").write("\n".join(lines) + "\n")
    print(f"{len(names)} models: {OUT}/rows.csv, rows.md")


if __name__ == "__main__":
    main()
