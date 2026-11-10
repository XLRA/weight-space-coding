"""Activation probes (plans/training-phase.md, "Do the edits survive training?"): after training, do E's edited
units still fire where they are meant to, and only there?

    python3 /train/probes.py build                  fixes the texts and their labels, once (wsp-train:1, no GPU)
    python3 /train/probes.py run <name> <weights>   one model's activations (wsp-train:1 on a GPU: run-probes.sh)
    python3 probes.py summary                       the tables over every model run so far (numpy only)

<weights> is a folder of .npy tensors laid over the original (a submission, a checkpoint, or combine.py's output),
or "original". Everything goes to train/analysis/probes/: texts.jsonl and index.json (build), <name>.json and
<name>.npz (run, "/" in the name written as "-"), summary.md and summary.csv (summary).

The texts are answers that O and E themselves wrote at their starting checks (train/evals/start-O, start-E), to the
validation problems and the panel's questions, never to held-out ones. About 200 are drawn once with a fixed seed,
so that each kind of position below is well represented. Each is the problem's prompt plus the answer, re-tokenized
and teacher-forced, and the units are read at every answer position.

The units are where pilot-14's edit E changed the weights (ΔE = bf16(E) − O, train/combine.py), with what the agent
built each to do (runs/pilot-10, -12, -13 and -14, final_answer.md):
- five loop detectors, whose down_proj columns pilot-10 wired to the end of the thinking: layer 10 neurons 59 and
  120, layer 13 neurons 7 and 16, layer 14 neuron 239; and their combined drive, each over its typical in-loop
  activation (pilot-10's a_n), averaged, so about 1 in a loop;
- pilot-10's answer-end neuron, layer 13 neuron 1420;
- the layer-23 neurons pilot-12 and pilot-14 repurposed: 3109 (the end-of-turn brake, meant to fire where the
  detectors do), 2664 (ends degenerate re-close cycles), and the pair 1965 + 2621 (boxes an answer statement; the two
  activations are summed, since their down columns are the same);
- pilot-13's name gates, rank-1 changes to the q_proj gate rows of layer 15 head 7, layer 19 heads 0, 2, 5, 6 and 7,
  and layer 23 heads 0 and 1: the detector each reads (ΔE's gate row over its strength, applied to the model's own
  q_proj input), and each head's gate (the mean sigmoid of its 256 gate logits).
An MLP unit's activation is its down_proj input, act(gate)·up, computed with the model's own rows.

Where each unit is meant to fire, its label (answer positions; the control is every other answer position unless
said otherwise):
- rep (the detectors, the drive, 3109): verbatim repetition: the 32 tokens up to the position appeared earlier in
  the answer. Control: positions whose 8 tokens up to them are new. Positions in between are left out.
- fence (1420): the answer's closing code fence where the turn ends. Other fences are left out.
- degen (2664): where 2664 ends a degenerate re-close cycle: the last position of a finished answer, kept whole,
  that ends in the degenerate state (the thinking has closed at least twice, and the segment before the latest
  close is under 60 tokens). The state's other positions are left out: the texts are E's own answers, so the
  re-closes left in them are where 2664 did not end the turn.
- statement (the pair): in a math answer, after the thinking closed, the position after "answer is" or "Answer:".
  The same words elsewhere (in the thinking, in multiple-choice, code or prose answers) are controls, also
  reported on their own.
- name (the layer-19 and layer-23 gates): pilot-13's positives for those detectors, the def position (the next
  token is the name's first piece) and word-start positions inside the name (the next piece begins with "_" or a
  capital letter). name-inner (the layer-15 gate): the word-start positions only. Mid-word positions and the name's
  last token, which pilot-13 fitted the detectors against, are controls.
Per model and unit: the mean activation on the label and on its control, and the AUROC between them, oriented so
that E's design fires high (detectors whose typical in-loop activation is negative are negated). The four layer-23
neurons are switches, silu of a thresholded readout. Their AUROC is taken on the gate's pre-activation (for the pair,
1965's), since silu dips below zero just under the threshold and near-misses would rank below positions where the
neuron is off; on_target and on_control give the share of positions where the activation (for the pair, the sum)
is above 0.5. The degen and name labels and these measures were set on 2026-11-10, after the sanity check on E
showed that the first ones didn't match the agent's designs (plans/training-phase.md, Deviations). On the label each
text counts once: a text's positions share a weight of 1, so the few degenerate answers with thousands of labelled
positions (fixed before any model was probed: 5 texts hold 65% of the degen and statement positions) don't
outweigh the rest. The control is counted by position. The unweighted AUROC is kept as auroc_positions.
"""
import bisect
import glob
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np

TRAIN = os.environ.get("WSP_TRAIN", "/train")
OUT = f"{TRAIN}/analysis/probes"
TEXTS = f"{OUT}/texts.jsonl"
PANEL = "/panel/panel.jsonl"
EDIT = "/edit"  # pilot-14's submission, mounted by run-probes.sh
SEED = 20261110
MAX_ANSWER = 12288  # answer tokens kept per text
LOOP_START = 4000  # a loop text's first repeat must come by this answer token
LOOP_KEEP = 1500  # answer tokens kept after it
QUOTA = {"loop": 40, "prose": 40, "code-E": 20, "code-O": 20, "math": 40, "reclose": 40, "choice": 20}

# pilot-10's typical in-loop activation a_n of each detector
DETECTORS = {"L10.59": (10, 59, -0.453), "L10.120": (10, 120, -0.619), "L13.7": (13, 7, -0.777),
             "L13.16": (13, 16, 1.085), "L14.239": (14, 239, -0.761)}
NEURONS = {**{u: (layer, n) for u, (layer, n, _) in DETECTORS.items()}, "L13.1420": (13, 1420),
           "L23.3109": (23, 3109), "L23.2664": (23, 2664), "L23.1965": (23, 1965), "L23.2621": (23, 2621)}
GATES = {"L15": ([7], 3.0), "L19": ([0, 2, 5, 6, 7], 2.0), "L23": ([0, 1], 3.0)}  # q_proj: (heads, strength)
UNITS = {**{u: ("rep", float(np.sign(a))) for u, (_, _, a) in DETECTORS.items()}, "drive": ("rep", 1.0),
         "L13.1420": ("fence", -1.0), "L23.3109": ("rep", 1.0), "L23.2664": ("degen", 1.0),
         "L23.pair": ("statement", 1.0),
         **{f"{g}.detector": ("name-inner" if g == "L15" else "name", 1.0) for g in GATES},
         **{f"{g}.h{h}.gate": ("name-inner" if g == "L15" else "name", 1.0)
            for g, (hs, _) in GATES.items() for h in hs}}  # unit: (label, sign)
LABELS = ["rep", "fence", "degen", "statement", "name", "name-inner"]
PRE = {"L23.3109.g": 3109, "L23.2664.g": 2664, "L23.1965.g": 1965, "L23.2621.g": 2621}  # layer-23 gate pre-activations
SWITCHES = {"L23.3109": "L23.3109.g", "L23.2664": "L23.2664.g", "L23.pair": "L23.1965.g"}  # unit: what its AUROC ranks
ON = 0.5  # a switch is on where its activation is above this
STATEMENT = re.compile(r"(?i)\banswer(?:\s+is\b|\s*:)")
DEF = re.compile(r"\bdef\s+([A-Za-z_]\w*)\s*\(")


def columns():
    """The captured columns, in capture order: the neurons layer by layer, the layer-23 gate pre-activations, the gate
    detectors, the head gates."""
    cols = [u for layer in sorted({v[0] for v in NEURONS.values()}) for u, (ll, _) in NEURONS.items() if ll == layer]
    return (cols + list(PRE) + [f"{g}.detector" for g in GATES]
            + [f"{g}.h{h}.gate" for g, (hs, _) in GATES.items() for h in hs])


# ---------------------------------------------------------------------------------------------------------- build
def label(a, off, s, suite, close_id):
    """Answer-relative positions: {label: (targets, left out)}, and the statement controls."""
    n = len(a)
    ends = [o[1] for o in off]
    starts = [o[0] for o in off]

    def tok_at(ch):  # the token holding character ch
        return bisect.bisect_right(ends, ch)

    lab = {}
    seen8, seen32, tgt, out = set(), set(), [], []
    for i in range(n):
        g8 = tuple(a[i - 7:i + 1]) if i >= 7 else None
        g32 = tuple(a[i - 31:i + 1]) if i >= 31 else None
        if g32 is not None and g32 in seen32:
            tgt.append(i)
        elif g8 is None or g8 in seen8:
            out.append(i)
        if g8 is not None:
            seen8.add(g8)
        if g32 is not None:
            seen32.add(g32)
    lab["rep"] = (tgt, out)

    first_close = s.find("</think>")
    fences = [m.start() for m in re.finditer("```", s)]
    after = [f for f in fences if first_close >= 0 and f > first_close]
    tgt = []
    if after and len(after) % 2 == 0 and not s[after[-1] + 3:].strip():
        tgt = [tok_at(after[-1] + 2)]
    out = sorted({tok_at(f + k) for f in fences for k in range(3)} - set(tgt))
    lab["fence"] = (tgt, out)

    closes = [i for i, t in enumerate(a) if t == close_id]
    tgt = []
    for k in range(1, len(closes)):
        if closes[k] - closes[k - 1] < 60:
            tgt.extend(range(closes[k], closes[k + 1] if k + 1 < len(closes) else n))
    lab["degen"] = (tgt, [])  # the state; record() makes the label, which needs the finish and the cut

    tgt, neg = [], []
    for m in STATEMENT.finditer(s):
        i = tok_at(m.end() - 1)
        (tgt if suite == "math" and 0 <= first_close < m.start() else neg).append(i)
    lab["statement"] = (tgt, [])

    tgt, defs, word = set(), set(), set()
    for m in DEF.finditer(s):
        s0, e0 = m.span(1)
        j = first_piece = tok_at(s0)
        while j < n and starts[j] < e0:
            if j > 0:
                tgt.add(j - 1)
                if j == first_piece:
                    defs.add(j - 1)
                elif s[starts[j]] == "_" or s[starts[j]].isupper():
                    word.add(j - 1)
            j += 1
    lab["name"] = (sorted(tgt), [])  # every name position; record() makes the labels
    lab["name-def"] = (sorted(defs), [])
    lab["name-word"] = (sorted(word), [])
    return lab, neg


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def build():
    sys.path.insert(0, TRAIN)
    import common
    tok = common.tok
    close_id = tok.convert_tokens_to_ids("</think>")
    val = {p["id"]: p["problem"] for p in common.load_problems(f"{TRAIN}/data/val.jsonl")}
    panel = {p["id"]: p for p in map(json.loads, open(PANEL, encoding="utf-8"))}
    sources = {m: {"val": f"{TRAIN}/evals/start-{m}/val.raw.jsonl.gz", "panel": f"{TRAIN}/evals/start-{m}/panel.raw.jsonl"}
               for m in ("O", "E")}

    def answers(model, part):
        path = sources[model][part]
        with (gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")) as f:
            for line in f:
                r = json.loads(line)
                r["model"], r["part"] = model, part
                if part == "val":
                    r["problem"], r["suite"] = val[r["id"]], r["id"].split("/")[0]
                else:
                    r["problem"], r["suite"] = panel[r["id"]]["problem"], panel[r["id"]]["suite"]
                yield r

    def after_close(s):
        i = s.find("</think>")
        return s[i:] if i >= 0 else ""

    stop = lambda r: r["finish_reason"] == "stop"  # noqa: E731
    pools = {
        "loop": [r for r in answers("O", "val") if r["finish_reason"] == "length"],
        "prose": [r for r in answers("E", "panel") if r["suite"] == "instructions" and stop(r) and "</think>" in r["text"]],
        "code-E": [r for r in answers("E", "val") if r["suite"] == "mbpp" and stop(r) and "def " in r["text"]
                   and r["text"].rstrip().endswith("```")],
        "code-O": [r for r in answers("O", "val") if r["suite"] == "mbpp" and stop(r) and "def " in r["text"]
                   and r["text"].rstrip().endswith("```")],
        "math": [r for r in answers("E", "panel") if r["suite"] == "math" and stop(r)
                 and STATEMENT.search(after_close(r["text"]))],
        "reclose": [r for p in ("val", "panel") for r in answers("E", p) if r["text"].count("</think>") >= 2],
        "choice": [r for r in answers("E", "panel") if r["suite"] == "choice" and stop(r)
                   and STATEMENT.search(after_close(r["text"]))],
    }

    def record(r, cat):
        s = r["text"]
        enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
        a, off = enc.input_ids, enc.offset_mapping
        lab, neg = label(a, off, s, r["suite"], close_id)
        cut = MAX_ANSWER
        if cat == "loop":
            if not lab["rep"][0] or lab["rep"][0][0] > LOOP_START:
                return None
            cut = lab["rep"][0][0] + LOOP_KEEP
        lab = {k: ([i for i in t if i < cut], [i for i in o if i < cut]) for k, (t, o) in lab.items()}
        neg = [i for i in neg if i < cut]
        first = a.index(close_id) if close_id in a[:cut] else cut
        ok = {"loop": len(lab["rep"][0]) >= 200,
              "prose": sum(i > first for i in lab["rep"][0]) >= 40,
              "code-E": bool(lab["fence"][0]) and bool(lab["name"][0]),
              "code-O": bool(lab["fence"][0]) and bool(lab["name"][0]),
              "math": bool(lab["statement"][0]),
              "reclose": bool(lab["degen"][0]),
              "choice": any(i > first for i in neg)}[cat]
        if not ok:
            return None
        # The labels (2026-11-10, after the sanity check on E; the selection above is unchanged): degen is the last
        # position of a finished answer kept whole that ends in the degenerate state, the state's other positions
        # left out; name is the def and word-start positions, name-inner the word-start ones.
        state = set(lab["degen"][0])
        end = [len(a) - 1] if r["finish_reason"] == "stop" and len(a) <= cut and len(a) - 1 in state else []
        lab["degen"] = (end, sorted(state - set(end)))
        word = lab.pop("name-word")[0]
        lab["name"] = (sorted(lab.pop("name-def")[0] + word), [])
        lab["name-inner"] = (word, [])
        p = common.prompt_ids(r["problem"])
        P = len(p)
        return {"key": f"{r['model']}/{r['part']}/{r['id']}#{r['sample']}", "category": cat, "suite": r["suite"],
                "finish_reason": r["finish_reason"], "n_tokens": r["n_tokens"], "answer_tokens": len(a),
                "kept_tokens": min(cut, len(a)), "prompt_tokens": P, "ids": p + a[:cut],
                "labels": {k: {"target": [P + i for i in t], "left_out": [P + i for i in o]} for k, (t, o) in lab.items()},
                "statement_controls": [P + i for i in neg]}

    texts, used, tried = [], set(), {}
    for cat, quota in QUOTA.items():
        pool = sorted(pools[cat], key=lambda r: (r["id"], r["sample"], r["model"], r["part"]))
        random.Random(f"{SEED}-{cat}").shuffle(pool)
        got = tried[cat] = 0
        for r in pool:
            key = f"{r['model']}/{r['part']}/{r['id']}#{r['sample']}"
            if key in used:
                continue
            tried[cat] += 1
            rec = record(r, cat)
            if rec:
                texts.append(rec)
                used.add(key)
                got += 1
                if got == quota:
                    break
        print(f"{cat}: {got} of {quota} from {tried[cat]} tried ({len(pools[cat])} in the pool)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{TEXTS}.tmp", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    os.replace(f"{TEXTS}.tmp", TEXTS)
    (tg, ct), neg = masks(texts)
    index = {"made": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "seed": SEED, "texts": len(texts),
             "texts_md5": md5(TEXTS), "answer_positions": int(len(neg)),
             "categories": {c: sum(t["category"] == c for t in texts) for c in QUOTA},
             "tried": tried, "pools": {c: len(p) for c, p in pools.items()},
             # n_tokens counts a finished answer's stop token, which its text leaves out
             "retokenized_length_matches": sum(t["answer_tokens"] == t["n_tokens"] - (t["finish_reason"] == "stop")
                                               for t in texts),
             "retokenized_length_largest_gap": max(abs(t["answer_tokens"] - t["n_tokens"]) for t in texts),
             "labels": {k: {"target": int(tg[k].sum()), "control": int(ct[k].sum())} for k in LABELS},
             "statement_controls": int(neg.sum()),
             "sources": {f"start-{m}/{os.path.basename(p)}": md5(p) for m, d in sources.items() for p in d.values()}}
    json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
    print(json.dumps(index, indent=1))


def masks(texts, with_text=False):
    """Over all answer positions, text by text: ({label: target mask}, {label: control mask}), statement controls,
    and if asked the text each position belongs to."""
    n = sum(len(t["ids"]) - t["prompt_tokens"] for t in texts)
    tg = {k: np.zeros(n, bool) for k in LABELS}
    ct = {k: np.ones(n, bool) for k in LABELS}
    neg = np.zeros(n, bool)
    text_of = np.zeros(n, np.int64)
    off = 0
    for j, t in enumerate(texts):
        text_of[off:off + len(t["ids"]) - t["prompt_tokens"]] = j
        base = off - t["prompt_tokens"]
        for k in LABELS:
            i = np.array(t["labels"][k]["target"], dtype=np.int64) + base
            o = np.array(t["labels"][k]["left_out"], dtype=np.int64) + base
            tg[k][i] = True
            ct[k][i] = False
            ct[k][o] = False
        neg[np.array(t["statement_controls"], dtype=np.int64) + base] = True
        off += len(t["ids"]) - t["prompt_tokens"]
    return ((tg, ct), neg, text_of) if with_text else ((tg, ct), neg)


def text_weights(mask, text_of):
    """Weights over the mask's positions that give each text a total of 1."""
    t = text_of[mask]
    return 1.0 / np.bincount(t)[t]


def auroc_weighted(t, w, c):
    """Σ w·P(control < t) / Σ w, ties counting half: the AUROC with the targets weighted."""
    if not len(t) or not len(c):
        return float("nan")
    cs = np.sort(c)
    below = np.searchsorted(cs, t, side="left")
    equal = np.searchsorted(cs, t, side="right") - below
    return float((w * (below + 0.5 * equal)).sum() / len(cs) / w.sum())


# ------------------------------------------------------------------------------------------------------------ run
def auroc(t, c):
    """P(a target value > a control value), ties counting half."""
    if not len(t) or not len(c):
        return float("nan")
    x = np.concatenate([t, c])
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    cut = np.flatnonzero(np.diff(xs) != 0) + 1
    lo = np.concatenate([[0], cut])
    hi = np.concatenate([cut, [len(xs)]])
    ranks = np.empty(len(x))
    ranks[order] = np.repeat((lo + hi + 1) / 2, hi - lo)
    return float((ranks[:len(t)].sum() - len(t) * (len(t) + 1) / 2) / (len(t) * len(c)))


def unit_values(X, cols):
    c = {name: X[:, i].astype(np.float64) for i, name in enumerate(cols)}
    c["drive"] = sum(c[u] / a for u, (_, _, a) in DETECTORS.items()) / len(DETECTORS)
    c["L23.pair"] = c["L23.1965"] + c["L23.2621"]
    return {u: c[u] for u in UNITS}


def stats(X, cols, texts):
    (tg, ct), neg, text_of = masks(texts, with_text=True)
    V = unit_values(X, cols)
    A = {**V, **{u: X[:, cols.index(g)].astype(np.float64) for u, g in SWITCHES.items()}}  # what each AUROC ranks
    w = {k: text_weights(tg[k], text_of) for k in LABELS}
    res = {}
    for u, (lab, sign) in UNITS.items():
        v, a, t, c = V[u], A[u], tg[lab], ct[lab]
        res[u] = {"label": lab, "n_target": int(t.sum()), "n_target_texts": int(len(np.unique(text_of[t]))),
                  "n_control": int(c.sum()),
                  "mean_target": float((w[lab] * v[t]).sum() / w[lab].sum()), "mean_control": float(v[c].mean()),
                  "sd_control": float(v[c].std()),
                  "auroc_of": "gate pre-activation" if u in SWITCHES else "activation",
                  "auroc": auroc_weighted(sign * a[t], w[lab], sign * a[c]),
                  "auroc_positions": auroc(sign * a[t], sign * a[c])}
        if u in SWITCHES:
            res[u]["on_target"] = float((w[lab] * (v[t] > ON)).sum() / w[lab].sum())
            res[u]["on_control"] = float((v[c] > ON).mean())
    v, a = V["L23.pair"], A["L23.pair"]
    res["L23.pair"]["statement_controls"] = {
        "n": int(neg.sum()), "mean": float(v[neg].mean()), "on": float((v[neg] > ON).mean()),
        "auroc_vs_them": auroc_weighted(a[tg["statement"]], w["statement"], a[neg])}
    return res


def run(name, weights, host_path):
    import torch
    sys.path.insert(0, TRAIN)
    import common
    t0 = time.time()
    texts = [json.loads(line) for line in open(TEXTS, encoding="utf-8")]
    model = common.build_model(None if weights == "original" else weights, trainable=False).eval()
    layers = model.model.layers

    def delta(n):  # ΔE on one tensor, as train/combine.py takes it
        return torch.from_numpy(np.load(f"{EDIT}/{n}.npy")).to(torch.bfloat16).float() - common.original(n).float()

    by_layer = {}
    for u, (layer, i) in NEURONS.items():
        by_layer.setdefault(layer, []).append(i)
    for layer, idx in by_layer.items():  # the units are where E changed the weights
        d = delta(f"{common.PREFIX}layers.{layer}.mlp.down_proj.weight")
        assert sorted((d.abs().sum(0) > 0).nonzero()[:, 0].tolist()) == sorted(idx), layer
    dets = {}
    for g, (heads, strength) in GATES.items():
        d = delta(f"{common.PREFIX}layers.{g[1:]}.self_attn.q_proj.weight")
        rows = (d.abs().sum(1) > 0).nonzero()[:, 0]
        assert sorted(set((rows // 512).tolist())) == heads and bool((rows % 512 >= 256).all()), g
        gate_rows = torch.cat([d[h * 512 + 256:h * 512 + 512] for h in heads])
        dets[g] = (gate_rows.mean(0) / strength).cuda()

    cap = {}
    for layer, idx in by_layer.items():
        ix = torch.tensor(idx, device="cuda")

        def pre(mod, inp, layer=layer, ix=ix):
            cap[f"mlp{layer}"] = inp[0][0].index_select(-1, ix).float()
        layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    ixg = torch.tensor(list(PRE.values()), device="cuda")

    def gpost(mod, inp, out):
        cap["g23"] = out[0].index_select(-1, ixg).float()
    layers[23].mlp.gate_proj.register_forward_hook(gpost)
    for g, (heads, _) in GATES.items():
        def post(mod, inp, out, g=g, heads=heads):
            o = out[0].float()
            cap[f"det{g}"] = (inp[0][0].float() @ dets[g])[:, None]
            cap[f"gate{g}"] = torch.stack([torch.sigmoid(o[:, h * 512 + 256:h * 512 + 512]).mean(-1) for h in heads], -1)
        layers[int(g[1:])].self_attn.q_proj.register_forward_hook(post)

    mats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for t in texts:
            ids = torch.tensor(t["ids"], device="cuda")
            model.model(input_ids=ids[None], use_cache=False)
            parts = ([cap[f"mlp{layer}"] for layer in sorted(by_layer)] + [cap["g23"]]
                     + [cap[f"det{g}"] for g in GATES] + [cap[f"gate{g}"] for g in GATES])
            mats.append(torch.cat(parts, -1)[t["prompt_tokens"]:].cpu().numpy())
            cap.clear()
    X = np.concatenate(mats)
    cols = columns()
    assert X.shape[1] == len(cols)
    file = name.replace("/", "-")
    np.savez_compressed(f"{OUT}/{file}.npz.tmp.npz", X=X.astype(np.float16), columns=np.array(cols))
    os.replace(f"{OUT}/{file}.npz.tmp.npz", f"{OUT}/{file}.npz")
    out = {"name": name, "weights": host_path, "texts_md5": md5(TEXTS), "texts": len(texts),
           "answer_positions": int(len(X)), "seconds": round(time.time() - t0),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "units": stats(X, cols, texts)}
    json.dump(out, open(f"{OUT}/{file}.json.tmp", "w"), indent=1)
    os.replace(f"{OUT}/{file}.json.tmp", f"{OUT}/{file}.json")
    print(f"{name}: {len(texts)} texts, {len(X)} positions in {out['seconds']}s; AUROC "
          + ", ".join(f"{u} {r['auroc']:.2f}" for u, r in out["units"].items()), flush=True)


# -------------------------------------------------------------------------------------------------------- summary
def order(name):
    m = re.match(r"([A-Z]+(?:-[A-Z]+\d*)?)(?:/(step|round)_(\d+))?$", name)
    arm, kind, k = (m.group(1), m.group(2), int(m.group(3))) if m and m.group(2) else (name, "", 0)
    rank = {"O": 0, "E": 1, "R-O": 2, "R-E": 3, "G-O": 4, "G-E": 5, "G-E2": 6}.get(arm, 7)
    return rank, arm, k


def summary():
    runs = [json.load(open(p)) for p in glob.glob(f"{OUT}/*.json") if not p.endswith("index.json")]
    runs.sort(key=lambda r: order(r["name"]))
    names = [r["name"] for r in runs]
    with open(f"{OUT}/summary.csv", "w") as f:
        f.write("model,unit,label,n_target,n_control,mean_target,mean_control,auroc,auroc_of,on_target,on_control\n")
        for r in runs:
            for u, s in r["units"].items():
                on = f"{s['on_target']:.4f},{s['on_control']:.6f}" if "on_target" in s else ","
                f.write(f"{r['name']},{u},{s['label']},{s['n_target']},{s['n_control']},{s['mean_target']:.4f},"
                        f"{s['mean_control']:.4f},{s['auroc']:.4f},{s['auroc_of']},{on}\n")
    e = next((r for r in runs if r["name"] == "E"), None)
    lines = [f"# Activation probes ({len(runs)} models, texts {runs[0]['texts_md5'][:8] if runs else ''})", "",
             "AUROC between each unit's label and its control (0.5 = no better than chance; for the layer-23 "
             "neurons, on the gate's pre-activation):", "",
             "| Unit | Label | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for u, (lab, _) in UNITS.items():
        lines.append(f"| {u} | {lab} | " + " | ".join(f"{r['units'][u]['auroc']:.2f}" for r in runs) + " |")
    lines += ["", f"The layer-23 switches: the share of positions where the activation is above {ON}, on the label / "
              "on its control:", "", "| Unit | Label | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for u in SWITCHES:
        lines.append(f"| {u} | {UNITS[u][0]} | " + " | ".join(
            f"{r['units'][u]['on_target']:.0%} / {r['units'][u]['on_control']:.3%}" for r in runs) + " |")
    if e:
        lines += ["", "Mean activation on the label, as a share of E's (sign included; the gates' mean minus the "
                  "control's):", "", "| Unit | E's mean | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for u in UNITS:
            ref = e["units"][u]
            gate = u.endswith(".gate")
            base = (lambda s: s["mean_target"] - s["mean_control"]) if gate else (lambda s: s["mean_target"])
            lines.append(f"| {u} | {base(ref):.3f} | " + " | ".join(
                f"{base(r['units'][u]) / base(ref):.2f}" if base(ref) else "" for r in runs) + " |")
    open(f"{OUT}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "run":
        run(*sys.argv[2:4], sys.argv[4] if len(sys.argv) > 4 else sys.argv[3])
    elif cmd == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
