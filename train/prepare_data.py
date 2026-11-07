"""Training data for the training phase (plans/training-phase.md, "The recipe" and "Data hygiene").

    python3 /opt/wsp/train/prepare_data.py

Reads the agent's practice set (practice/final: 319 MBPP, 1,400 APPS) and writes train/data/:
- val.jsonl: a fixed validation split, 64 MBPP and 200 APPS problems (APPS stratified by difficulty), never trained
  on and used only to choose checkpoints;
- train.jsonl: the rest, minus any problem that overlaps a held-out or panel problem;
- overlap.json: the overlap check and its hits.

The overlap check compares each practice problem with every held-out code problem (MBPP+, HumanEval+:
eval/final/problems.jsonl; LiveCodeBench: eval/final/suites/lcb.jsonl) and the panel's code tasks
(panel/final/panel.jsonl). Text is lower-cased and split into runs of letters and digits. A pair is a hit, and the
practice problem is dropped from both splits, if any of these holds:
- **13-grams:** at least a quarter of the smaller problem's informative 13-grams are shared. 13-grams found in 10 or
  more problems of one corpus are prompt boilerplate (the shared answer instructions) and are ignored. So are 13-grams
  made mostly of numbers and single letters (sample data, input formats such as "n m a 1 a 2 ... a n").
- **Task sentence:** the first lines (the task statement), as sets of content words (the template words "write a
  python function to find the given ..." removed, see STOP), have a Jaccard similarity of 0.75 or more.
- **Same function:** the entry points are the same, ignoring case, and the content-word Jaccard is 0.5 or more.
MBPP repeats many tasks across its splits, word for word or reworded (change_date_format, harmonic_sum), and has many
template variants (the sum of the fourth power against the fifth, even against odd, camel to snake case against snake
to camel). The rule drops the variants too. Gradient training can memorise a solution and reuse it on a held-out twin,
which an edit chosen by forward passes could not, so this keeps the comparison with the agent fair. The first version
(2026-11-07) flagged any shared 13-gram and caught 253 problems, nearly all on contest boilerplate and sample numbers.

Nothing here reads a held-out answer or score; the held-out problems' text is used only to find overlaps.
"""
import collections
import json
import random
import re

WSP = "/opt/wsp"
OUT = f"{WSP}/train/data"
SEED = 20261107
N = 13
BOILERPLATE = 10


def load(path, keep=lambda r: True):
    return [r for r in map(json.loads, open(path, encoding="utf-8")) if keep(r)]


def grams(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def frequent(docs):
    """13-grams found in BOILERPLATE or more documents of one corpus."""
    c = collections.Counter(g for d in docs for g in d)
    return {g for g, k in c.items() if k >= BOILERPLATE}


def informative(g):
    """False for 13-grams made mostly of numbers and single letters (sample data, input formats)."""
    return sum(1 for x in g.split() if x.isdigit() or len(x) == 1) < 7


STOP = set("write a an the python function functions to find given of in on for from and or is are by with whether check "
           "if using all that it be which into as its their each every program code return returns".split())


def first_words(r):
    """The content words of the task statement (the problem's first line)."""
    return set(re.findall(r"[a-z0-9]+", r["problem"].split("\n")[0].lower())) - STOP


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    mbpp = load(f"{WSP}/practice/final/mbpp_practice.jsonl")
    apps = load(f"{WSP}/practice/final/apps_practice.jsonl")
    held = {
        "mbpp+ and humaneval+": load(f"{WSP}/eval/final/problems.jsonl"),
        "livecodebench": load(f"{WSP}/eval/final/suites/lcb.jsonl"),
        "panel code": load(f"{WSP}/panel/final/panel.jsonl", lambda r: r["suite"] == "code"),
    }
    practice = mbpp + apps
    pg = {r["id"]: grams(r["problem"]) for r in practice}
    ignore = frequent(pg.values())
    hg, hrec = {}, {}
    for name, rs in held.items():
        docs = {r["id"]: grams(r["problem"]) for r in rs}
        ignore |= frequent(docs.values())
        for r in rs:
            hg[(name, r["id"])] = docs[r["id"]]
            hrec[(name, r["id"])] = r
    keep = lambda g: {x for x in g - ignore if informative(x)}
    pg = {k: keep(g) for k, g in pg.items()}
    hg = {k: keep(g) for k, g in hg.items()}
    index = collections.defaultdict(set)
    for key, g in hg.items():
        for x in g:
            index[x].add(key)
    hw = {k: first_words(r) for k, r in hrec.items()}
    hits = {}
    for r in practice:
        shared = collections.Counter(key for x in pg[r["id"]] for key in index.get(x, ()))
        words, entry = first_words(r), (r.get("entry_point") or "").lower()
        found = []
        for key, h in hrec.items():
            n = shared.get(key, 0)
            ratio = n / max(1, min(len(pg[r["id"]]), len(hg[key])))
            jac = jaccard(words, hw[key])
            same_fn = bool(entry) and entry == (h.get("entry_point") or "").lower()
            why = [w for w, hit in (("13-grams", ratio >= 0.25), ("task sentence", jac >= 0.75),
                                    ("same function", same_fn and jac >= 0.5)) if hit]
            if why:
                found.append({"corpus": key[0], "id": key[1], "why": why, "shared_13grams": n,
                              "ratio": round(ratio, 2), "jaccard": round(jac, 2),
                              "practice_task": r["problem"].split("\n")[0][:120],
                              "held_out_task": h["problem"].split("\n")[0][:120]})
        if found:
            hits[r["id"]] = found

    rng = random.Random(SEED)
    val_ids = set(rng.sample(sorted(r["id"] for r in mbpp), 64))
    by_level = collections.defaultdict(list)
    for r in apps:
        by_level[r["difficulty"]].append(r["id"])
    quotas = {lv: round(200 * len(ids) / len(apps)) for lv, ids in by_level.items()}
    quotas[max(quotas, key=quotas.get)] += 200 - sum(quotas.values())
    for lv in sorted(by_level):
        val_ids |= set(rng.sample(sorted(by_level[lv]), quotas[lv]))

    val = [r for r in practice if r["id"] in val_ids and r["id"] not in hits]
    train = [r for r in practice if r["id"] not in val_ids and r["id"] not in hits]
    for name, rs in (("val", val), ("train", train)):
        with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    count = lambda rs: dict(collections.Counter(r["source"] for r in rs))
    report = {
        "seed": SEED, "ngram": N, "boilerplate_min_docs": BOILERPLATE, "ignored_13grams": len(ignore),
        "practice": count(practice), "held_out_problems": {k: len(v) for k, v in held.items()},
        "apps_val_quotas": quotas, "hits": hits, "dropped": sorted(hits),
        "val": count(val), "train": count(train),
    }
    json.dump(report, open(f"{OUT}/overlap.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("hits",)}, indent=1))
    for i, h in hits.items():
        for x in h:
            print(f"hit {i} ~ {x['id']} ({', '.join(x['why'])}; ratio {x['ratio']}, jaccard {x['jaccard']})\n"
                  f"    {x['practice_task']}\n    {x['held_out_task']}")


if __name__ == "__main__":
    main()
