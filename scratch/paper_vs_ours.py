"""Pair our models with the original paper's published answers, problem by problem.

The paper (Takehi et al., 2026) published per-problem grades for the unedited model and for its
50-pass math edit (eval/raw/paper_samples/{unedited,p50c}/*/per_problem.json): MMLU-Pro's 1,000
questions (1 sample), IFEval (2 samples) and MATH-500 (4 samples), the same questions and sample
counts as our evaluation. Our graders agree with the paper's grades on 799 of its 800 published
answers (eval/evaluate.py, check_against_paper), so its grades are used as they are.

Each line is the mean over problems of (A's pass rate on the problem - B's), with its SE. The
three-suite mean's SE is sqrt(sum of SE^2) / 3. "O - paper-unedited" checks that the two
evaluations agree on the same model.

Run on the server: python3 scratch/paper_vs_ours.py
"""
import json, glob, statistics as st
from collections import defaultdict
R = "/opt/wsp/eval/results"
P = "/opt/wsp/eval/raw/paper_samples"
def ours(path):
    d = defaultdict(list)
    for l in open(path):
        r = json.loads(l)
        d[r["id"]].append(r["status"] == "pass")
    return {k: sum(v) / len(v) for k, v in d.items()}
def paper(arm, suite):
    name = {"mmlupro": "mmlupro1k"}.get(suite, suite)
    d = {}
    for f in glob.glob(f"{P}/{arm}/{name}_s*/per_problem.json"):
        for r in json.load(open(f)):
            d[r["id"]] = sum(map(float, r["correct"])) / len(r["correct"])
    return d
def paired(a, b):
    ids = sorted(set(a) & set(b))
    diffs = [a[i] - b[i] for i in ids]
    return 100 * sum(diffs) / len(diffs), 100 * st.stdev(diffs) / len(diffs) ** 0.5, len(ids)
models = {"O": "pilot-14-pp15/gen_{s}_original.jsonl", "E": "pilot-14-pp15/gen_{s}_edited.jsonl",
          "G-O": "G-O-pp15/gen_{s}_edited.jsonl"}
summ = defaultdict(list)
for s in ("mmlupro", "ifeval", "math500"):
    pu, pe = paper("unedited", s), paper("p50c", s)
    m = {k: ours(f"{R}/{v.format(s=s)}") for k, v in models.items()}
    absl = "  ".join(f"{k} {100*sum(x.values())/len(x):.1f}" for k, x in [("paper-unedited", pu), ("paper-edit", pe)] + list(m.items()))
    print(f"{s}: {absl}")
    for k, a, b in [("O - paper-unedited", m["O"], pu), ("E - paper-edit", m["E"], pe), ("G-O - paper-edit", m["G-O"], pe),
                    ("paper-edit - paper-unedited", pe, pu), ("E - O", m["E"], m["O"])]:
        d, se, n = paired(a, b)
        summ[k].append((d, se))
        print(f"    {k:28s} {d:+6.1f} (SE {se:.1f})  n={n}")
print("three-suite means:")
for k, v in summ.items():
    print(f"    {k:28s} {sum(d for d, _ in v)/3:+6.2f} (SE {(sum(se*se for _, se in v))**0.5/3:.2f})")
