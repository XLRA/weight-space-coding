"""Data for the paper's figures that paper/data doesn't already hold. It reads the held-out evaluations in
eval/results/ (on the server; their raw answers aren't in the repository) and writes small tables to
paper/figures/data/:

  pairs.csv          paired differences between models, per suite and subset, from eval/compare.py
  shapes.csv         per model and suite: pass rate, answers cut at the 32k cap, loop-flagged answers, finished
                     answers that close the thinking two or more times (and how many of those pass), mean length,
                     syntax errors
  paper_compare.csv  scores and paired differences against the original paper's published answers
                     (as scratch/paper_vs_ours.py)
  overview.csv       every evaluated model's mean paired difference with O over the five suites, with its SE

Run on the server: cd /opt/wsp && venv/bin/python paper/figure_data.py
paper/figures.py draws the figures from these tables and paper/data/.
"""
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "eval", "results")
OUT = os.path.join(ROOT, "paper", "figures", "data")

# name: (evaluation folder, which of its two models). O's answers are the ones graded in E's evaluation, as in
# RESULTS.md: one LiveCodeBench answer of O's sits at its tests' time limit and was graded differently elsewhere.
MODELS = {
    "O": ("pilot-14-pp15", "original"),
    "E": ("pilot-14-pp15", "edited"),
    "R-O": ("R-O-pp15", "edited"),
    "G-O": ("G-O-pp15", "edited"),
    "G-E2": ("G-E2-pp15", "edited"),
    "T": ("T-pp15", "edited"),
    "pilot-20": ("pilot-20-pp15", "edited"),
}
PAIRS = [("E", "O"), ("R-O", "O"), ("G-O", "O"), ("R-O", "E"), ("G-O", "E"), ("G-E2", "E"), ("G-E2", "G-O"),
         ("T", "G-O"), ("pilot-20", "G-O"), ("pilot-20", "T")]
STEMS = {"code": "gen_{}", "mmlupro": "gen_mmlupro_{}", "ifeval": "gen_ifeval_{}", "lcb": "gen_lcb_{}",
         "math500": "gen_math500_{}"}
# The overview figure's models, each from its <name>-pp15 evaluation and paired with O. compare.py prints the
# five-suite mean to two decimals, and one that lands on x.x5 there can then show the wrong first decimal
# (pilot-19's +21.1511 prints as +21.15, which was drawn as +21.1), so overview() keeps full precision.
OVERVIEW = ["pilot-10", "pilot-12", "pilot-13", "combo-12-13", "pilot-14", "pilot-15", "pilot-16", "pilot-17",
            "pilot-18", "pilot-19", "R-O", "G-O", "G-E2", "T", "pilot-20"]


def spec(name):
    folder, side = MODELS[name]
    return f"results/{folder}" + (":original" if side == "original" else "")


def pairs():
    rows = []
    for a, b in PAIRS:
        out = subprocess.run([sys.executable, "compare.py", spec(a), spec(b)], cwd=os.path.join(ROOT, "eval"),
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 9 and cells[0] not in ("Suite", "---") and not cells[0].startswith("-"):
                suite, subset, pa, pb, diff, se, n, abetter, bbetter = cells
                rows.append({"a": a, "b": b, "suite": suite, "subset": subset, "a_pass": pa.rstrip("%"),
                             "b_pass": pb.rstrip("%"), "diff": diff, "se": se, "problems": n,
                             "a_better": abetter, "b_better": bbetter})
        m = re.search(r"Mean of the five suites' A - B: ([+-]?[\d.]+) \(SE ([\d.]+)\)", out)
        if m:
            rows.append({"a": a, "b": b, "suite": "mean5", "subset": "all", "a_pass": "", "b_pass": "",
                         "diff": m.group(1), "se": m.group(2), "problems": "", "a_better": "", "b_better": ""})
    return rows


def shapes():
    rows = []
    for name, (folder, side) in MODELS.items():
        for suite, stem in STEMS.items():
            base = os.path.join(RES, folder, stem.format(side))
            raw = {}
            for line in open(base + ".raw.jsonl"):
                r = json.loads(line)
                raw[(r["id"], r["sample"])] = (r["text"].count("</think>"), r.get("finish_reason"))
            graded = [json.loads(line) for line in open(base + ".jsonl")]
            n = len(graded)
            cut = fin = re2 = re2_pass = loop = syntax = passed = tokens = 0
            for g in graded:
                closes, reason = raw[(g["id"], g["sample"])]
                passed += g["status"] == "pass"
                syntax += g["status"] == "syntax_error"
                loop += bool(g.get("loop"))
                tokens += g["n_tokens"]
                if reason == "length":
                    cut += 1
                    continue
                fin += 1
                if closes >= 2:
                    re2 += 1
                    re2_pass += g["status"] == "pass"
            rows.append({"model": name, "suite": suite, "answers": n, "pass": round(100 * passed / n, 2),
                         "cut": round(100 * cut / n, 2), "loop": round(100 * loop / n, 2),
                         "reclose": round(100 * re2 / max(1, fin), 2), "reclose_answers": re2,
                         "reclose_pass": round(100 * re2_pass / max(1, re2), 1),
                         "mean_tokens": round(tokens / n), "syntax_errors": syntax})
    return rows


def paper_compare():
    def paper(arm, suite):
        name = {"mmlupro": "mmlupro1k"}.get(suite, suite)
        d = {}
        for f in glob.glob(os.path.join(ROOT, "eval", "raw", "paper_samples", arm, f"{name}_s*", "per_problem.json")):
            for r in json.load(open(f)):
                d[r["id"]] = sum(map(float, r["correct"])) / len(r["correct"])
        return d

    def ours(name, suite):
        folder, side = MODELS[name]
        d = defaultdict(list)
        for line in open(os.path.join(RES, folder, STEMS[suite].format(side) + ".jsonl")):
            r = json.loads(line)
            d[r["id"]].append(r["status"] == "pass")
        return {k: sum(v) / len(v) for k, v in d.items()}

    rows = []
    for suite in ("mmlupro", "ifeval", "math500"):
        m = {"paper-unedited": paper("unedited", suite), "paper-edit": paper("p50c", suite)}
        m.update({k: ours(k, suite) for k in ("O", "E", "G-O")})
        for k, v in m.items():
            rows.append({"kind": "score", "a": k, "b": "", "suite": suite, "value": round(100 * sum(v.values()) / len(v), 2),
                         "se": "", "problems": len(v)})
        for a, b in (("O", "paper-unedited"), ("paper-edit", "paper-unedited"), ("E", "paper-edit"), ("G-O", "paper-edit")):
            ids = sorted(set(m[a]) & set(m[b]))
            diffs = [m[a][i] - m[b][i] for i in ids]
            rows.append({"kind": "pair", "a": a, "b": b, "suite": suite, "value": round(100 * sum(diffs) / len(diffs), 2),
                         "se": round(100 * statistics.stdev(diffs) / len(diffs) ** 0.5, 2), "problems": len(ids)})
    return rows


def overview():
    def scores(folder, side, stem):
        d = defaultdict(list)
        for line in open(os.path.join(RES, folder, stem.format(side) + ".jsonl")):
            r = json.loads(line)
            d[r["id"]].append(r["status"] == "pass")
        return {k: (sum(v) / len(v), len(v)) for k, v in d.items()}

    rows, base = [], {stem: scores(*MODELS["O"], stem) for stem in STEMS.values()}
    for name in OVERVIEW:
        diffs, ses = [], []
        for stem in STEMS.values():
            a, o = scores(f"{name}-pp15", "edited", stem), base[stem]
            assert set(a) == set(o) and all(a[k][1] == o[k][1] for k in a), (name, stem)
            d = [100 * (a[k][0] - o[k][0]) for k in sorted(a)]
            diffs.append(statistics.mean(d))
            ses.append(statistics.stdev(d) / len(d) ** 0.5)
        rows.append({"model": name, "eval": f"{name}-pp15", "diff": round(statistics.mean(diffs), 4),
                     "se": round(sum(s * s for s in ses) ** 0.5 / len(ses), 4)})
    return rows


def write(name, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    write("pairs.csv", pairs())
    write("paper_compare.csv", paper_compare())
    write("shapes.csv", shapes())
    write("overview.csv", overview())
