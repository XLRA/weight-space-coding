#!/usr/bin/env python3
"""Held-out evaluation of a submitted edit (host side; never shown to the agent).

    evaluate.py prepare [--suites all]        build final/ and check every grader (reference answers)
    evaluate.py run <name> --submission DIR [--suites code] [--samples N] [--models original,edited]
    evaluate.py report <name>                 grade and summarize results/<name>/ again

Suites (default samples per problem; `run` takes them in this order, each for both models before
the next):
  code     MBPP+ (378) and HumanEval+ (164)                                   4
  mmlupro  MMLU-Pro, the paper's fixed 1,000 questions                         1
  ifeval   IFEval (541), strict prompt-level                                   2
  lcb      LiveCodeBench release_v6 additions (175, 2025-01..04)               4
  math500  MATH-500                                                            4
`--suites all` runs every suite; the paper's Experiment 8 used the samples above for MATH-500,
IFEval and MMLU-Pro (prompts and readings: suites.py). `run` generates with the editing runs'
protocol (eval_gen.py, on the 5090, llama-swap paused), grades in the no-GPU sandbox, and writes
results/<name>/report.md and report.json: the score averaged over samples, the cut-at-token-limit
share next to every score, and the paired difference edited - original over problems with its
standard error.
"""
import argparse
import ast
import fcntl
import gzip
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from collections import Counter

import numpy as np

import suites as S

WSP = "/opt/wsp"
EVAL = f"{WSP}/eval"
MODEL_DIR = f"{WSP}/models/qwen3.5-0.8b"
HARNESS = f"{WSP}/harness"
IMAGE = "wsp-runtime:1"
GRADE_IMAGE = "wsp-grade:1"  # Dockerfile.grade: wsp-runtime:1 + math-verify and the IFEval checkers' dependencies
GPU = "nvidia.com/gpu=GPU-<uuid>"  # the RTX 5090
F = "`" * 3
INSTRUCTION = ("Write a complete, self-contained Python 3 solution. Put your final code in a single ```python block "
               "at the end of your answer.")
FUNC_TIMEOUT = 30  # seconds for a problem's whole EvalPlus test script
SUITES = {"code": 4, "mmlupro": 1, "ifeval": 2, "lcb": 4, "math500": 4}
# Rows of each suite's report table: (label, which graded rows).
SETS = {"code": [("all", None), ("mbpp+", ("source", "mbpp+")), ("humaneval+", ("source", "humaneval+"))],
        "lcb": [("all", None)] + [(d, ("difficulty", d)) for d in ("easy", "medium", "hard")],
        "ifeval": [("all", None)], "mmlupro": [("all", None)], "math500": [("all", None)]}
TITLES = {"code": "MBPP+ and HumanEval+ (pass@1)", "lcb": "LiveCodeBench v6 additions (pass@1)",
          "ifeval": "IFEval (strict prompt-level accuracy)", "mmlupro": "MMLU-Pro, 1,000 questions (accuracy)",
          "math500": "MATH-500 (accuracy)"}
SANDBOX = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g,mode=1777",
           "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "16g",
           "--pids-limit", "512", "--cpus", "12", "-w", "/tmp"]


def sh(*cmd, check=True, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def running(name):
    r = sh("docker", "inspect", "-f", "{{.State.Running}}", name, check=False)
    return r.returncode == 0 and r.stdout.strip() == "true"


def mbpp_entry(code, test):
    defs = [n.name for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)]
    tail = test[test.rfind("for i, (inp, exp)"):]
    return next((d for d in defs if f"{d}(" in tail), defs[0] if defs else None)


def prepare(suites):
    if "code" in suites:
        prepare_code()
    if set(suites) - {"code"}:
        prepare_suites([s for s in suites if s != "code"])


def prepare_code():
    import pandas as pd
    probs = []
    for _, r in pd.read_parquet(f"{EVAL}/raw/mbppplus.parquet").iterrows():
        probs.append({"id": f"mbpp+/{r['task_id']}", "source": "mbpp+", "split": "test", "difficulty": "plus",
                      "problem": f"{r['prompt']}\nYour code should pass this test:\n{list(r['test_list'])[0]}\n\n"
                                 f"{INSTRUCTION}",
                      "test_format": "function", "entry_point": mbpp_entry(r["code"], r["test"]),
                      "tests": [r["test"]], "test_setup": "", "reference": r["code"]})
    for _, r in pd.read_parquet(f"{EVAL}/raw/humanevalplus.parquet").iterrows():
        probs.append({"id": f"humaneval+/{r['task_id'].split('/')[1]}", "source": "humaneval+", "split": "test",
                      "difficulty": "plus",
                      "problem": f"Write the following Python function.\n\n{F}python\n{r['prompt'].rstrip()}\n{F}\n\n"
                                 f"{INSTRUCTION}",
                      "test_format": "function", "entry_point": r["entry_point"],
                      "tests": [r["test"], f"check({r['entry_point']})"], "test_setup": "",
                      "reference": r["prompt"] + r["canonical_solution"]})
    os.makedirs(f"{EVAL}/final", exist_ok=True)
    with open(f"{EVAL}/final/problems.jsonl", "w") as f:
        for p in probs:
            f.write(json.dumps(p) + "\n")
    # The reference solutions must pass under this grader, or the grader is wrong.
    ref_dir = f"{EVAL}/results/_references"
    os.makedirs(ref_dir, exist_ok=True)
    with open(f"{ref_dir}/gen_reference.raw.jsonl", "w") as f:
        for p in probs:
            f.write(json.dumps({"id": p["id"], "sample": 0, "text": f"</think>\n\n{F}python\n{p['reference']}\n{F}\n",
                                "n_tokens": 0, "finish_reason": "stop"}) + "\n")
    rows = grade(ref_dir, "gen_reference.raw.jsonl", "code")
    bad = [(r["id"], r["status"], r.get("exc")) for r in rows if r["status"] != "pass"]
    by = Counter(r["source"] for r in rows if r["status"] == "pass")
    print(f"{len(probs)} problems; references passing: {dict(by)}; failing: {len(bad)}")
    for b in bad[:20]:
        print("   ", b)
    json.dump({"problems": len(probs), "reference_pass": dict(by), "reference_fail": bad},
              open(f"{EVAL}/final/prepare_stats.json", "w"), indent=1)


def _raw(text, finish="stop"):
    return {"text": text, "n_tokens": 0, "finish_reason": finish}


def prepare_suites(suites):
    """Build final/suites/ from raw/, then check each grader in the sandbox: reference answers must pass
    and a wrong control must fail (sample 0 / 1), and the paper's published answers of its unedited and
    50-pass models must get the paper's own grades (sample 0 / 1, IFEval, MMLU-Pro and MATH-500)."""
    out = f"{EVAL}/final/suites"
    counts = S.prepare(f"{EVAL}/raw", out)
    print("suites:", counts)
    ref_dir = f"{EVAL}/results/_references"
    os.makedirs(ref_dir, exist_ok=True)
    stats = {"problems": counts}
    for suite in suites:
        probs = [json.loads(l) for l in open(f"{out}/{suite}.jsonl")]
        rows = []
        for i, p in enumerate(probs):
            if suite == "lcb":
                path = f"/suites/{p['tests_file']}"
                if p["test_format"] == "stdio":
                    good = (f"import sys, json\n_T = {{c['input'].strip(): c['output'] for c in "
                            f"json.load(open({path!r}))['cases']}}\nsys.stdout.write(_T[sys.stdin.read().strip()])\n")
                    bad = "print(0)\n"
                else:
                    good = (f"import json\n_T = {{json.dumps([json.loads(x) for x in c['input'].split(chr(10))]): "
                            f"json.loads(c['output']) for c in json.load(open({path!r}))['cases']}}\n"
                            f"class Solution:\n    def {p['func_name']}(self, *args):\n"
                            f"        return _T[json.dumps(list(args))]\n")
                    bad = f"class Solution:\n    def {p['func_name']}(self, *args):\n        return None\n"
                pair = [f"</think>\n\n{F}python\n{good}{F}\n", f"</think>\n\n{F}python\n{bad}{F}\n"]
            elif suite == "mmlupro":
                wrong = S.L[(S.L.index(p["answer"]) + 1) % 10]
                pair = [f"</think>\n\nThe answer is ({p['answer']}).", f"</think>\n\nThe answer is ({wrong})."]
            elif suite == "math500":
                other = probs[(i + 1) % len(probs)]["answer"]
                pair = [f"</think>\n\nSo the answer is $\\boxed{{{p['answer']}}}$.",
                        f"</think>\n\nSo the answer is $\\boxed{{{other}}}$."]
            else:  # IFEval has no reference answers; the paper's graded answers below check it
                continue
            rows += [{"id": p["id"], "sample": s, **_raw(t)} for s, t in enumerate(pair)]
        st = {}
        if rows:
            with open(f"{ref_dir}/gen_{suite}_reference.raw.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            g = grade(ref_dir, f"gen_{suite}_reference.raw.jsonl", suite)
            ref_fail = [(r["id"], r["status"], r.get("exc")) for r in g if r["sample"] == 0 and r["status"] != "pass"]
            ctl_pass = [r["id"] for r in g if r["sample"] == 1 and r["status"] == "pass"]
            st.update(reference_pass=sum(r["sample"] == 0 for r in g) - len(ref_fail), reference_fail=ref_fail,
                      control_pass=ctl_pass)
            print(f"{suite}: references passing {st['reference_pass']} of {len(probs)}; wrong controls passing "
                  f"{len(ctl_pass)}")
            for b in ref_fail[:10]:
                print("   ", b)
        if suite in ("ifeval", "mmlupro", "math500"):
            st["paper"] = check_against_paper(suite, ref_dir)
        stats[suite] = st
    json.dump(stats, open(f"{EVAL}/final/suites_prepare_stats.json", "w"), indent=1)


def check_against_paper(suite, ref_dir):
    """Grade the paper's published first answers (first 20 problems of each shard; sample 0 = unedited,
    1 = its 50-pass edit) and compare with the grades it recorded for them."""
    name = {"mmlupro": "mmlupro1k"}.get(suite, suite)
    shards = 10 if suite == "math500" else 5
    rows, theirs = [], {}
    for s, run in enumerate(("unedited", "p50c")):
        for k in range(shards):
            d = f"{EVAL}/raw/paper_samples/{run}/{name}_s{k}of{shards}"
            rec = {r["id"]: r["correct"][0] for r in json.load(open(f"{d}/per_problem.json"))}
            for r in map(json.loads, gzip.open(f"{d}/first_samples.jsonl.gz", "rt")):
                rows.append({"id": r["id"], "sample": s, **_raw(r["text"])})
                theirs[(r["id"], s)] = bool(rec[r["id"]])
    with open(f"{ref_dir}/gen_{suite}_paper.raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    g = grade(ref_dir, f"gen_{suite}_paper.raw.jsonl", suite)
    mine = {(r["id"], r["sample"]): r["status"] == "pass" for r in g}
    differ = sorted(k for k in theirs if mine[k] != theirs[k])
    res = {"answers": len(theirs), "agree": len(theirs) - len(differ), "paper_correct": sum(theirs.values()),
           "ours_correct": sum(mine.values()), "differ": differ}
    print(f"{suite}: the paper's {len(theirs)} published answers: grades agree on {res['agree']} "
          f"(correct: paper {res['paper_correct']}, ours {res['ours_correct']})")
    for k in differ[:10]:
        print(f"    differ: {k} paper={theirs[k]} ours={mine[k]}")
    return res


def grade(out_dir, raw_name, suite):
    """Grade a raw generation file in the no-GPU sandbox; writes the graded .jsonl next to it."""
    if suite == "code":
        cmd = [*SANDBOX, "-e", f"WSP_FUNC_TIMEOUT={FUNC_TIMEOUT}", "-v", f"{EVAL}/final:/practice:ro",
               "-v", f"{HARNESS}:/harness:ro", "-v", f"{out_dir}:/work:ro",
               IMAGE, "python3", "-W", "ignore", "/harness/grade_run.py", f"/work/{raw_name}"]
    else:
        cmd = [*SANDBOX, "-e", "PYTHONPATH=/evalbin:/harness", "-v", f"{EVAL}/final/suites:/suites:ro",
               "-v", f"{HARNESS}:/harness:ro", "-v", f"{EVAL}:/evalbin:ro", "-v", f"{out_dir}:/work:ro",
               GRADE_IMAGE, "python3", "-W", "ignore", "/evalbin/grade_suites.py", suite, f"/work/{raw_name}"]
    r = sh(*cmd, timeout=6 * 3600)
    rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    n_raw = sum(1 for _ in open(f"{out_dir}/{raw_name}"))
    if len(rows) != n_raw:
        raise RuntimeError(f"grader returned {len(rows)} of {n_raw} rows: {r.stderr[-800:]}")
    with open(f"{out_dir}/{raw_name.replace('.raw.jsonl', '.jsonl')}", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def run(name, submission, spec, models, limit, presence):
    out = f"{EVAL}/results/{name}"
    os.makedirs(out, exist_ok=True)
    sub = f"{out}/submission"
    if os.path.exists(sub):
        shutil.rmtree(sub)
    shutil.copytree(submission, sub)
    tensors = [f for f in os.listdir(sub) if f.endswith(".npy")]
    if "edited" in models and not tensors:
        raise SystemExit(f"{submission} has no .npy edits")
    missing = [s for s in spec if s != "code" and not os.path.exists(f"{EVAL}/final/suites/{s}.jsonl")]
    if missing:
        raise SystemExit(f"run `evaluate.py prepare --suites {','.join(missing)}` first")
    json.dump({"submission": os.path.realpath(submission), "tensors": sorted(tensors), "suites": spec,
               "models": models, "limit_per_set": limit, "presence_penalty": presence},
              open(f"{out}/config.json", "w"), indent=1)
    lock = open(f"{WSP}/runs/.gpu.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another run holds the GPU (/opt/wsp/runs/.gpu.lock)")
    stopped = False
    name_c = f"wsp-eval-{name}"
    # pkill -TERM: exit through the finally below, so the container goes and llama-swap comes back
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("stopped (SIGTERM)"))
    try:
        if running("llama-swap"):
            print("stopping llama-swap to free the 5090", flush=True)
            sh("docker", "stop", "llama-swap", timeout=180)
            stopped = True
        sh("docker", "rm", "-f", name_c, check=False)
        subprocess.run(["docker", "run", "--rm", "--name", name_c, "--network", "none", "--shm-size", "8g",
                        "--memory", "24g", "--device", GPU, "-e", "PYTHONPATH=/harness", "-e", "HF_HUB_OFFLINE=1",
                        "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1", "-e", f"WSP_PRESENCE={presence}",
                        "-v", f"{MODEL_DIR}:/model:ro", "-v", f"{EVAL}/final:/eval:ro", "-v", f"{out}:/out",
                        "-v", f"{sub}:/submission:ro", "-v", f"{HARNESS}:/harness:ro", "-v", f"{EVAL}:/evalbin:ro",
                        "-v", f"{WSP}/cache:/root/.cache", IMAGE, "python3", "-W", "ignore", "/evalbin/eval_gen.py",
                        ",".join(models), json.dumps(spec), str(limit)], check=True)
    finally:
        sh("docker", "rm", "-f", name_c, check=False)
        if stopped:
            sh("docker", "start", "llama-swap", check=False)
            print("llama-swap started again", flush=True)
    report(name)


def summarize(rows):
    n = len(rows)
    c = Counter(r["status"] for r in rows)
    s = {"answers": n, "pass@1": round(100 * c["pass"] / n, 2),
         "cut": round(100 * sum(r.get("cut", r["status"] == "cut_at_max_tokens") for r in rows) / n, 2),
         "loop": round(100 * sum(r["loop"] for r in rows) / n, 2),
         "mean_tokens": round(float(np.mean([r["n_tokens"] for r in rows])), 1),
         "failures": dict(c.most_common())}
    if any("inst_total" in r for r in rows):  # IFEval: instruction-level accuracy too
        s["instruction_level"] = round(100 * sum(r["inst_ok"] for r in rows) / sum(r["inst_total"] for r in rows), 2)
    return s


def paired(o_rows, e_rows):
    O = {(r["id"], r["sample"]): r["status"] == "pass" for r in o_rows}
    E = {(r["id"], r["sample"]): r["status"] == "pass" for r in e_rows}
    keys = sorted(set(O) & set(E))
    per = {}
    for k in keys:
        per.setdefault(k[0], []).append(float(E[k]) - float(O[k]))
    d = np.array([np.mean(v) for v in per.values()])
    return {"problems": len(d), "diff": round(100 * d.mean(), 2),
            "se": round(100 * d.std(ddof=1) / math.sqrt(len(d)), 2) if len(d) > 1 else None,
            "fail_to_pass": sum(not O[k] and E[k] for k in keys), "pass_to_fail": sum(O[k] and not E[k] for k in keys)}


def report(name):
    out = f"{EVAL}/results/{name}"
    graded = {}  # suite -> model -> graded rows
    for fn in sorted(os.listdir(out)):
        if fn.startswith("gen_") and fn.endswith(".raw.jsonl"):
            stem = fn[4:-10]
            suite, model = stem.split("_", 1) if "_" in stem else ("code", stem)
            graded.setdefault(suite, {})[model] = grade(out, fn, suite)
    order = {"original": 0, "edited": 1}
    graded = {s: dict(sorted(ms.items(), key=lambda kv: order.get(kv[0], 2))) for s, ms in graded.items()}
    cfg = json.load(open(f"{out}/config.json")) if os.path.exists(f"{out}/config.json") else {}
    spec = cfg.get("suites") or {"code": cfg.get("samples")}  # results from before the suites
    res = {"name": name, "config": cfg, "models": {}, "paired": {}}
    lines = [f"# Held-out evaluation: {name}", "",
             f"Submission {cfg.get('submission')} ({len(cfg.get('tensors', []))} tensors); samples per problem "
             + ", ".join(f"{s} {n}" for s, n in spec.items()) + f"; presence penalty {cfg.get('presence_penalty', 0)}"
             + (f"; SMOKE TEST, first {cfg['limit_per_set']} problems per set only" if cfg.get("limit_per_set") else "")
             + ".", "", "Cut = answers that hit the 32,768-token limit. Paired diff = edited - original, averaged per "
             "problem over its samples, with its standard error over problems."]
    for suite in [s for s in SUITES if s in graded]:
        lines += ["", f"## {TITLES[suite]}", "", "| Set | Model | Answers | Score | Cut | Loop-flagged | Mean tokens |",
                  "|---|---|---|---|---|---|---|"]
        sel = {label: (lambda r, k=key: k is None or r[k[0]] == k[1]) for label, key in SETS[suite]}
        for m, rows in graded[suite].items():
            for label, keep in sel.items():
                s = summarize([r for r in rows if keep(r)])
                res["models"].setdefault(suite, {}).setdefault(m, {})[label] = s
                lines.append(f"| {label} | {m} | {s['answers']} | {s['pass@1']:.1f}% | {s['cut']:.1f}% | "
                             f"{s['loop']:.1f}% | {s['mean_tokens']:,.0f} |")
        if "original" in graded[suite] and "edited" in graded[suite]:
            lines += ["", "| Set | Paired diff | SE | Problems | Fail->pass | Pass->fail |", "|---|---|---|---|---|---|"]
            for label, keep in sel.items():
                p = paired(*[[r for r in graded[suite][m] if keep(r)] for m in ("original", "edited")])
                res["paired"].setdefault(suite, {})[label] = p
                lines.append(f"| {label} | {p['diff']:+.2f} points | {p['se']} | {p['problems']} | {p['fail_to_pass']} | "
                             f"{p['pass_to_fail']} |")
        lines.append("")
        for m in graded[suite]:
            s = res["models"][suite][m]["all"]
            extra = (f"; instruction-level accuracy {s['instruction_level']:.1f}%" if "instruction_level" in s else "")
            lines.append(f"- {m}: " + ", ".join(f"{k} {v}" for k, v in s["failures"].items()) + extra)
    json.dump(res, open(f"{out}/report.json", "w"), indent=1)
    open(f"{out}/report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--suites", default="all")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--submission", required=True)
    r.add_argument("--suites", default="code", help="comma-separated, or all: " + ",".join(SUITES))
    r.add_argument("--samples", type=int, help="samples per problem for every chosen suite (default: per suite)")
    r.add_argument("--models", default="original,edited")
    r.add_argument("--limit", type=int, default=0, help="only the first N problems of each set (smoke tests)")
    r.add_argument("--presence-penalty", type=float, default=0.0,
                   help="0 = the editing runs' coding setting; the paper's Experiment 8 used 1.5")
    p = sub.add_parser("report")
    p.add_argument("name")
    a = ap.parse_args()
    if a.cmd in ("prepare", "run"):
        chosen = list(SUITES) if a.suites == "all" else a.suites.split(",")
        if set(chosen) - set(SUITES):
            raise SystemExit(f"--suites takes all or some of {','.join(SUITES)}")
        chosen = [s for s in SUITES if s in chosen]
    if a.cmd == "prepare":
        prepare(chosen)
    elif a.cmd == "run":
        models = a.models.split(",")
        if set(models) - {"original", "edited"}:
            raise SystemExit("--models takes original and/or edited")
        run(a.name, a.submission, {s: a.samples or SUITES[s] for s in chosen}, models, a.limit, a.presence_penalty)
    else:
        report(a.name)


if __name__ == "__main__":
    sys.exit(main())
