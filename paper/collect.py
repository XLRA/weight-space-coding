"""Collects the data for a write-up of this project (README, "Paper data") into /opt/wsp/paper/data/.

    /opt/wsp/venv/bin/python /opt/wsp/paper/collect.py   (safe to run any time, also during runs; it only reads the runs)

Writes:
- runs.csv        one row per agent run: label, what it started from, brief, GPU, presence penalty, passes used of
                  the budget, turns, conversations, code runs, output tokens, usage-limit waits, start, end, hours;
- passes.csv      one row per forward request: run, request k, pass number, when, type, model, texts or problems,
                  GPU seconds, GPU, whether it ran, and whether the edit at that moment was snapshotted;
- passes.jsonl    the same, with the request (minus its texts) and the summary the agent got back, in full;
- evals.csv       one row per held-out evaluation, suite and subset: original and edited score, paired difference
                  and its standard error, number of problems, flips, and each model's share of answers cut at the
                  token limit or loop-flagged and its mean length;
- lineage.csv     run -> the edit it started from, so the chain of edits can be drawn; the training arms too
                  (arm -> its start);
- edits.csv       the anatomy of each run's final edit and of every per-pass snapshot: per tensor, the change
                  against the original weights (Frobenius norm, relative norm, largest entry, entries changed), its
                  numerical rank, and the share of it in the top 1 and top 4 singular directions;
- environment.json  the model (repository, revision, weights sha256), the runtime's vLLM, transformers, torch,
                  numpy and Python versions, the local GPUs and driver, the Colab GPUs used;
- panels.csv      one row per panel pass and suite (README, "Panel"): run, request k, pass, baseline, questions,
                  edited and baseline pass rates, paired difference and its standard error, flips, cut and mean
                  lengths; null-check runs (panel-*) give the panel's noise;
- panel.csv       the panel's questions without their text: label q, id, suite, source, difficulty, about;
- panel_stats.json  how the panel was drawn (eval/prepare_panel.py);
- runs/<run>/     the small texts of each run (brief, report, explanations, label, config, state, manifest), its
                  evaluation reports (report.md, report.json), its full agent transcript (tool_calls.jsonl,
                  messages/), the harness copies driver.py made at each start (code/) and its panel passes per
                  question without the answers (panel/panel_<k>.jsonl, and the graded rows panel/gen_*.jsonl and
                  panel/base_*.jsonl);
- trainings.csv   one row per training arm (train/runs/<arm>, plans/training-phase.md): kind (rft or grpo), start,
                  status (done, running, stopped), first and "done" times, hours, launches and GPUs, rounds or steps
                  done, and the recipe (its config.json) as JSON;
- rft_rounds.csv  one row per RFT round: what it sampled from, problems, answers, pass rate (also by source), problems
                  with a pass, answers kept, cut and loop shares, mean lengths, tokens and seconds sampled, and the
                  fine-tuning: steps, tokens, first and last NLL per token, seconds, peak memory;
- rft_steps.csv, grpo_steps.csv  one row per optimizer step (round_<r>/sft.jsonl, steps.jsonl);
- checkpoints.csv one row per checkpoint check and start check (train/evals): weights, GPU, panel pass rate per suite
                  with its cut share and mean length, panel mean, validation pass rate (also by source), problems
                  with a pass, cut share, mean length; and paired against the arm's start: per-suite and panel-mean
                  differences with standard errors, the validation difference with its standard error, and whether
                  the panel mean is within 2.5 points of the start's (the checkpoint rule);
- train/          the small training records: each arm's config, launches, log (timestamped lines and errors), round
                  summaries and fine-tuning steps or GRPO steps, checkpoint manifests; each check's summary and
                  generation record; the step-1 checks (log-prob agreement, weight push). The sampled answers stay in
                  /opt/wsp/train/runs and the checks' answers in /opt/wsp/train/evals (too large to copy);
- /opt/wsp/paper/code/<time>/  the harness, evaluation, training and paper code, copied whenever it changed.
Edits kept by driver.py at each request (snapshot_edit) are in /opt/wsp/paper/edits/<md5>.npy, listed per run in
runs/<run>/edits.jsonl. Runs before 2026-11-06 20:30 have only their final edit (runs/<run>/submission).
Test runs (e2e-*, test-*) are left out.
"""
import csv
import glob
import json
import os
import re
import hashlib
import shutil
import subprocess
from datetime import datetime

WSP = "/opt/wsp"
OUT = f"{WSP}/paper/data"
MODEL = f"{WSP}/models/qwen3.5-0.8b"
CACHE = f"{OUT}/.cache.json"
PANEL = f"{WSP}/panel/final/panel.jsonl"
SKIP = re.compile(r"^(e2e|test|smoke)")
STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ")
TRAIN = f"{WSP}/train"
PANEL_SUITES = ("choice", "math", "instructions", "code")
GPU_BY_MEMORY = {"0.65": "5080", "0.85": "5090"}  # run-ckpt.sh gives vLLM 0.65 of the 5080 and 0.85 of the 5090


def load(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def lines(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def jsonl(path):
    out = []
    for line in lines(path):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def hours(a, b):
    try:
        fmt = "%Y-%m-%d %H:%M:%S"
        return round((datetime.strptime(b, fmt) - datetime.strptime(a, fmt)).total_seconds() / 3600, 2)
    except (TypeError, ValueError):
        return ""


def runs():
    for d in sorted(glob.glob(f"{WSP}/runs/*/")):
        name = os.path.basename(d.rstrip("/"))
        if not SKIP.match(name) and os.path.exists(f"{d}config.json"):
            yield name, d


def run_row(name, d):
    cfg, st = load(f"{d}config.json", {}), load(f"{d}state.json", {})
    log = lines(f"{d}driver.log")
    stamps = [m.group(1) for m in (STAMP.match(l) for l in log) if m]
    label = "\n".join(lines(f"{d}label.txt")).strip()
    tok = st.get("tokens") or {}
    return {
        "run": name, "label": label, "status": st.get("status", ""), "backend": cfg.get("backend", "claude-code"),
        "start_from": os.path.basename(os.path.dirname((cfg.get("start_from") or "").rstrip("/"))) or "original",
        "brief": os.path.basename(cfg.get("brief") or ""), "gpu": cfg.get("gpu", "5090"),
        "model": cfg.get("model", ""), "effort": cfg.get("effort", ""),
        "presence_penalty": cfg.get("presence_penalty", 0), "passes_used": st.get("passes_used", ""),
        "passes_budget": cfg.get("passes", ""), "requests": st.get("requests", ""), "turns": st.get("turn", ""),
        "conversations": st["conv"] + 1 if isinstance(st.get("conv"), int) else "", "code_runs": st.get("code_runs", ""),
        "output_tokens": tok.get("output_tokens", ""), "cache_read_tokens": tok.get("cache_read_input_tokens", ""),
        "limit_waits": st.get("limit_waits", ""), "limit_wait_hours": round(st.get("limit_wait_seconds", 0) / 3600, 2),
        "started": stamps[0] if stamps else "", "last_event": stamps[-1] if stamps else "",
        "hours": hours(stamps[0], stamps[-1]) if stamps else "",
    }


def pass_rows(name, d):
    asked = {}
    for l in lines(f"{d}driver.log"):
        m = re.match(r"^(\S+ \S+) request (\d+): (\S+)", l)
        if m:
            asked[int(m.group(2))] = m.group(1)
    results = {}  # k -> the summary the agent got, from the forward tool calls
    for t in jsonl(f"{d}tool_calls.jsonl"):
        if t.get("kind") != "forward":
            continue
        res = t.get("result") or ""
        m = re.search(r"\(request (\d+)", res) or re.search(r"\[request (\d+) failed", res)
        if m:
            results[int(m.group(1))] = (t.get("turn"), res)
    edits = {e.get("k"): e for e in jsonl(f"{d}edits.jsonl")}
    out = []
    for path in sorted(glob.glob(f"{d}requests/fwd_*.json"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        if path.endswith(".result.json"):
            continue
        k = int(re.findall(r"\d+", os.path.basename(path))[0])
        req = load(path, {}) or {}
        res = load(f"{d}requests/fwd_{k}.result.json", {}) or {}
        turn, summary = results.get(k, ("", ""))
        m = re.search(r"\[forward pass (\d+) of", summary)
        size = (len(req.get("texts") or []) or len(req.get("problems") or req.get("prompts") or [])
                or len(req.get("items") or []) or "")
        out.append({
            "run": name, "k": k, "pass": int(m.group(1)) if m else "", "turn": turn, "requested": asked.get(k, ""),
            "type": req.get("type", ""), "model": req.get("model", ""), "size": size,
            "gpu_seconds": res.get("wall_seconds", ""), "gpu": (res.get("colab") or {}).get("gpu", "") or "local",
            "ran": bool(res.get("ok")) and bool(m), "edit_snapshot": k in edits,
            "_request": {x: y for x, y in req.items() if x not in ("texts", "items")}, "_summary": summary,
        })
    return out


def panel_rows(name, d, prows):
    """One row per panel pass (work/panel_<k>.jsonl, written by driver.py's summarize_panel) and suite."""
    passno = {p["k"]: p["pass"] for p in prows if p["run"] == name}
    out = []
    for path in sorted(glob.glob(f"{d}work/panel_*.jsonl"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        k = int(re.findall(r"\d+", os.path.basename(path))[0])
        rows = jsonl(path)
        for suite in sorted({r["suite"] for r in rows}):
            rs = [r for r in rows if r["suite"] == suite]
            n = len(rs)
            e = [r["status"] == "pass" for r in rs]
            b = [r["base_status"] == "pass" for r in rs]
            diff = [100 * (x - y) for x, y in zip(e, b)]
            mean = sum(diff) / n
            se = (sum((x - mean) ** 2 for x in diff) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else ""
            out.append({"run": name, "k": k, "pass": passno.get(k, ""), "baseline": rs[0].get("baseline", ""),
                        "suite": suite, "questions": n, "edited": round(100 * sum(e) / n, 2),
                        "base": round(100 * sum(b) / n, 2), "diff": round(mean, 2),
                        "se": round(se, 2) if se != "" else "",
                        "fail_to_pass": sum(x and not y for x, y in zip(e, b)),
                        "pass_to_fail": sum(y and not x for x, y in zip(e, b)),
                        "cut": sum(bool(r["cut"]) for r in rs), "base_cut": sum(bool(r["base_cut"]) for r in rs),
                        "mean_tokens": round(sum(r["tokens"] for r in rs) / n),
                        "base_mean_tokens": round(sum(r["base_tokens"] for r in rs) / n)})
    return out


def panel_questions():
    """The panel's questions without their text (the full file is rebuilt by eval/prepare_panel.py)."""
    out, count = [], {}
    for p in jsonl(PANEL):  # the same q and about as driver.py
        count[p["suite"]] = count.get(p["suite"], 0) + 1
        about = (p["instruction_id_list"] if p["suite"] == "instructions" else
                 f"{p.get('subject', '')}, {p['difficulty']}" if p["suite"] == "math" else p["difficulty"])
        out.append({"q": f"{p['suite']}/{count[p['suite']]}", "id": p["id"], "suite": p["suite"],
                    "source": p.get("source", ""), "difficulty": p.get("difficulty", ""),
                    "about": ";".join(about) if isinstance(about, list) else about})
    return out


def metric(block):
    for key in ("pass@1", "accuracy", "acc", "score", "strict_prompt", "prompt_strict"):
        if isinstance(block, dict) and key in block:
            return key, block[key]
    return "", ""


def eval_rows():
    out = []
    for rep_path in sorted(glob.glob(f"{WSP}/eval/results/*/report.json")):
        name = os.path.basename(os.path.dirname(rep_path))
        if SKIP.match(name):
            continue
        rep = load(rep_path, {}) or {}
        cfg = rep.get("config", {})
        run = os.path.basename(os.path.dirname(cfg.get("submission", "").rstrip("/")))
        for suite, subsets in (rep.get("paired") or {}).items():
            for sub, p in subsets.items():
                mo = (rep.get("models") or {}).get(suite, {})
                bo, be = (mo.get("original") or {}).get(sub) or {}, (mo.get("edited") or {}).get(sub) or {}
                key, orig = metric(bo)
                _, edit = metric(be)
                out.append({"eval": name, "run": run, "presence_penalty": cfg.get("presence_penalty", ""),
                            "suite": suite, "subset": sub, "metric": key, "original": orig, "edited": edit,
                            "diff": p.get("diff", ""), "se": p.get("se", ""), "problems": p.get("problems", ""),
                            "fail_to_pass": p.get("fail_to_pass", ""), "pass_to_fail": p.get("pass_to_fail", ""),
                            # How often each model hit the 32,768-token limit or was loop-flagged (% of answers), and
                            # its mean answer length: most of the edits' gain is answers that now finish.
                            "original_cut": bo.get("cut", ""), "edited_cut": be.get("cut", ""),
                            "original_loop": bo.get("loop", ""), "edited_loop": be.get("loop", ""),
                            "original_mean_tokens": bo.get("mean_tokens", ""),
                            "edited_mean_tokens": be.get("mean_tokens", "")})
    return out


def label_of(path, start=""):
    """A training start or a checked folder as a short label: original, pilot-14, R-O/round_1, G-E/step_25.
    Paths inside wsp-train:1 (/train/...) are mapped to the host; "start" (an RFT round's source) is the arm's start."""
    path = (path or "").rstrip("/")
    if path.startswith("/train/"):
        path = TRAIN + path[len("/train"):]
    if path in ("", "None", "original"):
        return "original"
    if path in ("start", "/start"):
        return start or "start"
    for pat in (rf"{WSP}/runs/([^/]+)/submission$", rf"{TRAIN}/runs/([^/]+)/(round_\d+)/checkpoint$",
                rf"{TRAIN}/runs/([^/]+)/checkpoints/(step_\d+)$"):
        m = re.match(pat, path)
        if m:
            return "/".join(m.groups())
    return path


def container_up(name):
    try:
        return bool(subprocess.run(["docker", "ps", "-q", "--filter", f"name=^{name}$"], capture_output=True,
                                   text=True, timeout=30).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def train_arms():
    for d in sorted(glob.glob(f"{TRAIN}/runs/*/")):
        arm = os.path.basename(d.rstrip("/"))
        cfg = load(f"{d}config.json")
        if not SKIP.match(arm) and cfg:
            yield arm, d, cfg


def training_rows():
    """Rows of trainings.csv, rft_rounds.csv, rft_steps.csv and grpo_steps.csv, from train/runs/<arm>/."""
    arms, rounds, rsteps, gsteps = [], [], [], []
    for arm, d, cfg in train_arms():
        kind = "rft" if "rounds" in cfg else "grpo"
        launches = jsonl(f"{d}launches.jsonl")  # one line per launch (run-train.sh)
        start = label_of(launches[0].get("start")) if launches else ("" if cfg.get("start") else "original")
        log = [l for l in lines(f"{d}train.log") if STAMP.match(l)]
        t = [STAMP.match(l).group(1) for l in log]
        done = next((STAMP.match(l).group(1) for l in log if l.endswith(" done")), "")
        row = {"arm": arm, "kind": kind, "start": start,
               "status": "done" if done else "running" if container_up(f"wsp-train-{arm}") else "stopped",
               "started": t[0] if t else "", "ended": done, "hours": hours(t[0], done) if t and done else "",
               "launches": len(launches), "gpus": " ".join(sorted({x.get("gpu", "") for x in launches})),
               "rounds_done": "", "steps_done": "", "torch": cfg.get("torch", ""),
               "recipe": json.dumps({k: v for k, v in cfg.items() if k not in ("arm", "start", "started", "torch")},
                                    sort_keys=True)}
        if kind == "rft":
            row["rounds_done"] = 0
            for rd in sorted(glob.glob(f"{d}round_*/"), key=lambda p: int(p.rstrip("/").rsplit("_", 1)[1])):
                r = int(rd.rstrip("/").rsplit("_", 1)[1])
                s, sft = load(f"{rd}summary.json", {}) or {}, jsonl(f"{rd}sft.jsonl")
                chunks = [re.search(r": (\d+) tokens in (\d+)s", l) for l in log
                          if l[20:].startswith(f"round {r}: sampled")]
                chunks = [m for m in chunks if m]
                ckpt = os.path.exists(f"{rd}checkpoint/manifest.json")
                row["rounds_done"] += ckpt
                src = s.get("pass_rate_by_source") or {}
                # rft.py records "start" or "round_<n>"; a round still sampling has no summary yet.
                sf = s.get("sampled_from") or ""
                sf = "" if not s else f"{arm}/{sf}" if re.fullmatch(r"round_\d+", sf) else label_of(sf, start)
                rounds.append({
                    "arm": arm, "round": r, "sampled_from": sf,
                    "problems": s.get("problems", ""), "answers": s.get("answers", ""),
                    "pass_rate": s.get("pass_rate", ""), "pass_rate_mbpp": src.get("mbpp", ""),
                    "pass_rate_apps": src.get("apps", ""), "problems_with_a_pass": s.get("problems_with_a_pass", ""),
                    "kept_answers": s.get("kept_answers", ""), "cut_share": s.get("cut_share", ""),
                    "loop_share": s.get("loop_share", ""), "mean_tokens": s.get("mean_tokens", ""),
                    "mean_tokens_kept": s.get("mean_tokens_kept", ""),
                    "sampled_tokens": sum(int(m.group(1)) for m in chunks) if chunks else "",
                    "sampling_seconds": sum(int(m.group(2)) for m in chunks) if chunks else "",
                    "sft_steps": len(sft), "sft_tokens": sum(x.get("tokens", 0) for x in sft),
                    "nll_first": sft[0].get("nll_per_token", "") if sft else "",
                    "nll_last": sft[-1].get("nll_per_token", "") if sft else "",
                    "sft_seconds": round(sum(x.get("seconds", 0) for x in sft), 1),
                    "sft_peak_gb": max((x.get("peak_gb", 0) for x in sft), default=""), "checkpoint": ckpt})
                rsteps += [{"arm": arm, "round": r, **x} for x in sft]
        else:
            steps = jsonl(f"{d}steps.jsonl")
            row["steps_done"] = len(steps)
            for x in steps:
                gsteps.append({"arm": arm, **{k: v for k, v in x.items() if k not in ("seconds", "status")},
                               **{f"seconds_{k}": v for k, v in (x.get("seconds") or {}).items()},
                               "status": json.dumps(x.get("status") or {}, sort_keys=True)})
        arms.append(row)
    return arms, rounds, rsteps, gsteps


def checks():
    for p in sorted(glob.glob(f"{TRAIN}/evals/*/summary.json") + glob.glob(f"{TRAIN}/evals/*/*/summary.json")):
        d = os.path.dirname(p)
        name = os.path.relpath(d, f"{TRAIN}/evals")
        if not SKIP.match(name):
            yield name, d


def checkpoint_rows():
    """Rows of checkpoints.csv: every checkpoint check and start check (train/evals/<name>/summary.json)."""
    out = []
    for name, d in checks():
        s, g = load(f"{d}/summary.json", {}) or {}, load(f"{d}/gen.json", {}) or {}
        v, vb = s.get("val") or {}, s.get("vs_base") or {}
        row = {"check": name, "arm": name.split("/")[0] if "/" in name else "", "weights": label_of(g.get("weights")),
               "gpu": GPU_BY_MEMORY.get(str(g.get("gpu_memory", "")), ""), "finished": g.get("finished", ""),
               "answers": g.get("answers", ""), "tokens": g.get("tokens", ""),
               "generate_seconds": g.get("generate_seconds", ""), "panel_mean": s.get("panel_mean", "")}
        for suite in PANEL_SUITES:
            b = (s.get("panel") or {}).get(suite) or {}
            row.update({f"panel_{suite}": b.get("pass", ""), f"panel_{suite}_cut": b.get("cut", ""),
                        f"panel_{suite}_mean_tokens": b.get("mean_tokens", "")})
        src = v.get("by_source") or {}
        row.update({"val_pass": v.get("pass", ""), "val_pass_mbpp": src.get("mbpp", ""),
                    "val_pass_apps": src.get("apps", ""), "val_problems_with_a_pass": v.get("problems_with_a_pass", ""),
                    "val_cut": v.get("cut", ""), "val_mean_tokens": v.get("mean_tokens", ""),
                    "base": vb.get("base", "")})
        for suite in PANEL_SUITES:
            b = (vb.get("panel") or {}).get(suite) or {}
            row.update({f"panel_{suite}_diff": b.get("diff", ""), f"panel_{suite}_se": b.get("se", "")})
        pm, vv = vb.get("panel_mean") or {}, vb.get("val") or {}
        row.update({"panel_mean_diff": pm.get("diff", ""), "panel_mean_se": pm.get("se", ""),
                    "val_diff": vv.get("diff", ""), "val_se": vv.get("se", ""),
                    "panel_within_2_5": vb.get("panel_within_2_5", "")})
        out.append(row)
    return out


def copy_training_texts():
    """paper/data/train/: the small training records (see the docstring)."""
    dest = f"{OUT}/train"

    def put(src, rel, keep=None):
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(f"{dest}/{rel}"), exist_ok=True)
        if keep is None:
            shutil.copyfile(src, f"{dest}/{rel}")
        else:
            with open(f"{dest}/{rel}", "w") as f:
                f.write("".join(l + "\n" for l in lines(src) if keep(l)))

    for arm, d, cfg in train_arms():
        for fn in ("config.json", "launches.jsonl", "steps.jsonl"):
            put(f"{d}{fn}", f"runs/{arm}/{fn}")
        put(f"{d}train.log", f"runs/{arm}/train.log", lambda l: STAMP.match(l) or "Error" in l)
        for rd in glob.glob(f"{d}round_*/"):
            r = os.path.basename(rd.rstrip("/"))
            for fn in ("summary.json", "sft.jsonl"):
                put(f"{rd}{fn}", f"runs/{arm}/{r}/{fn}")
            put(f"{rd}checkpoint/manifest.json", f"runs/{arm}/{r}/checkpoint_manifest.json")
        for m in glob.glob(f"{d}checkpoints/step_*/manifest.json"):
            put(m, f"runs/{arm}/checkpoints/{os.path.basename(os.path.dirname(m))}_manifest.json")
    for name, d in checks():
        for fn in ("summary.json", "gen.json"):
            put(f"{d}/{fn}", f"evals/{name}/{fn}")
    for p in glob.glob(f"{TRAIN}/checks/*.json"):
        put(p, f"checks/{os.path.basename(p)}")


def write_csv(path, rows):
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        # Every column any row has, in order of first appearance (GRPO step rows can differ).
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def copy_texts(name, d):
    dest = f"{OUT}/runs/{name}"
    os.makedirs(dest, exist_ok=True)
    for fn in ("brief.md", "final_answer.md", "explanations.md", "label.txt", "config.json", "state.json",
               "edits.jsonl", "submission/manifest.json", "tool_calls.jsonl"):
        if os.path.exists(f"{d}{fn}"):
            shutil.copyfile(f"{d}{fn}", f"{dest}/{fn.replace('/', '_')}")
    for sub in ("messages", "code"):
        if os.path.isdir(f"{d}{sub}"):
            shutil.copytree(f"{d}{sub}", f"{dest}/{sub}", dirs_exist_ok=True)
    panel = sorted(glob.glob(f"{d}work/panel_*.jsonl")) + [p for p in sorted(glob.glob(f"{d}panel/*.jsonl"))
                                                           if not p.endswith(".raw.jsonl")]
    for p in panel:  # per question, without question or answer texts
        os.makedirs(f"{dest}/panel", exist_ok=True)
        shutil.copyfile(p, f"{dest}/panel/{os.path.basename(p)}")
    for rep in glob.glob(f"{WSP}/eval/results/*/report.json"):
        cfg = (load(rep, {}) or {}).get("config", {})
        if os.path.basename(os.path.dirname(cfg.get("submission", "").rstrip("/"))) == name:
            ev = os.path.basename(os.path.dirname(rep))
            for fn in ("report.md", "report.json"):
                src = f"{os.path.dirname(rep)}/{fn}"
                if os.path.exists(src):
                    shutil.copyfile(src, f"{dest}/eval-{ev}-{fn}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


class Original:
    """The unedited weights, read straight from the safetensors file (bf16 -> float32), by tensor name."""

    def __init__(self):
        with open(f"{MODEL}/model.safetensors.index.json") as f:
            idx = json.load(f)["weight_map"]
        self.files = {k: f"{MODEL}/{v}" for k, v in idx.items()}
        self.headers = {}

    def get(self, name):
        import numpy as np
        path = self.files[name]
        if path not in self.headers:
            with open(path, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.headers[path] = (8 + n, json.loads(f.read(n)))
        base, head = self.headers[path]
        meta = head[name]
        a, b = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + a)
            raw = f.read(b - a)
        if meta["dtype"] == "BF16":
            x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            x = np.frombuffer(raw, {"F32": np.float32, "F16": np.float16}[meta["dtype"]]).astype(np.float32)
        return x.reshape(meta["shape"])


def anatomy(orig, name, path):
    import numpy as np
    w = orig.get(name)
    e = np.load(path).astype(np.float32)
    dlt = e - w
    nrm = float(np.linalg.norm(dlt))
    row = {"tensor": name, "shape": "x".join(map(str, e.shape)), "delta_norm": round(nrm, 6),
           "relative_norm": round(nrm / (float(np.linalg.norm(w)) or 1.0), 6),
           "max_abs": round(float(np.abs(dlt).max()) if dlt.size else 0.0, 6),
           "entries_changed": int(np.count_nonzero(dlt)), "rank": "", "top1_share": "", "top4_share": ""}
    if dlt.ndim == 2 and nrm > 0:
        sv = np.linalg.svd(dlt, compute_uv=False)
        energy = sv ** 2 / float((sv ** 2).sum())
        row.update(rank=int((sv > sv[0] * 1e-3).sum()), top1_share=round(float(energy[0]), 4),
                   top4_share=round(float(energy[:4].sum()), 4))
    return row


def edit_rows(cache):
    out = []
    try:
        orig = Original()
    except (OSError, ValueError, KeyError) as e:
        print(f"edits.csv skipped: cannot read the original weights ({e})")
        return out
    jobs = []
    for name, d in runs():
        for p in sorted(glob.glob(f"{d}submission/*.npy")):
            jobs.append((name, "final", os.path.basename(p)[:-4], p))
        for e in jsonl(f"{d}edits.jsonl"):
            for t, h in (e.get("tensors") or {}).items():
                jobs.append((name, f"request {e.get('k')}", t, f"{WSP}/paper/edits/{h}.npy"))
    for run, when, tensor, path in jobs:
        if not os.path.exists(path) or tensor not in orig.files:
            continue
        key = f"{md5(path)}:{tensor}"
        if key not in cache:
            cache[key] = anatomy(orig, tensor, path)
        out.append({"run": run, "edit": when, **cache[key]})
    return out


def environment(cache):
    env = cache.get("environment") or {}
    revs = set()
    for fn in glob.glob(f"{MODEL}/.cache/huggingface/download/*.metadata"):
        first = lines(fn)[:1]
        if first:
            revs.add(first[0].strip())
    shas = {}
    for p in sorted(glob.glob(f"{MODEL}/*.safetensors")):
        if not cache.get("sha:" + p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for block in iter(lambda: f.read(1 << 22), b""):
                    h.update(block)
            cache["sha:" + p] = h.hexdigest()
        shas[os.path.basename(p)] = cache["sha:" + p]
    env["model"] = {"repo": "Qwen/Qwen3.5-0.8B", "revision": sorted(revs), "weights_sha256": shas}
    if not env.get("runtime"):
        try:
            r = subprocess.run(["docker", "run", "--rm", "wsp-runtime:1", "python3", "-c",
                                "import vllm, transformers, torch, numpy, sys; print(vllm.__version__, "
                                "transformers.__version__, torch.__version__, numpy.__version__, sys.version.split()[0])"],
                               capture_output=True, text=True, timeout=600)
            v = (r.stdout.strip().splitlines() or [""])[-1].split()
            if len(v) == 5:
                env["runtime"] = dict(zip(("vllm", "transformers", "torch", "numpy", "python"), v), image="wsp-runtime:1")
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        env["local_gpus"] = [l.strip() for l in g.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    env["agent"] = "Claude Code headless; model and effort per run in runs.csv"
    cache["environment"] = env
    return env


def code_history():
    """Copies the harness, evaluation and paper code to /opt/wsp/paper/code/<time>/ when it changed."""
    files = sorted(glob.glob(f"{WSP}/harness/*.py") + glob.glob(f"{WSP}/harness/*.sh")
                   + glob.glob(f"{WSP}/harness/tests/*.py") + glob.glob(f"{WSP}/harness/briefs/*.md")
                   + glob.glob(f"{WSP}/eval/*.py") + glob.glob(f"{WSP}/paper/*.py")
                   + glob.glob(f"{TRAIN}/*.py") + glob.glob(f"{TRAIN}/*.sh") + glob.glob(f"{TRAIN}/Dockerfile")
                   + glob.glob(f"{TRAIN}/freeze.txt"))
    manifest = {os.path.relpath(p, WSP): md5(p) for p in files}
    old = sorted(glob.glob(f"{WSP}/paper/code/*/manifest.json"))
    if old and load(old[-1]) == manifest:
        return os.path.basename(os.path.dirname(old[-1]))
    dest = f"{WSP}/paper/code/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    for rel in manifest:
        os.makedirs(os.path.dirname(f"{dest}/{rel}"), exist_ok=True)
        shutil.copyfile(f"{WSP}/{rel}", f"{dest}/{rel}")
    with open(f"{dest}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    return os.path.basename(dest)


def main():
    os.makedirs(OUT, exist_ok=True)
    rrows, prows, panrows = [], [], []
    for name, d in runs():
        rrows.append(run_row(name, d))
        prows += pass_rows(name, d)
        panrows += panel_rows(name, d, prows)
        copy_texts(name, d)
    write_csv(f"{OUT}/panels.csv", panrows)
    if os.path.exists(PANEL):
        write_csv(f"{OUT}/panel.csv", panel_questions())
        shutil.copyfile(os.path.join(os.path.dirname(PANEL), "stats.json"), f"{OUT}/panel_stats.json")
    erows = eval_rows()
    write_csv(f"{OUT}/runs.csv", rrows)
    write_csv(f"{OUT}/passes.csv", prows)
    with open(f"{OUT}/passes.jsonl", "w") as f:
        for p in prows:
            f.write(json.dumps({**{k: v for k, v in p.items() if not k.startswith("_")},
                                "request": p["_request"], "summary": p["_summary"]}) + "\n")
    write_csv(f"{OUT}/evals.csv", erows)
    arms, rounds, rsteps, gsteps = training_rows()
    write_csv(f"{OUT}/trainings.csv", arms)
    write_csv(f"{OUT}/rft_rounds.csv", rounds)
    write_csv(f"{OUT}/rft_steps.csv", rsteps)
    write_csv(f"{OUT}/grpo_steps.csv", gsteps)
    crows = checkpoint_rows()
    write_csv(f"{OUT}/checkpoints.csv", crows)
    copy_training_texts()
    write_csv(f"{OUT}/lineage.csv", [{"run": r["run"], "started_from": r["start_from"]} for r in rrows]
              + [{"run": a["arm"], "started_from": a["start"]} for a in arms])
    cache = load(CACHE, {}) or {}
    drows = edit_rows(cache)
    write_csv(f"{OUT}/edits.csv", drows)
    env = environment(cache)
    env["colab_gpus"] = sorted({p["gpu"] for p in prows if p["gpu"] != "local"})
    with open(f"{OUT}/environment.json", "w") as f:
        json.dump(env, f, indent=1)
    with open(CACHE, "w") as f:
        json.dump(cache, f)
    code = code_history()
    print(f"{len(rrows)} runs, {len(prows)} requests, {len(panrows)} panel rows, {len(erows)} evaluation rows, "
          f"{len(drows)} edited tensors, {len(arms)} training arms, {len(rounds)} RFT rounds, {len(gsteps)} GRPO "
          f"steps, {len(crows)} checks; code copy {code} -> {OUT}")


if __name__ == "__main__":
    main()
