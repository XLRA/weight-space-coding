"""Progress of an agent run: python3 progress.py <run> [--watch SECONDS]

Passes used, turns, code runs and tokens from runs/<run>/state.json; the forward pass running now and the
answers it has written so far; the agent's latest explanation; the last lines of driver.log; the 5090 and
llama-swap. --watch redraws every SECONDS until Ctrl-C (the run goes on).
"""
import calendar
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time

RUNS = "/opt/wsp/runs"


def count(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return None


def bar(done, total, width=20):
    n = round(width * done / total) if total else 0
    return "#" * n + "." * (width - n)


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def minutes(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min"


def stamp(line):
    try:
        return calendar.timegm(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def current_request(run, now):
    """The forward pass running now: the highest fwd_N.json without a result, with its outputs so far."""
    req = f"{run}/requests"
    ns = sorted(int(m.group(1)) for f in os.listdir(req) if (m := re.fullmatch(r"fwd_(\d+)\.json", f)))
    if not ns or os.path.exists(f"{req}/fwd_{ns[-1]}.result.json"):
        return []
    n = ns[-1]
    r = json.load(open(f"{req}/fwd_{n}.json"))
    probs = r.get("problems")
    if probs is None and r.get("problems_file"):
        try:
            probs = json.load(open(r["problems_file"].replace("/work", f"{run}/work", 1)))
        except (OSError, ValueError):
            probs = None
    k = r.get("n", 1)
    what = f"{r['type']}, model {r['model']}"
    if r.get("baseline"):
        what += f", baseline {r['baseline']}"
    if probs is not None:
        what += f", {len(probs)} problems x {k} sample{'s' if k != 1 else ''}"
    if r.get("texts") is not None:
        what += f", {len(r['texts'])} texts"
    lines = [f"Forward pass {n} running for {minutes(now - os.path.getmtime(f'{req}/fwd_{n}.json'))}: {what}"]
    if r["type"] == "generate":
        variants = [r.get("baseline", "original"), "edited"] if r["model"] == "both" else [r["model"]]
        total = len(probs) * k if probs is not None else None
        before_done = True
        for v in variants:
            f = f"{run}/work/gen_{n}_{v}.raw.jsonl" if len(variants) > 1 else f"{run}/work/gen_{n}.raw.jsonl"
            d = count(f)
            if d is None:
                lines.append(f"  {v:<9} {bar(0, 1)} {'generating' if before_done else 'waiting'}")
                before_done = False
            else:
                lines.append(f"  {v:<9} {bar(d, total or d)} {d}/{total or '?'} answers, done")
        lines.append("  (each model's answers are written when all of them are finished)")
    return lines


def latest_explanation(run, width, max_lines):
    try:
        text = open(f"{run}/explanations.md", errors="replace").read()
    except FileNotFoundError:
        return []
    parts = re.split(r"^## (Turn \d+ \(\w+\))\s*$", text, flags=re.M)
    if len(parts) < 3:
        return []
    head, body = parts[-2], parts[-1].strip()
    out = []
    for para in body.split("\n"):
        out += textwrap.wrap(para, width - 2) or [""]
    if len(out) > max_lines:
        out = out[:max_lines - 1] + ["..."]
    return [f"Agent, {head}:"] + ["  " + l for l in out]


def show(name):
    run = f"{RUNS}/{name}"
    width = min(shutil.get_terminal_size((110, 40)).columns, 130)
    now = time.time()
    st = json.load(open(f"{run}/state.json"))
    cfg = json.load(open(f"{run}/config.json"))
    log = open(f"{run}/driver.log", errors="replace").read().splitlines()
    start = stamp(log[0]) if log else None
    alive = sh("pgrep", "-f", f"[d]river.py {name}( |$)") not in ("", "?")
    status = st["status"] if alive or st["status"] != "running" else "running, but the driver is NOT RUNNING"
    lines = [f"Agent run {name}  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}  {status}"
             + (f", started {minutes(now - start)} ago" if start else ""),
             f"{cfg['model']} effort {cfg['effort']}, presence penalty {cfg.get('presence_penalty', 0)}"
             + (f", from {cfg['start_from']}" if cfg.get("start_from") else ""), ""]
    p, P = st["passes_used"], cfg["passes"]
    lines.append(f"forward passes {bar(p, P)} {p}/{P}")
    lines.append(f"code runs      {bar(st['code_runs'], cfg['max_code_runs'])} {st['code_runs']}/{cfg['max_code_runs']}")
    tok = st["tokens"]
    lines.append(f"turn {st['turn']} (conversation {st['conv'] + 1}, turn {st['conv_turns']} of it; a fresh one "
                 f"every {cfg['handoff_every']}), {tok['output_tokens']:,} output tokens, context {st['context']:,}")
    if st.get("limit_waits"):
        lines.append(f"waited for the Claude usage limit {st['limit_waits']} times, "
                     f"{minutes(st['limit_wait_seconds'])} in all")
    lines.append("")
    lines += current_request(run, now) or ["No forward pass running; the agent is working in its kernel."]
    lines.append("")
    lines += latest_explanation(run, width, 9)
    lines.append("")
    lines.append("driver.log:")
    lines += ["  " + l[:width - 2] for l in log[-6:]]
    lines.append("")
    gpu = sh("nvidia-smi", "-i", "1", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader")
    swap = sh("docker", "inspect", "-f", "{{.State.Status}}", "llama-swap")
    lines.append(f"GPU: {gpu}   llama-swap: {swap} (stopped while the run uses the 5090)")
    if st["status"] == "done":
        sub = os.listdir(f"{run}/submission")
        lines.append(f"finished: {len(sub)} files in {run}/submission, report in {run}/final_answer.md")
    return "\n".join(lines)


def main():
    name = sys.argv[1]
    every = int(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else 0
    while True:
        text = show(name)
        if not every:
            print(text)
            return
        print("\033[H\033[2J" + text + f"\n\nredraws every {every} s; Ctrl-C to stop watching (the run goes on)",
              flush=True)
        time.sleep(every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
