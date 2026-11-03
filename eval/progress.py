"""Progress of an evaluation: python3 progress.py <name> [--watch SECONDS]

Answers done per suite and model (the raw files grow one batch of up to 512 answers at a time), the last
progress line from eval/<name>.out, the 5090 and llama-swap. --watch redraws every SECONDS until Ctrl-C.
"""
import json
import os
import re
import subprocess
import sys
import time

EVAL = "/opt/wsp/eval"


def count(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def bar(done, total, width=24):
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


def show(name):
    out = f"{EVAL}/results/{name}"
    cfg = json.load(open(f"{out}/config.json"))
    now = time.time()
    running = sh("pgrep", "-f", f"[e]valuate.py run {name}") not in ("", "?")
    lines = [f"Evaluation {name}  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now))}  "
             f"{'running' if running else 'NOT RUNNING'} for {minutes(now - os.path.getmtime(out + '/config.json'))}",
             f"submission {cfg['submission']}", ""]
    lines.append(f"{'suite':<9} {'samples':>7}   " + "   ".join(f"{m:<36}" for m in cfg["models"]))
    done_all = total_all = 0
    for suite, k in cfg["suites"].items():
        probs = count(f"{EVAL}/final/problems.jsonl" if suite == "code" else f"{EVAL}/final/suites/{suite}.jsonl")
        total = probs * k
        cells = []
        for m in cfg["models"]:
            d = count(f"{out}/gen_{m}.raw.jsonl" if suite == "code" else f"{out}/gen_{suite}_{m}.raw.jsonl")
            done_all, total_all = done_all + d, total_all + total
            cells.append(f"{bar(d, total)} {d:>5}/{total:<5}")
        lines.append(f"{suite:<9} {k:>7}   " + "   ".join(f"{c:<36}" for c in cells))
    lines += ["", f"answers done {done_all:,} of {total_all:,} (the edited model's are much shorter, so time is not "
              "proportional)"]
    log = f"{EVAL}/{name}.out"
    if os.path.exists(log):
        prog = [l.strip() for l in open(log, errors="replace") if re.match(r"\w+ / \w+: ", l)]
        if prog:
            lines.append(f"last: {prog[-1]}  ({minutes(now - os.path.getmtime(log))} ago)")
            if running:
                lines.append(f"current batch (up to 512 answers) running for {minutes(now - os.path.getmtime(log))}; "
                             "the bars move only when it ends, ~40-50 min when most answers loop to 32k tokens")
    gpu = sh("nvidia-smi", "-i", "1", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader")
    swap = sh("docker", "inspect", "-f", "{{.State.Status}}", "llama-swap")
    lines.append(f"GPU: {gpu}   llama-swap: {swap}")
    report = f"{out}/report.md"
    if os.path.exists(report):
        lines.append(f"finished: {report}")
    return "\n".join(lines)


def main():
    name = sys.argv[1]
    every = int(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else 0
    while True:
        text = show(name)
        if not every:
            print(text)
            return
        print("\033[H\033[2J" + text + f"\n\nredraws every {every} s; Ctrl-C to stop watching (the evaluation goes on)",
              flush=True)
        time.sleep(every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
