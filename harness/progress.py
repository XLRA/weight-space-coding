"""Live view of agent runs: python3 progress.py <run> [<run> ...] [--watch SECONDS]

For each run:
- what it works on (runs/<run>/label.txt, if there is one), its status and age, GPU, model and settings;
- passes and code runs used; the agent's turn, tokens and context;
- what is happening now (the forward pass running, with each model's answers, or the agent working in
  its kernel) and the last driver.log event;
- the start of the agent's latest explanation, and its evaluations (matched by submission directory), or
  that its hand-off will start one when the agent finishes.
Below the runs:
- evaluations running now or finished in the last day (eval/results/<name>, not smoke or e2e tests): GPU, how long, which suite
  and how many answers (from eval/<name>.out), or when done the edited model's overall score per suite;
- both GPUs (temperature, power, use, memory, the jobs running on each now, llama-swap);
- the temperature logger's warnings (gpu_temps.py).

--watch redraws every SECONDS in place, on the terminal's alternate screen, fitted to the window (the
explanation is shortened first when the window is small). Ctrl-C stops watching; the runs go on.
With one run it shows more of the explanation and the last lines of driver.log.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpus import GPUS  # noqa: E402

RUNS = "/opt/wsp/runs"
EVAL = "/opt/wsp/eval"
SUITE_NAMES = {"code": "code", "mmlupro": "MMLU-Pro", "ifeval": "IFEval", "lcb": "LiveCodeBench", "math500": "MATH-500"}
TEMP_WARNINGS = f"{RUNS}/gpu-temps-warnings.log"
COLOR = sys.stdout.isatty()
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LABEL = 12  # width of the label column


def style(code):
    return lambda s: f"\x1b[{code}m{s}\x1b[0m" if COLOR else str(s)


bold, dim, green, yellow, red, cyan, blue = (style(c) for c in ("1", "2", "32", "33", "31", "36", "34"))


def visible(s):
    return len(ANSI.sub("", s))


def clip(s, width):
    """s cut to width visible characters, keeping its colour codes."""
    if visible(s) <= width:
        return s
    out, n, i = [], 0, 0
    while i < len(s) and n < width - 1:
        m = ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + "…" + ("\x1b[0m" if COLOR else "")


def spread(left, right, width):
    return left + " " * max(width - visible(left) - visible(right), 2) + right


def row(label, text):
    return f"   {dim(label.ljust(LABEL))}{text}"


def bar(done, total, width=20):
    n = min(width, round(width * done / total)) if total else 0
    return cyan("█" * n) + dim("░" * (width - n))


def rule(width):
    return dim("─" * width)


def count(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return None


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def minutes(s):
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min"


def ago(s):
    return "just now" if s < 60 else f"{minutes(s)} ago"


def stamp(line):
    try:
        return calendar.timegm(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def k(n):
    return f"{n / 1000:,.0f}k" if n >= 10000 else f"{n:,}"


def model_name(m):
    x = re.fullmatch(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", m)
    return f"{x.group(1).capitalize()} {x.group(2)}{'.' + x.group(3) if x.group(3) else ''}" if x else m


def start_name(path):
    if not path:
        return "from the original model"
    m = re.search(r"runs/([^/]+)/submission", path)
    return f"from {m.group(1)}'s edit" if m else f"from {path}"


def forward_pass(run, now):
    """(what, answers) for the forward pass running now, or None when the agent is in its kernel."""
    req = f"{run}/requests"
    try:
        ns = sorted(int(m.group(1)) for f in os.listdir(req) if (m := re.fullmatch(r"fwd_(\d+)\.json", f)))
    except FileNotFoundError:
        return None
    if not ns or os.path.exists(f"{req}/fwd_{ns[-1]}.result.json"):
        return None
    n = ns[-1]
    path = f"{req}/fwd_{n}.json"
    r = json.load(open(path))

    def items(key):
        v = r.get(key)
        if v is None and r.get(f"{key}_file"):
            try:
                v = json.load(open(r[f"{key}_file"].replace("/work", f"{run}/work", 1)))
            except (OSError, ValueError):
                return None
        return v or []

    what = [f"pass {n}", r["type"].replace("_", "-")]
    what.append(f"{r.get('baseline', 'original')} vs edited" if r["model"] == "both" else f"model {r['model']}")
    if r["type"] == "generate":
        probs, own = items("problems"), items("prompts")
        if probs is not None and own is not None:
            parts = ([f"{len(probs)} problems"] if probs else []) + ([f"{len(own)} prompts"] if own else [])
            k_ = r.get("n", 1)
            what.append(" + ".join(parts) + (f" × {k_}" if k_ != 1 else ""))
    else:
        texts = items("texts")
        if texts:
            what.append(f"{len(texts)} texts")
    what.append(f"running {minutes(now - os.path.getmtime(path))}")
    answers = None
    if r["type"] == "generate":
        variants = [r.get("baseline", "original"), "edited"] if r["model"] == "both" else [r["model"]]
        states, first_open = [], True
        for v in variants:
            d = count(f"{run}/work/gen_{n}_{v}.raw.jsonl" if len(variants) > 1 else f"{run}/work/gen_{n}.raw.jsonl")
            if d is not None:
                states.append(f"{v} {green('✓')} {d} answers")
            elif first_open:
                states.append(f"{v} {yellow('▸ generating')}")
                first_open = False
            else:
                states.append(f"{v} {dim('waiting')}")
        answers = "    ".join(states)
    return " · ".join(what), answers


def explanation(run):
    """(turn heading, latest explanation as one paragraph), or (None, "")."""
    try:
        text = open(f"{run}/explanations.md", errors="replace").read()
    except FileNotFoundError:
        return None, ""
    parts = re.split(r"^## Turn (\d+) \((\w+)\)\s*$", text, flags=re.M)
    if len(parts) < 4:
        return None, ""
    body = re.sub(r"\*\*|`", "", parts[-1])
    body = re.sub(r"^\s*[-*]\s+", "", body, flags=re.M)
    return f"turn {parts[-3]}, {parts[-2]}", " ".join(body.split())


def pad(s, width):
    return s + " " * max(width - visible(s), 0)


def clock(t, now):
    """HH:MM UTC, with the date when it isn't today."""
    fmt = "%H:%M" if time.gmtime(t)[:3] == time.gmtime(now)[:3] else "%m-%d %H:%M"
    return time.strftime(fmt, time.gmtime(t)) + " UTC"


def running_jobs():
    """{card: [label, ...]} for the agent runs and evaluations running now."""
    jobs = {}
    for line in sh("pgrep", "-af", "driver.py").splitlines():
        if m := re.match(r"\d+ \S*python\S* (?:\S*/)?driver\.py (\S+)", line):
            try:
                card = json.load(open(f"{RUNS}/{m.group(1)}/config.json")).get("gpu", "5090")
            except (OSError, ValueError):
                card = "?"
            jobs.setdefault(card, []).append(f"agent {m.group(1)}")
    for line in sh("pgrep", "-af", "evaluate.py run").splitlines():
        if m := re.match(r"\d+ \S*python\S* (?:\S*/)?evaluate\.py run (\S+)", line):
            g = re.search(r"--gpu (\S+)", line)
            jobs.setdefault(g.group(1) if g else "5090", []).append(f"eval {m.group(1)}")
    return jobs


def eval_progress(name, suites, models):
    """Where a running evaluation is, from the progress lines in eval/<name>.out."""
    try:
        text = open(f"{EVAL}/{name}.out", errors="replace").read().replace("\r", "\n")
    except FileNotFoundError:
        return "starting"
    last = None
    for m in re.finditer(r"^(\w+) / (\w+): (\d+)/(\d+) in", text, flags=re.M):
        last = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if not last:
        return "loading the model, then " + (SUITE_NAMES.get(suites[0], suites[0]) if suites else "the first suite")
    suite, model, done, total = last
    k = suites.index(suite) + 1 if suite in suites else 0
    if done < total:
        return f"suite {k} of {len(suites)}, {SUITE_NAMES.get(suite, suite)}: {done:,}/{total:,} answers"
    if model == "original" and "edited" in models:
        return f"suite {k} of {len(suites)}, {SUITE_NAMES.get(suite, suite)}: edited answers starting"
    if k and k < len(suites):
        return f"suite {k + 1} of {len(suites)}, {SUITE_NAMES.get(suites[k], suites[k])}: first batch generating"
    return "grading the answers and writing the report"


def eval_scores(report):
    """The edited model's overall score per suite, from a report.md."""
    out = []
    for section in re.split(r"^## ", open(report, errors="replace").read(), flags=re.M)[1:]:
        title = section.split("\n", 1)[0]
        short = next((v for k_, v in (("MBPP", "code"), ("MMLU", "MMLU-Pro"), ("IFEval", "IFEval"),
                                       ("LiveCode", "LCB"), ("MATH", "MATH")) if title.startswith(k_)), title[:8])
        if m := re.search(r"^\| all \| edited \| \d+ \| ([\d.]+%)", section, flags=re.M):
            out.append(f"{short} {m.group(1)}")
    return out


def evaluations(now, keep=24 * 3600):
    """Evaluations running now or finished in the last day, running ones first."""
    running = {m.group(1) for line in sh("pgrep", "-af", "evaluate.py run").splitlines()
               if (m := re.match(r"\d+ \S*python\S* (?:\S*/)?evaluate\.py run (\S+)", line))}
    try:
        names = os.listdir(f"{EVAL}/results")
    except FileNotFoundError:
        return []
    out = []
    for name in names:
        d, rep = f"{EVAL}/results/{name}", f"{EVAL}/results/{name}/report.md"
        if name not in running and (name.startswith(("smoke", "e2e")) or
                                    not (os.path.exists(rep) and now - os.path.getmtime(rep) < keep)):
            continue
        try:
            cfg = json.load(open(f"{d}/config.json"))
            start = os.path.getmtime(f"{d}/config.json")
        except (OSError, ValueError):
            cfg, start = {}, None
        e = {"name": name, "gpu": cfg.get("gpu", "5090"), "submission": cfg.get("submission", ""),
             "running": name in running, "start": start}
        if e["running"]:
            e["progress"] = eval_progress(name, list(cfg.get("suites", {})), cfg.get("models", []))
        elif os.path.exists(rep):
            e["done"], e["scores"] = os.path.getmtime(rep), eval_scores(rep)
        out.append(e)
    out.sort(key=lambda e: (0, e["start"] or 0) if e["running"] else (1, -e.get("done", 0)))
    return out


def eval_block(evs, now):
    out = []
    width = max((len(e["name"]) for e in evs), default=0) + 2
    for e in evs[:5]:
        if e["running"]:
            state = yellow(f"▸ running {minutes(now - e['start'])}" if e["start"] else "▸ running")
            detail = e["progress"]
        elif "done" in e:
            state = green(f"✓ done {clock(e['done'], now)}")
            detail = "  ".join(e["scores"])
        else:
            state, detail = red("✗ stopped without a report"), ""
        out.append(row("Evaluations" if not out else "",
                       f"{e['name']:<{width}}{dim('RTX ' + e['gpu'])}  {pad(state, 23)}{detail}"))
    return out or [row("Evaluations", dim("none running or finished in the last day"))]


def run_evaluation(name, run, evs, now):
    """The Evaluation row of a run: its own evaluations (by submission), or its waiting hand-off."""
    sub = os.path.realpath(f"{run}/submission")
    mine = [e for e in evs if e["submission"] and os.path.realpath(e["submission"]) == sub]
    if mine:
        parts = []
        for e in mine:
            if e["running"]:
                parts.append(f"{e['name']} " + yellow(f"running on the RTX {e['gpu']}"))
            elif "done" in e:
                parts.append(f"{e['name']} " + green(f"done {clock(e['done'], now)}"))
        return " · ".join(parts) + dim("  (see Evaluations below)")
    if any(sh("pgrep", "-f", pat) not in ("", "?") for pat in (f"after-{name}[.]sh", f"after-run[.]sh {name} ")):
        return dim("starts by itself when the agent finishes")
    return None


def run_block(name, now, width, expl_lines, detail, evs):
    run = f"{RUNS}/{name}"
    try:
        st = json.load(open(f"{run}/state.json"))
        cfg = json.load(open(f"{run}/config.json"))
    except (OSError, ValueError) as e:
        return [f" {red('●')} {bold(name)}  {red(f'cannot read the run: {e}')}"]
    try:
        log = open(f"{run}/driver.log", errors="replace").read().splitlines()
    except FileNotFoundError:
        log = []
    try:
        label = open(f"{run}/label.txt").read().strip()
    except FileNotFoundError:
        label = ""
    start = stamp(log[0]) if log else None
    alive = sh("pgrep", "-f", f"[d]river.py {name}( |$)") not in ("", "?")
    status = st["status"]
    if status == "running" and not alive:
        dot, word = red("●"), red("driver not running")
    elif status == "running":
        dot, word = green("●"), green("agent running")
    elif status == "done":
        dot, word = blue("✓"), blue("agent finished")
    else:
        dot, word = yellow("●"), yellow(status)
    end = now if status == "running" else (stamp(log[-1]) or now) if log else now
    age = f" · {minutes(end - start)}" if start else ""
    out = [spread(f" {dot} {bold(name)}  {label}", f"{word}{dim(age)}", width),
           dim(f"   RTX {cfg.get('gpu', '5090')} · {model_name(cfg['model'])} {cfg['effort']} effort · "
               f"presence penalty {cfg.get('presence_penalty', 0)} · {start_name(cfg.get('start_from'))}")]

    p, P, c, C = st["passes_used"], cfg["passes"], st["code_runs"], cfg["max_code_runs"]
    passes, code = f"{bar(p, P)} {p}/{P}", f"{bar(c, C)} {c}/{C}"
    if width >= 92:
        out.append(row("Passes", passes + "     " + dim("Code runs  ") + code))
    else:
        out += [row("Passes", passes), row("Code runs", code)]
    tok = st["tokens"]
    if cfg.get("handoff_every"):  # runs before 2026-11-05 19:00 UTC restart every N turns
        agent = (f"turn {st['turn']} · conversation {st['conv'] + 1} ({st['conv_turns']}/{cfg['handoff_every']} "
                 f"turns) · {k(tok['output_tokens'])} output tokens · context {k(st['context'])}")
    else:
        agent = (f"turn {st['turn']} · conversation {st['conv'] + 1} ({st['conv_turns']} turns) · "
                 f"{k(tok['output_tokens'])} output tokens · context {k(st['context'])}"
                 f"/{k(cfg.get('max_context', 400_000))}")
    if st.get("limit_waits"):
        agent += yellow(f" · waited for the usage limit {st['limit_waits']}× ({minutes(st['limit_wait_seconds'])})")
    out.append(row("Agent", agent))

    fp = forward_pass(run, now) if status == "running" else None
    if fp:
        out.append(row("Now", yellow(fp[0])))
        if fp[1]:
            out.append(row("", fp[1]))
    elif status == "running":
        out.append(row("Now", "the agent is working in its kernel"))
    elif status == "done":
        files = len(os.listdir(f"{run}/submission")) if os.path.isdir(f"{run}/submission") else 0
        out.append(row("Result", f"{files} files in submission/ · report in final_answer.md"))
    if log:
        t = stamp(log[-1])
        out.append(row("Last event", dim(f"{ago(now - t) if t else ''} · {re.sub(r';.*', '', log[-1][20:])}")))

    head, body = explanation(run)
    if body and expl_lines:
        wrapped = textwrap.wrap(body, max(width - 3 - LABEL - len(head) - 2, 30))
        if len(wrapped) > expl_lines:
            wrapped = wrapped[:expl_lines]
            wrapped[-1] += " …"
        out.append(row("Latest", dim(f"{head}  ") + wrapped[0]))
        out += [row("", " " * (len(head) + 2) + w) for w in wrapped[1:]]

    ev = run_evaluation(name, run, evs, now)
    if ev:
        out.append(row("Evaluation", ev))
    if detail and log:
        out.append(row("driver.log", dim(log[-4][11:] if len(log) >= 4 else log[0][11:])))
        out += [row("", dim(l[11:])) for l in log[-3:]]
    return out


def gpu_block():
    users = running_jobs()
    q = sh("nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
           "--format=csv,noheader,nounits")
    rows = {r[0]: r for r in (l.split(", ") for l in q.splitlines()) if len(r) == 6}
    out = []
    for card, g in GPUS.items():
        r = rows.get(str(g["index"]))
        label = "GPUs" if not out else ""
        if not r:
            out.append(row(label, f"RTX {card}  {red('not visible to nvidia-smi')}"))
            continue
        try:
            t, p, u, mu, mt = float(r[1]), float(r[2]), int(r[3]), float(r[4]) / 1000, float(r[5]) / 1000
        except ValueError:
            out.append(row(label, f"RTX {card}  {', '.join(r[1:])}"))
            continue
        temp = f"{t:>3.0f} °C"
        temp = red(temp) if t >= 83 else yellow(temp) if t >= 75 else temp
        who = ", ".join(users.get(card, [])) or dim("idle")
        if g["llama_swap"]:
            swap = sh("docker", "inspect", "-f", "{{.State.Status}}", "llama-swap")
            extra = f"llama-swap {'stopped' if swap == 'exited' else swap}"
        else:
            extra = "shared: other services"
        out.append(row(label, f"RTX {card}  {temp}  {p:>4.0f} W  {u:>3} %  {mu:4.1f} / {mt:4.1f} GB   "
                              f"{pad(who, 24)}  {dim(extra)}"))
    return out


def temp_line():
    alive = sh("pgrep", "-f", "/opt/wsp/harness/gpu_temps[.]py") not in ("", "?")
    if not alive:
        return row("Temps", yellow("the temperature logger is not running: bash /opt/wsp/harness/start-gpu-temps.sh"))
    try:
        lines = open(TEMP_WARNINGS, errors="replace").read().splitlines()
    except FileNotFoundError:
        lines = []
    active, since = {}, None
    for l in lines:
        msg = l[20:]
        if msg.startswith("logger started"):
            active, since = {}, l[11:16]
        elif m := re.match(r"(WARNING|STILL) ([^:]+): ", msg):
            active[m.group(2)] = l
        elif m := re.match(r"OK ([^:]+): ", msg):
            active.pop(m.group(1), None)
    if active:
        return row("Temps", red(" | ".join(f"{l[11:16]} {l[20:]}" for l in active.values())))
    warned = sum(1 for l in lines if l[20:].startswith("WARNING"))
    note = f"{warned} warning{'s' if warned != 1 else ''}, all cleared" if warned else "no warnings"
    return row("Temps", green(note) + dim(f" · logged every 30 s since {since} UTC" if since else ""))


def frame(names, cols, rows, every):
    """The screen as lines, at most rows of them (None: no limit)."""
    width = max(min(cols - 1, 130), 60)
    now = time.time()
    single = len(names) == 1
    header = spread(f" {bold('Agent runs and evaluations')}  {dim('Qwen3.5-0.8B weight editing')}",
                    time.strftime("%H:%M:%S UTC", time.gmtime(now)), width)
    footer = dim(f" redraws every {every} s · Ctrl-C stops watching; the runs carry on") if every else None
    evs = evaluations(now)
    tail = [rule(width)] + eval_block(evs, now) + [rule(width)] + gpu_block() + [temp_line()]
    for expl in ([8, 4, 2, 0] if single else [3, 2, 1, 0]):
        lines = [header, rule(width)]
        for i, name in enumerate(names):
            if i:
                lines.append("")
            lines += run_block(name, now, width, expl, single, evs)
        lines += tail + ([rule(width), footer] if footer else [])
        if rows is None or len(lines) <= rows:
            break
    if rows is not None:
        lines = lines[:rows]
    return [clip(l, width) for l in lines]


def main():
    args = sys.argv[1:]
    every = 0
    if "--watch" in args:
        i = args.index("--watch")
        every = int(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__)
    if not every:
        print("\n".join(frame(args, shutil.get_terminal_size((120, 40)).columns, None, 0)))
        return
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l")  # alternate screen, cursor hidden
    try:
        while True:
            size = shutil.get_terminal_size((120, 40))
            lines = frame(args, size.columns, size.lines, every)
            out.write("\x1b[H" + "\n".join(l + "\x1b[K" for l in lines) + "\x1b[J")
            out.flush()
            end = time.time() + every
            while time.time() < end and shutil.get_terminal_size((120, 40)) == size:
                time.sleep(0.5)
    finally:
        out.write("\x1b[?25h\x1b[?1049l")
        out.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
