"""Live view of agent runs and the training phase: python3 progress.py [<run> ...] [--watch [SECONDS]]

For each run:
- what it works on (runs/<run>/label.txt, if there is one), its status and age, GPU, model and settings;
- passes and code runs used; the agent's turn, tokens and context;
- Now, what the driver is doing at this moment, with how long it has been at it:
  - the agent thinking and writing its next reply (what it replies to, since when, from Claude Code's live
    transcript), with how long its replies have taken so far;
  - running a code run in the kernel, against the kernel's time limit;
  - a forward pass: stopping llama-swap, starting the GPU server, then the request on the GPU (each model's
    answers: generating, waiting, done and how long it took) and grading its answers, with how long earlier
    passes of that kind took;
  - waiting for the usage limit to reset, a Claude Code error and its retry, a kernel restart, a handoff,
    or a driver that is not running;
- Containers: the kernel's up time, memory (now and peak, against its limit) and CPU; the GPU server's
  state (off, loading, busy, or idle and when it is given back);
- the last driver.log event, the start of the agent's latest explanation, and its evaluations (matched by
  submission directory, with their progress), or that its hand-off will start one when the agent finishes.
Below the runs:
- evaluations running now or finished in the last day (eval/results/<name>, not smoke or e2e tests): GPU, how long,
  which suite, how many answers, the speed and about how long the suite has left (from eval/<name>.out), or when
  done the edited model's overall score per suite;
- the training phase's jobs (train/, plans/training-phase.md) running, grading, waiting for their GPU's lock, or
  finished in the last day: GRPO runs (step, pass rate, groups kept, answer length, time per step and left,
  checkpoints), RFT runs (the round's sampling or fine-tuning, and each finished round's pass rate), the log-prob
  agreement check (its agreement, timing and kernels), and checkpoint checks (stage, then the panel mean and validation
  pass rate, paired with the arm's start);
- both GPUs (temperature, power, use, memory, the jobs running on each now and how many wait for it, llama-swap);
- the temperature logger's warnings (gpu_temps.py).

--watch [SECONDS] in a terminal, with rich installed: the live dashboard of progress_ui.py (a panel per GPU with live
bars and estimates, the queue, the checkpoint checks, the evaluations, and the runs named here; the jobs are read every
SECONDS, 5 by default, and the GPUs every second). q or Ctrl-C stops watching; the runs go on.
Without rich, with WSP_PLAIN=1, or when the output is not a terminal, --watch redraws this plain view every SECONDS in
place, on the terminal's alternate screen, fitted to the window (the explanation is shortened first when the window is
small, then the runs' last lines, so the GPUs stay in view). With one run it shows more of the explanation and the last
lines of driver.log.
The runs are optional. --snapshot FILE.svg [--width N] [--height N] draws one screen of the dashboard to a file (.txt
for plain text), for tests.
"""
import calendar
import glob
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


# ---------- what a running run is doing now ----------
# A turn of the driver: it writes turns/tNNNN/prompt.txt and calls Claude Code (the agent thinks, then writes its
# reply), writes answer.md, appends the reply's explanation to explanations.md and runs its block (code in the
# kernel, or a forward request in requests/ for the GPU server), then writes run.json. Claude Code writes its
# transcript as it goes: the user message, an api-request entry when the request is sent, and when the reply is
# done one assistant entry per content block (the thinking and the text, written together).

CLAUDE_PROJECTS = "/home/wspagent/.claude/projects"
_seen = {}   # kept between redraws: finished turns' times, earlier passes' times, CPU samples, container ids
_frame = {}  # kept for one redraw (frame() lays the screen out several times when it is short of rows)


def secs(s):
    """A duration that moves at a 10-second redraw: 45 s, 3 min 05 s, 12 min, 1 h 05 min."""
    s = int(max(s, 0))
    if s < 60:
        return f"{s} s"
    return f"{s // 60} min {s % 60:02d} s" if s < 600 else minutes(s)


def iso(ts):
    try:
        return calendar.timegm(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return None


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def hms(t):
    return time.strftime("%H:%M:%S UTC", time.gmtime(t))


def tail_jsonl(path, nbytes=1 << 20):
    """The JSON lines in the last nbytes of a file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - nbytes, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    out = []
    for l in lines[1:] if size > nbytes else lines:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def reply_progress(name, since):
    """The agent's reply in progress, from Claude Code's live transcript: when the request went out and how
    often; None when the transcript has not got this turn's message. (The thinking and the text are written
    together when the reply is done, so the transcript cannot tell the two apart while they happen.)"""
    d = f"{CLAUDE_PROJECTS}/{re.sub(r'[^A-Za-z0-9]', '-', f'{RUNS}/{name}/agent')}"
    try:
        path = max((f"{d}/{f}" for f in os.listdir(d) if f.endswith(".jsonl")), key=os.path.getmtime, default=None)
    except OSError:
        return None
    if not path or (mtime(path) or 0) < since - 5:
        return None
    key = ("transcript", path)
    size = os.path.getsize(path)
    if _seen.get(key, (None,))[0] == size:
        return _seen[key][1]
    entries = tail_jsonl(path)
    users = [i for i, e in enumerate(entries) if e.get("type") == "user"]
    out = None
    if users and (iso(entries[users[-1]].get("timestamp")) or 0) >= since - 5:
        after = entries[users[-1] + 1:]
        sent = [iso(e.get("timestamp")) for e in after if e.get("type") == "api-request"]
        out = {"sent": next((t for t in reversed(sent) if t), None) or iso(entries[users[-1]].get("timestamp")),
               "requests": len(sent)}
    _seen[key] = (size, out)
    return out


def replying_to(tdir):
    """What the agent's current reply answers, from the turn's prompt."""
    try:
        text = open(f"{tdir}/prompt.txt", errors="replace").read()
    except OSError:
        return "its next message"
    cut = text.startswith("[Your reply was cut at the")
    if cut:
        text = text.split("\n", 1)[-1]
    if m := re.match(r"\[code run (\d+) of \d+", text):
        what = f"code run {m.group(1)}'s output"
        if "ran past the" in text[:400]:
            what += " (it hit the time limit)"
        elif "kernel died" in text[:400]:
            what += " (the kernel died)"
    elif m := re.match(r"\[forward pass (\d+) of \d+ used \(request (\d+)\)", text):
        what = f"pass {m.group(1)}'s result"
    elif m := re.match(r"\[request (\d+) failed", text):
        what = f"request {m.group(1)}'s failure"
    elif text.startswith("[forward request refused"):
        what = "a refused forward request"
    elif m := re.search(r"SESSION RESTART \(([^)]*)\)", text):
        what = f"the restart message ({m.group(1)})"
    elif text.startswith(("Your reply has", "Your block")):
        what = "a note that its reply ran nothing"
    elif len(text) > 5000:
        what = "the opening prompt"
    else:
        what = "its last message"
    return what + (", after a reply cut at the output limit" if cut else "")


def turn_seconds(run):
    """{turn: seconds the agent took to reply} for the finished turns."""
    got = _seen.setdefault(("turns", run), {})
    try:
        names = os.listdir(f"{run}/turns")
    except OSError:
        return got
    for n in names:
        t = int(n[1:]) if re.fullmatch(r"t\d+", n) else None
        if t is not None and t not in got:
            try:
                got[t] = json.load(open(f"{run}/turns/{n}/run.json"))["seconds"]
            except (OSError, ValueError, KeyError):
                pass
    return got


def typical_reply(run, upto):
    ts = sorted(v for t, v in turn_seconds(run).items() if t < upto)
    if len(ts) < 3:
        return None
    longest = max((v, t) for t, v in turn_seconds(run).items() if t < upto)
    return (f"replies so far: half within {secs(ts[len(ts) // 2])}, 9 in 10 within "
            f"{secs(ts[min(len(ts) - 1, int(len(ts) * 0.9))])}, longest {secs(longest[0])} (turn {longest[1]})")


def latest_kind(run, turn):
    """The block kind of the given turn's reply (python or forward) if it had a runnable block."""
    try:
        text = open(f"{run}/explanations.md", errors="replace").read()
    except OSError:
        return None
    heads = re.findall(r"^## Turn (\d+) \((\w+)\)\s*$", text, flags=re.M)
    return heads[-1][1] if heads and int(heads[-1][0]) == turn else None


def container(name):
    """{"cg": cgroup folder, "started": epoch} of a running container, or None."""
    c = _seen.get(("ctr", name))
    if c and os.path.isdir(c["cg"]):
        return c
    parts = sh("docker", "inspect", "-f", "{{.Id}} {{.State.Running}} {{.State.StartedAt}}", name).split()
    if len(parts) != 3 or parts[1] != "true":
        _seen.pop(("ctr", name), None)
        return None
    c = {"cg": f"/sys/fs/cgroup/system.slice/docker-{parts[0]}.scope", "started": iso(parts[2])}
    _seen[("ctr", name)] = c
    return c


def usage(name):
    """A running container's memory (GB, now and peak) and CPU cores (averaged since a sample at least 5 s
    earlier; None until there is one), or None when it is not running."""
    c = container(name)
    if not c:
        return None

    def read(f):
        try:
            return open(f"{c['cg']}/{f}").read()
        except OSError:
            return None
    mem, peak, stat = read("memory.current"), read("memory.peak"), read("cpu.stat") or ""
    out = {"started": c["started"], "mem": int(mem) / 1e9 if mem else None, "peak": int(peak) / 1e9 if peak else None,
           "cores": None}
    if m := re.search(r"usage_usec (\d+)", stat):
        t, u = time.time(), int(m.group(1))
        prev = _seen.get(("cpu", c["cg"]))
        if prev is None or t - prev[0] >= 5:
            cores = (u - prev[1]) / 1e6 / (t - prev[0]) if prev else None
            _seen[("cpu", c["cg"])] = (t, u, cores)
        else:
            cores = prev[2]
        out["cores"] = cores
    return out


def request(run, k):
    """Request k as the driver wrote it, and its wall time once done (cached: neither changes)."""
    key = ("req", run, k)
    if key in _seen:
        return _seen[key]
    try:
        r = json.load(open(f"{run}/requests/fwd_{k}.json"))
    except (OSError, ValueError):
        return None, None
    try:
        wall = json.load(open(f"{run}/requests/fwd_{k}.result.json")).get("wall_seconds")
    except (OSError, ValueError):
        return r, None
    _seen[key] = (r, wall)
    return r, wall


def kind_of(r):
    return "panel" if r.get("panel") else r["type"]


def earlier_passes(run, k, kind):
    walls = [w for j in range(1, k) for r, w in [request(run, j)] if r and w and kind_of(r) == kind]
    if not walls:
        return None
    lo, hi = min(walls), max(walls)
    span = secs(lo) if len(walls) == 1 else f"{secs(lo)} to {secs(hi)}"
    return f"earlier {kind.replace('_', '-')} passes took {span}"


def describe_request(r):
    what = ["panel" if r.get("panel") else r["type"].replace("_", "-")]
    what.append(f"{r.get('baseline', 'original')} vs edited" if r["model"] == "both" else f"model {r['model']}")
    if r.get("panel"):
        what.append(f"{len(r.get('items', []))} questions ({', '.join(r['panel'])})")
    elif r["type"] == "generate":
        parts = [f"{len(r[key])} {key if len(r[key]) != 1 else key[:-1]}" for key in ("problems", "prompts")
                 if r.get(key)]
        n = r.get("n", 1)
        what.append(" + ".join(parts) + (f" × {n}" if n != 1 else ""))
    else:
        what.append(f"{len(r.get('texts') or r.get('chats') or [])} texts")
    return " · ".join(what)


def generation_states(run, k, r, since, now, graded=False):
    """Each model's answers in a generation: generating (and for how long), waiting, done, graded."""
    variants = [r.get("baseline", "original"), "edited"] if r["model"] == "both" else [r["model"]]
    states, t0, first_open = [], since, True
    for v in variants:
        stem = f"{run}/work/gen_{k}_{v}" if len(variants) > 1 else f"{run}/work/gen_{k}"
        if r.get("panel") and not os.path.exists(f"{stem}.raw.jsonl"):  # the driver moves panel answers out of /work
            stem = stem.replace(f"{run}/work/", f"{run}/panel/")
        done = mtime(f"{stem}.raw.jsonl")
        if done is not None:
            took = (f" {'generated ' if graded else ''}in {secs(done - t0)}") if done > t0 else ""
            mark = (green("✓ graded") if os.path.exists(f"{stem}.jsonl") else yellow("▸ grading")) if graded else \
                green("✓")
            states.append(f"{v} {mark} {count(stem + '.raw.jsonl')} answers{took}")
            t0 = done
        elif first_open:
            note = f" (the first model took {secs(t0 - since)})" if states and t0 > since else ""
            states.append(f"{v} {yellow('▸ generating')} {secs(now - t0)}{note}")
            first_open = False
        else:
            states.append(f"{v} {dim('waiting')}")
    return "    ".join(states)


def is_remote(cfg):
    return bool(GPUS.get(cfg.get("gpu", "5090"), {}).get("remote"))


def colab_gpu(run):
    """"Colab (L4)" and so on: the GPU the worker last reported, or the one the run asks for (A100) before any report."""
    name = (colab(run)[0].get("gpu") or {}).get("name", "")
    return f"Colab ({name.replace('NVIDIA ', '').split()[0] if name else 'A100'})"


def colab(run):
    """(the worker's last report as the relay copied it, the relay's phase), each {} when missing."""
    out = []
    for f in ("colab-worker.json", "colab-relay.json"):
        try:
            out.append(json.load(open(f"{run}/requests/{f}")))
        except (OSError, ValueError):
            out.append({})
    return out


def worker_line(w, now):
    """The Colab worker's state from its last report, or a hint when there is none."""
    if not w:
        return "no Colab worker has reported: run colab_runner.ipynb in Colab (README, Colab runner)"
    age = now - float(w.get("updated", now))
    g = w.get("gpu") or {}
    text = f"Colab worker: {w.get('state', '?')}" + (f" on {g['name']}" if g.get("name") else "")
    text += f" · report {secs(age)} old"
    if w.get("error"):
        text += f" · last error: {str(w['error']).splitlines()[-1][:120]}"
    return red(text) if age > 300 else text


def remote_generation_states(r, w, k, since, now):
    """generation_states for a pass on Colab, from the worker's report of the answer files done there."""
    variants = [r.get("baseline", "original"), "edited"] if r["model"] == "both" else [r["model"]]
    ps = w.get("pass") or {}
    files = ps.get("done_files", {}) if ps.get("k") == k else {}
    start = float(ps.get("placed", since)) if ps.get("k") == k else since
    states, t0, first_open = [], 0, True
    for v in variants:
        f = files.get(f"gen_{k}_{v}.raw.jsonl" if len(variants) > 1 else f"gen_{k}.raw.jsonl")
        if f:
            states.append(f"{v} {green('✓')} {f['answers']} answers in {secs(f['after'] - t0)}")
            t0 = f["after"]
        elif first_open:
            states.append(f"{v} {yellow('▸ generating')} {secs(now - start - t0)}")
            first_open = False
        else:
            states.append(f"{v} {dim('waiting')}")
    return "    ".join(states)


def colab_forward_rows(name, run, st, cfg, now, since, head, p, P):
    """forward_rows for a run whose GPU server is a Colab notebook (gpus.py "colab")."""
    req, k = f"{run}/requests", st["requests"]
    w, rl = colab(run)
    path = f"{req}/fwd_{k}.json"
    if (mtime(path) or 0) < since - 1:
        g = container(f"wsp-gpu-{name}")
        if g and not os.path.exists(f"{req}/server.ready"):
            return [row("Now", yellow(head + f"waiting for the Colab worker for pass {p}, "
                                      f"{secs(now - (g['started'] or now))} so far · relay: {rl.get('phase', 'starting')}")),
                    row("", worker_line(w if w.get("session") else {}, now))]
        return [row("Now", head + "checking the forward request")]
    r, _ = request(run, k)
    if r is None:
        return [row("Now", head + f"request {k}")]
    t_req = mtime(path)
    res = mtime(f"{req}/fwd_{k}.result.json")
    what = f"request {k}: {describe_request(r)}"
    if res is None:
        ph = rl.get("phase", "") if rl.get("k") == k else ""
        if ph.startswith(("sending", "receiving")):
            rows = [row("Now", yellow(head + f"pass {p} of {P} on Colab: {ph}, "
                                      f"{secs(now - float(rl.get('since', now)))} · {what}"))]
        else:
            rows = [row("Now", yellow(head + f"pass {p} of {P} on Colab, {secs(now - t_req)} so far · {what}"))]
            if r["type"] == "generate" and w.get("busy") == k:
                rows.append(row("", remote_generation_states(r, w, k, t_req, now)))
            elif w.get("busy") != k:
                rows.append(row("", dim("the worker has not picked the request up yet · ") + worker_line(w, now)))
        if hint := earlier_passes(run, k, kind_of(r)):
            rows.append(row("", dim(hint)))
        return rows
    if r["type"] == "generate":
        return [row("Now", yellow(head + f"grading pass {p}'s answers in the sandbox, {secs(now - res)} so far · {what}")),
                row("", generation_states(run, k, r, t_req, now, graded=True))]
    return [row("Now", yellow(head + f"summarizing pass {p}'s result, {secs(now - res)} so far · {what}"))]


def forward_rows(name, run, st, cfg, log, now, since, turn):
    """The Now rows while a forward block is handled (since: when the reply arrived)."""
    req, k, p, P = f"{run}/requests", st["requests"], st["passes_used"] + 1, cfg["passes"]
    head = f"turn {turn}: "
    if is_remote(cfg):
        return colab_forward_rows(name, run, st, cfg, now, since, head, p, P)
    path = f"{req}/fwd_{k}.json"
    if (mtime(path) or 0) < since - 1:  # this turn's request is not written yet
        last = log[-1] if log else ""
        if "stopping llama-swap" in last and (stamp(last) or 0) >= since - 1:
            return [row("Now", yellow(head + f"stopping llama-swap to free the RTX {cfg.get('gpu', '5090')} for pass {p}"))]
        g = container(f"wsp-gpu-{name}")
        if g and not os.path.exists(f"{req}/server.ready"):
            return [row("Now", yellow(head + f"starting the GPU server for pass {p} · {secs(now - g['started'])}")),
                    row("", dim("loading the model and compiling; earlier starts took 1 to 3 minutes"))]
        return [row("Now", head + "checking the forward request")]
    r, _ = request(run, k)
    if r is None:
        return [row("Now", head + f"request {k}")]
    t_req = mtime(path)
    res = mtime(f"{req}/fwd_{k}.result.json")
    what = f"request {k}: {describe_request(r)}"
    if res is None:
        rows = [row("Now", yellow(head + f"pass {p} of {P} on the GPU, {secs(now - t_req)} so far · {what}"))]
        if r["type"] == "generate":
            rows.append(row("", generation_states(run, k, r, t_req, now)))
        if hint := earlier_passes(run, k, kind_of(r)):
            rows.append(row("", dim(hint)))
        return rows
    if r["type"] == "generate":
        return [row("Now", yellow(head + f"grading pass {p}'s answers in the sandbox, {secs(now - res)} so far · {what}")),
                row("", generation_states(run, k, r, t_req, now, graded=True))]
    return [row("Now", yellow(head + f"summarizing pass {p}'s result, {secs(now - res)} so far · {what}"))]


def activity(name, run, st, cfg, log, now, alive):
    """The Now rows of a running run: what its driver is doing at this moment."""
    if not alive:
        return [row("Now", red(f"the driver is not running · to resume: driver.py {name} in /opt/wsp/harness "
                               "(it keeps a kernel that still answers)"))]
    try:
        turns = sorted(int(n[1:]) for n in os.listdir(f"{run}/turns") if re.fullmatch(r"t\d+", n))
    except OSError:
        turns = []
    if not turns:
        return [row("Now", "starting: the kernel, then the opening prompt")]
    T = turns[-1]
    tdir = f"{run}/turns/t{T:04d}"
    last = log[-1] if log else ""
    lt = stamp(last) or 0
    if st.get("waiting_until"):
        try:
            until = time.mktime(time.strptime(st["waiting_until"], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            until = None
        when = f" · until {clock(until, now)}, in {secs(until - now)}" if until else ""
        rows = [row("Now", yellow(f"turn {T}: waiting for the Claude usage limit to reset{when}"))]
        if m := re.search(r"usage limit \((.*)\); waiting", last):
            rows.append(row("", dim(m.group(1))))
        return rows
    prompt_t = mtime(f"{tdir}/prompt.txt") or now
    answer_t = mtime(f"{tdir}/answer.md")
    if answer_t is None:  # the agent is replying
        if (m := re.search(r"Claude Code error \((.*)\); retrying in (\d+) s", last)) and lt >= prompt_t - 1:
            at = lt + int(m.group(2))
            return [row("Now", red(f"turn {T}: Claude Code failed; " +
                                   (f"retrying at {hms(at)}, in {secs(at - now)}" if at > now else "retrying now"))),
                    row("", dim(m.group(1)))]
        what = replying_to(tdir)
        rp = reply_progress(name, prompt_t)
        line = f"turn {T}: the agent is thinking and writing, {secs(now - (rp['sent'] if rp else prompt_t))} so far"
        if rp and rp["requests"] > 1:
            line += f" (request sent {rp['requests']} times)"
        rows = [row("Now", f"{line} · its reply to {what}")]
        if hint := typical_reply(run, T):
            rows.append(row("", dim(hint)))
        return rows
    if mtime(f"{tdir}/run.json") is not None:
        if "handoff after" in last:
            return [row("Now", f"starting conversation {st['conv'] + 2}: {last[20:]}")]
        return [row("Now", f"turn {T} done · starting turn {T + 1}")]
    kind = latest_kind(run, T)
    if kind == "python":
        n, lim = st["code_runs"] + 1, cfg.get("kernel_seconds", 900)
        if lt >= answer_t - 1 and ("kernel time limit" in last or "kernel died" in last):
            why = "it ran past the time limit" if "time limit" in last else "it died"
            return [row("Now", red(f"turn {T}: restarting the kernel after code run {n}: {why}"))]
        ran = now - answer_t
        line = f"turn {T}: running code run {n} of {cfg['max_code_runs']} in the kernel · {secs(ran)}"
        line += f" of the {lim // 60}-minute limit"
        return [row("Now", red(line) if ran > 0.8 * lim else line)]
    if kind == "forward":
        return forward_rows(name, run, st, cfg, log, now, answer_t, T)
    return [row("Now", f"turn {T}: handling the reply (no runnable block)")]


def containers_row(name, run, cfg, now):
    """The run's kernel (up time, memory against its limit, CPU) and GPU server (loading, busy or idle)."""
    parts = []
    u = usage(f"wsp-kernel-{name}")
    if u:
        lim = cfg.get("kernel_memory_gb", 20)
        s = f"kernel: up {minutes(now - u['started'])}" if u["started"] else "kernel: up"
        if u["mem"] is not None:
            mem = f"{u['mem']:.1f}/{lim} GB" + (f" (peak {u['peak']:.1f})" if u["peak"] else "")
            s += ", " + (red(mem) if u["mem"] > 0.85 * lim else yellow(mem) if (u["peak"] or 0) > 0.85 * lim else mem)
        if u["cores"] is not None:
            s += f", CPU {u['cores']:.1f}/12 cores"
        parts.append(s)
    else:
        parts.append(red("kernel not running"))
    req = f"{run}/requests"
    g = container(f"wsp-gpu-{name}")
    if is_remote(cfg):
        w, rl = colab(run)
        if not g:
            parts.append(dim("Colab relay: off, the next pass starts it and waits for the notebook"))
        elif not os.path.exists(f"{req}/server.ready"):
            parts.append(yellow(f"Colab relay: waiting for the worker, {secs(now - (g['started'] or now))}"))
        else:
            parts.append(f"Colab relay: up {minutes(now - (g['started'] or now))}, {rl.get('phase', '?')}")
        return row("Containers", "    ".join(parts))
    if not g:
        parts.append(dim("GPU server: off, the next pass starts it"))
    elif not os.path.exists(f"{req}/server.ready"):
        parts.append(yellow(f"GPU server: loading, {secs(now - (g['started'] or now))}"))
    else:
        try:
            done = [f for f in os.listdir(req) if f.endswith(".result.json")]
            busy = any(re.fullmatch(r"fwd_\d+\.json", f) and f"{f[:-5]}.result.json" not in done for f in os.listdir(req))
        except OSError:
            done, busy = [], False
        if busy:
            parts.append(yellow("GPU server: busy"))
        else:
            used = max([mtime(f"{req}/{f}") or 0 for f in done] + [mtime(f"{req}/server.ready") or 0])
            idle, limit = now - used, cfg.get("gpu_idle_minutes", 30)
            note = (yellow(f", over {limit}: freed at the next turn") if idle > limit * 60 else
                    dim(f", freed at {limit}"))
            parts.append(f"GPU server: idle {minutes(idle)}{note}")
    return row("Containers", "    ".join(parts))


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
    path = f"{EVAL}/{name}.out"
    try:
        text = open(path, errors="replace").read().replace("\r", "\n")
    except FileNotFoundError:
        return "starting"
    since = time.time() - (mtime(path) or time.time())  # since the last progress line
    last, before = None, {}
    # one line per batch of answers: "<suite> / <model>: <done>/<total> in <the batch's seconds> s (<rate> tok/s)"
    for m in re.finditer(r"^(\w+) / (\w+): (\d+)/(\d+) in (\d+) s(?: \(([\d,.]+) tok/s\))?", text, flags=re.M):
        prev = before.get(m.group(1, 2), 0) if last and last[:2] == m.group(1, 2) else 0
        last = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6), prev
        before[m.group(1, 2)] = int(m.group(3))
    if not last:
        return "loading the model, then " + (SUITE_NAMES.get(suites[0], suites[0]) if suites else "the first suite")
    suite, model, done, total, took, rate, prev = last
    k = suites.index(suite) + 1 if suite in suites else 0
    if done < total:
        batch = done - prev
        left = max(took * (total - done) / batch - since, 0) if batch > 0 else None
        return (f"suite {k} of {len(suites)}, {SUITE_NAMES.get(suite, suite)}, {model}: {done:,}/{total:,} answers"
                + (f" · {rate} tok/s" if rate else "")
                + (f" · about {minutes(left + 59)} left in this suite" if left is not None else ""))
    if model == "original" and "edited" in models:
        return f"suite {k} of {len(suites)}, {SUITE_NAMES.get(suite, suite)}: edited answers starting"
    if k and k < len(suites):
        return (f"suite {k + 1} of {len(suites)}, {SUITE_NAMES.get(suites[k], suites[k])}: first batch generating "
                f"for {secs(since)}")
    return f"grading the answers and writing the report · {secs(since)}"


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
                parts.append(f"{e['name']} " + yellow(f"running on the RTX {e['gpu']}: {e['progress']}"))
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
           dim(f"   {colab_gpu(run) if is_remote(cfg) else 'RTX ' + cfg.get('gpu', '5090')} · "
               f"{model_name(cfg['model'])} {cfg['effort']} effort · "
               f"presence penalty {cfg.get('presence_penalty', 0)} · {start_name(cfg.get('start_from'))}")]

    p, P, c, C = st["passes_used"], cfg["passes"], st["code_runs"], cfg["max_code_runs"]
    passes, code = f"{bar(p, P)} {p}/{P}", f"{bar(c, C)} {c}/{C}"
    if width >= 92:
        out.append(row("Passes", passes + "     " + dim("Code runs  ") + code))
    else:
        out += [row("Passes", passes), row("Code runs", code)]
    tok = st["tokens"]
    turn = st["turn"]
    if status == "running":  # state.json is saved at the end of a turn: the turn under way is the newest folder
        try:
            turn = max([turn] + [int(n[1:]) for n in os.listdir(f"{run}/turns") if re.fullmatch(r"t\d+", n)])
        except OSError:
            pass
    conv_turns = st["conv_turns"] + turn - st["turn"]
    if cfg.get("handoff_every"):  # runs before 2026-11-05 19:00 UTC restart every N turns
        agent = (f"turn {turn} · conversation {st['conv'] + 1} ({conv_turns}/{cfg['handoff_every']} "
                 f"turns) · {k(tok['output_tokens'])} output tokens · context {k(st['context'])}")
    else:
        agent = (f"turn {turn} · conversation {st['conv'] + 1} ({conv_turns} turns) · "
                 f"{k(tok['output_tokens'])} output tokens · context {k(st['context'])}"
                 f"/{k(cfg.get('max_context', 400_000))}")
    if st.get("context_warned"):
        agent += yellow(" · told the conversation restarts soon")
    if st.get("limit_waits"):
        agent += yellow(f" · waited for the usage limit {st['limit_waits']}× ({minutes(st['limit_wait_seconds'])})")
    out.append(row("Agent", agent))

    if status == "running":
        key = ("live", name)
        if key not in _frame:
            _frame[key] = activity(name, run, st, cfg, log, now, alive) + [containers_row(name, run, cfg, now)]
        out += _frame[key]
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


# --- the training phase (train/, plans/training-phase.md) ---

TRAIN = "/opt/wsp/train"
TRAIN_KINDS = {"grpo": "GRPO", "rft": "RFT", "check": "", "ckpt": "check"}
STAMPED = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d ")


def tail_text(path, nbytes=1 << 16):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(f.tell() - nbytes, 0))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def failure(lines):
    """The last exception line of a log, or its last line."""
    for l in reversed(lines):
        if re.match(r"[\w.]*(Error|Exception)\b", l.strip()):
            return l.strip()
    return next((l.strip() for l in reversed(lines) if l.strip()), "no log")


def gpu_in_log(path):
    """The GPU a launcher waited for, from the first lines of its log."""
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("utf-8", "replace").splitlines()
    except OSError:
        return "?"
    return next((m.group(1) for l in head if (m := re.search(r"waiting for the (\S+)$", l))), "?")


def json_or(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def arg(args, flag, default):
    m = re.search(rf"{flag} (\S+)", args)
    return m.group(1) if m else default


def train_processes():
    """The training phase's launchers alive now: [{"pid", "kind", "gpu", "name", "args"}]."""
    out = []
    pat = r"/opt/wsp/train/(run-train|run-grpo|run-ckpt)[.]sh|flock .* --name wsp-train-check-"
    for line in sh("pgrep", "-af", pat).splitlines():
        if m := re.match(r"(\d+) bash /opt/wsp/train/run-train\.sh (\S+) (grpo|rft) (\S+)(.*)", line):
            out.append({"pid": m[1], "kind": m[3], "gpu": m[2], "name": m[4], "args": m[5]})
        elif m := re.match(r"(\d+) bash /opt/wsp/train/run-grpo\.sh (\S+) (\S+)(.*)", line):
            out.append({"pid": m[1], "kind": "grpo", "gpu": m[2], "name": m[3], "args": m[4]})
        elif m := re.match(r"(\d+) flock \S+ docker run --rm --name wsp-train-check-(push-)?(\S+)", line):
            out.append({"pid": m[1], "kind": "check", "gpu": m[3], "name": f"{'push' if m[2] else 'log-probs'}-{m[3]}",
                        "args": ""})
        elif m := re.match(r"(\d+) bash /opt/wsp/train/run-ckpt\.sh (\S+) (\S+) \S+ ?(\S*)", line):
            out.append({"pid": m[1], "kind": "ckpt", "gpu": m[2], "name": m[3], "args": m[4]})
    return out


def ckpt_stage(name):
    """The last stage run-ckpt.sh logged for a check: waiting, grading the panel, grading the validation answers."""
    for l in reversed(tail_text(f"{TRAIN}/evals/{name}/log")):
        for key in ("grading the validation answers", "grading the panel", "waiting for the"):
            if STAMPED.match(l) and key in l:
                return "waiting" if key == "waiting for the" else key
    return "waiting"


def grpo_detail(name, args, j, now):
    d = f"{TRAIN}/runs/{name}"
    total = json_or(f"{d}/config.json", {}).get("steps") or int(arg(args, "--steps", 150))
    rows = tail_jsonl(f"{d}/steps.jsonl")
    if j["state"] == "failed":
        return red(clip(failure(tail_text(f"{d}/train.log")), 70)) + (dim(f" · after step {rows[-1]['step']}") if rows else "")
    if j["state"] == "stopped":  # train/runs/<arm>/stopped holds why (a stop rule, or two failed attempts)
        why = (open(f"{d}/stopped").read().strip() if os.path.exists(f"{d}/stopped") else "") or "stopped"
        return yellow(clip(why, 80)) + (dim(f" · after step {rows[-1]['step']}") if rows else "")
    if not rows:
        cfg = json_or(f"{d}/config.json", {})
        shape = (f"{total} steps · {cfg.get('prompts') or arg(args, '--prompts', 64)} problems × "
                 f"{cfg.get('group') or arg(args, '--group', 8)} answers, at most "
                 f"{int(cfg.get('max_new') or arg(args, '--max-new', 16384)):,} tokens")
        if j["state"] == "queued":  # what it waits for first: lines are cut at 130 columns
            held = f"{d}/stopped"  # put there before the run started: it is on hold, and the file says why
            what = yellow(clip(open(held).read().strip(), 100)) if os.path.exists(held) else dim(shape)
            return (f"after {j['after']} · " if j.get("after") else "") + what
        return shape + (dim(" · loading vLLM and the trainer, then step 1") if j["state"] == "running" else "")
    r, last = rows[-1], rows[-5:]
    per = sum(x["seconds"]["total"] for x in last) / len(last)
    n_ck = len([x for x in os.listdir(f"{d}/checkpoints")]) if os.path.isdir(f"{d}/checkpoints") else 0
    s = (f"{bar(r['step'], total, 10)} step {r['step']}/{total} · pass {100 * r['pass_rate']:.0f}% · "
         f"kept {r['groups_kept']}/{r['problems']} groups · {r['mean_tokens']:,.0f} tok/answer · {secs(per)}/step")
    if j["state"] == "running" and r["step"] < total:
        s += f" · ≈ {minutes((total - r['step']) * per)} left"
    if j["state"] == "done":
        s += f" · pass {100 * rows[0]['pass_rate']:.0f}% → {100 * r['pass_rate']:.0f}%" if len(rows) > 1 else ""
    return s + (f" · {n_ck} checkpoint{'s' if n_ck != 1 else ''}" if n_ck else "")


def rft_detail(name, args, j, now):
    d = f"{TRAIN}/runs/{name}"
    cfg = json_or(f"{d}/config.json", {})
    rounds = cfg.get("rounds") or int(arg(args, "--rounds", 2))
    done = [s for r in range(1, rounds + 1) if (s := json_or(f"{d}/round_{r}/summary.json"))]
    parts = [f"round {s['round']}: pass {100 * s['pass_rate']:.0f}%, {s['kept_answers']:,} kept" for s in done]
    lines = tail_text(f"{d}/train.log")
    if j["state"] == "failed":
        return red(clip(failure(lines), 70))
    if j["state"] == "queued":  # e.g. a run resuming at a later round (queue.json's "note" says so)
        return ((f"after {j['after']} · " if j.get("after") else "") + (j.get("note") or f"{rounds} rounds")
                + "".join(f" · {p}" for p in parts))
    if j["state"] == "waiting":
        n = cfg.get("problems") or arg(args, "--problems", "all")
        return f"{rounds} rounds · {n} problems × {cfg.get('samples') or arg(args, '--samples', 8)} answers"
    stamped = [l[20:] for l in lines if STAMPED.match(l)]
    events = [l for l in stamped if re.match(r"round \d+: ", l)]
    now_s = ""
    if j["state"] == "running":
        now_s = events[-1] if events else "loading vLLM"
        if stamped and stamped[-1].startswith("vLLM now holds"):  # sampling has begun; no chunk is logged yet
            r = 1 + max([int(m[1]) for l in events if (m := re.match(r"round (\d+): (checkpoint saved|already done)", l))],
                        default=0)
            now_s = f"round {r}: sampling the first {cfg.get('chunk') or 256} problems"
    return " · ".join(([now_s] if now_s else []) + parts) or f"{rounds} rounds"


def push_detail(gpu, j):
    if j["state"] == "waiting":
        return "does vLLM get the trainer's weights each step? a large change taken through a step"
    if j["state"] == "running":
        return "taking a large change through a training step's sleep, wake and push"
    out = json_or(f"{TRAIN}/checks/push-{gpu}.json")
    if j["state"] == "failed" or not out:
        return red(clip(failure(tail_text(f"{TRAIN}/checks/push-{gpu}.log")), 70))
    s = (f"vLLM vs trainer {out['vllm_vs_trainer_before']['mean_abs']:.4f} before, "
         f"{out['vllm_vs_trainer_after_change']['mean_abs']:.4f} after a change that moved vLLM by "
         f"{out['vllm_moved_by_the_change']['mean_abs']:.3f}, {out['vllm_vs_trainer_after_undo']['mean_abs']:.4f} undone")
    return ("push works · " if out["pass"] else red("push FAILED · ")) + s


def check_detail(name, args, j, now):
    gpu = name.rsplit("-", 1)[-1]
    if name.startswith("push-"):
        return push_detail(gpu, j)
    if j["state"] == "waiting":
        return "trainer vs vLLM log-probs on 50 answers, and fwd+bwd timing"
    if j["state"] == "running":
        return "sampling 50 answers, then scoring them with the trainer"
    out = json_or(f"{TRAIN}/checks/logprobs-{gpu}.json")
    if j["state"] == "failed" or not out:
        return red(clip(failure(tail_text(f"{TRAIN}/checks/logprobs-{gpu}.log")), 70))
    a, fb = out["agreement"], out["fwd_bwd"]
    if "kernels" in out:  # recorded from transformers' decorator: fla, or the slow reference path
        fast = out["kernels"].get("torch_chunk_gated_delta_rule", "").startswith("fla.")
    else:
        fast = not any("gated_delta" in w for w in out["linear_attention_fallback_warnings"])
    kernels = "delta rule on fla" if fast else red("delta rule on the slow reference path")
    times = ", ".join(f"{int(n) // 1024}k {v['seconds']} s" for n, v in sorted(fb.items(), key=lambda x: int(x[0])))
    peak = max(v["peak_gb"] for v in fb.values())
    recompiles = any(v.get("new_length_seconds", 0) > 3 * v["seconds"] + 0.5 for v in fb.values())
    return (f"mean |Δ| {a['mean_abs']:.4f} · p99 {a['abs_p50_p90_p99_p999'][2]:.3f} · over 0.1: "
            f"{100 * a['share_abs_over_0.1']:.1f}% · fwd+bwd {times}, peak {peak} GB · {kernels}"
            + (yellow(" · recompiles for each new length") if recompiles else ""))


def ckpt_detail(name, args, j, now):
    d = f"{TRAIN}/evals/{name}"
    base = args or json_or(f"{d}/summary.json", {}).get("vs_base", {}).get("base")
    if j["state"] == "waiting":
        return "panel 1,000 + validation 252 × 4" + (dim(f" · paired with {base}") if base else "")
    if j["state"] == "running":
        return "generating 2,008 answers (the panel, then the validation problems)"
    if j["state"] == "grading":
        return j["stage"]
    s = json_or(f"{d}/summary.json")
    if j["state"] == "failed" or not s:
        return red(clip(failure(tail_text(f"{d}/log")), 70))
    key = "val2" if "val2" in s else "val"  # val2: on train/data2's tests, which the rules read from 2026-11-08
    out = f"panel mean {s['panel_mean']:.1f}% · validation {s[key]['pass']:.1f}%" + (" (new tests)" if key == "val2" else "")
    if v := s.get("vs_base"):
        pm, vv = v["panel_mean"], v.get(key, v["val"])
        out += (f" · against {v['base']}: panel {pm['diff']:+.1f} (SE {pm['se']:.1f}), "
                f"validation {vv['diff']:+.1f} (SE {vv['se']:.1f})")
    return out


TRAIN_DETAIL = {"grpo": grpo_detail, "rft": rft_detail, "check": check_detail, "ckpt": ckpt_detail}


def train_jobs(now, keep=24 * 3600):
    """The training phase's jobs: running, grading, or waiting for their GPU's lock, then those finished in the last
    day, newest first. [{"name", "kind", "gpu", "state", "t", "detail"}]; state is running, grading, waiting, done or
    failed, and t is when it started (live) or ended (finished)."""
    procs = train_processes()
    boxes = set(sh("docker", "ps", "--format", "{{.Names}}").split())
    starts = {}
    if procs:
        for l in sh("ps", "-o", "pid=,etimes=", "-p", ",".join(p["pid"] for p in procs)).splitlines():
            if len(l.split()) == 2 and l.split()[1].isdigit():
                starts[l.split()[0]] = now - int(l.split()[1])
    jobs, live = [], set()
    for p in procs:
        kind, name, gpu = p["kind"], p["name"], p["gpu"]
        live.add((kind, name))
        box = {"check": f"wsp-train-check-{'push-' if name.startswith('push-') else ''}{gpu}",
               "ckpt": f"wsp-ckpt-{name.replace('/', '-')}"}.get(kind, f"wsp-train-{name}")
        j = {"name": name, "kind": kind, "gpu": gpu, "t": starts.get(p["pid"], now), "args": p["args"]}
        if box in boxes:
            j["state"] = "running"
            j["t"] = iso(sh("docker", "inspect", "-f", "{{.State.StartedAt}}", box)) or j["t"]
        elif kind == "ckpt" and (stage := ckpt_stage(name)) != "waiting":
            j["state"], j["stage"] = "grading", stage
        else:
            j["state"] = "waiting"
        jobs.append(j)
    qt = mtime(f"{TRAIN}/queue.json") or 0
    for q in json_or(f"{TRAIN}/queue.json", []) or []:  # queued by a chain (train/grpo-queue-*.sh), not started yet
        if isinstance(q, dict) and q.get("kind") in TRAIN_DETAIL and (q["kind"], q.get("name")) not in live:
            if (mtime(f"{TRAIN}/runs/{q['name']}/train.log") or 0) > qt:  # it has run since it was queued
                continue
            after = q.get("after", "")  # a hook run before it (train/queue/after-<after>.sh) names itself in "# shows:"
            shows = next((l[8:].strip() for l in tail_text(f"{TRAIN}/queue/after-{after}.sh")[:5]
                          if l.startswith("# shows:")), "") if after else ""
            held = os.path.exists(f"{TRAIN}/runs/{q['name']}/stopped")  # the chain skips it when its turn comes
            jobs.append({"name": q["name"], "kind": q["kind"], "gpu": q.get("gpu", "?"), "t": now, "state": "queued",
                         "args": q.get("args", ""), "after": f"{after} and {shows}" if shows else after,
                         "note": "(on hold)" if held else q.get("note", "")})
    queued = {j["name"] for j in jobs if j["state"] == "queued"}  # an earlier attempt of these is not their end
    ended = []
    for log in glob.glob(f"{TRAIN}/runs/*/train.log"):
        name, t = os.path.basename(os.path.dirname(log)), mtime(log)
        cfg = json_or(f"{os.path.dirname(log)}/config.json", {})
        kind = "rft" if "rounds" in cfg or os.path.isdir(f"{os.path.dirname(log)}/round_1") else "grpo"
        if any(n == name for k_, n in live if k_ in ("grpo", "rft")) or name in queued or not t or now - t > keep:
            continue
        lines = tail_text(log)
        ok = any(STAMPED.match(l) and l[20:] == "done" for l in lines[-50:])
        stopped = os.path.exists(f"{os.path.dirname(log)}/stopped")
        ended.append({"name": name, "kind": kind, "gpu": gpu_in_log(log), "t": t,
                      "state": "done" if ok else "stopped" if stopped else "failed", "args": ""})
    for log in glob.glob(f"{TRAIN}/checks/*.log"):
        m, t = re.fullmatch(r"(logprobs|push)-(\w+)\.log", os.path.basename(log)), mtime(log)
        if not m:  # e.g. logprobs-5080.run1.log, a kept earlier run
            continue
        gpu, name = m[2], f"{'push' if m[1] == 'push' else 'log-probs'}-{m[2]}"
        if ("check", name) in live or not t or now - t > keep:
            continue
        ok = os.path.exists(log[:-4] + ".json") and mtime(log[:-4] + ".json") >= t - 5
        ended.append({"name": name, "kind": "check", "gpu": gpu, "t": t, "state": "done" if ok else "failed", "args": ""})
    for log in glob.glob(f"{TRAIN}/evals/**/log", recursive=True):
        name, t = os.path.relpath(os.path.dirname(log), f"{TRAIN}/evals"), mtime(log)
        if ("ckpt", name) in live or not t or now - t > keep:
            continue
        ok = os.path.exists(f"{os.path.dirname(log)}/summary.json")
        ended.append({"name": name, "kind": "ckpt", "gpu": gpu_in_log(log), "t": t,
                      "state": "done" if ok else "failed", "args": ""})
    order = {"running": 0, "grading": 1, "waiting": 2, "queued": 3}
    jobs.sort(key=lambda j: (order[j["state"]], j["t"]))
    ended.sort(key=lambda j: -j["t"])
    for j in jobs + ended:
        try:
            j["detail"] = TRAIN_DETAIL[j["kind"]](j["name"], j["args"], j, now)
        except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as e:
            j["detail"] = red(f"cannot read its progress: {type(e).__name__}")
    return jobs + ended


def train_label(j):
    if j["kind"] == "check":
        return "push check" if j["name"].startswith("push-") else "log-prob check"
    return f"{j['name']} {TRAIN_KINDS[j['kind']]}"


def train_block(jobs, now, most=10):
    if not jobs:
        return [row("Training", dim("nothing running, waiting or finished in the last day (train/)"))]
    width = max(len(train_label(j)) for j in jobs[:most]) + 2
    out = []
    for j in jobs[:most]:
        state = {"running": lambda: yellow(f"▸ running {minutes(now - j['t'])}"),
                 "grading": lambda: yellow("▸ grading"),
                 "waiting": lambda: dim(f"◷ waiting {minutes(now - j['t'])}"),
                 "queued": lambda: dim("◷ queued"),
                 "done": lambda: green(f"✓ done {clock(j['t'], now)}"),
                 "stopped": lambda: yellow(f"■ stopped {clock(j['t'], now)}"),
                 "failed": lambda: red(f"✗ failed {clock(j['t'], now)}")}[j["state"]]()
        out.append(row("Training" if not out else "",
                       f"{train_label(j):<{width}}{dim('RTX ' + j['gpu'])}  {pad(state, 23)}{j['detail']}"))
    if len(jobs) > most:
        out.append(row("", dim(f"and {len(jobs) - most} more")))
    return out


def gpu_block(train=()):
    users = running_jobs()
    for j in train:
        if j["state"] == "running":
            users.setdefault(j["gpu"], []).append({"check": train_label(j), "ckpt": f"check {j['name']}"}
                                                  .get(j["kind"], f"train {j['name']}"))
    waiting, queued = {}, {}
    for j in train:
        if j["state"] == "waiting":
            waiting[j["gpu"]] = waiting.get(j["gpu"], 0) + 1
        elif j["state"] == "queued":
            queued[j["gpu"]] = queued.get(j["gpu"], 0) + 1
    q = sh("nvidia-smi", "--query-gpu=index,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
           "--format=csv,noheader,nounits")
    rows = {r[0]: r for r in (l.split(", ") for l in q.splitlines()) if len(r) == 6}
    out = []
    for card, g in GPUS.items():
        if g.get("remote"):
            continue
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
        if waiting.get(card):
            who += dim(f" +{waiting[card]} waiting")
        if queued.get(card):
            who += dim(f" +{queued[card]} queued")
        if g["llama_swap"]:
            swap = sh("docker", "inspect", "-f", "{{.State.Status}}", "llama-swap")
            extra = f"llama-swap {'stopped' if swap == 'exited' else swap}"
        else:
            extra = "shared: other services"
        out.append(row(label, f"RTX {card}  {temp}  {p:>4.0f} W  {u:>3} %  {mu:4.1f} / {mt:4.1f} GB   "
                              f"{pad(who, 24)}  {dim(extra)}"))
    for card, g in GPUS.items():
        if g.get("remote"):
            out += colab_rows(card, g, users.get(card, []))
    return out


def colab_rows(card, g, users):
    """The Colab GPU's row, from the worker's last report, for each run on it (none: no row)."""
    out, now = [], time.time()
    for label in users:
        run = f"{RUNS}/{label.split()[-1]}"
        w, _ = colab(run)
        gpu = w.get("gpu") or {}
        if not gpu:
            out.append(row("", f"Colab  {pad(label, 24)}  {dim(worker_line(w, now))}"))
            continue
        t = float(gpu.get("temp", 0))
        temp = f"{t:>3.0f} °C"
        temp = red(temp) if t >= 83 else yellow(temp) if t >= 75 else temp
        age = now - float(w.get("updated", now))
        up = (now - float(w["started"])) / 3600 if w.get("started") else 0
        rate = next((r for k, r in g.get("units_by_gpu", {}).items() if k in gpu.get("name", "")), g["units_per_hour"])
        extra = f"report {secs(age)} old, up {up:.1f} h ≈ {up * rate:.0f} units"
        out.append(row("", f"{gpu.get('name', 'Colab').replace('NVIDIA ', '')[:14]}  {temp}  {float(gpu.get('power', 0)):>4.0f} W  "
                           f"{int(gpu.get('util', 0)):>3} %  {float(gpu.get('mem_used_gb', 0)):4.1f} / "
                           f"{float(gpu.get('mem_total_gb', 0)):4.1f} GB   {pad(label, 24)}  "
                           + (red(extra) if age > 300 else dim(extra))))
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
    _frame.clear()
    single = len(names) == 1
    header = spread(f" {bold('Agent runs and evaluations')}  {dim('Qwen3.5-0.8B weight editing')}",
                    time.strftime("%H:%M:%S UTC", time.gmtime(now)), width)
    footer = dim(f" redraws every {every} s · Ctrl-C stops watching; the runs carry on") if every else None
    evs, tj = evaluations(now), train_jobs(now)
    tail = ([rule(width)] + eval_block(evs, now) + [rule(width)] + train_block(tj, now) + [rule(width)]
            + gpu_block(tj) + [temp_line()] + ([rule(width), footer] if footer else []))
    for expl in ([8, 4, 2, 0] if single else [3, 2, 1, 0]):
        body = []
        for i, name in enumerate(names):
            if i:
                body.append("")
            body += run_block(name, now, width, expl, single, evs)
        if rows is None or 2 + len(body) + len(tail) <= rows:
            break
    if rows is not None and 2 + len(body) + len(tail) > rows:
        # Still too tall with the shortest explanation: cut the runs' last lines (driver.log), not the GPUs below.
        keep = max(rows - 2 - len(tail), 0)
        body = body[:keep - 1] + [dim("   … (window too short for the rest of the run)")] if keep else []
    lines = [header, rule(width)] + body + tail
    if rows is not None:
        lines = lines[:rows]
    return [clip(l, width) for l in lines]


def main():
    args = sys.argv[1:]
    every, opts = 0, {}
    if "--watch" in args:
        i = args.index("--watch")
        n = args[i + 1] if i + 1 < len(args) and args[i + 1].isdigit() else None
        every = int(n) if n else 5
        del args[i:i + (2 if n else 1)]
    for flag in ("--snapshot", "--width", "--height"):
        if flag in args:
            i = args.index(flag)
            opts[flag] = args[i + 1]
            del args[i:i + 2]
    if any(a.startswith("-") for a in args):
        sys.exit(__doc__)
    try:
        import progress_ui as ui  # the live dashboard (rich); without rich, the plain view below
    except ImportError:
        ui = None
    if "--snapshot" in opts:
        if not ui:
            sys.exit("--snapshot needs rich: /opt/wsp/venv/bin/python -m pip install rich")
        ui.snapshot(args, opts["--snapshot"], int(opts.get("--width", 130)), int(opts.get("--height", 50)))
        return
    if every and ui and sys.stdout.isatty() and not os.environ.get("WSP_PLAIN"):
        ui.watch(args, every)
        return
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
