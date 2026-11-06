"""Live view of agent runs: python3 progress.py <run> [<run> ...] [--watch SECONDS]

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


def gpu_block():
    users = running_jobs()
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
