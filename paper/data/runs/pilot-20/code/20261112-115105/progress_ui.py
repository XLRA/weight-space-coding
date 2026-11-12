"""The live dashboard of progress.py --watch, drawn with rich: python3 progress.py [<run> ...] --watch [SECONDS]

From the top:
- Header: how many jobs run and are queued, how many ended in the last day, llama-swap, the time, and a heartbeat
  that turns while the screen is live.
- One panel per GPU: the job holding it, with a bar, how long it has run and about how long it has left, and the
  results it has so far; then, lined up across the panels, what comes next on that GPU, the card's temperature,
  power, use and memory, read every second, and its power over the last ten minutes.
- Queue: the training jobs queued, in the order they will run, then the runs that ended in the last day.
- Checkpoint checks: panel and validation pass rates, and the paired differences from the arm's start with their SE
  (green when above 2 SE, red when below -2 SE).
- Held-out evaluations finished in the last day, by suite.
- The agent runs named on the command line: all of progress.py's view while one runs, one line once it has ended.
- Footer, on the window's last row: the keys, what the bar colours mean, the temperature logger, and how old the
  readings are.
The screen is fitted to the window once a second. A short window gives up, in turn: the evaluations past the third,
the runs that ended, the chart's second row, the checks past the third, the other evaluations, the chart, the other
checks, then the queue; the GPU panels, the agent runs and the footer always show.
The jobs are read every SECONDS (5 by default, r reads them now) and the GPUs every second. The screen is drawn ten
times a second, and only the lines that changed are written. q or Ctrl-C stops watching; the runs go on.

The bars are cell backgrounds, so they are solid whatever the terminal's font: the part the logs confirm, then the
estimated part (≈) in a darker tone. A cell the bar only partly covers takes a blend of the two colours, so the bar
moves by fractions of a cell. A light sweeps along a running job's bar; a block slides along the bar of a stage with
no measure of progress (loading, grading, a check past its reference time). The colours follow the terminal's own
background and text colours when it reports them (OSC 11 and 10), else a dark scheme.

Estimates fill the gaps between log lines, since the logs say little while a model generates:
- An RFT round logs each chunk of problems when it ends. Every round samples the same chunks, and a chunk's time
  follows its problems, so each chunk is predicted from the last full round's time for it, scaled by how this
  round's finished chunks compare; in a first round, from the round's time per problem so far.
- A checkpoint check logs nothing until it has generated every answer, so its bar runs against the generation time
  of the same arm's last check, or of its start's check.
- A GRPO step moves at the pace of the last five steps.
Each estimate stops at 99% until the log confirms it.

progress.py --snapshot FILE.svg|FILE.txt [--width N] [--height N] draws one screen to a file, for tests.
"""
import collections
import functools
import glob
import importlib
import math
import os
import re
import select
import sys
import threading
import time
import traceback

from rich import box
from rich.color import ColorSystem
from rich.console import Console, Group
from rich.measure import Measurement
from rich.padding import Padding
from rich.panel import Panel
from rich.segment import Segment
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

_main = sys.modules.get("__main__")
P = _main if hasattr(_main, "train_jobs") else importlib.import_module("progress")

SPARK = "▁▂▃▄▅▆▇█"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
LOAD = 150  # seconds from a check's container starting to its first answer: vLLM loads and compiles
LANE_MIN = 56  # the narrowest a GPU panel may be when two sit side by side
FPS = 10
CHART_MIN = 10  # minutes of power in each GPU panel


# ---------- colours ----------

SCHEMES = {
    "dark": dict(bg="#1e1e1e", fg="#d4d4d4", ok="#98c379", warn="#e5c07b", bad="#e06c75", acc="#61afef",
                 est="#8ab4d8", dim="#8b929e", faint="#5c6370", lab="#a0a6b0", head="#e8e8e8",
                 fill="#61afef", fill_est="#3b6a94", shine="#ffffff", chart="#4a86b8"),
    "light": dict(bg="#ffffff", fg="#1f2328", ok="#2e7d32", warn="#9a6700", bad="#c62828", acc="#1f6fd1",
                  est="#3a6ea5", dim="#5f6670", faint="#8c939c", lab="#4b525c", head="#1f2328",
                  fill="#2f7fd8", fill_est="#9dc3ec", shine="#ffffff", chart="#4f8fdb"),
}
PAL = {}


def rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t):
    """The colour t of the way from a to b."""
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(rgb(a), rgb(b)))


def luminance(h):
    c = [v / 255 for v in rgb(h)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def set_palette(bg=None, fg=None):
    """The colours, fitted to the terminal's background and text colours when they are known. Returns the theme."""
    PAL.clear()
    PAL.update(SCHEMES["light" if bg and luminance(bg) > 0.4 else "dark"])
    PAL.update({k: v for k, v in (("bg", bg), ("fg", fg)) if v})
    PAL["track"] = mix(PAL["bg"], PAL["fg"], 0.13)  # a bar's empty part: just off the background
    PAL["band"] = mix(PAL["bg"], PAL["fg"], 0.07)  # the header's background
    return Theme({"ok": PAL["ok"], "warn": PAL["warn"], "bad": "bold " + PAL["bad"], "run": "bold " + PAL["acc"],
                  "acc": PAL["acc"], "est": PAL["est"], "dim": PAL["dim"], "faint": PAL["faint"],
                  "lab": PAL["lab"], "head": "bold " + PAL["head"], "band": "on " + PAL["band"]})


THEME = set_palette()


@functools.lru_cache(maxsize=4096)
def on(colour):
    return Style(bgcolor=colour)


def seed(name):
    """A fixed offset per job, so bars side by side do not sweep in step."""
    return sum(map(ord, name)) % 97 / 10


# ---------- formatting ----------

def human(n):
    n = float(n)
    for unit, d in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= d:
            return f"{n / d:.1f}{unit}"
    return f"{n:.0f}"


def ticking(s):
    """A running time that ticks every second: 4:07:12, or 7:12 under an hour."""
    s = int(max(s, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def about(s):
    """An estimated time, to the minute."""
    return "≈ " + (P.minutes(s + 59) if s >= 60 else "1 min")


def hhmm(t, now):
    return P.clock(t, now).replace(" UTC", "")


def pct(x):
    return f"{100 * x:.1f}%"


def signed(diff, se):
    """A paired difference with its SE, coloured when it is beyond 2 SE."""
    style = "ok" if diff > 2 * se else "bad" if diff < -2 * se else "dim"
    return Text(f"{diff:+.1f} ±{se:.1f}", style=style)


class Gauge:
    """A bar drawn with cell backgrounds, so it is solid whatever the font. done: the part the logs confirm; est: up
    to where the estimate has got, in a darker tone; a cell partly covered takes a blend of the colours, so the bar
    moves by fractions of a cell. While the job runs (live) a soft light sweeps along the filled part. pulse: a block
    slides along the bar, for a stage with no measure of progress."""

    def __init__(self, done=0.0, est=None, now=None, pulse=False, live=True, seed=0.0):
        self.done = min(max(done, 0.0), 1.0)
        self.est = min(max(self.done if est is None else est, self.done), 1.0)
        self.now, self.pulse, self.live, self.seed = now, pulse, live, seed

    def __rich_measure__(self, console, options):
        return Measurement(4, options.max_width)

    def colours(self, w):
        now = time.time() if self.now is None else self.now
        track = PAL["track"]
        if self.pulse:
            seg, period = max(4.0, w / 5), 2.0 + w / 40
            left = ((now + self.seed) % period) / period * (w + seg) - seg
            out = []
            for i in range(w):
                x = (i + 0.5 - left) / seg
                out.append(mix(track, PAL["fill"], min(1.0, 1.5 * (1 - abs(2 * x - 1)))) if 0 <= x <= 1 else track)
            return out
        nd, ne = self.done * w, self.est * w
        out = []
        for i in range(w):
            c, ce, cd = track, min(max(ne - i, 0.0), 1.0), min(max(nd - i, 0.0), 1.0)
            if ce:
                c = mix(c, PAL["fill_est"], ce)
            if cd:
                c = mix(c, PAL["fill"], cd)
            out.append(c)
        if self.live and ne > 0:
            span = max(ne, 8.0)
            period = 1.4 + span / 30
            p = ((now + self.seed) % period) / period * (span + 12) - 6
            for i in range(min(w, math.ceil(ne))):
                d = abs(i + 0.5 - p)
                if d < 6:
                    out[i] = mix(out[i], PAL["shine"], 0.32 * (1 - d / 6) ** 2 * min(1.0, ne - i))
        return out

    def __rich_console__(self, console, options):
        cells = self.colours(max(options.max_width, 1))
        if console.no_color:
            yield Segment("".join("░" if c == PAL["track"] else "█" for c in cells))
            return
        run, n = cells[0], 0
        for c in cells:
            if c == run:
                n += 1
            else:
                yield Segment(" " * n, on(run))
                run, n = c, 1
        yield Segment(" " * n, on(run))


class Chart:
    """Power over the last `minutes`, `rows` lines tall, in a quieter tone than the bars: one column per equal slice of
    time, on fixed boundaries so the columns do not shift as the clock runs. A slice with no reading repeats the one
    before (from before the window, for the first), so it has no holes."""

    def __init__(self, points, now, top, rows=1, minutes=CHART_MIN):
        self.points, self.now, self.top, self.rows, self.minutes = points, now, top, rows, minutes

    def __rich_measure__(self, console, options):
        return Measurement(4, options.max_width)

    def __rich_console__(self, console, options):
        w = max(options.max_width, 1)
        span = self.minutes * 60 / w
        start = (self.now // span + 1) * span - w * span
        sums, counts, before = [0.0] * w, [0] * w, (None, None)
        for t, v in self.points:
            i = int((t - start) // span)
            if 0 <= i < w:
                sums[i] += v
                counts[i] += 1
            elif i < 0 and (before[0] is None or t > before[0]):
                before = (t, v)
        vals, last = [], before[1]
        for i in range(w):
            if counts[i]:
                last = sums[i] / counts[i]
            vals.append(last)
        style = Style(color=PAL["chart"])
        for r in range(self.rows - 1, -1, -1):  # the top line first
            line = []
            for v in vals:
                if v is None or not self.top:
                    line.append(" ")
                    continue
                k = round(min(v / self.top, 1.0) * 8 * self.rows) - 8 * r
                if r == 0 and v > 0:
                    k = max(k, 1)  # a reading above zero always shows
                line.append(" " if k <= 0 else "█" if k >= 8 else SPARK[k - 1])
            yield Segment("".join(line), style)
            if r:
                yield Segment.line()


def bar_cell(b, right):
    """A bar that takes the row's width, with a short text at its right."""
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(ratio=1, no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    t.add_row(b, right)
    return t


def kv(rows):
    """Rows of (label, renderable): a label column, then the rest of the width."""
    t = Table.grid(padding=(0, 1), expand=True)
    t.add_column(width=10, no_wrap=True, style="lab")
    t.add_column(ratio=1)  # long values wrap onto a second line rather than lose their end
    for r in rows:
        t.add_row(*r)
    return t


# ---------- readings ----------

NVQ = "index,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total"


def read_gpus():
    """{nvidia-smi index: {temp, power, limit, util, mem, total}} now."""
    out = {}
    for line in P.sh("nvidia-smi", f"--query-gpu={NVQ}", "--format=csv,noheader,nounits").splitlines():
        r = [x.strip() for x in line.split(",")]
        if len(r) == 7:
            try:
                out[r[0]] = dict(zip(("temp", "power", "limit", "util", "mem", "total"), map(float, r[1:])))
            except ValueError:
                pass
    return out


def history(minutes=60):
    """{card: [(t, power)]} over the last minutes, from the temperature logger (a row per card every 30 s)."""
    out, cut = {}, time.time() - minutes * 60
    for l in P.tail_text(f"{P.RUNS}/gpu-temps.csv", 1 << 17):
        r = l.split(",")
        if len(r) < 11 or not r[0].startswith("RTX "):
            continue
        try:
            t = time.mktime(time.strptime(r[1][:19], "%Y/%m/%d %H:%M:%S"))  # nvidia-smi's local time
            p = float(r[6])
        except ValueError:
            continue
        if t >= cut:
            out.setdefault(r[0][4:], []).append((t, p))
    return out


def spark(points, now, minutes, width, top):
    """(t, value) points as block characters, one column per equal slice of the last minutes up to now."""
    if width < 4 or not top:
        return Text("")
    span, start = minutes * 60 / width, now - minutes * 60
    cols = [[] for _ in range(width)]
    for t, v in points:
        i = int((t - start) / span)
        if 0 <= i < width:
            cols[i].append(v)
    return Text("".join(" " if not c else SPARK[min(int(sum(c) / len(c) / top * 8), 7)] for c in cols), style="acc")


SAMPLED = re.compile(r"round (\d+): sampled problems (\d+)-(\d+) of (\d+): (\d+) tokens in (\d+)s \((\d+) tok/s\)")


def rft_live(name):
    """Where an RFT run is, as times and counts the screen turns into a live estimate."""
    d = f"{P.TRAIN}/runs/{name}"
    cfg = P.json_or(f"{d}/config.json", {})
    R, chunk = cfg.get("rounds") or 2, cfg.get("chunk") or 256
    rounds, holds, done, first_t, session = {}, {}, False, None, None
    for l in P.tail_text(f"{d}/train.log", 1 << 22):
        if not P.STAMPED.match(l):
            continue
        t, msg = P.stamp(l), l[20:]
        first_t = first_t or t
        if m := SAMPLED.match(msg):
            rounds.setdefault(int(m[1]), {"chunks": []})["chunks"].append(
                {"a": int(m[2]), "b": int(m[3]), "of": int(m[4]), "tokens": int(m[5]), "s": int(m[6]), "rate": int(m[7]),
                 "t": t})
        elif m := re.match(r"round (\d+): pass rate", msg):
            rounds.setdefault(int(m[1]), {"chunks": []})["graded"] = t
        elif m := re.match(r"round (\d+): fine-tuning on (\d+) answers", msg):
            rounds.setdefault(int(m[1]), {"chunks": []}).update(ft=t, kept=int(m[2]))
        elif m := re.match(r"round (\d+): step (\d+)/(\d+),", msg):
            rounds.setdefault(int(m[1]), {"chunks": []}).update(step=int(m[2]), steps=int(m[3]), step_t=t)
        elif m := re.match(r"round (\d+): checkpoint saved", msg):
            rounds.setdefault(int(m[1]), {"chunks": []})["saved"] = t
        elif m := re.match(r"round (\d+): already done", msg):  # a resumed run skipping a round: keep its own save time
            rounds.setdefault(int(m[1]), {"chunks": []}).setdefault("saved", t)
        elif msg.startswith("vLLM now holds"):  # the next round's sampling begins
            holds[1 + max([r for r, x in rounds.items() if "saved" in x], default=0)] = t
        elif msg.startswith("waiting for the "):  # each start or resume of the run
            session = t
        elif msg == "done":
            done = True
    saved = [r for r, x in rounds.items() if "saved" in x]
    cur = max(saved, default=0) + 1
    x = rounds.get(cur, {"chunks": []})
    total = next((c["of"] for r in sorted(rounds, reverse=True) for c in rounds[r]["chunks"]), None)
    spp_of = lambda r: (sum(c["s"] for c in rounds[r]["chunks"]) / sum(c["b"] - c["a"] + 1 for c in rounds[r]["chunks"])
                        if rounds.get(r, {}).get("chunks") else None)
    spp = spp_of(cur) or next((spp_of(r) for r in sorted(rounds, reverse=True) if spp_of(r)), None)
    post = next((rounds[r]["saved"] - rounds[r]["chunks"][-1]["t"] for r in sorted(saved, reverse=True)
                 if rounds[r]["chunks"]), 600)  # sampling's end to the checkpoint: grading and fine-tuning
    # Every round samples the same problems in the same chunks, and a chunk's time follows its problems (the first
    # is short, the rest about even), so a chunk is predicted by the last full round's time for it, scaled by how
    # this round's finished chunks compare with that round's.
    secs = lambda r: {c["a"]: c["s"] for c in rounds.get(r, {}).get("chunks", [])}
    full = lambda r: rounds[r]["chunks"] and rounds[r]["chunks"][-1]["b"] >= rounds[r]["chunks"][-1]["of"]
    prev = next((r for r in sorted(rounds, reverse=True) if r < cur and full(r)), None)
    pred, done_s = None, sum(secs(cur).values())
    if prev is not None:
        ps, cs = secs(prev), secs(cur)
        common = [a for a in cs if a in ps]
        ratio = sum(cs[a] for a in common) / sum(ps[a] for a in common) if common else 1.0
        pred = {a: s * ratio for a, s in ps.items()}
    round_pred = done_s or (sum(pred.values()) if pred else spp * total if spp and total else None)
    out = {"kind": "rft", "round": min(cur, R), "rounds": R, "total": total, "spp": spp, "post": post,
           "round_pred": round_pred, "started": P.stamp(cfg.get("started", "")) or first_t, "session": session,
           "results": [s for r in range(1, R + 1) if (s := P.json_or(f"{d}/round_{r}/summary.json"))]}
    if done or cur > R:
        return {**out, "stage": "done"}
    if "step" in x or "ft" in x:
        out.update(stage="fine-tuning", step=x.get("step", 0), steps=x.get("steps"), ft=x.get("ft"),
                   step_t=x.get("step_t"))
    elif "graded" in x:
        out.update(stage="fine-tuning", step=0, steps=None, ft=x["graded"])
    elif x["chunks"] and x["chunks"][-1]["b"] >= x["chunks"][-1]["of"]:
        out.update(stage="grading", since=x["chunks"][-1]["t"])
    elif x["chunks"] or cur in holds:
        b = x["chunks"][-1]["b"] if x["chunks"] else 0
        c = min(chunk, (total or b + chunk) - b)
        if pred and b + 1 in pred:
            chunk_pred, rest_pred = pred[b + 1], sum(v for a, v in pred.items() if a > b + 1)
        elif spp and total:
            chunk_pred, rest_pred = spp * c, spp * max(total - b - c, 0)
        else:
            chunk_pred = rest_pred = None
        out.update(stage="sampling", b=b, anchor=x["chunks"][-1]["t"] if x["chunks"] else holds[cur], c=c,
                   tokens=sum(c["tokens"] for c in x["chunks"]), rate=x["chunks"][-1]["rate"] if x["chunks"] else None,
                   chunk_pred=chunk_pred, rest_pred=rest_pred,
                   round_pred=done_s + (chunk_pred or 0) + (rest_pred or 0) if chunk_pred else round_pred)
    else:
        out.update(stage="loading", since=session or first_t)
    return out


def ckpt_ref(name, base):
    """The finished check whose generation time a running check is measured against: the same arm's latest, else the
    base's. (name, seconds) or None."""
    arm, best = name.split("/")[0], None
    for g in glob.glob(f"{P.TRAIN}/evals/**/gen.json", recursive=True):
        n = os.path.relpath(os.path.dirname(g), f"{P.TRAIN}/evals")
        j = P.json_or(g, {})
        if n == name or not j.get("generate_seconds") or not (n.split("/")[0] == arm or n == base):
            continue
        key = (n.split("/")[0] == arm, j.get("finished", ""))
        if best is None or key > best[0]:
            best = (key, n, j["generate_seconds"])
    return best[1:] if best else None


def grpo_live(name, args):
    d = f"{P.TRAIN}/runs/{name}"
    cfg = P.json_or(f"{d}/config.json", {})
    out = {"kind": "grpo", "total": cfg.get("steps") or int(P.arg(args, "--steps", 150)), "step": 0,
           "ckpts": len(os.listdir(f"{d}/checkpoints")) if os.path.isdir(f"{d}/checkpoints") else 0}
    rows = P.tail_jsonl(f"{d}/steps.jsonl")
    if rows:
        last = rows[-5:]
        out.update(step=rows[-1]["step"], anchor=P.mtime(f"{d}/steps.jsonl"),
                   per=sum(x["seconds"]["total"] for x in last) / len(last), passes=[x["pass_rate"] for x in rows],
                   kept=rows[-1]["groups_kept"], problems=rows[-1]["problems"], tokens=rows[-1]["mean_tokens"])
    return out


def watchers():
    """The checkpoint watchers alive now: [(arm, base, card)]."""
    out = []
    for line in P.sh("pgrep", "-af", "watch-ckpts[.]sh").splitlines():
        if "watch-ckpts.sh " in line:
            a = line.split("watch-ckpts.sh ", 1)[1].split(" ")
            if len(a) >= 3:
                out.append((a[0], a[1], a[2]))
    return out


def temps():
    """(style, text): the temperature logger's state."""
    if P.sh("pgrep", "-f", "/opt/wsp/harness/gpu_temps[.]py") in ("", "?"):
        return "warn", "temperature logger not running"
    try:
        lines = open(P.TEMP_WARNINGS, errors="replace").read().splitlines()
    except OSError:
        lines = []
    active = {}
    for l in lines:
        msg = l[20:]
        if msg.startswith("logger started"):
            active = {}
        elif m := re.match(r"(WARNING|STILL) ([^:]+): ", msg):
            active[m[2]] = msg
        elif m := re.match(r"OK ([^:]+): ", msg):
            active.pop(m[1], None)
    if active:
        return "bad", " | ".join(active.values())
    return "ok", "temperatures fine"


def checks():
    """The checkpoint checks with a summary, newest first."""
    out = []
    for path in glob.glob(f"{P.TRAIN}/evals/**/summary.json", recursive=True):
        d = os.path.dirname(path)
        s = P.json_or(path)
        if not s:
            continue
        fin = P.json_or(f"{d}/gen.json", {}).get("finished")
        t = P.stamp(fin) if fin else P.mtime(path)
        out.append({"name": os.path.relpath(d, f"{P.TRAIN}/evals"), "gpu": P.gpu_in_log(f"{d}/log"), "t": t or 0,
                    "s": s})
    return sorted(out, key=lambda c: -c["t"])


def run_info(name, now, width, evs):
    """An agent run named on the command line: its full view while it runs, a summary once it has ended."""
    run = f"{P.RUNS}/{name}"
    st, cfg = P.json_or(f"{run}/state.json"), P.json_or(f"{run}/config.json")
    if not st or not cfg:
        return {"name": name, "error": "cannot read its state.json and config.json"}
    if st["status"] == "running":
        return {"name": name, "running": True, "lines": P.run_block(name, now, width, 3, False, evs)}
    log = P.tail_text(f"{run}/driver.log", 1 << 15)
    stamped = [P.stamp(l) for l in log if P.STAMPED.match(l)]
    mine = [e for e in evs if e["submission"] and os.path.realpath(e["submission"]) == os.path.realpath(f"{run}/submission")]
    return {"name": name, "running": False, "status": st["status"], "end": stamped[-1] if stamped else None,
            "start": P.stamp(open(f"{run}/driver.log", errors="replace").readline()) if log else None,
            "passes": (st["passes_used"], cfg["passes"]), "code": (st["code_runs"], cfg["max_code_runs"]),
            "tokens": st["tokens"]["output_tokens"], "turn": st["turn"], "gpu": cfg.get("gpu", "5090"), "evals": mine}


def gather(names, width):
    """Everything read from disk and processes, every few seconds (the GPUs are read separately)."""
    now = time.time()
    P._frame.clear()
    tj = P.train_jobs(now)
    evs = P.evaluations(now)
    live = {}
    for j in tj:
        try:
            if j["kind"] == "rft" and j["state"] == "running":
                live[j["name"]] = rft_live(j["name"])
            elif j["kind"] == "grpo" and j["state"] in ("running", "done"):
                live[j["name"]] = grpo_live(j["name"], j["args"])
            elif j["kind"] == "ckpt" and j["state"] in ("running", "grading"):
                base = j["args"] or None
                live[j["name"]] = {"kind": "ckpt", "base": base, "ref": ckpt_ref(j["name"], base)}
        except Exception as e:  # a log in a shape not foreseen: fall back to progress.py's one-line detail
            live[j["name"]] = {"error": f"{type(e).__name__}: {e}"}
    for e in evs:
        if e["running"]:
            m = re.search(r"([\d,]+)/([\d,]+) answers", e.get("progress", ""))
            e["frac"] = int(m[1].replace(",", "")) / int(m[2].replace(",", "")) if m else None
    swap = P.sh("docker", "inspect", "-f", "{{.State.Status}}", "llama-swap")
    return {"t": now, "tj": tj, "evs": evs, "live": live, "users": P.running_jobs(), "watchers": watchers(),
            "swap": "stopped" if swap == "exited" else swap, "temps": temps(), "checks": checks(),
            "runs": [run_info(n, now, width, evs) for n in names]}


class Data:
    """The readings, refreshed by two threads: the GPUs every second, the rest every `every` seconds."""

    def __init__(self, names, every):
        self.names, self.every = names, every
        self.stop, self.poke = threading.Event(), threading.Event()
        self.gpus, self.slow, self.hist, self.error = {}, None, {}, None
        self.samples = {c: collections.deque(maxlen=3600) for c in P.GPUS}  # (t, power) every second, per card
        self.gpus_t = self.hist_t = 0
        self.width = 130
        self.fit, self.lane_fit = {}, {}  # the layout chosen for the window, worked out once a second

    def start(self):
        for f in (self.gpu_loop, self.slow_loop):
            threading.Thread(target=f, daemon=True).start()

    def gpu_loop(self):
        while not self.stop.is_set():
            g, t = read_gpus(), time.time()
            self.gpus, self.gpus_t = g, t
            for c, info in P.GPUS.items():
                r = g.get(str(info.get("index")))
                if r:
                    self.samples[c].append((t, r["power"]))
            self.stop.wait(1)

    def slow_loop(self):
        while not self.stop.is_set():
            try:
                self.slow, self.error = gather(self.names, self.width), None
                if time.time() - self.hist_t > 30:
                    self.hist, self.hist_t = history(), time.time()
            except Exception:  # keep the screen up and say what failed
                self.error = traceback.format_exc(limit=3)
            self.poke.wait(self.every)
            self.poke.clear()

    def power(self, card):
        """The card's power readings: the logger's every 30 s, then this screen's every second."""
        live = list(self.samples.get(card, ()))
        first = live[0][0] if live else float("inf")
        return [p for p in self.hist.get(card, []) if p[0] < first] + live


# ---------- the screen ----------

def header(D, now, width):
    s = D.slow or {}
    tj = s.get("tj", [])
    n = lambda *st: sum(j["state"] in st for j in tj)
    wide = width >= 120
    left = Text.assemble((" ◆ wsp ", "head"), ("training phase   " if wide else " ", "dim"))
    for count, glyph, word, style in ((n("running", "grading") + sum(e["running"] for e in s.get("evs", [])), "▸",
                                       "running", "run"), (n("waiting"), "◷", "waiting", "warn"),
                                      (n("queued"), "◷", "queued", "lab"), (n("done"), "✓", "done", "ok"),
                                      (n("failed"), "✗", "failed", "bad"), (n("stopped"), "■", "stopped", "warn")):
        if count or word in ("running", "queued"):
            today = " today" if wide and word in ("done", "failed", "stopped") else ""
            left.append(f"{glyph} {count} {word}{today}  ", style=style if count else "faint")
    swap = s.get("swap", "?")
    right = Text.assemble(("llama-swap ", "dim"), (swap, "warn" if swap == "stopped" else "ok"), ("  ", ""),
                          (time.strftime("%H:%M:%S UTC", time.gmtime(now)), "head"), " ",
                          (SPIN[int(now * 8) % len(SPIN)], "acc"), " ")
    t = Table.grid(padding=(0, 2), expand=True)  # a cut left part ends in … two spaces before the right
    t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column(justify="right", no_wrap=True)
    t.add_row(left, right, style="band")  # a row's style fills its background; a table's only reaches its borders
    return t


def title_row(glyph, name, kind, extra, right, style="run"):
    t = Table.grid(padding=(0, 2), expand=True)
    t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column(justify="right", no_wrap=True)
    t.add_row(Text.assemble((f"{glyph} ", style), (name, "head"), (f" {kind}" if kind else "", "lab"), (extra, "dim")),
              right)
    return t


def running_for(t, now):
    return Text.assemble(("running ", "dim"), (ticking(now - t), "acc"))


def note(text, style="dim"):
    return ("", Text(text, style=style))


def eta(left, now):
    return f"{about(left)} · around {hhmm(now + left, now)}"


def rft_rows(j, L, now):
    rows, stage, total, sd = [], L["stage"], L.get("total"), seed(j["name"])
    left_round = None
    if stage == "sampling":
        c, b, cp = L["c"], L["b"], L.get("chunk_pred")
        if cp and total:
            in_chunk = now - L["anchor"]
            est = b + min(in_chunk / cp, 0.99) * c
            left_s = max(cp - in_chunk, 0) + L["rest_pred"]
            left_round = left_s + L["post"]
            rows.append(("sampling", bar_cell(Gauge(b / total, est / total, now, seed=sd),
                                              Text(f"≈ {est:,.0f} / {total:,} problems", style="est"))))
            rows.append(note(" · ".join(x for x in (
                f"{L['rate'] / 1000:.1f}k tok/s" if L.get("rate") else None, f"sampling {about(left_s)} more") if x)))
        else:
            rows.append(("sampling", bar_cell(Gauge(now=now, pulse=True, seed=sd),
                                              Text(f"{b:,} / {total or '?'} problems", style="dim"))))
    elif stage == "grading":
        left_round = max(L["post"] - (now - L["since"]), 0)
        rows.append(("grading", bar_cell(Gauge(now=now, pulse=True, seed=sd),
                                         Text(f"{total:,} problems × 8 answers", style="dim"))))
    elif stage == "fine-tuning":
        steps, step = L.get("steps"), L.get("step", 0)
        if steps:
            per = (L["step_t"] - L["ft"]) / step if step and L.get("step_t") else 15
            since = now - (L.get("step_t") or L["ft"])
            left_round = max((steps - step) * per - since, 0) + 30
            rows.append(("fine-tune", bar_cell(Gauge(step / steps, (step + min(since / per, 0.99)) / steps, now,
                                                     seed=sd), Text(f"step {step} / {steps}", style="dim"))))
        else:
            left_round = max(L["post"] - (now - L["ft"]), 0)
            rows.append(("fine-tune", bar_cell(Gauge(now=now, pulse=True, seed=sd), Text("starting", style="dim"))))
    elif stage == "loading":
        rows.append(("loading", bar_cell(Gauge(now=now, pulse=True, seed=sd),
                                         Text("vLLM and the round's weights", style="dim"))))
    if left_round is not None:
        if L["round"] >= L["rounds"]:
            rows.append(("done in", Text(f"{eta(left_round, now)} (round {L['round']}'s checkpoint)", style="est")))
        elif L.get("round_pred"):  # the next round samples about as long as this one
            run_left = left_round + LOAD + L["round_pred"] + L["post"]
            rows.append(("done in", Text(f"{eta(run_left, now)} (round {L['round']}'s checkpoint {about(left_round)})",
                                         style="est")))
    for s in L["results"]:
        rows.append((f"round {s['round']}", Text(
            f"pass {pct(s['pass_rate'])} · {s['kept_answers']:,} answers kept · {100 * s['cut_share']:.0f}% cut",
            style="dim")))
    return rows


def ckpt_rows(j, L, now):
    ref, sd = L.get("ref"), seed(j["name"])
    if j["state"] == "grading":
        return [("grading", bar_cell(Gauge(now=now, pulse=True, seed=sd),
                                     Text(j.get("stage", "").replace("grading the ", ""), style="dim")))]
    if now - j["t"] < LOAD:
        return [("loading", bar_cell(Gauge(now=now, pulse=True, seed=sd), Text("vLLM and the checkpoint", style="dim")))]
    gen = now - j["t"] - LOAD
    if ref and gen < ref[1]:
        return [("generating", bar_cell(Gauge(0, gen / ref[1], now, seed=sd),
                                        Text(f"≈ {100 * gen / ref[1]:.0f}% of 2,008 answers", style="est"))),
                ("done in", Text(eta(ref[1] - gen + 120, now), style="est")),
                note(f"at {ref[0]}'s pace ({P.minutes(ref[1])}), then 2 min of grading")]
    return [("generating", bar_cell(Gauge(now=now, pulse=True, seed=sd), Text("2,008 answers", style="dim"))),
            note(f"past {ref[0]}'s {P.minutes(ref[1])}; nothing is logged until the last answer"
                 if ref else "no earlier check to pace it; nothing is logged until the last answer")]


def grpo_rows(j, L, now):
    total, step, sd = L["total"], L["step"], seed(j["name"])
    if not step:
        return [("loading", bar_cell(Gauge(now=now, pulse=True, seed=sd),
                                     Text(f"vLLM and the trainer, then step 1 of {total}", style="dim")))]
    frac = min((now - L["anchor"]) / L["per"], 0.99) if L.get("per") and j["state"] == "running" else 0
    passes = L.get("passes", [])
    trend = spark([(i, p) for i, p in enumerate(passes)], len(passes) - 0.5, len(passes) / 60, min(len(passes), 30),
                  max(passes + [0.01]))
    rows = [("steps", bar_cell(Gauge(step / total, (step + frac) / total, now, live=j["state"] == "running", seed=sd),
                               Text(f"{step} / {total}", style="dim"))),
            ("pass rate", Text.assemble(trend, (f"  {pct(passes[0])} → {pct(passes[-1])}", "dim"))),
            note(f"kept {L['kept']}/{L['problems']} groups · {human(L['tokens'])} tok/answer · "
                 f"{P.secs(L['per'])} a step · {L['ckpts']} checkpoint{'s' if L['ckpts'] != 1 else ''}")]
    if j["state"] == "running" and step < total:
        rows.append(("done in", Text(eta(max((total - step) * L["per"] - (now - L["anchor"]), 0), now), style="est")))
    return rows


def arm_start(name):
    """An arm's start, from its name: O for G-O, E for R-E and for G-E2 (G-E's rerun)."""
    return name.split("-")[-1].rstrip("0123456789")


def job_block(j, L, now):
    """A running job's rows in its GPU's panel."""
    kind = {"rft": "RFT", "grpo": "GRPO", "ckpt": "check"}.get(j["kind"], "")
    name = P.train_label(j) if j["kind"] == "check" else j["name"]
    extra, since = "", j["t"]
    if j["kind"] == "rft" and L.get("round"):
        extra = f" · from {arm_start(j['name'])} · round {L['round']} of {L['rounds']}"
        since = L.get("session") or L.get("started") or j["t"]  # since this start of the run, not its first
        if L.get("session") and L.get("started") and L["session"] - L["started"] > 600:
            extra += f" · resumed {hhmm(L['session'], now)}"
    elif j["kind"] == "grpo":
        extra = f" · from {arm_start(j['name'])}"
    elif j["kind"] == "ckpt":
        extra = f" · against {L['base']}" if L.get("base") else " · unpaired"
    try:
        if "error" in L:
            raise ValueError(L["error"])
        rows = {"rft": rft_rows, "ckpt": ckpt_rows, "grpo": grpo_rows}[j["kind"]](j, L, now)
    except Exception:  # a shape not foreseen: progress.py's one-line detail instead
        rows = [("", Text.from_ansi(j.get("detail", "")))]
    return Group(title_row("▸", name, kind, extra, running_for(since, now)), kv(rows))


def lane_parts(card, D, now, chart_rows):
    """A GPU panel's two halves: what runs on it (top); then what comes next, the card and its power (bottom)."""
    s = D.slow or {}
    g = P.GPUS[card]
    tj = s.get("tj", [])
    parts = []
    holders = [j for j in tj if j["gpu"] == card and j["state"] in ("running", "grading")]
    for j in holders:
        parts.append(job_block(j, s.get("live", {}).get(j["name"], {}), now))
    evs = [e for e in s.get("evs", []) if e["running"] and e["gpu"] == card]
    for e in evs:
        b = (Gauge(e["frac"], now=now, seed=seed(e["name"])) if e.get("frac") is not None else
             Gauge(now=now, pulse=True, seed=seed(e["name"])))
        parts.append(Group(title_row("▸", e["name"], "evaluation", "", running_for(e["start"] or now, now)),
                           kv([("progress", b), note(P.ANSI.sub("", e.get("progress", "")))])))
    for u in s.get("users", {}).get(card, []):
        if u.startswith("agent "):
            parts.append(title_row("▸", u[6:], "agent run", " · its view is below", Text("")))
    if not parts:
        parts.append(Text.assemble(("○ ", "faint"), ("idle", "dim")))
    nxt = [f"{j['name']} (waiting {P.minutes(now - j['t'])})" for j in tj if j["gpu"] == card and j["state"] == "waiting"]
    shown = {j.get("after", "").split(" and ")[0] for j in tj if j["state"] == "queued"}
    for j in holders:  # a running arm's hook (train/queue/after-<arm>.sh), unless a queued job shows it below
        if j.get("hook") and j["name"] not in shown:
            nxt.append(f"{j['hook']} (after {j['name']})")
    for j in tj:
        if j["gpu"] == card and j["state"] == "queued":
            if " and " in j.get("after", ""):  # a hook that runs before it
                nxt.append(j["after"].split(" and ", 1)[1])
            nxt.append(j["name"] + (f" {j['note'].split(',')[0]}" if j.get("note") else ""))
    w = sorted(f"{arm}'s" for arm, base, c in s.get("watchers", []) if c == card)
    if w:
        nxt.append(f"checks of {' and '.join(w)} new checkpoints")
    rows = [("next", Text(" → ".join(nxt), style="lab") if nxt else Text("nothing queued", style="faint"))]
    r = D.gpus.get(str(g["index"]))
    if r:
        temp = Text(f"{r['temp']:.0f} °C", style="bad" if r["temp"] >= 83 else "warn" if r["temp"] >= 75 else "")
        rows.append(("card", Text.assemble(temp, f"  {r['power']:.0f} / {r['limit']:.0f} W  ",
                                           (f"{r['util']:.0f} %", "acc" if r["util"] >= 50 else ""),
                                           f"  {r['mem'] / 1024:.1f} / {r['total'] / 1024:.1f} GB")))
        if chart_rows:
            label = Text("power", style="lab")
            if chart_rows > 1:
                label.append(f"\n{CHART_MIN} min", style="faint")
            rows.append((label, Chart(D.power(card), now, r["limit"], chart_rows)))
    else:
        rows.append(("card", Text("not visible to nvidia-smi", style="bad")))
    title = Text.assemble((f" RTX {card} ", "head"),
                          (f"· llama-swap {s.get('swap', '?')} " if g.get("llama_swap") else
                           "· shared with other services ", "dim"))
    return Group(*parts), kv(rows), title, "acc" if holders or evs else "faint"


def lane(card, D, now, chart_rows, gap=0, height=None):
    top, bottom, title, border = lane_parts(card, D, now, chart_rows)
    body = Group(Padding(top, (0, 0, gap, 0)) if gap > 0 else top, bottom)
    return Panel(body, title=title, title_align="left", box=box.ROUNDED, border_style=border, padding=(0, 1),
                 height=height)


class Side:
    """Renderables side by side at fixed widths, one space apart."""

    def __init__(self, items):
        self.items = items  # [(renderable, width)]

    def __rich_console__(self, console, options):
        cols = [console.render_lines(r, options.update_width(w)) for r, w in self.items]
        for i in range(max(len(c) for c in cols)):
            for k, (c, (_, w)) in enumerate(zip(cols, self.items)):
                if k:
                    yield Segment(" ")
                yield from c[i] if i < len(c) else [Segment(" " * w)]
            yield Segment.line()


def lanes(console, D, now, width, chart_rows):
    """The GPU panels: side by side when there is room, the same height, with their card rows lined up at the
    bottom; else one above the other."""
    cards = [c for c, g in P.GPUS.items() if not g.get("remote")]
    if width >= 2 * LANE_MIN + 1 and len(cards) == 2:
        widths = ((width - 1) // 2, width - 1 - (width - 1) // 2)
        key = (width, chart_rows, int(now), id(D.slow))
        fit = D.lane_fit.get(key)
        if fit is None:
            hs = []
            for c, w in zip(cards, widths):
                top, bottom, _, _ = lane_parts(c, D, now, chart_rows)
                opts = console.options.update_width(w - 4)  # inside the border and the padding
                hs.append((len(console.render_lines(top, opts, pad=False)),
                           len(console.render_lines(bottom, opts, pad=False))))
            inner = max(a + b for a, b in hs)
            if len(D.lane_fit) > 8:
                D.lane_fit.clear()
            fit = D.lane_fit[key] = {"gaps": [inner - a - b for a, b in hs], "height": inner + 2}
        return [Side([(lane(c, D, now, chart_rows, gap, fit["height"]), w)
                      for c, gap, w in zip(cards, fit["gaps"], widths)])]
    return [lane(c, D, now, chart_rows) for c in cards]


STATE = {"queued": ("◷", "lab"), "done": ("✓", "ok"), "failed": ("✗", "bad"), "stopped": ("■", "warn"),
         "waiting": ("◷", "warn")}


def queue_panel(D, now, width, ended=True):
    tj = (D.slow or {}).get("tj", [])
    rows = [j for j in tj if j["state"] in ("queued", "waiting")]
    if ended:
        rows += [j for j in tj if j["state"] in ("done", "failed", "stopped") and j["kind"] in ("rft", "grpo")]
        rows += [j for j in tj if j["state"] == "failed" and j["kind"] not in ("rft", "grpo")]
    if not rows:
        return None
    items = []
    for j in rows:
        glyph, style = STATE.get(j["state"], ("·", "dim"))
        when = (f"after {j['after']}" if j["state"] == "queued" and j.get("after") else
                f"waiting {P.minutes(now - j['t'])}" if j["state"] == "waiting" else
                f"{j['state']} {hhmm(j['t'], now)}")
        detail = re.sub(r"^[█░]+ ", "", re.sub(r"^after .*? · ", "", P.ANSI.sub("", j.get("detail", ""))))
        items.append((Text(glyph, style=style), Text(j["name"], style="head"),
                      Text(f"{P.TRAIN_KINDS.get(j['kind']) or 'check'} · {j['gpu']}", style="lab"),
                      Text(when, style="lab" if j["state"] == "queued" else style), Text(detail, style="dim")))
    if width >= 110:  # one row per job; the long cells wrap rather than lose their ends
        t = Table(box=None, padding=(0, 1), expand=True, show_header=False, pad_edge=False)
        t.add_column(width=1, no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(no_wrap=True)
        t.add_column(max_width=30)
        t.add_column(ratio=1)
        for it in items:
            t.add_row(*it)
    else:  # narrow: the name, kind and when on one line, the details under them
        t = Table.grid(padding=(0, 1), expand=True)
        t.add_column(width=1, no_wrap=True)
        t.add_column(ratio=1)
        for glyph, name, kind, when, detail in items:
            head = Text.assemble(name, "  ", kind, "  ", when)
            t.add_row(glyph, Group(head, detail) if detail.plain else head)
    return Panel(t, title=Text(" Queue · then what ended in the last day " if ended else " Queue ", style="head"),
                 title_align="left", box=box.ROUNDED, border_style="faint", padding=(0, 1))


SUITES = (("choice", "choice"), ("math", "math"), ("instructions", "instr."), ("code", "code"))


def table(*cols):
    t = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=True, header_style="lab", border_style="faint",
              pad_edge=False, show_edge=False)
    for c in cols:
        t.add_column(**c)
    return t


def checks_panel(D, now, width, limit):
    cs = (D.slow or {}).get("checks", [])[:limit]
    if not cs:
        return None
    wide = width >= 118
    num = dict(justify="right", no_wrap=True)
    t = table(dict(header="check", no_wrap=True, style="head"), dict(header="done", no_wrap=True, style="dim"),
              dict(header="panel", **num),
              *([dict(header=short, style="dim", **num) for _, short in SUITES] if wide else []),
              dict(header="valid.", **num), dict(header="Δ panel", **num), dict(header="Δ valid.", **num),
              dict(header="against", no_wrap=True, style="dim", ratio=1))
    for c in cs:
        s, v = c["s"], c["s"].get("vs_base")
        key = "val2" if "val2" in s else "val"  # val2: on train/data2's tests, which the rules read from 2026-11-08
        row = [c["name"], hhmm(c["t"], now), f"{s['panel_mean']:.1f}"]
        if wide:
            row += [f"{s['panel'][k]['pass']:.1f}" for k, _ in SUITES]
        row.append(f"{s[key]['pass']:.1f}")
        if v:
            vv = v.get(key, v["val"])
            row += [signed(v["panel_mean"]["diff"], v["panel_mean"]["se"]), signed(vv["diff"], vv["se"]), v["base"]]
        else:
            row += [Text("—", style="faint"), Text("—", style="faint"),
                    Text("a start" if c["name"].startswith("start-") else "not paired yet", style="faint")]
        t.add_row(*row)
    return Panel(t, title=Text(" Checkpoint checks ", style="head"), title_align="left",
                 subtitle=Text(" pass rates in %; validation on the new tests; Δ ± SE against the arm's start, "
                               "coloured beyond 2 SE ", style="faint"),
                 subtitle_align="right", box=box.ROUNDED, border_style="faint", padding=(0, 1))


EVAL_COLS = ("code", "MMLU-Pro", "IFEval", "LCB", "MATH")


def evals_panel(D, now, limit):
    evs = [e for e in (D.slow or {}).get("evs", []) if not e["running"] and "done" in e][:limit]
    if not evs:
        return None
    num = dict(justify="right", no_wrap=True)
    t = table(dict(header="evaluation", no_wrap=True, style="head"), dict(header="GPU", no_wrap=True, style="dim"),
              dict(header="done", no_wrap=True, style="dim"), *[dict(header=h, **num) for h in EVAL_COLS],
              dict(header="", ratio=1))
    for e in evs:
        sc = dict(x.rsplit(" ", 1) for x in e.get("scores", []) if " " in x)
        t.add_row(e["name"], e["gpu"], hhmm(e["done"], now), *[sc.get(k, "—").rstrip("%") for k in EVAL_COLS], "")
    return Panel(t, title=Text(" Held-out evaluations, last day ", style="head"), title_align="left",
                 subtitle=Text(" the edited model's pass rate, % ", style="faint"), subtitle_align="right",
                 box=box.ROUNDED, border_style="faint", padding=(0, 1))


def runs_panel(D, now):
    out = []
    for r in (D.slow or {}).get("runs", []):
        if r.get("error"):
            out.append(Text(f" ● {r['name']}: {r['error']}", style="bad"))
        elif r["running"]:
            out.append(Panel(Text.from_ansi("\n".join(r["lines"])), title=Text(f" {r['name']} ", style="head"),
                             title_align="left", box=box.ROUNDED, border_style="acc", padding=(0, 0)))
        else:
            dur = f" · {P.minutes(r['end'] - r['start'])}" if r.get("end") and r.get("start") else ""
            ev = " · ".join(f"{e['name']} done {hhmm(e['done'], now)}" for e in r["evals"] if "done" in e)
            glyph, style = ("✓", "ok") if r["status"] == "done" else ("●", "warn")
            g = Table.grid(padding=(0, 1), expand=True)  # wrapped lines hang under the name
            g.add_column(width=2, no_wrap=True, justify="right")
            g.add_column(ratio=1)
            g.add_row(Text(glyph, style=style), Text.assemble(
                (r["name"], "head"),
                (f"  agent {'finished' if r['status'] == 'done' else r['status']} "
                 f"{hhmm(r['end'], now) if r.get('end') else ''}{dur}", style),
                (f" · passes {r['passes'][0]}/{r['passes'][1]} · code runs {r['code'][0]}/{r['code'][1]} · "
                 f"{P.k(r['tokens'])} output tokens" + (f" · {ev}" if ev else ""), "dim")))
            out.append(g)
    return out


def footer(D, now, width):
    s = D.slow or {}
    st, msg = s.get("temps", ("dim", "…"))
    age = now - s["t"] if s else None
    right = ", ".join(x for x in (f"GPUs every 1 s · jobs every {D.every} s" if width >= 120 else "",
                                  f"read {age:.0f} s ago" if age is not None else "") if x)
    t = Table.grid(padding=(0, 2), expand=True)
    t.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    t.add_column(justify="right", no_wrap=True)
    t.add_row(Text.assemble((" q ", "head"), ("quit  ", "dim"), ("r ", "head"), ("refresh   ", "dim"),
                            ("  ", "on " + PAL["fill"]), (" logged  ", "dim"), ("  ", "on " + PAL["fill_est"]),
                            (" ≈ estimated   ", "dim"), ("● ", st), (msg, "dim")),
              Text(right + " ", style="faint"), style="band")
    return t


FULL = dict(chart=2, checks=6, evals=5, ended=True, queue=True)
# In a short window the GPU panels, the agent runs and the footer always show; the rest is added back in this order,
# each with the largest count that still fits (so a window gives them up from the end of the list first).
ADD = (("queue", (True,)), ("checks", (3, 2, 1)), ("chart", (1,)), ("evals", (3, 2, 1)), ("checks", (6, 5, 4)),
       ("chart", (2,)), ("ended", (True,)), ("evals", (5, 4)))


def always(D, now, width):
    """The blocks every window shows: the header, then (below the rest) the agent runs, an error, the footer."""
    out = [header(D, now, width), *runs_panel(D, now)]
    if D.error:
        out.append(Text(" " + D.error.strip().splitlines()[-1], style="bad"))
    return out + [footer(D, now, width)]


def optional(console, D, now, width, kind, v):
    """One block that may be left out, as a list of renderables: the GPU panels with v chart rows, the queue (v: with
    the runs that ended), the last v checks or the last v evaluations."""
    if kind == "chart":
        return lanes(console, D, now, width, v)
    p = (queue_panel(D, now, width, v) if kind == "queue" else checks_panel(D, now, width, v) if kind == "checks" else
         evals_panel(D, now, v))
    return [p] if p is not None else []


def fit(console, D, now, width, height):
    """The most of the screen the window holds: each block's height is measured on its own, then the extras are
    added in the order of ADD while the whole fits."""
    opts = console.options.update_width(width)
    rows = lambda rs: sum(len(console.render_lines(r, opts, pad=False)) for r in rs)
    fixed, memo = rows(always(D, now, width)), {}

    def h(kind, v):
        if (kind, v) not in memo:
            memo[kind, v] = rows(optional(console, D, now, width, kind, v))
        return memo[kind, v]

    def total(lv):
        return (fixed + h("chart", lv["chart"]) + (h("queue", lv["ended"]) if lv["queue"] else 0) +
                (h("checks", lv["checks"]) if lv["checks"] else 0) + (h("evals", lv["evals"]) if lv["evals"] else 0))
    if total(FULL) <= height:
        return FULL
    lv = dict(chart=0, checks=0, evals=0, ended=False, queue=False)
    for kind, values in ADD:
        lv[kind] = next((v for v in values if total({**lv, kind: v}) <= height), lv[kind])
    return lv


def compose(console, D, now, width, lv):
    parts = always(D, now, width)
    middle = optional(console, D, now, width, "chart", lv["chart"])
    for kind in ("queue", "checks", "evals"):
        if lv[kind]:
            middle += optional(console, D, now, width, kind, lv["ended"] if kind == "queue" else lv[kind])
    return Group(parts[0], *middle, *parts[1:])


def frame(console, D, now):
    """The whole screen, fitted to the window once a second (see ADD)."""
    width, height = console.size
    if D.slow is None:
        return Group(header(D, now, width), Text("\n  reading the jobs …", style="dim"))
    key = (width, height, int(now), id(D.slow))
    if D.fit.get("key") != key:
        D.fit = {"key": key, "lv": FULL if height is None else fit(console, D, now, width, height)}
    return compose(console, D, now, width, D.fit["lv"])


# ---------- the terminal ----------

ESCAPES = re.compile(rb"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)?|\[[0-?]*[ -/]*[@-~]|.)", re.S)
SYSTEMS = {"truecolor": ColorSystem.TRUECOLOR, "256": ColorSystem.EIGHT_BIT, "standard": ColorSystem.STANDARD,
           "windows": ColorSystem.WINDOWS}


def cbreak(fd):
    """Keys arrive one at a time and are not echoed. Returns a function that puts the terminal back, or None."""
    try:
        import termios
        import tty
        if not os.isatty(fd):
            return None
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, OSError, ValueError, AttributeError):
        return None


def query_colours(fd, timeout=0.3):
    """The terminal's background and text colours as #rrggbb, when it answers OSC 11 and OSC 10; else None."""
    try:
        os.write(sys.stdout.fileno(), b"\x1b]11;?\x1b\\\x1b]10;?\x1b\\")
        buf, end = b"", time.time() + timeout
        while buf.count(b"rgb:") < 2 and time.time() < end:
            if select.select([fd], [], [], max(0.0, end - time.time()))[0]:
                buf += os.read(fd, 512)
    except OSError:
        return None, None

    def colour(n):
        m = re.search(rb"\x1b\]%d;rgb:([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})/([0-9a-fA-F]{1,4})" % n, buf)
        return "#" + "".join("%02x" % round(int(x, 16) / (16 ** len(x) - 1) * 255) for x in m.groups()) if m else None
    return colour(11), colour(10)


def keys(D, fd):
    """q stops watching, r reads the jobs now; whatever the terminal sends back to queries is skipped."""
    def loop():
        while not D.stop.is_set():
            if select.select([fd], [], [], 0.2)[0]:
                data = ESCAPES.sub(b"", os.read(fd, 256)).lower()
                if b"q" in data:
                    D.stop.set()
                elif b"r" in data:
                    D.poke.set()
    threading.Thread(target=loop, daemon=True).start()


def ansi(line, system):
    return "".join(s.style.render(s.text, color_system=system) if s.style else s.text for s in line if not s.control)


def watch(names, every):
    fd = sys.stdin.fileno()
    restore = cbreak(fd)
    bg, fg = query_colours(fd) if restore else (None, None)
    truecolor = bool(bg) or os.environ.get("COLORTERM") in ("truecolor", "24bit")  # a terminal that answers has it
    console = Console(theme=set_palette(bg, fg), highlight=False, color_system="truecolor" if truecolor else "auto")
    system = SYSTEMS.get(console.color_system)
    D = Data(names, every)
    D.width = console.size.width
    D.start()
    if restore:
        keys(D, fd)
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J")
    prev, size = [], None
    try:
        while not D.stop.is_set():
            now = time.time()
            w, h = os.get_terminal_size(out.fileno())
            if (w, h) != size:  # a new size: redraw everything
                size, prev = (w, h), []
                console.size = (w, h)
                D.width = w
                out.write("\x1b[H\x1b[2J")
            pin = D.slow is not None  # the footer keeps to the window's last row
            try:
                g = frame(console, D, now)
            except Exception:
                g, pin = Text(traceback.format_exc(limit=4), style="bad"), False
            lines = [ansi(l, system) for l in console.render_lines(g, console.options.update_width(w))[:h]]
            pad = [" " * w] * (h - len(lines))  # the panels take no height from the options: pad here
            lines = lines[:-1] + pad + lines[-1:] if pin and lines else lines + pad
            out.write("".join(f"\x1b[{i + 1};1H{l}" for i, l in enumerate(lines)
                              if i >= len(prev) or prev[i] != l))  # only the lines that changed
            out.flush()
            prev = lines
            D.stop.wait(max(0.0, 1 / FPS - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        D.stop.set()
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")
        out.flush()
        if restore:
            restore()


def snapshot(names, path, width, height):
    """One screen, drawn to an SVG file (or a text file when path ends in .txt)."""
    console = Console(theme=THEME, highlight=False, record=True, width=width, height=height, force_terminal=True,
                      color_system="truecolor", file=open(os.devnull, "w"))
    D = Data(names, 5)
    D.width = width
    D.gpus, D.hist = read_gpus(), history()
    D.slow = gather(names, width)
    console.print(frame(console, D, time.time()))
    if path.endswith(".txt"):
        open(path, "w").write(console.export_text())
    else:
        console.save_svg(path, title=f"progress.py --watch  ({width}×{height})")
