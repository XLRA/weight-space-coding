#!/usr/bin/env python3
"""Agent driver for one weight-editing run (host side, the server).

    driver.py <run> [--passes 50] [--model claude-opus-5-5] [--effort max]   (Claude Code on a Claude subscription)
    driver.py <run> --backend anthropic --max-usd 300 [--passes 50] ...     (Claude API, billed per token)
    driver.py <run> --backend replay --replay replies.json [--passes 2]    (harness tests, no model)

    [--presence-penalty 1.5]  the generation passes' presence penalty (default 0), also stated in the prompt
    [--start-from DIR]        a new run starts with DIR's edit (a submission/) in /work/edited, and keeps a
                              read-only copy in /start that "model": "start" and "baseline": "start" run
    [--brief FILE]            text added to the prompt under THIS RUN (copied to runs/<run>/brief.md)
    [--practice DIR]          the practice problems, mounted read-only at /practice (default practice/final); an
                              about.txt in DIR replaces the prompt's description of the two files
    [--gpu 5090|5080]         the GPU of the run's GPU server (default 5090; see gpus.py)
    [--panel]                 offer the agent the panel: {"type": "panel"} answers a fixed set of questions beyond
                              code (MMLU-Pro, MATH, IFEval-style, MBPP test tasks; none in the evaluation or the
                              practice set) and reports per suite the paired difference edited - start (README, "Panel")

Starts the kernel container for /opt/wsp/runs/<run>/, sends the editor prompt (prompt.txt), and
loops. Each agent reply must hold one ```python block, run in the kernel, or one ```forward
block, validated here, run by the GPU server, graded in a no-GPU sandbox and summarized. The run
ends when the forward-pass budget is used and the agent replies without a block; /work/edited is
then copied to submission/. Running the driver again on a run that did not finish resumes it in
a fresh conversation (a handoff), with the last result the agent had not answered. The kernel is
fresh, unless the old driver was stopped with SIGKILL (kill -KILL -<its process group>) and its
kernel still answers: then the kernel and the GPU server are kept, so its variables survive.

The GPU server starts when the agent first asks for a forward pass. On the 5090, llama-swap is
stopped while the run holds the card; on the 5080, other services keep running beside it.
The run gives the GPU back (and starts llama-swap again) when it has not used it for
--gpu-idle-minutes, while it waits for a subscription usage limit to reset, and when the driver
exits. One run or evaluation per GPU at a time.

Backends: claude-code uses Claude Code's headless mode on the Claude subscription that the user
wspagent signed in with (`claude auth login`); anthropic uses the Messages API with the key in
$ANTHROPIC_API_KEY or /opt/wsp/secrets/anthropic.env (KEY=value, mode 600).
"""
import argparse
import fcntl
import glob
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_client  # noqa: E402
from gpus import GPUS  # noqa: E402

WSP = "/opt/wsp"
MODEL_DIR = f"{WSP}/models/qwen3.5-0.8b"
PRACTICE = f"{WSP}/practice/final"
HARNESS = f"{WSP}/harness"
CACHE = f"{WSP}/cache"
SECRETS = f"{WSP}/secrets/anthropic.env"
COLAB_KEY = f"{WSP}/secrets/colab"  # key: what the Colab worker must send to the relay (--gpu colab)
IMAGE = "wsp-runtime:1"
GRADE_IMAGE = "wsp-grade:1"  # wsp-runtime:1 plus math-verify and the IFEval checkers' dependencies (eval/Dockerfile.grade)
EVAL = f"{WSP}/eval"
# The panel's questions and grading data (eval/prepare_panel.py); never mounted where the agent can read it.
# Tests point WSP_PANEL at a small panel and WSP_PANEL_MAX_TOKENS lower.
PANEL = os.environ.get("WSP_PANEL", f"{WSP}/panel/final/panel.jsonl")
PANEL_SUITES = ("choice", "math", "instructions", "code")  # MMLU-Pro, MATH, IFEval-style, MBPP
OUT_CHARS = 10_000
MAX_NEW = 32768
PANEL_MAX_TOKENS = int(os.environ.get("WSP_PANEL_MAX_TOKENS", MAX_NEW))
FULL_ATTN = {3, 7, 11, 15, 19, 23}
FORWARD_TIMEOUT = 5 * 3600
CATEGORIES = ["cut_at_max_tokens", "no_code", "syntax_error", "wrong_interface", "runtime_error", "timeout",
              "wrong_answer"]
# $ per million tokens: input, output, cache read, cache write (1-hour TTL)
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 8.0)}
DEFAULTS = dict(passes=50, backend="claude-code", replay=None, model="claude-opus-5-5", effort="max", max_usd=None,
                handoff_every=0, max_turns=1000, max_code_runs=300, kernel_seconds=900, kernel_memory_gb=20,
                max_output_tokens=128000, max_context=400_000, gpu_idle_minutes=30,
                claude_bin="/home/wspagent/.local/bin/claude", presence_penalty=0.0, start_from=None, brief=None,
                gpu="5090", panel=False, practice=PRACTICE)
AGENT_USER = "wspagent"
# Claude Code's replies when a subscription limit is reached, e.g. "Claude AI usage limit reached|<epoch>"
# or "You've hit your session limit · resets 5am".
LIMIT = re.compile(r"usage limit|limit reached|hit your (?:\w+ )?limit|out of (?:extra )?usage|resets? (?:at )?\d",
                   re.I)
AUTH = re.compile(r"not logged in|please run /login|/login|invalid api key|oauth token (?:has )?expired|"
                  r"authentication (?:failed|error)", re.I)

OPEN = re.compile(r"^```(python|forward)[ \t]*$")
CLOSE = re.compile(r"^```[ \t]*$")


class Refused(Exception):
    """A forward request that is invalid: sent back, no pass used."""


def sh(*cmd, check=True, timeout=None):
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def running(name):
    r = sh("docker", "inspect", "-f", "{{.State.Running}}", name, check=False)
    return r.returncode == 0 and r.stdout.strip() == "true"


def ascii(s):
    return s.encode("ascii", "replace").decode()


def cut(s, n=OUT_CHARS):
    return s if len(s) <= n else s[:n] + f"\n[output cut: showing the first {n:,} of {len(s):,} characters]"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def find_blocks(text):
    """[(kind, body, start offset, closed)] of the ```python / ```forward blocks in a reply.

    With one opening line, the block runs to the last line that is just ``` (so code may hold such
    lines inside strings). With several, each runs to the next such line.
    """
    lines = text.split("\n")
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1
    opens = [i for i, ln in enumerate(lines) if OPEN.match(ln.strip())]
    if len(opens) == 1:
        i = opens[0]
        closes = [j for j in range(i + 1, len(lines)) if CLOSE.match(lines[j].strip())]
        end = closes[-1] if closes else len(lines)
        return [(OPEN.match(lines[i].strip()).group(1), "\n".join(lines[i + 1:end]), starts[i], bool(closes))]
    blocks, i = [], 0
    while i < len(lines):
        m = OPEN.match(lines[i].strip())
        if not m:
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not CLOSE.match(lines[j].strip()):
            j += 1
        blocks.append((m.group(1), "\n".join(lines[i + 1:j]), starts[i], j < len(lines)))
        i = j + 1
    return blocks


# ---------- agent backends ----------

class AnthropicBackend:
    """Claude through the Messages API: adaptive thinking at the configured effort, streamed,
    prompt-cached, with server-side fallbacks. Assistant turns are appended unchanged."""

    def __init__(self, cfg, log):
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY") and os.path.exists(SECRETS):
            if os.stat(SECRETS).st_mode & 0o077:
                log(f"warning: {SECRETS} is readable by other users; chmod 600 it")
            for line in open(SECRETS):
                key, _, value = line.strip().partition("=")
                if key == "ANTHROPIC_API_KEY":
                    os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit(f"no API key: set ANTHROPIC_API_KEY or put it in {SECRETS}")
        self.anthropic, self.cfg, self.log = anthropic, cfg, log
        self.client = anthropic.Anthropic(max_retries=4, timeout=3600.0)

    def respond(self, messages):
        a = self.anthropic
        kw = dict(model=self.cfg["model"], max_tokens=self.cfg["max_output_tokens"], messages=messages,
                  thinking={"type": "adaptive", "display": "summarized"},
                  output_config={"effort": self.cfg["effort"]},
                  cache_control={"type": "ephemeral", "ttl": "1h"},
                  betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        for attempt in range(8):
            try:
                with self.client.beta.messages.stream(**kw) as stream:
                    msg = stream.get_final_message()
                break
            except a.APIStatusError as e:
                if e.status_code < 500 and e.status_code not in (408, 409, 429):
                    raise
                err = e
            except a.APIConnectionError as e:
                err = e
            wait = min(60 * 2 ** attempt, 1800)
            self.log(f"API error ({type(err).__name__}: {str(err)[:300]}); retrying in {wait} s")
            time.sleep(wait)
        else:
            raise RuntimeError("the API kept failing; resume the run later")
        content = list(msg.content)
        raw = [b.model_dump(mode="json", exclude_none=True) for b in content]
        usage = Counter()
        for it in (getattr(msg.usage, "iterations", None) or [msg.usage]):
            for f in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                usage[f] += getattr(it, f, 0) or 0
        out = dict(stop_reason=msg.stop_reason, model=msg.model, raw=raw, usage=dict(usage),
                   context=(msg.usage.input_tokens or 0) + (msg.usage.cache_read_input_tokens or 0)
                   + (msg.usage.cache_creation_input_tokens or 0))
        if msg.stop_reason == "refusal":
            return dict(out, refusal=True, text="", thinking="", content=None,
                        stop_details=str(getattr(msg, "stop_details", None)))
        # After a mid-output fallback, reasoning blocks before the last fallback marker are not echoed back.
        marks = [i for i, b in enumerate(content) if b.type == "fallback"]
        hist = content if not marks else [b for i, b in enumerate(content) if i > marks[-1] or b.type not in
                                          ("thinking", "redacted_thinking", "tool_use")]
        return dict(out, refusal=False, content=hist,
                    text="".join(b.text for b in content if b.type == "text"),
                    thinking="\n\n".join(getattr(b, "thinking", "") or "" for b in content if b.type == "thinking"))


def limit_wait(text):
    """Seconds to wait after a usage-limit reply: until the reset time when it gives one, else 15 minutes."""
    m = re.search(r"\|(\d{10})\b", text)
    if m:
        return int(min(max(int(m.group(1)) - time.time() + 60, 60), 7 * 86400))
    return 900


class ClaudeCodeBackend:
    """The agent through Claude Code's headless mode (claude -p) on a Claude subscription, run as the
    unprivileged user wspagent, who signed in once with `claude auth login`.

    Built-in tools, plugins, hooks, MCP servers and slash commands are off and the system prompt is
    replaced, so the agent can act only through the blocks in its replies. Claude Code keeps each
    conversation as a session on disk, so only the newest message is sent; its transcript (with the
    thinking) is copied to messages/. When the usage limit is reached, the driver waits for the reset
    (on_wait) and sends the turn again.
    """

    SYSTEM = ("You are the research agent in an automated experiment. You have no tools: you act only by writing "
              "code blocks in your replies, as the user's messages describe, and the harness answers each block "
              "with its result.")

    def __init__(self, cfg, log, run_dir, on_wait):
        self.cfg, self.log, self.on_wait, self.run_dir = cfg, log, on_wait, run_dir
        self.cwd = f"{run_dir}/agent"
        os.makedirs(self.cwd, exist_ok=True)
        shutil.chown(self.cwd, AGENT_USER, AGENT_USER)
        self.session = None
        r = self.call(["auth", "status", "--json"], timeout=120)
        try:
            status = json.loads(r.stdout)
        except ValueError:
            status = {}
        if not status.get("loggedIn"):
            raise SystemExit(f"Claude Code is not signed in for {AGENT_USER}. Sign in once with:\n"
                             f"  runuser -u {AGENT_USER} -- bash -lc 'claude auth login'")
        log(f"Claude Code signed in ({status.get('authMethod')}, {status.get('subscriptionType', 'plan unknown')})")

    def call(self, args, stdin=None, timeout=None):
        env = ["env", "-i", f"HOME=/home/{AGENT_USER}", f"PATH=/home/{AGENT_USER}/.local/bin:/usr/bin:/bin",
               "LANG=C.UTF-8", "TERM=dumb", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"]
        return subprocess.run(["runuser", "-u", AGENT_USER, "--", *env, self.cfg["claude_bin"], *args],
                              input=stdin, capture_output=True, text=True, cwd=self.cwd, timeout=timeout)

    def new_session(self):
        self.session = str(uuid.uuid4())

    def transcript(self):
        found = glob.glob(f"/home/{AGENT_USER}/.claude/projects/*/{self.session}.jsonl")
        return found[0] if found else None

    def respond(self, messages):
        if self.session is None:
            self.new_session()
        errors = 0
        while True:
            args = ["-p", "--output-format", "json", "--tools", "", "--system-prompt", self.SYSTEM,
                    "--model", self.cfg["model"], "--effort", self.cfg["effort"], "--safe-mode",
                    "--strict-mcp-config", "--disable-slash-commands", "--permission-prompts", "none"]
            args += ["--resume", self.session] if self.transcript() else ["--session-id", self.session]
            stderr = ""
            try:
                r = self.call(args, stdin=messages[-1]["content"], timeout=4 * 3600)
                stderr, out = r.stderr, r.stdout.strip()
                try:
                    res = json.loads(out.splitlines()[-1]) if out else {}
                except ValueError:
                    res = {"is_error": True, "result": out[-2000:]}
            except subprocess.TimeoutExpired:
                res = {"is_error": True, "result": "no reply within 4 hours"}
            text = str(res.get("result") or "")
            if res.get("type") == "result" and not res.get("is_error"):
                break
            blob = f"{text} {stderr[-2000:]}".strip()
            if LIMIT.search(blob):
                self.on_wait(limit_wait(blob), blob[:300])
                continue
            if AUTH.search(blob):
                raise SystemExit(f"Claude Code sign-in problem (sign in again as {AGENT_USER}): {blob[:500]}")
            errors += 1
            if errors >= 8:
                raise RuntimeError(f"Claude Code kept failing: {blob[:500]}")
            wait = min(60 * 2 ** (errors - 1), 1800)
            self.log(f"Claude Code error ({blob[:300]}); retrying in {wait} s")
            time.sleep(wait)
        if self.transcript():
            shutil.copy(self.transcript(), f"{self.run_dir}/messages/claude_{self.session}.jsonl")
        u = res.get("usage") or {}
        usage = {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                 "cache_creation_input_tokens")}
        block = {"type": "text", "text": text}
        return dict(refusal=False, text=text, thinking="", content=[block], raw=[block], usage=usage,
                    stop_reason=res.get("subtype", "success"),
                    model=",".join(res.get("modelUsage") or {}) or self.cfg["model"],
                    context=usage["input_tokens"] + usage["cache_read_input_tokens"]
                    + usage["cache_creation_input_tokens"], notional_usd=res.get("total_cost_usd"))


class ReplayBackend:
    """Scripted replies (a JSON list of strings) for testing the harness without the API; after the
    last one it gives a final answer without a block."""

    def __init__(self, cfg, log):
        self.replies, self.i = json.load(open(cfg["replay"])), 0

    def respond(self, messages):
        text = self.replies[self.i] if self.i < len(self.replies) else "Final answer (replay): the script is done."
        self.i += 1
        block = {"type": "text", "text": text}
        return dict(refusal=False, text=text, thinking="", content=[block], raw=[block], usage={},
                    stop_reason="end_turn", model="replay", context=0)


# ---------- the run ----------

class Run:
    def __init__(self, name, args):
        self.name, self.dir = name, f"{WSP}/runs/{name}"
        for sub in ("work", "work/edited", "start", "ctl", "requests", "turns", "messages", "submission"):
            os.makedirs(f"{self.dir}/{sub}", exist_ok=True)
        self.cfg = dict(DEFAULTS)
        if os.path.exists(f"{self.dir}/config.json"):
            self.cfg.update(json.load(open(f"{self.dir}/config.json")))
        self.cfg.update({k: v for k, v in vars(args).items() if v is not None and k in DEFAULTS})
        if self.cfg["backend"] == "anthropic" and self.cfg["max_usd"] is None:
            raise SystemExit("pass --max-usd: the driver stops (resumably) once the estimated API spend reaches it")
        for key in ("start_from", "brief", "practice"):
            if self.cfg[key]:
                self.cfg[key] = os.path.realpath(self.cfg[key])
        if not glob.glob(f"{self.cfg['practice']}/*.jsonl"):
            raise SystemExit(f"--practice {self.cfg['practice']} has no .jsonl problem files")
        json.dump(self.cfg, open(f"{self.dir}/config.json", "w"), indent=1)
        self.state = dict(status="new", turn=0, conv=0, conv_turns=0, passes_used=0, code_runs=0, requests=0,
                          usd=0.0, tokens={}, invalid_streak=0, refusals=0, turns_after_budget=0, context=0,
                          limit_waits=0, limit_wait_seconds=0)
        if os.path.exists(f"{self.dir}/state.json"):
            self.state.update(json.load(open(f"{self.dir}/state.json")))
        start = self.cfg["start_from"]
        if start and not glob.glob(f"{self.dir}/start/*.npy"):
            files = [f for f in os.listdir(start) if f.endswith(".npy") or f == "manifest.json"]
            if not any(f.endswith(".npy") for f in files):
                raise SystemExit(f"--start-from {start} has no .npy edits")
            for f in files:
                shutil.copy(f"{start}/{f}", f"{self.dir}/start/{f}")
                if self.state["status"] == "new":
                    shutil.copy(f"{start}/{f}", f"{self.dir}/work/edited/{f}")
            report = f"{os.path.dirname(start)}/final_answer.md"  # the earlier run's own report, if it has one
            if os.path.exists(report):
                shutil.copy(report, f"{self.dir}/start/report.md")
        self.has_start = bool(glob.glob(f"{self.dir}/start/*.npy"))
        if self.cfg["brief"] and not os.path.exists(f"{self.dir}/brief.md"):
            shutil.copy(self.cfg["brief"], f"{self.dir}/brief.md")
        self.kernel_name, self.gpu_name = f"wsp-kernel-{name}", f"wsp-gpu-{name}"
        self.sock = f"{self.dir}/ctl/kernel.sock"
        self.problems = {}
        for fn in sorted(os.listdir(self.cfg["practice"])):
            if fn.endswith(".jsonl"):
                for line in open(f"{self.cfg['practice']}/{fn}"):
                    self.problems[json.loads(line)["id"]] = True
        prompt = open(f"{HARNESS}/prompt.txt").read().strip()
        about = f"{self.cfg['practice']}/about.txt"  # another practice set describes its own files
        if os.path.exists(about):
            head, tail = "and reference (a solution that passes the tests): ", " Choose whichever serve your experiments."
            i, j = prompt.find(head), prompt.find(tail)
            assert prompt.count(head) == prompt.count(tail) == 1 and i < j, \
                "prompt.txt changed: place the practice text by hand"
            prompt = prompt[:i + len(head)] + open(about).read().strip() + prompt[j:]
        self.panel, self.panel_suites = [], []
        if self.cfg["panel"]:
            if not os.path.exists(PANEL):
                raise SystemExit(f"--panel: {PANEL} does not exist (eval/prepare_panel.py builds it)")
            self.panel, count = [], Counter()
            for p in map(json.loads, open(PANEL)):  # q: a stable label without the question, for /work/panel_<k>.jsonl
                count[p["suite"]] += 1
                about = (p["instruction_id_list"] if p["suite"] == "instructions" else
                         f"{p.get('subject', '')}, {p['difficulty']}" if p["suite"] == "math" else p["difficulty"])
                self.panel.append({"id": p["id"], "suite": p["suite"], "problem": p["problem"], "about": about,
                                   "q": f"{p['suite']}/{count[p['suite']]}"})
            self.panel_suites = [x for x in PANEL_SUITES if any(p["suite"] == x for p in self.panel)]
            os.makedirs(f"{self.dir}/panel", exist_ok=True)
            prompt = self.panel_prompt(prompt)
        if os.path.exists(f"{self.dir}/brief.md"):
            prompt += "\n\nTHIS RUN.\n" + open(f"{self.dir}/brief.md").read().strip()
        for key in ("passes", "max_code_runs", "kernel_seconds", "kernel_memory_gb", "handoff_every"):
            prompt = prompt.replace("{{%s}}" % {"passes": "N_PASSES"}.get(key, key.upper()), str(self.cfg[key]))
        if self.cfg["handoff_every"]:
            rule = f"Every {self.cfg['handoff_every']} turns, or sooner when the conversation grows long,"
        else:
            rule = (f"When the conversation grows past about {self.cfg['max_context']:,} tokens (you are warned a "
                    f"little before),")
        prompt = prompt.replace("{{HANDOFF_RULE}}", rule)
        pp = float(self.cfg["presence_penalty"])
        prompt = prompt.replace("{{PRESENCE}}", f"presence penalty {pp:g}" if pp else "no presence penalty")
        assert "{{" not in prompt, "unfilled placeholder in prompt.txt"
        self.prompt = prompt
        self.stopped_llama = False
        self.gpu_used_at = None  # when the GPU server last finished a pass; None while it is not running
        self.messages = []

    def panel_prompt(self, prompt):
        """prompt.txt with the panel's description (prompt_panel.txt) as forward pass (c)."""
        text = open(f"{HARNESS}/prompt_panel.txt").read().rstrip()
        names = {"choice": "multiple choice", "math": "competition math", "instructions": "instruction following",
                 "code": "code"}
        sizes = [f"{sum(p['suite'] == x for p in self.panel)} {names[x]} ({x})" for x in self.panel_suites]
        base = "start" if self.has_start else "original"
        key = (', and "baseline": "start" (the default: this run\'s starting edit) or "original" (the unedited model)'
               if self.has_start else " (the baseline is the unedited model, \"original\")")
        if os.path.exists(f"{self.dir}/panel/base_{base}.jsonl"):  # put there before the run started, or a resume
            gen = f"The {base} answers are already generated, and every panel pass reuses them"
        else:
            gen = (f"The {base} answers are generated in your first panel pass, which therefore takes about twice as "
                   f"long, and later passes reuse them")
        for k, v in {"PANEL_TOTAL": f"{len(self.panel):,}", "PANEL_SIZES": ", ".join(sizes),
                     "PANEL_NAMES": ", ".join(f'"{x}"' for x in self.panel_suites), "PANEL_BASE_KEY": key,
                     "PANEL_BASE": base, "PANEL_BASE_GEN": gen}.items():
            text = text.replace("{{%s}}" % k, v)
        head = "A forward pass is one of two things"
        assert head in prompt and "\n\nHOW TO WORK." in prompt, "prompt.txt changed: place the panel text by hand"
        prompt = prompt.replace(head, "A forward pass is one of three things")
        return prompt.replace("\n\nHOW TO WORK.", f"\n{text}\n\nHOW TO WORK.", 1)

    # --- bookkeeping ---

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(f"{self.dir}/driver.log", "a") as f:
            f.write(line + "\n")

    def save_state(self):
        with open(f"{self.dir}/state.json.tmp", "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(f"{self.dir}/state.json.tmp", f"{self.dir}/state.json")

    def append(self, path, text):
        with open(f"{self.dir}/{path}", "a") as f:
            f.write(text)

    def add_message(self, role, content, raw=None):
        self.messages.append({"role": role, "content": content})
        self.append(f"messages/conv_{self.state['conv']:03d}.jsonl",
                    json.dumps({"role": role, "content": raw or content}) + "\n")

    def charge(self, usage):
        """Counts tokens; for the API backend also the estimated spend (a subscription is not billed per token)."""
        tot = self.state["tokens"]
        for k, v in usage.items():
            tot[k] = tot.get(k, 0) + v
        if self.cfg["backend"] != "anthropic":
            return 0.0
        p_in, p_out, p_read, p_write = PRICES.get(self.cfg["model"], PRICES["claude-opus-5-5"])
        usd = (usage.get("input_tokens", 0) * p_in + usage.get("output_tokens", 0) * p_out
               + usage.get("cache_read_input_tokens", 0) * p_read
               + usage.get("cache_creation_input_tokens", 0) * p_write) / 1e6
        self.state["usd"] = round(self.state["usd"] + usd, 4)
        return usd

    # --- containers ---

    def lock_gpu(self):
        path = f"{WSP}/runs/{GPUS[self.cfg['gpu']]['lock']}"
        self.lock = open(path, "w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another run or evaluation holds the {self.cfg['gpu']} ({path})")

    def pause_llama_swap(self):
        if GPUS[self.cfg["gpu"]]["llama_swap"] and running("llama-swap"):
            self.log("stopping llama-swap to free the 5090")
            sh("docker", "stop", "llama-swap", timeout=180)
            self.stopped_llama = True

    def restore_llama_swap(self):
        if self.stopped_llama:
            r = sh("docker", "start", "llama-swap", check=False)
            self.log("llama-swap started again" if r.returncode == 0 else f"could not start llama-swap: {r.stderr}")
            self.stopped_llama = False

    def ensure_gpu(self):
        if not running(self.gpu_name):
            self.pause_llama_swap()
            self.start_gpu()
            self.gpu_used_at = time.time()

    def remove_gpu_container(self):
        """A Colab relay is stopped gently first: it then tells the notebook, which shuts its runtime down."""
        if GPUS[self.cfg["gpu"]].get("remote") and running(self.gpu_name):
            sh("docker", "stop", "-t", "45", self.gpu_name, check=False, timeout=120)
        sh("docker", "rm", "-f", self.gpu_name, check=False)

    def release_gpu(self, why):
        """Stop the GPU server and give the GPU back (on the 5090, to llama-swap); the next forward pass starts it
        again."""
        if self.gpu_used_at is not None or running(self.gpu_name):
            self.remove_gpu_container()
            self.log(f"GPU released ({why})")
        self.gpu_used_at = None
        self.restore_llama_swap()

    def snapshot_code(self):
        """For the paper: copies the harness this start (or resume) runs with into <run>/code/<time>/, with an md5
        manifest, unless it is identical to the last copy. Never stops the run."""
        try:
            import hashlib
            files = sorted(glob.glob(f"{HARNESS}/*.py") + glob.glob(f"{HARNESS}/*.sh") + glob.glob(f"{HARNESS}/tests/*.py")
                           + [p for p in [self.cfg.get("brief")] if p and os.path.exists(p)])
            manifest = {os.path.relpath(p, HARNESS) if p.startswith(HARNESS) else os.path.basename(p):
                        hashlib.md5(open(p, "rb").read()).hexdigest() for p in files}
            old = sorted(glob.glob(f"{self.dir}/code/*/manifest.json"))
            if old and json.load(open(old[-1])) == manifest:
                return
            dest = f"{self.dir}/code/{time.strftime('%Y%m%d-%H%M%S')}"
            for p, rel in zip(files, manifest):
                os.makedirs(os.path.dirname(f"{dest}/{rel}"), exist_ok=True)
                shutil.copyfile(p, f"{dest}/{rel}")
            json.dump(manifest, open(f"{dest}/manifest.json", "w"), indent=1)
        except Exception as e:  # noqa: BLE001 - paper data must never stop a run
            self.log(f"code snapshot failed: {type(e).__name__}: {e}")

    def snapshot_edit(self, k, req):
        """For the paper (README, "Paper data"): keeps the tensors in /work/edited at request k, stored once by md5
        under /opt/wsp/paper/edits/, and appends {k, time, type, model, tensors: {name: md5}} to <run>/edits.jsonl.
        Never stops the run: a failure is only logged."""
        try:
            import hashlib
            store = "/opt/wsp/paper/edits"
            os.makedirs(store, exist_ok=True)
            tensors = {}
            for p in sorted(glob.glob(f"{self.dir}/work/edited/*.npy")):
                h = hashlib.md5()
                with open(p, "rb") as f:
                    for block in iter(lambda: f.read(1 << 22), b""):
                        h.update(block)
                md5 = h.hexdigest()
                tensors[os.path.basename(p)[:-4]] = md5
                dest = f"{store}/{md5}.npy"
                if not os.path.exists(dest):
                    shutil.copyfile(p, dest + ".tmp")
                    os.replace(dest + ".tmp", dest)
            with open(f"{self.dir}/edits.jsonl", "a") as f:
                f.write(json.dumps({"k": k, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "type": req.get("type"),
                                    "model": req.get("model"), "tensors": tensors}) + "\n")
        except Exception as e:  # noqa: BLE001 - paper data must never stop a run
            self.log(f"edit snapshot of request {k} failed: {type(e).__name__}: {e}")

    def wait_for_limit(self, seconds, why):
        """The agent's subscription hit its usage limit: wait for the reset, without holding the GPU."""
        self.log(f"usage limit ({why}); waiting {seconds / 60:.0f} min")
        self.state["limit_waits"] += 1
        self.state["waiting_until"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + seconds))
        self.save_state()
        if seconds >= self.cfg["gpu_idle_minutes"] * 60:
            self.release_gpu("waiting for the usage limit to reset")
        time.sleep(seconds)
        self.state["limit_wait_seconds"] += seconds
        self.state.pop("waiting_until", None)
        self.save_state()

    def start_kernel(self):
        sh("docker", "rm", "-f", self.kernel_name, check=False)
        if os.path.exists(self.sock):
            os.remove(self.sock)
        mem = f"{self.cfg['kernel_memory_gb']}g"
        sh("docker", "run", "-d", "--name", self.kernel_name, "--network", "none", "--read-only",
           "--tmpfs", "/tmp:rw,size=4g,mode=1777", "--memory", mem, "--memory-swap", mem, "--pids-limit", "1024",
           "--cpus", "12", "-e", "OMP_NUM_THREADS=12", "-e", "HOME=/tmp", "-e", "HF_HUB_OFFLINE=1",
           "-v", f"{MODEL_DIR}:/model:ro", "-v", f"{self.cfg['practice']}:/practice:ro", "-v", f"{self.dir}/work:/work",
           "-v", f"{self.dir}/start:/start:ro", "-v", f"{self.dir}/ctl:/ctl", "-v", f"{HARNESS}:/harness:ro",
           IMAGE, "python3", "-W", "ignore", "/harness/kernel_server.py")
        t0 = time.time()
        while time.time() - t0 < 300:
            if os.path.exists(self.sock):
                try:
                    kernel_client.execute(self.sock, "1", 30)
                    return
                except (ConnectionError, TimeoutError, OSError):
                    pass
            if not running(self.kernel_name):
                break
            time.sleep(2)
        logs = sh("docker", "logs", "--tail", "40", self.kernel_name, check=False)
        raise RuntimeError(f"the kernel did not start:\n{logs.stdout}{logs.stderr}")

    def start_gpu(self):
        sh("docker", "rm", "-f", self.gpu_name, check=False)
        req = f"{self.dir}/requests"
        for fn in os.listdir(req):  # a request left without a result by a crash is not re-run
            if fn.startswith("fwd_") and fn.endswith(".json") and not fn.endswith(".result.json") and \
                    not os.path.exists(f"{req}/{fn[:-5]}.result.json"):
                os.replace(f"{req}/{fn}", f"{req}/{fn}.abandoned")
        for fn in ("server.ready", "server.alive"):
            if os.path.exists(f"{req}/{fn}"):
                os.remove(f"{req}/{fn}")
        gpu = GPUS[self.cfg["gpu"]]
        if gpu.get("remote"):
            return self.start_colab_relay(gpu)
        sh("docker", "run", "-d", "--name", self.gpu_name, "--network", "none", "--shm-size", "8g",
           "--memory", "20g", "--device", gpu["device"], "-e", "PYTHONPATH=/harness", "-e", "HF_HUB_OFFLINE=1",
           "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1", "-e", f"WSP_PRESENCE={self.cfg['presence_penalty']}",
           "-e", f"WSP_GPU_MEMORY={gpu['run_memory']}",
           "-v", f"{MODEL_DIR}:/model:ro", "-v", f"{self.cfg['practice']}:/practice:ro", "-v", f"{self.dir}/work:/work",
           "-v", f"{self.dir}/start:/start:ro", "-v", f"{req}:/requests", "-v", f"{HARNESS}:/harness:ro",
           "-v", f"{CACHE}:/root/.cache",
           IMAGE, "python3", "-W", "ignore", "/harness/gpu_server.py")
        t0 = time.time()
        while time.time() - t0 < 900:
            if os.path.exists(f"{req}/server.ready"):
                self.log(f"GPU server ready after {time.time() - t0:.0f} s")
                return
            if not running(self.gpu_name):
                break
            time.sleep(3)
        logs = sh("docker", "logs", "--tail", "60", self.gpu_name, check=False)
        raise RuntimeError(f"the GPU server did not start:\n{logs.stdout[-3000:]}{logs.stderr[-3000:]}")

    def start_colab_relay(self, gpu):
        """The GPU server is a Colab notebook: the relay (colab_relay.py) stands in for it under the same container
        name. It is an HTTP server on the Cloudflare tunnel's Docker network, which the notebook's worker polls
        through the tunnel, and it writes server.ready once the worker serves it. A relay that loses the worker
        exits, which wait_result treats like a crashed GPU server."""
        if not os.path.exists(f"{COLAB_KEY}/key"):
            raise RuntimeError(f"--gpu colab needs the relay's key in {COLAB_KEY}/key (README, Colab runner)")
        req = f"{self.dir}/requests"
        sh("docker", "run", "-d", "--name", self.gpu_name, "--memory", "4g", "--cpus", "2",
           "--network", gpu["relay_network"], "--network-alias", gpu["relay_alias"],
           "-e", f"WSP_RUN={self.name}", "-e", f"WSP_PRESENCE={self.cfg['presence_penalty']}",
           "-e", f"WSP_GPU_MEMORY={gpu['run_memory']}", "-e", f"WSP_PORT={gpu['relay_port']}",
           "-v", f"{MODEL_DIR}:/model:ro", "-v", f"{self.cfg['practice']}:/practice:ro", "-v", f"{self.dir}/work:/work",
           "-v", f"{self.dir}/start:/start:ro", "-v", f"{req}:/requests", "-v", f"{HARNESS}:/harness:ro",
           "-v", f"{COLAB_KEY}:/colab:ro",
           IMAGE, "python3", "-u", "-W", "ignore", "/harness/colab_relay.py")
        self.log("waiting for the Colab worker: run colab_runner.ipynb in Colab on an A100 runtime (README, "
                 "Colab runner)")
        t0 = said = time.time()
        while time.time() - t0 < gpu["start_timeout"]:
            if os.path.exists(f"{req}/server.ready"):
                self.log(f"Colab worker ready after {(time.time() - t0) / 60:.0f} min")
                return
            if not running(self.gpu_name):
                break
            if time.time() - said > 3600:
                said = time.time()
                self.log(f"still waiting for the Colab worker ({(time.time() - t0) / 3600:.0f} h)")
            time.sleep(5)
        logs = sh("docker", "logs", "--tail", "60", self.gpu_name, check=False)
        raise RuntimeError(f"the Colab relay did not get a worker:\n{logs.stdout[-3000:]}{logs.stderr[-3000:]}")

    def stop_containers(self):
        self.remove_gpu_container()
        sh("docker", "rm", "-f", self.kernel_name, check=False)

    # --- conversation ---

    def unseen_result(self):
        """On a resume: the last message of the last conversation, if it is a block's result that the agent never
        answered (the driver was stopped while waiting for the reply)."""
        try:
            lines = open(f"{self.dir}/messages/conv_{self.state['conv']:03d}.jsonl").read().splitlines()
        except OSError:
            return None
        if len(lines) < 2:  # only the conversation's first message
            return None
        last = json.loads(lines[-1])
        return last["content"] if last["role"] == "user" and isinstance(last["content"], str) else None

    def kernel_alive(self):
        if not running(self.kernel_name) or not os.path.exists(self.sock):
            return False
        try:
            kernel_client.execute(self.sock, "1", 30)
            return True
        except (ConnectionError, TimeoutError, OSError):
            return False

    def new_conversation(self, reason=None, kernel_kept=True, carry=True, pending=None):
        """Start a conversation: the prompt, or for a handoff the prompt, notes, edits and the result
        of the last block if the agent has not seen it yet."""
        if pending is None and carry and len(self.messages) > 1 and self.messages[-1]["role"] == "user":
            pending = self.messages[-1]["content"]
        if self.messages or self.state["turn"]:
            self.state["conv"] += 1
        self.state["conv_turns"], self.state["context"], self.state["context_warned"] = 0, 0, False
        self.messages = []
        if hasattr(self.backend, "new_session"):
            self.backend.new_session()
        if reason is None:
            self.add_message("user", self.prompt)
            return
        notes_path = f"{self.dir}/work/notes.md"
        notes = open(notes_path, errors="replace").read() if os.path.exists(notes_path) else "(there is no notes file)"
        if len(notes) > 40_000:
            notes = notes[:8_000] + "\n[... middle of the notes cut ...]\n" + notes[-32_000:]
        s, c = self.state, self.cfg
        kernel = ("The kernel kept its variables." if kernel_kept else
                  "The kernel was restarted, so its variables are lost (files in /work are kept).")
        self.add_message("user", f"{self.prompt}\n\n---\nSESSION RESTART ({reason}). This is a new conversation in the "
                         f"same run; the earlier conversation is not shown to you. {kernel} Forward passes used: "
                         f"{s['passes_used']} of {c['passes']}. Code runs used: {s['code_runs']} of "
                         f"{c['max_code_runs']}. Turns so far: {s['turn']}.\n\nYour notes, /work/notes.md:\n<<<\n"
                         f"{notes}\n>>>\n\n{self.manifest_text()}\n"
                         + (f"\nThe result of your last block, which the earlier conversation did not show you:"
                            f"\n<<<\n{pending}\n>>>\n" if pending else ""))

    def manifest_text(self):
        """The handoff's list of edits: /work/edited/manifest.json, and in a run with a start only the tensors whose
        values differ from it (the kernel adds changed_from_start to each entry)."""
        path = f"{self.dir}/work/edited/manifest.json"
        text = open(path).read().strip() if os.path.exists(path) else "{}"
        head = "Your current edits (/work/edited/manifest.json; changed_entries = entries that differ from the original"
        if not self.has_start:
            return f"{head}):\n{text}"
        try:
            entries = json.loads(text)
            same = [n for n, e in entries.items() if e["changed_from_start"] == 0]
        except (ValueError, AttributeError, KeyError, TypeError):  # the start's own manifest, before any save_edit
            return f"{head}):\n{text}"
        listed = {n: e for n, e in entries.items() if e["changed_from_start"] != 0}
        dropped = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{self.dir}/start/*.npy")
                         if os.path.basename(p)[:-4] not in entries)
        return (f"{head}, changed_from_start = entries that differ from the starting model in /start; "
                f"{len(same)} more tensors hold the starting model's values and are not listed):\n"
                f"{json.dumps(listed, indent=1)}"
                + (f"\nTensors of the start that drop_edit put back to the original: {dropped}" if dropped else ""))

    def run(self):
        self.lock_gpu()
        resumed = self.state["status"] != "new"
        if self.state["status"] == "done":
            raise SystemExit("this run is finished; start a new one")
        if self.cfg["backend"] == "claude-code":
            self.backend = ClaudeCodeBackend(self.cfg, self.log, self.dir, self.wait_for_limit)
        else:
            self.backend = (ReplayBackend if self.cfg["backend"] == "replay" else AnthropicBackend)(self.cfg, self.log)
        self.snapshot_code()
        if not os.path.exists(f"{self.dir}/work/eval_prompt_format.txt"):
            shutil.copy(f"{WSP}/baseline/eval_prompt_format.txt", f"{self.dir}/work/eval_prompt_format.txt")
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))
        try:
            # A driver stopped with SIGKILL leaves its containers running: a resume keeps a kernel that still
            # answers (its variables survive) and the GPU server, and gives llama-swap back when it ends.
            kept = resumed and self.kernel_alive()
            self.log(f"{'keeping the running' if kept else 'starting the'} kernel for run {self.name} "
                     f"({'resumed' if resumed else 'new'}, {self.cfg['backend']} backend, {self.cfg['model']}, "
                     f"effort {self.cfg['effort']}, GPU {self.cfg['gpu']}, restart past "
                     f"{self.cfg['max_context']:,} tokens"
                     + (f" or every {self.cfg['handoff_every']} turns)" if self.cfg["handoff_every"] else ")"))
            if not kept:
                self.start_kernel()
            if resumed and running(self.gpu_name):
                self.gpu_used_at = time.time()
                self.stopped_llama = bool(GPUS[self.cfg["gpu"]]["llama_swap"]) and not running("llama-swap")
                self.log("keeping the running GPU server")
            if resumed:
                self.new_conversation("the driver was restarted", kernel_kept=kept, pending=self.unseen_result())
            else:
                self.new_conversation()
            self.state["status"] = "running"
            self.save_state()
            self.loop()
        except BaseException as e:
            if self.state["status"] == "running":
                self.state["status"] = f"stopped: {type(e).__name__}: {str(e)[:300]}"
            self.log(f"stopped: {type(e).__name__}: {e}")
            raise
        finally:
            self.save_state()
            self.stop_containers()
            self.restore_llama_swap()

    def stop(self, why):
        self.state["status"] = f"stopped: {why}"
        self.log(f"stopping: {why} (run the driver again to resume)")

    def loop(self):
        s, c = self.state, self.cfg
        while True:
            if c["max_usd"] is not None and s["usd"] >= c["max_usd"]:
                return self.stop(f"estimated spend ${s['usd']:.2f} reached --max-usd {c['max_usd']}")
            if s["turn"] >= c["max_turns"]:
                return self.stop(f"turn limit {c['max_turns']}")
            if (c["handoff_every"] and s["conv_turns"] >= c["handoff_every"]) or s["context"] > c["max_context"]:
                self.log(f"handoff after {s['conv_turns']} turns ({s['context']:,} tokens of context)")
                self.new_conversation(f"routine restart after {s['conv_turns']} turns, at {s['context']:,} tokens")
            s["turn"] += 1
            s["conv_turns"] += 1
            tdir = f"{self.dir}/turns/t{s['turn']:04d}"
            os.makedirs(tdir, exist_ok=True)
            if self.gpu_used_at is not None and time.time() - self.gpu_used_at > c["gpu_idle_minutes"] * 60:
                self.release_gpu(f"unused for {c['gpu_idle_minutes']} minutes")
            last_user = self.messages[-1]["content"]
            open(f"{tdir}/prompt.txt", "w").write(last_user if isinstance(last_user, str) else json.dumps(last_user))
            t0 = time.time()
            reply = self.backend.respond(self.messages)
            usd = self.charge(reply["usage"])
            s["context"] = reply["context"]
            meta = dict(turn=s["turn"], conv=s["conv"], model=reply["model"], stop_reason=reply["stop_reason"],
                        usage=reply["usage"], usd=round(usd, 4), seconds=round(time.time() - t0, 1))
            if reply.get("notional_usd") is not None:  # Claude Code's API-price equivalent; not billed
                meta["notional_usd"] = reply["notional_usd"]
            if reply["refusal"]:
                json.dump(dict(meta, stop_details=reply.get("stop_details")), open(f"{tdir}/run.json", "w"), indent=1)
                s["refusals"] += 1
                self.log(f"turn {s['turn']}: the API declined the request ({reply.get('stop_details')})")
                if s["refusals"] > 2:
                    return self.stop("the API declined three times")
                self.new_conversation("the provider declined to continue the previous conversation; the result "
                                      "of your last block is not shown, in case it caused that", carry=False)
                self.save_state()
                continue
            self.add_message("assistant", reply["content"], reply["raw"])
            open(f"{tdir}/answer.md", "w").write(reply["text"])
            if reply["thinking"]:
                open(f"{tdir}/thinking.md", "w").write(reply["thinking"])
            feedback, kind = self.handle(reply, s["turn"])
            meta.update(block=kind, passes_used=s["passes_used"], code_runs=s["code_runs"], usd_total=s["usd"])
            json.dump(meta, open(f"{tdir}/run.json", "w"), indent=1)
            spent = (f"${s['usd']:.2f}" if c["backend"] == "anthropic" else
                     f"{s['tokens'].get('output_tokens', 0):,} output tokens so far")
            self.log(f"turn {s['turn']}: {kind}; passes {s['passes_used']}/{c['passes']}, code runs "
                     f"{s['code_runs']}, {meta['seconds']:.0f} s, {spent}")
            if feedback is None:
                return self.finish(reply["text"])
            if s["invalid_streak"] >= 8:
                return self.stop("eight replies in a row without a runnable block")
            if c["handoff_every"] and s["conv_turns"] == c["handoff_every"] - 3:
                feedback += ("\n\n[The conversation will be restarted in 3 turns: make sure /work/notes.md is "
                             "up to date.]")
            elif s["context"] > 0.9 * c["max_context"] and not s.get("context_warned"):
                # about 40,000 tokens before the restart at 400,000: a dozen turns at the usual growth
                s["context_warned"] = True
                self.log(f"turn {s['turn']}: warned that the conversation ({s['context']:,} tokens) restarts soon")
                feedback += (f"\n\n[This conversation is at {s['context']:,} tokens and will be restarted when it "
                             f"passes {c['max_context']:,}: make sure /work/notes.md is up to date.]")
            self.add_message("user", feedback)
            self.save_state()

    def finish(self, text):
        open(f"{self.dir}/final_answer.md", "w").write(text)
        sub = f"{self.dir}/submission"
        shutil.rmtree(sub)
        shutil.copytree(f"{self.dir}/work/edited", sub)
        self.state["status"] = "done"
        self.log(f"run finished: {len(os.listdir(sub))} files in submission/, final answer in final_answer.md")

    # --- one reply ---

    def handle(self, reply, turn):
        """Returns (the next user message, or None when the run is over; what the reply held)."""
        s, c = self.state, self.cfg
        note = (f"[Your reply was cut at the {c['max_output_tokens']:,}-token output limit.]\n"
                if reply["stop_reason"] == "max_tokens" else "")
        blocks = find_blocks(reply["text"])
        budget_done = s["passes_used"] >= c["passes"]
        if budget_done:
            s["turns_after_budget"] += 1
        if not blocks:
            if budget_done:
                return None, "final answer"
            s["invalid_streak"] += 1
            return (note + f"Your reply has no ```python or ```forward block, so nothing was run. The session "
                    f"continues until all {c['passes']} forward passes are used ({s['passes_used']} used so far): "
                    f"reply with exactly one block."), "no block"
        if len(blocks) > 1:
            s["invalid_streak"] += 1
            return (note + f"Your reply has {len(blocks)} blocks, so nothing was run. Send exactly one ```python "
                    f"or ```forward block per reply (inside code, build lines that start with ``` from '`' * 3)."
                    ), "several blocks"
        kind, body, start, closed = blocks[0]
        if not closed:
            s["invalid_streak"] += 1
            return note + "Your block is not closed by a line ```, so it was not run.", f"unclosed {kind}"
        explanation = reply["text"][:start].strip()
        if len(explanation.split()) < 10:
            s["invalid_streak"] += 1
            return (note + "Your block came without an explanation, so it was not run. Before each block, explain "
                    "in plain English what you are about to do, why, and what result would change your plan."
                    ), f"{kind} without explanation"
        s["invalid_streak"] = 0
        self.append("explanations.md", f"## Turn {turn} ({kind})\n\n{explanation}\n\n")
        t0 = time.time()
        result = self.run_python(body) if kind == "python" else self.run_forward(body)
        self.append("tool_calls.jsonl", json.dumps({"turn": turn, "kind": kind, "body": body,
                                                    "result": result[:20000],
                                                    "seconds": round(time.time() - t0, 1)}) + "\n")
        if budget_done and s["turns_after_budget"] >= 40:
            result += ("\n\n[All forward passes have been used for a while: make your final choice now and reply "
                       "without a block.]")
        if budget_done and s["turns_after_budget"] >= 50:
            return None, "final answer (forced)"
        return note + result, kind

    def run_python(self, code):
        s, c = self.state, self.cfg
        if s["code_runs"] >= c["max_code_runs"]:
            return f"You have used all {c['max_code_runs']} code runs; nothing was run."
        s["code_runs"] += 1
        head = f"[code run {s['code_runs']} of {c['max_code_runs']}]"
        try:
            r = kernel_client.execute(self.sock, code, c["kernel_seconds"] + 30)
            out = r["output"]
            head = f"[code run {s['code_runs']} of {c['max_code_runs']}, {r['seconds']:.1f} s]"
        except TimeoutError:
            self.log("kernel time limit: restarting it")
            self.start_kernel()
            out = (f"[The code ran past the {c['kernel_seconds']}-second limit. The kernel was restarted, so its "
                   f"variables are lost; files in /work are kept and the predefined names are defined again.]")
        except (ConnectionError, OSError) as e:
            self.log(f"kernel died ({type(e).__name__}: {e}): restarting it")
            self.start_kernel()
            out = (f"[The kernel died, probably at the {c['kernel_memory_gb']} GB memory limit, and was restarted, "
                   f"so its variables are lost; files in /work are kept.]")
        return f"{head}\n{cut(ascii(out)) if out.strip() else '(no output)'}"

    # --- forward passes ---

    def list_arg(self, req, key, kind, required=False):
        """A list given inline (key) or as a JSON file under /work (key_file)."""
        if key in req and f"{key}_file" in req:
            raise Refused(f"give {key} or {key}_file, not both")
        if f"{key}_file" in req:
            path = req[f"{key}_file"]
            if not isinstance(path, str) or not path.startswith("/work/"):
                raise Refused(f"{key}_file must be a path under /work")
            host = os.path.realpath(f"{self.dir}/work/{path[len('/work/'):]}")
            if not host.startswith(os.path.realpath(f"{self.dir}/work") + "/") or not os.path.isfile(host):
                raise Refused(f"{key}_file: {path} does not exist")
            try:
                value = json.load(open(host))
            except ValueError as e:
                raise Refused(f"{key}_file: {path} is not valid JSON ({e})")
        elif key in req:
            value = req[key]
        elif required:
            raise Refused(f"missing {key}")
        else:
            return []
        if not isinstance(value, list) or not all(isinstance(x, kind) for x in value):
            raise Refused(f"{key} must be a list of {'strings' if kind is str else 'lists'}")
        return value

    @staticmethod
    def int_list(value, lo, hi, what):
        if not isinstance(value, list) or not all(isinstance(x, int) and not isinstance(x, bool) and lo <= x <= hi
                                                  for x in value):
            raise Refused(f"{what} must be a list of integers from {lo} to {hi}")
        return sorted(set(value))

    def validate(self, req):
        if not isinstance(req, dict):
            raise Refused("the block must hold one JSON object")
        weights = ("original", "edited", "start") if self.has_start else ("original", "edited")
        if req.get("type") == "panel":
            if not self.panel:
                raise Refused('this run has no panel: "type" must be "generate" or "teacher_forced"')
            unknown = set(req) - {"type", "suites", "baseline"}
            if unknown:
                raise Refused(f"unknown keys {sorted(unknown)}")
            suites = req.get("suites", self.panel_suites)
            if not isinstance(suites, list) or not suites or len(set(suites)) < len(suites) or \
                    not all(x in self.panel_suites for x in suites):
                raise Refused('"suites" must be a list of distinct suites from ' +
                              ", ".join(f'"{x}"' for x in self.panel_suites))
            bases = ["start", "original"] if self.has_start else ["original"]
            baseline = req.get("baseline", bases[0])
            if baseline not in bases:
                raise Refused('"baseline" must be ' + " or ".join(f'"{b}"' for b in bases))
            return {"type": "panel", "suites": [x for x in self.panel_suites if x in suites], "baseline": baseline}
        if req.get("type") == "generate":
            unknown = set(req) - {"type", "model", "problems", "problems_file", "prompts", "prompts_file", "n",
                                  "max_tokens", "activations", "baseline"}
            if unknown:
                raise Refused(f"unknown keys {sorted(unknown)}")
            if req.get("model") not in weights + ("both",):
                raise Refused('"model" must be ' + ", ".join(f'"{w}"' for w in weights) + ' or "both"')
            baseline, bases = req.get("baseline", "original"), [w for w in weights if w != "edited"]
            if "baseline" in req and req["model"] != "both":
                raise Refused('"baseline" goes with "model": "both"')
            if baseline not in bases:
                raise Refused('"baseline" must be ' + " or ".join(f'"{w}"' for w in bases))
            problems, prompts = self.list_arg(req, "problems", str), self.list_arg(req, "prompts", str)
            bad = [p for p in problems if p not in self.problems]
            if bad:
                raise Refused(f"{len(bad)} unknown problem ids, for example {bad[:3]}")
            if len(set(problems)) < len(problems):
                raise Refused("problems has duplicates (use n for more samples per problem)")
            if any(not p.strip() for p in prompts):
                raise Refused("prompts must be non-empty strings")
            n, max_tokens = req.get("n", 1), req.get("max_tokens", MAX_NEW)
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                raise Refused('"n" must be a positive integer')
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= MAX_NEW:
                raise Refused(f'"max_tokens" must be an integer from 1 to {MAX_NEW}')
            items, cap = len(problems) + len(prompts), 256 if req["model"] == "both" else 512
            if items == 0:
                raise Refused("no problems or prompts")
            if items * n > cap:
                raise Refused(f"{items} items x n={n} = {items * n} generations per model (at most {cap})")
            out = {"type": "generate", "model": req["model"], "problems": problems, "prompts": prompts, "n": n,
                   "max_tokens": max_tokens}
            if req["model"] == "both":
                out["baseline"] = baseline
            acts = req.get("activations")
            if acts is not None:
                if not isinstance(acts, dict) or set(acts) - {"layers", "mlp"}:
                    raise Refused('"activations" must be {"layers": [...], "mlp": [...]}')
                if "mlp" in acts and not isinstance(acts["mlp"], list):
                    raise Refused('"mlp" takes a list of decoder layers 0..23')
                layers = self.int_list(acts.get("layers", []), 0, 24, "activations.layers (hidden indices)")
                mlp = self.int_list(acts.get("mlp", []), 0, 23, "activations.mlp (decoder layers)")
                if layers or mlp:
                    out["activations"] = {"layers": layers, "mlp": mlp}
            return out
        if req.get("type") == "teacher_forced":
            unknown = set(req) - {"type", "model", "texts", "texts_file", "chats", "chats_file", "enable_thinking",
                                  "return"}
            if unknown:
                raise Refused(f"unknown keys {sorted(unknown)}")
            if req.get("model") not in weights:
                raise Refused('"model" must be ' + " or ".join(f'"{w}"' for w in weights))
            texts, chats = self.list_arg(req, "texts", str), self.list_arg(req, "chats", list)
            if bool(texts) == bool(chats):
                raise Refused("give a non-empty texts (or texts_file) or chats (or chats_file), not both")
            for chat in chats:
                if not chat or not all(isinstance(m, dict) and set(m) == {"role", "content"} and m["role"] in
                                       ("system", "user", "assistant") and isinstance(m["content"], str)
                                       for m in chat):
                    raise Refused('each chat must be a list of {"role": system|user|assistant, "content": "..."}')
            n = len(texts or chats)
            if n > 1024:
                raise Refused(f"{n} texts (at most 1,024)")
            ret = req.get("return", {"token_logprobs": True})
            if not isinstance(ret, dict) or set(ret) - {"token_logprobs", "topk", "hidden_states", "mlp_act",
                                                        "attn_layers"}:
                raise Refused('"return" takes token_logprobs, topk, hidden_states, mlp_act and attn_layers')
            topk = ret.get("topk", 0)
            if not isinstance(topk, int) or isinstance(topk, bool) or not 0 <= topk <= 50:
                raise Refused('"topk" must be an integer from 0 to 50')
            attn = self.int_list(ret.get("attn_layers", []), 0, 23, "attn_layers")
            if set(attn) - FULL_ATTN:
                raise Refused(f"attn_layers: only the full-attention layers {sorted(FULL_ATTN)} have attention maps")
            if attn and n > 2:
                raise Refused("attn_layers: at most 2 texts")
            out = {"type": "teacher_forced", "model": req["model"],
                   "return": {"token_logprobs": bool(ret.get("token_logprobs", True)), "topk": topk,
                              "hidden_states": self.int_list(ret.get("hidden_states", []), 0, 24, "hidden_states"),
                              "mlp_act": self.int_list(ret.get("mlp_act", []), 0, 23, "mlp_act"),
                              "attn_layers": attn}}
            if texts:
                out["texts"] = texts
            else:
                out["chats"], out["enable_thinking"] = chats, bool(req.get("enable_thinking", False))
            return out
        raise Refused('"type" must be "generate" or "teacher_forced"' + (' or "panel"' if self.panel else ""))

    def run_forward(self, body):
        s, c = self.state, self.cfg
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as e:
            return f"[forward request refused, no pass used] The block is not valid JSON: {e}"
        try:
            req = self.validate(raw)
        except Refused as e:
            return f"[forward request refused, no pass used] {e}"
        if s["passes_used"] >= c["passes"]:
            return f"[forward request refused] All {c['passes']} forward passes are used."
        self.ensure_gpu()
        s["requests"] += 1
        k = s["requests"]
        self.save_state()
        path = f"{self.dir}/requests/fwd_{k}.json"
        gpu_req = self.panel_request(req) if req["type"] == "panel" else req
        with open(path + ".tmp", "w") as f:
            json.dump(gpu_req, f)
        os.replace(path + ".tmp", path)
        self.log(f"request {k}: {req['type']}/{gpu_req['model']}")
        self.snapshot_edit(k, {"type": req["type"], "model": gpu_req["model"]})
        res, err = self.wait_result(k, path)
        self.gpu_used_at = time.time()
        if req["type"] == "panel":
            self.hide_panel_outputs(k)
        if err:
            return f"[request {k} failed, no pass used] {err}"
        if not res["ok"]:
            trace = "\n".join(res.get("trace", "").strip().splitlines()[-6:])
            return f"[request {k} failed, no pass used] {ascii(res['error'])[:3000]}\n{ascii(trace)[-1500:]}"
        s["passes_used"] += 1
        head = f"[forward pass {s['passes_used']} of {c['passes']} used (request {k}), {res['wall_seconds']:.0f} s]"
        try:
            body = (self.summarize_generation(k, req, res) if req["type"] == "generate" else
                    self.summarize_panel(k, req, res) if req["type"] == "panel" else self.summarize_tf(k, req))
        except Exception as e:  # noqa: BLE001 - the pass ran; report what went wrong in summarizing it
            self.log(f"summary of request {k} failed: {type(e).__name__}: {e}")
            body = f"(the pass ran, but the harness failed to summarize it: {type(e).__name__}: {e})"
        if s["passes_used"] >= c["passes"]:
            body += (f"\n\nThat was your last forward pass: all {c['passes']} are used. You can still run ```python "
                     "blocks, for example to put your final choice in /work/edited. When you are done, reply "
                     "without a block with your final report, as described in HOW THE SESSION RUNS.")
        return f"{head}\n{body}"

    def wait_result(self, k, path):
        res_path = f"{self.dir}/requests/fwd_{k}.result.json"
        t0 = last = time.time()
        while not os.path.exists(res_path):
            time.sleep(2)
            if time.time() - last < 30 or os.path.exists(res_path):
                continue
            last = time.time()
            why = None
            if not running(self.gpu_name):
                why = "the forward-pass server crashed"
            elif time.time() - t0 > FORWARD_TIMEOUT:
                why = f"it ran past the {FORWARD_TIMEOUT // 3600}-hour limit"
            if why:
                logs = sh("docker", "logs", "--tail", "80", self.gpu_name, check=False)
                open(f"{self.dir}/requests/fwd_{k}.crash.log", "w").write(logs.stdout + logs.stderr)
                os.replace(path, path + ".abandoned")
                self.log(f"request {k}: {why}; restarting the GPU server")
                self.start_gpu()
                return None, f"{why} and was restarted."
        return json.load(open(res_path)), None

    def grade(self, raw_name):
        r = sh("docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g,mode=1777",
               "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "16g", "--pids-limit", "512", "--cpus", "12", "-w", "/tmp",
               "-v", f"{self.cfg['practice']}:/practice:ro", "-v", f"{HARNESS}:/harness:ro", "-v", f"{self.dir}/work:/work:ro",
               IMAGE, "python3", "-W", "ignore", "/harness/grade_run.py", f"/work/{raw_name}", timeout=7200)
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
        n_raw = sum(1 for _ in open(f"{self.dir}/work/{raw_name}"))
        if len(rows) != n_raw:
            raise RuntimeError(f"grader returned {len(rows)} of {n_raw} rows: {r.stderr[-500:]}")
        with open(f"{self.dir}/work/{raw_name.replace('.raw.jsonl', '.jsonl')}", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return rows

    @staticmethod
    def describe(rows, label):
        out = []
        graded = [r for r in rows if r["status"] != "ungraded"]
        if graded:
            n, cnt = len(graded), Counter(r["status"] for r in graded)
            out.append(f"{label}: {n} graded answers, pass {pct(cnt['pass'], n)} ({cnt['pass']}/{n}); mean length "
                       f"{np.mean([r['n_tokens'] for r in graded]):,.0f} tokens; cut at max_tokens "
                       f"{pct(cnt['cut_at_max_tokens'], n)}; loop-flagged {pct(sum(r['loop'] for r in graded), n)}")
            groups = defaultdict(list)
            for r in graded:
                groups[f"{r['source']}/{r['difficulty']}"].append(r["status"] == "pass")
            out.append("  by source/difficulty: " + " | ".join(f"{g} {pct(sum(v), len(v))} of {len(v)}"
                                                              for g, v in sorted(groups.items())))
            fails = [f"{k} {cnt[k]}" for k in CATEGORIES if cnt[k]]
            out.append("  failures: " + (", ".join(fails) or "none"))
            exc = Counter(r.get("exc") or "?" for r in graded if r["status"] == "runtime_error")
            if exc:
                out.append("  runtime errors by type: " + ", ".join(f"{k} {v}" for k, v in exc.most_common(8)))
        own = [r for r in rows if r["status"] == "ungraded"]
        if own:
            out.append(f"{label}, your prompts: {len(own)} answers (not graded); mean length "
                       f"{np.mean([r['n_tokens'] for r in own]):,.0f} tokens; cut at max_tokens "
                       f"{pct(sum(r['finish_reason'] == 'length' for r in own), len(own))}; loop-flagged "
                       f"{pct(sum(r['loop'] for r in own), len(own))}")
        return out

    @staticmethod
    def paired(o_rows, e_rows, base="original"):
        O = {(r["id"], r["sample"]): r for r in o_rows if r["status"] != "ungraded"}
        E = {(r["id"], r["sample"]): r for r in e_rows if r["status"] != "ungraded"}
        keys = sorted(set(O) & set(E))
        if not keys:
            return []

        def stat(ks):
            per = defaultdict(list)
            for key in ks:
                per[key[0]].append(float(E[key]["status"] == "pass") - float(O[key]["status"] == "pass"))
            d = np.array([np.mean(v) for v in per.values()])
            se = 100 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            return len(d), 100 * d.mean(), se

        n_items, m, se = stat(keys)
        up = sum(O[k]["status"] != "pass" and E[k]["status"] == "pass" for k in keys)
        down = sum(O[k]["status"] == "pass" and E[k]["status"] != "pass" for k in keys)
        cut_d = 100 * np.mean([float(E[k]["finish_reason"] == "length") - float(O[k]["finish_reason"] == "length")
                               for k in keys])
        len_d = np.mean([E[k]["n_tokens"] - O[k]["n_tokens"] for k in keys])
        out = [f"paired, edited - {base}, over {n_items} problems ({len(keys)} answer pairs): pass {m:+.1f} points "
               f"(SE {se:.1f}); fail->pass {up}, pass->fail {down}; cut at max_tokens {cut_d:+.1f} points; mean "
               f"length {len_d:+,.0f} tokens"]
        sources = sorted({O[k]["source"] for k in keys})
        if len(sources) > 1:
            parts = []
            for src in sources:
                ni, mi, si = stat([k for k in keys if O[k]["source"] == src])
                parts.append(f"{src} {mi:+.1f} (SE {si:.1f}, {ni} problems)")
            out.append("  by source: " + " | ".join(parts))
        return out

    def summarize_generation(self, k, req, res):
        rows, lines, files = {}, [], []
        for v, raw in res["outputs"].items():
            rows[v] = self.grade(os.path.basename(raw))
            files.append(raw.replace(".raw.jsonl", ".jsonl"))
            lines += self.describe(rows[v], v)
            folder = res.get("activations", {}).get(v)
            if folder:
                idx_path = f"{self.dir}/work/{os.path.basename(folder)}/index.json"
                idx = json.load(open(idx_path))
                status = {(r["id"], r["sample"]): r["status"] for r in rows[v]}
                for smp in idx["samples"]:
                    smp["status"] = status[(smp["id"], smp["sample"])]
                json.dump(idx, open(idx_path, "w"))
                files.append(folder + "/")
        if len(rows) == 2:
            base = req.get("baseline", "original")
            lines += self.paired(rows[base], rows["edited"], base)
        secs = ", ".join(f"{v} {t:.0f} s" for v, t in res.get("seconds", {}).items())
        lines.append(f"Files: {', '.join(files)}. Time: {secs}.")
        return "\n".join(lines)

    def summarize_tf(self, k, req):
        folder = f"{self.dir}/work/fwd_{k}"
        meta = json.load(open(f"{folder}/meta.json"))
        lens = meta["lengths"]
        parts = []
        for name, fn in meta["files"].items():
            a = np.load(f"{folder}/{fn}", mmap_mode="r")
            parts.append(f"{name} {tuple(a.shape)} {a.dtype}")
        line = (f"Read {len(lens)} texts ({sum(lens):,} tokens; {min(lens):,} to {max(lens):,} each) with the "
                f"{req['model']} weights. /work/fwd_{k}/: {'; '.join(parts)}.")
        if "token_logprobs" in meta["files"]:
            lp = np.load(f"{folder}/token_logprobs.npy", mmap_mode="r")
            line += f" Mean log-prob per token: {np.nanmean(lp):.3f}."
        if meta.get("hidden_index"):
            line += f" hidden_states indices: {meta['hidden_index']}."
        if meta.get("mlp_layer"):
            line += f" mlp_act layers: {meta['mlp_layer']}."
        if meta.get("attn_layer"):
            line += f" attn layers: {meta['attn_layer']}."
        return line

    # --- the panel (README, "Panel") ---

    def panel_cache(self, base):
        """The baseline's graded panel answers by question id (runs/<run>/panel/base_<base>.jsonl): the baseline's
        weights never change in a run, so its answers are generated once."""
        path = f"{self.dir}/panel/base_{base}.jsonl"
        return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}

    def panel_request(self, req):
        """The GPU server's request for a panel pass: the edited model on the chosen suites, and the baseline as well
        ("model": "both") while its answers to them are not cached yet."""
        items = [p for p in self.panel if p["suite"] in req["suites"]]
        cached = self.panel_cache(req["baseline"])
        out = {"type": "generate", "model": "edited", "items": [[p["id"], p["problem"]] for p in items], "n": 1,
               "max_tokens": PANEL_MAX_TOKENS, "panel": req["suites"]}
        if not all(p["id"] in cached for p in items):
            out.update(model="both", baseline=req["baseline"])
        return out

    def hide_panel_outputs(self, k):
        """Moves a panel pass's answers out of /work, into runs/<run>/panel/, which the agent cannot read."""
        for path in glob.glob(f"{self.dir}/work/gen_{k}.raw.jsonl") + glob.glob(f"{self.dir}/work/gen_{k}_*.raw.jsonl"):
            os.replace(path, f"{self.dir}/panel/{os.path.basename(path)}")

    def grade_panel(self, raw_name):
        """Grades runs/<run>/panel/<raw_name> in the no-GPU sandbox with the evaluation's graders (grade_panel.py)."""
        r = sh("docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=2g,mode=1777",
               "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--memory", "16g", "--pids-limit", "512", "--cpus", "12", "-w", "/tmp",
               "-e", "PYTHONPATH=/harness:/evalbin", "-v", f"{PANEL}:/panel/panel.jsonl:ro",
               "-v", f"{HARNESS}:/harness:ro", "-v", f"{EVAL}/suites.py:/evalbin/suites.py:ro",
               "-v", f"{EVAL}/ifeval:/evalbin/ifeval:ro", "-v", f"{self.dir}/panel:/work:ro",
               GRADE_IMAGE, "python3", "-W", "ignore", "/harness/grade_panel.py", f"/work/{raw_name}", timeout=7200)
        rows = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
        n_raw = sum(1 for _ in open(f"{self.dir}/panel/{raw_name}"))
        if len(rows) != n_raw:
            raise RuntimeError(f"the panel grader returned {len(rows)} of {n_raw} rows: {r.stderr[-500:]}")
        with open(f"{self.dir}/panel/{raw_name.replace('.raw.jsonl', '.jsonl')}", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return rows

    @staticmethod
    def panel_suite(suite, e, b, base):
        """One suite's line: edited rows e and baseline rows b, in the same question order."""
        n = len(e)
        pe = np.array([r["status"] == "pass" for r in e], float)
        pb = np.array([r["status"] == "pass" for r in b], float)
        d = pe - pb
        m, se = 100 * d.mean(), (100 * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan"))

        def share(rows, test):
            return pct(sum(bool(test(r)) for r in rows), n)

        none = "no_code" if suite == "code" else "no_answer"
        line = (f"  {suite} ({n}): edited {pct(pe.sum(), n)} vs {base} {pct(pb.sum(), n)}, {m:+.1f} points (SE "
                f"{se:.1f}); fail->pass {int(((pe == 1) & (pb == 0)).sum())}, pass->fail "
                f"{int(((pe == 0) & (pb == 1)).sum())}; cut at max_tokens "
                f"{share(e, lambda r: r['finish_reason'] == 'length')} ({base} "
                f"{share(b, lambda r: r['finish_reason'] == 'length')}); no answer "
                f"{share(e, lambda r: r['status'] == none)} ({share(b, lambda r: r['status'] == none)}); "
                f"loop-flagged {share(e, lambda r: r['loop'])} ({share(b, lambda r: r['loop'])}); mean length "
                f"{np.mean([r['n_tokens'] for r in e]):,.0f} tokens "
                f"({np.mean([r['n_tokens'] for r in e]) - np.mean([r['n_tokens'] for r in b]):+,.0f})")
        if suite == "instructions":
            ok = [sum(r.get("inst_ok", 0) for r in rows) / max(sum(r.get("inst_total", 0) for r in rows), 1)
                  for rows in (e, b)]
            line += f"; single constraints followed {100 * ok[0]:.1f}% ({100 * ok[1]:.1f}%)"
        if suite == "code":
            ce, cb = Counter(r["status"] for r in e), Counter(r["status"] for r in b)
            fails = [f"{c} {ce[c]} ({cb[c]})" for c in CATEGORIES if ce[c] or cb[c]]
            line += "; failures: " + (", ".join(fails) or "none")
        return line, m, se

    def summarize_panel(self, k, req, res):
        base = req["baseline"]
        rows = {v: self.grade_panel(os.path.basename(raw)) for v, raw in res["outputs"].items()}
        cache = self.panel_cache(base)
        if base in rows:
            cache.update({r["id"]: r for r in rows[base]})
            path = f"{self.dir}/panel/base_{base}.jsonl"
            with open(path + ".tmp", "w") as f:
                for r in cache.values():
                    f.write(json.dumps(r) + "\n")
            os.replace(path + ".tmp", path)
        E = {r["id"]: r for r in rows["edited"]}
        with open(f"{self.dir}/work/panel_{k}.jsonl", "w") as f:
            for p in self.panel:
                if p["id"] in E:
                    e, b = E[p["id"]], cache[p["id"]]
                    f.write(json.dumps({"q": p["q"], "suite": p["suite"], "about": p["about"], "baseline": base,
                                        "status": e["status"], "base_status": b["status"], "tokens": e["n_tokens"],
                                        "base_tokens": b["n_tokens"], "cut": e["finish_reason"] == "length",
                                        "base_cut": b["finish_reason"] == "length"}) + "\n")
        lines = [f"Panel: {len(E):,} questions, one answer each, edited - {base} paired over the same questions (the "
                 f"answers are not shown; in brackets: {base}):"]
        diffs = []
        for suite in req["suites"]:
            ids = sorted(i for i in E if E[i]["suite"] == suite)
            line, m, se = self.panel_suite(suite, [E[i] for i in ids], [cache[i] for i in ids], base)
            lines.append(line)
            diffs.append((m, se))
        if len(diffs) > 1:
            m = float(np.mean([d[0] for d in diffs]))
            se = math.sqrt(sum(d[1] ** 2 for d in diffs)) / len(diffs)
            lines.append(f"  mean over the {len(diffs)} suites: {m:+.1f} points (SE {se:.1f})")
        secs = ", ".join(f"{v} {t:.0f} s" for v, t in res.get("seconds", {}).items())
        lines.append(f"Per question: /work/panel_{k}.jsonl. Time: {secs}." +
                     (f" The {base} answers were generated in this pass; later panel passes reuse them."
                      if base in rows else ""))
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--passes", type=int)
    ap.add_argument("--backend", choices=["claude-code", "anthropic", "replay"])
    ap.add_argument("--replay")
    ap.add_argument("--model")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--handoff-every", type=int,
                    help="also restart the conversation every N turns (default 0: only at --max-context)")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--max-code-runs", type=int)
    ap.add_argument("--kernel-seconds", type=int)
    ap.add_argument("--kernel-memory-gb", type=int, help="the kernel container's memory limit (default 20)")
    ap.add_argument("--max-context", type=int,
                    help="restart the conversation once its context passes this many tokens (default 400000)")
    ap.add_argument("--gpu-idle-minutes", type=float)
    ap.add_argument("--claude-bin", help="the claude executable (tests use a stand-in)")
    ap.add_argument("--presence-penalty", type=float)
    ap.add_argument("--start-from", help="a submission/ folder whose edit a new run starts from")
    ap.add_argument("--brief", help="a text file added to the prompt under THIS RUN")
    ap.add_argument("--practice", help="the practice problems' folder, mounted read-only at /practice "
                                       "(default practice/final; an about.txt in it describes its files in the prompt)")
    ap.add_argument("--gpu", choices=sorted(GPUS),
                    help="the GPU of the run's GPU server (default 5090; colab: a Colab notebook, README)")
    ap.add_argument("--panel", action="store_true", default=None,
                    help='offer the agent {"type": "panel"}, the cross-suite check (README, "Panel")')
    args = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run):
        raise SystemExit("run names use letters, digits, '.', '_' and '-'")
    Run(args.run, args).run()


if __name__ == "__main__":
    main()
