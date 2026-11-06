"""Checks every Now state of progress.py against made-up run folders (run on the server from /tmp/ptest)."""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/tmp/ptest")
import progress as P  # noqa: E402

P.COLOR = False
P.RUNS = "/tmp/ptest/runs"
P.EVAL = "/tmp/ptest/eval"
shutil.rmtree(P.RUNS, ignore_errors=True)
shutil.rmtree(P.EVAL, ignore_errors=True)
now = time.time()
CFG = {"passes": 20, "max_code_runs": 300, "kernel_seconds": 900, "kernel_memory_gb": 14, "gpu_idle_minutes": 30,
       "gpu": "5080"}
fake_ctr = {}
real_container = P.container
P.container = lambda name: fake_ctr.get(name)


def put(path, text, age=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    os.utime(path, (now - age, now - age))


def logline(age, msg):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now - age)) + " " + msg


def case(title, files, st=None, log=(), alive=True, ctr=None, cfg=None):
    name = "fake-" + str(len(os.listdir(P.RUNS)) if os.path.isdir(P.RUNS) else 0)
    run = f"{P.RUNS}/{name}"
    os.makedirs(f"{run}/turns", exist_ok=True)
    os.makedirs(f"{run}/requests", exist_ok=True)
    for rel, (text, age) in files.items():
        put(f"{run}/{rel}", text, age)
    fake_ctr.clear()
    fake_ctr.update({k.replace("NAME", name): v for k, v in (ctr or {}).items()})
    s = dict(dict(code_runs=11, passes_used=2, requests=3, conv=0), **(st or {}))
    P._frame.clear()
    c = cfg or CFG
    rows = P.activity(name, run, s, c, list(log), now, alive) + [P.containers_row(name, run, c, now)]
    print(f"--- {title}")
    for r in rows:
        print(r)


GEN = json.dumps({"type": "generate", "model": "both", "baseline": "start", "problems": ["a", "b"],
                  "prompts": ["x"], "n": 4, "max_tokens": 32768})
TF = json.dumps({"type": "teacher_forced", "model": "start", "texts": ["t"] * 62, "return": {}})
TURN = "turns/t0012/"
case("no turn yet", {})
case("usage limit", {TURN + "prompt.txt": ("[code run 11 of 300, 0.1 s]\nok", 60)},
     st={"waiting_until": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + 1500))},
     log=[logline(60, "usage limit (Claude AI usage limit reached|1762450000); waiting 25 min")])
files = {f"turns/t{t:04d}/run.json": (json.dumps({"seconds": 10 * t}), 0) for t in range(1, 12)}
case("thinking after a code run", dict(files, **{TURN + "prompt.txt": ("[code run 11 of 300, 0.1 s]\nok", 75)}))
case("thinking after a cut reply and a timeout", {TURN + "prompt.txt": (
    "[Your reply was cut at the 128,000-token output limit.]\n[code run 11 of 300]\n[The code ran past the 900-second "
    "limit. The kernel was restarted]", 5)})
case("thinking after a pass", {TURN + "prompt.txt": ("[forward pass 2 of 20 used (request 3), 18 s]\nRead 62", 30)})
case("thinking, restart message", {TURN + "prompt.txt": ("P" * 6000 + "\n---\nSESSION RESTART (routine restart after "
                                                          "40 turns, at 300,000 tokens). This is", 200)})
case("thinking, opening prompt", {TURN + "prompt.txt": ("P" * 6000, 20)})
case("Claude Code error", {TURN + "prompt.txt": ("[code run 11 of 300, 0.1 s]", 100)},
     log=[logline(30, "Claude Code error (API Error: 529 overloaded); retrying in 120 s")])
case("code running", {TURN + "prompt.txt": ("x", 100), TURN + "answer.md": ("x", 95),
                      "explanations.md": ("## Turn 12 (python)\n\nthis and that\n", 95)})
case("code near the limit", {TURN + "prompt.txt": ("x", 900), TURN + "answer.md": ("x", 800),
                             "explanations.md": ("## Turn 12 (python)\n\nthis and that\n", 800)})
case("kernel restart", {TURN + "prompt.txt": ("x", 999), TURN + "answer.md": ("x", 940),
                        "explanations.md": ("## Turn 12 (python)\n\nx\n", 940)},
     log=[logline(2, "kernel time limit: restarting it")])
FWD = {TURN + "prompt.txt": ("x", 200), TURN + "answer.md": ("x", 100),
       "explanations.md": ("## Turn 12 (forward)\n\nx\n", 100), "requests/fwd_3.json": (TF, 3000),
       "requests/fwd_3.result.json": ('{"ok": true, "wall_seconds": 75}', 2900)}
case("stopping llama-swap", FWD, log=[logline(90, "stopping llama-swap to free the 5090")])
case("GPU server loading", FWD, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 70}})
case("checking the request", FWD)
FWD["requests/server.ready"] = ("x", 5000)
GEN_P = dict(FWD, **{"requests/fwd_4.json": (GEN, 90), "work/gen_4_start.raw.jsonl": ("{}\n" * 12, 50)})
case("generation, second model", GEN_P, st={"requests": 4},
     ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}, "wsp-kernel-NAME": {"cg": "/none", "started": now - 9000}})
GEN_P2 = dict(FWD, **{"requests/fwd_4.json": (GEN, 90)})
case("generation, first model", GEN_P2, st={"requests": 4})
GEN_G = dict(GEN_P, **{"work/gen_4_edited.raw.jsonl": ("{}\n" * 12, 30), "requests/fwd_4.result.json": ("{}", 29),
                       "work/gen_4_start.jsonl": ("{}\n" * 12, 20)})
case("grading", GEN_G, st={"requests": 4})
case("teacher-forced on the GPU", dict(FWD, **{"requests/fwd_4.json": (TF, 50)}), st={"requests": 4},
     ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})
case("teacher-forced summarizing", dict(FWD, **{"requests/fwd_4.json": (TF, 50), "requests/fwd_4.result.json": ("{}", 4)}),
     st={"requests": 4})
case("between turns", {TURN + "prompt.txt": ("x", 30), TURN + "answer.md": ("x", 20), TURN + "run.json": ("{}", 1)})
case("handoff", {TURN + "prompt.txt": ("x", 30), TURN + "answer.md": ("x", 20), TURN + "run.json": ("{}", 1)},
     log=[logline(1, "handoff after 40 turns (401,000 tokens of context)")])
case("no runnable block", {TURN + "prompt.txt": ("x", 30), TURN + "answer.md": ("x", 1)})
case("driver dead", {}, alive=False)
case("GPU server idle past the limit", {"requests/server.ready": ("x", 2500), "requests/fwd_3.json": (TF, 2400),
                                        "requests/fwd_3.result.json": ("{}", 2000)},
     ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})

# A run whose GPU server is a Colab notebook.
COL = dict(CFG, gpu="colab")
W = {"state": "ready", "session": "s1", "updated": now - 20, "started": now - 5400, "busy": None,
     "gpu": {"name": "NVIDIA A100-SXM4-40GB", "temp": 61, "power": 240, "util": 97, "mem_used_gb": 30.1,
             "mem_total_gb": 40.0}}
CF = {TURN + "prompt.txt": ("x", 200), TURN + "answer.md": ("x", 100),
      "explanations.md": ("## Turn 12 (forward)\n\nx\n", 100)}
case("colab: waiting, no worker", CF, cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 400}})
case("colab: waiting, worker loading", dict(CF, **{"requests/colab-worker.json": (json.dumps(dict(W, state="loading the model")), 5),
     "requests/colab-relay.json": (json.dumps({"k": None, "phase": "preparing the session", "since": now - 30}), 30)}),
     cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 400}})
CR = dict(CF, **{"requests/server.ready": ("x", 3000), "requests/fwd_4.json": (GEN, 90)})
case("colab: uploading tensors", dict(CR, **{"requests/colab-relay.json": (json.dumps({"k": 4, "phase": "sending the request and 9 edit tensors (54 MB)", "since": now - 12}), 12),
     "requests/colab-worker.json": (json.dumps(W), 5)}), st={"requests": 4}, cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})
case("colab: not picked up", dict(CR, **{"requests/colab-relay.json": (json.dumps({"k": 4, "phase": "sending the request and 0 edit tensors (0 MB)", "since": now - 40}), 40),
     "requests/colab-worker.json": (json.dumps(W), 5)}), st={"requests": 4}, cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})
WB = dict(W, state="busy", busy=4, **{"pass": {"k": 4, "placed": now - 80, "done_files": {"gen_4_start.raw.jsonl": {"after": 50, "answers": 12}}}})
case("colab: generating", dict(CR, **{"requests/colab-relay.json": (json.dumps({"k": 4, "phase": "waiting for the Colab worker", "since": now - 85}), 85),
     "requests/colab-worker.json": (json.dumps(WB), 5)}), st={"requests": 4}, cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})
case("colab: downloading", dict(CR, **{"requests/colab-relay.json": (json.dumps({"k": 4, "phase": "receiving the outputs: 3 MB so far", "since": now - 4}), 4),
     "requests/colab-worker.json": (json.dumps(dict(W, updated=now - 400)), 5)}), st={"requests": 4}, cfg=COL, ctr={"wsp-gpu-NAME": {"cg": "/none", "started": now - 4000}})
case("colab: relay off", {"requests/fwd_3.json": (TF, 2400), "requests/fwd_3.result.json": ("{}", 2000)}, cfg=COL)
withw = [n for n in sorted(os.listdir(P.RUNS)) if os.path.exists(f"{P.RUNS}/{n}/requests/colab-worker.json")]
print("--- colab GPU rows:")
for r in P.colab_rows("colab", P.GPUS["colab"], [f"agent {n}" for n in withw[-2:]] + ["agent fake-0"]):
    print(r)

# The evaluation's progress line and estimate.
os.makedirs(P.EVAL, exist_ok=True)
put(f"{P.EVAL}/ev.out", "code: reusing\ncode / edited: 512/2168 in 200 s (8390 tok/s)\n"
    "code / edited: 1024/2168 in 210 s (8100 tok/s)\n", 60)
print("--- evaluation:", P.eval_progress("ev", ["code", "mmlupro"], ["edited"]))
put(f"{P.EVAL}/ev.out", "code / edited: 2168/2168 in 90 s (8000 tok/s)\n", 30)
print("--- evaluation:", P.eval_progress("ev", ["code", "mmlupro"], ["edited"]))

# Live containers: memory and, on the second sample, CPU.
P.container = real_container
for i in range(2):
    P._frame.clear()
    print("--- live pilot-17 kernel:", P.usage("wsp-kernel-pilot-17"))
    if i == 0:
        time.sleep(6)
