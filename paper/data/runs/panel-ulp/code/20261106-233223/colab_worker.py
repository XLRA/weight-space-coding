"""Colab side of a run whose GPU server is in Colab (gpus.py "colab"; colab_relay.py describes the relay's calls).

The notebook colab_runner.ipynb installs vLLM and calls main(url, key) with the relay's address (the server's
Cloudflare tunnel hostname) and key. bootstrap(url, key) does the notebook's steps in one call, for a one-line cell
in any notebook (README, Colab runner). The worker makes every connection; the relay only answers. It:
- reports to the relay every 15 s, from a thread with its own connection: its state, the GPU's readings and which of
  the busy pass's answer files are done. The answer gives the relay's session and whether the GPU was released;
- for a new session, copies the session's files into /content/wsp (the harness: gpu_server.py and wsp_ext.py; the
  model's small files; the practice problems; the starting edit). The weights come from Hugging Face at the pinned
  revision, checked against the server's sha256;
- links /model, /work, /requests, /practice, /start and /harness to those folders, and runs the server's own
  gpu_server.py as a subprocess, unchanged, with the session's presence penalty and GPU memory share;
- asks for the next request; makes /work/edited the request's set of tensors (downloading the ones it does not
  have, by md5) and writes /requests/fwd_<k>.json; when gpu_server.py has written the result, sends the outputs
  (gen_<k>*, fwd_<k>/) in pieces of at most 32 MB, then the result, and deletes the outputs here;
- shuts the Colab runtime down (google.colab.runtime.unassign, so that it stops using compute units) when it is not
  in a pass and the relay says the GPU was released (the driver found it idle for --gpu-idle-minutes, waits for a
  usage limit, or the run ended), or the relay could not be reached for GONE seconds. The next forward pass then
  waits until the notebook is run again.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import traceback

import requests

ROOT = "/content/wsp"
LINKS = ("model", "work", "requests", "practice", "start", "harness")
PIECE = 32 << 20        # Cloudflare takes request bodies up to 100 MB
BEAT = 15
RELEASED_GRACE = 120    # a relay restarted right after a release (a new session) keeps the runtime
GONE = 1800


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


class Relay:
    """The relay's calls (colab_relay.py), over one HTTP session; one instance per thread."""

    def __init__(self, url, key, headers=None):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {key}", "User-Agent": "wsp-colab-worker/1", **(headers or {})})

    @staticmethod
    def brief(r):
        """An error answer in one line: an HTML page (Cloudflare's, for example) by its title."""
        text = r.text or ""
        if "html" in r.headers.get("Content-Type", "") or text.lstrip().startswith("<"):
            m = re.search(r"<title[^>]*>(.*?)</title>", text, re.S | re.I)
            return f"a web page: {' '.join(m.group(1).split())[:150]}" if m else "a web page"
        return " ".join(text.split())[:300]

    def call(self, method, path, timeout=120, ok=(200,), **kw):
        r = self.s.request(method, self.url + path, timeout=timeout, **kw)
        if r.status_code not in ok:
            raise RuntimeError(f"{method} {path}: HTTP {r.status_code}: {self.brief(r)}")
        return r

    def json(self, path, **kw):
        return self.call("GET", path, **kw).json()

    def download(self, path, dest, timeout=600):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with self.s.get(self.url + path, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                raise RuntimeError(f"GET {path}: HTTP {r.status_code}: {self.brief(r)}")
            with open(dest + ".part", "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        os.replace(dest + ".part", dest)

    def upload(self, k, rel, path):
        size, offset = os.path.getsize(path), 0
        with open(path, "rb") as f:
            while offset < size or (size == 0 and offset == 0):
                data = f.read(PIECE)
                for attempt in range(5):
                    try:
                        self.call("PUT", f"/v1/out/{k}/{rel}", params={"offset": offset}, data=data, timeout=600)
                        break
                    except (requests.RequestException, RuntimeError) as e:
                        if "HTTP 410" in str(e) or attempt == 4:
                            raise
                        log(f"fwd_{k}: sending {rel} at {offset}: {e}; trying again")
                        time.sleep(10 * (attempt + 1))
                offset += len(data)
                if size == 0:
                    break


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def gpu_readings():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,temperature.gpu,power.draw,utilization.gpu,memory.used,"
                              "memory.total,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        name, t, p, u, mu, mt, drv = [x.strip() for x in out.split(",")]
        return {"name": name, "temp": float(t), "power": float(p), "util": int(float(u)),
                "mem_used_gb": round(float(mu) / 1024, 1), "mem_total_gb": round(float(mt) / 1024, 1), "driver": drv}
    except Exception:  # noqa: BLE001 - the report goes out without readings
        return {}


class Worker:
    def __init__(self, make_relay, root=ROOT, link=True, on_idle=None, server_script=None):
        self.r, self.hb = make_relay(), make_relay()  # one connection per thread
        self.root, self.link, self.on_idle = root, link, on_idle
        self.server_script = server_script  # tests: a stand-in for the session's gpu_server.py
        self.status = {"state": "starting", "session": None, "busy": None, "error": None, "started": time.time(),
                       "passes": 0}
        self.session, self.server, self.server_key = None, None, None
        self.placed, self.local_md5 = {}, {}
        self.relay = {"session": None, "released": False, "at": None}  # the last heartbeat answer
        self.reached, self.released_seen = time.time(), None
        self.lock = threading.Lock()
        os.makedirs(root, exist_ok=True)

    def p(self, sub):
        return f"{self.root}/{sub}"

    # --- reports ---

    def report(self, **kw):
        with self.lock:
            self.status.update(kw)

    def pass_progress(self):
        """For the watcher: the busy pass's answer files that are done, seconds after the pass was placed."""
        if not self.placed:
            return {}
        k = min(self.placed)
        out = {}
        try:
            for name in os.listdir(self.p("work")):
                if re.fullmatch(rf"gen_{k}([._].*)?\.raw\.jsonl", name):
                    path = self.p(f"work/{name}")
                    out[name] = {"after": round(os.path.getmtime(path) - self.placed[k]),
                                 "answers": sum(1 for _ in open(path, errors="replace"))}
        except OSError:
            pass
        return {"pass": {"k": k, "placed": self.placed[k], "done_files": out}}

    def beat(self):
        with self.lock:
            body = dict(self.status, updated=time.time(), gpu=gpu_readings())
        body.update(self.pass_progress())
        ans = self.hb.call("POST", "/v1/heartbeat", json=body, timeout=60).json()
        now = time.time()
        with self.lock:
            self.relay = dict(ans, at=now)
            self.reached = now

    def heartbeat(self):
        failing, logged = None, 0  # the first failed report of an outage, and when one was last logged
        while True:
            try:
                self.beat()
                if failing:
                    log(f"the relay answers again after {(time.time() - failing) / 60:.0f} min")
                failing = None
            except Exception as e:  # noqa: BLE001 - keep reporting; check_idle counts the silence
                with self.lock:
                    self.relay["at"] = None
                now = time.time()
                failing = failing or now
                if now - logged >= 300:  # the first failure, then every 5 minutes
                    logged = now
                    log(f"report failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(BEAT)

    # --- session ---

    def mirror(self, area, files, only=None):
        """Makes the local folder hold the session's files of one area (by md5), deleting other files there."""
        local = self.p(area)
        os.makedirs(local, exist_ok=True)
        for name, h in files.items():
            path = f"{local}/{name}"
            if not (os.path.exists(path) and md5(path) == h):
                self.r.download(f"/v1/file/{area}/{name}", path)
                if md5(path) != h:
                    raise RuntimeError(f"{area}/{name} arrived damaged")
        for name in os.listdir(local):
            if os.path.isfile(f"{local}/{name}") and name not in files and (only is None or only(name)):
                os.remove(f"{local}/{name}")

    def adopt(self, s):
        log(f"session {s['session']} of run {s['run']}: copying the harness, model files, practice problems and "
            "starting edit")
        self.report(state="copying the session's files", session=None)
        for area in ("harness", "practice", "start"):
            self.mirror(area, s["files"][area], only=(lambda n: n.endswith(".npy")) if area == "start" else None)
        self.mirror("model", s["files"]["model"], only=lambda n: not n.endswith(".safetensors"))
        for name, sha in s["model"]["weights"].items():
            path = self.p(f"model/{name}")
            if not os.path.exists(path) or sha256(path) != sha:
                self.report(state="downloading the model weights")
                from huggingface_hub import hf_hub_download
                got = hf_hub_download(s["model"]["repo"], name, revision=s["model"]["revision"],
                                      cache_dir=self.p("hf-cache"))
                shutil.copy(got, path)
                if sha256(path) != sha:
                    raise RuntimeError(f"{name} from {s['model']['repo']}@{s['model']['revision']} does not match "
                                       "the server's copy")
        for sub in ("work/edited", "requests"):
            os.makedirs(self.p(sub), exist_ok=True)
        if self.link:
            for name in LINKS:
                if os.path.islink(f"/{name}") or not os.path.exists(f"/{name}"):
                    if os.path.islink(f"/{name}"):
                        os.remove(f"/{name}")
                    os.symlink(self.p(name), f"/{name}")
        # Requests of an earlier session are dropped; the GPU server is kept unless what it loads changed.
        for f in os.listdir(self.p("requests")):
            if re.fullmatch(r"fwd_\d+(\.result)?\.json", f):
                os.remove(self.p(f"requests/{f}"))
        for k in list(self.placed):
            self.clean(k)
        key = {"files": {a: s["files"][a] for a in ("harness", "model", "start")}, "weights": s["model"]["weights"],
               "presence": s["presence"], "gpu_memory": s["gpu_memory"]}
        if self.server is None or self.server.poll() is not None or key != self.server_key:
            self.start_server(s)
            self.server_key = key
        self.session = s
        self.report(session=s["session"], state="ready" if self.server_ready() else "loading the model")

    def start_server(self, s):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(60)
            except subprocess.TimeoutExpired:
                self.server.kill()
        ready = self.p("requests/server.ready")
        if os.path.exists(ready):
            os.remove(ready)
        env = dict(os.environ, PYTHONPATH=self.p("harness"), HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                   DO_NOT_TRACK="1", WSP_PRESENCE=str(s["presence"]), WSP_GPU_MEMORY=str(s["gpu_memory"]))
        self.server_log = open(self.p("gpu_server.log"), "a")
        log("starting gpu_server.py (loading the model takes a few minutes)")
        script = self.server_script or self.p("harness/gpu_server.py")
        self.server = subprocess.Popen(["python3", "-W", "ignore", script], env=env, stdout=self.server_log,
                                       stderr=subprocess.STDOUT, cwd=self.root)

    def server_ready(self):
        return os.path.exists(self.p("requests/server.ready")) and self.server is not None and self.server.poll() is None

    # --- requests ---

    def local_md5_of(self, path):
        st = os.stat(path)
        key = (path, st.st_size, st.st_mtime_ns)
        if key not in self.local_md5:
            self.local_md5[key] = md5(path)
        return self.local_md5[key]

    def place(self, env):
        k = env["k"]
        self.report(state="fetching the request", busy=k)
        edited = self.p("work/edited")
        for name, h in env["edited"].items():
            path = f"{edited}/{name}.npy"
            if os.path.exists(path) and self.local_md5_of(path) == h:
                continue
            self.r.download(f"/v1/tensor/{h}", path)
            if self.local_md5_of(path) != h:
                raise RuntimeError(f"request {k}: tensor {name} arrived damaged")
        for name in os.listdir(edited):
            if name.endswith(".npy") and name[:-4] not in env["edited"]:
                os.remove(f"{edited}/{name}")
        path = self.p(f"requests/fwd_{k}.json")
        with open(path + ".tmp", "w") as out:
            json.dump(env["request"], out)
        os.replace(path + ".tmp", path)
        self.placed[k] = time.time()
        self.report(state="busy", busy=k)
        log(f"fwd_{k}: on the GPU ({env['request']['type']}/{env['request']['model']})")

    def outputs(self, k):
        """The files gpu_server.py wrote for request k, relative to /work."""
        pat = re.compile(rf"^(gen_{k}([._].*)?|fwd_{k})$")
        out = []
        for name in os.listdir(self.p("work")):
            if not pat.match(name):
                continue
            full = self.p(f"work/{name}")
            if os.path.isdir(full):
                for dirpath, _, files in os.walk(full):
                    out += [os.path.relpath(f"{dirpath}/{f}", self.p("work")) for f in files]
            else:
                out.append(name)
        return sorted(out)

    def send(self, k):
        res = json.load(open(self.p(f"requests/fwd_{k}.result.json")))
        rels = self.outputs(k)
        files = {rel: {"size": os.path.getsize(self.p(f"work/{rel}")), "md5": md5(self.p(f"work/{rel}"))} for rel in rels}
        mb = sum(f["size"] for f in files.values()) / 1e6
        log(f"fwd_{k}: done on the GPU in {res.get('wall_seconds')} s; sending {len(rels)} file{'' if len(rels) == 1 else 's'} ({mb:,.0f} MB)")
        self.report(state="sending the outputs", busy=k, error=None)
        todo = rels
        for attempt in range(4):
            try:
                for rel in todo:
                    self.r.upload(k, rel, self.p(f"work/{rel}"))
                r = self.r.call("POST", f"/v1/done/{k}", json={"files": files, "result": res,
                                                               "gpu": gpu_readings().get("name")},
                                ok=(200, 409, 410), timeout=600)
            except RuntimeError as e:
                if "HTTP 410" not in str(e):
                    raise
                r = None
            if r is None or r.status_code == 410:
                log(f"fwd_{k}: the driver no longer waits for it; dropping its outputs")
                break
            if r.status_code == 200:
                with self.lock:
                    self.status["passes"] += 1
                log(f"fwd_{k}: outputs delivered")
                break
            todo = r.json().get("send_again", [])
            log(f"fwd_{k}: {len(todo)} files arrived damaged; sending them again")
        else:
            raise RuntimeError(f"fwd_{k}: the outputs could not be delivered")
        self.clean(k)
        self.report(state="ready", busy=None)

    def clean(self, k):
        for rel in self.outputs(k):
            top = self.p(f"work/{rel.split('/')[0]}")
            if os.path.isdir(top):
                shutil.rmtree(top, ignore_errors=True)
            elif os.path.exists(top):
                os.remove(top)
        for f in (f"requests/fwd_{k}.json", f"requests/fwd_{k}.result.json"):
            if os.path.exists(self.p(f)):
                os.remove(self.p(f))
        self.placed.pop(k, None)

    # --- idle ---

    def check_idle(self):
        """Shuts the runtime down when the run no longer needs it (see the module's docstring)."""
        if self.on_idle is None or self.placed:
            return
        with self.lock:
            rel, reached = dict(self.relay), self.reached
        now, why = time.time(), None
        if rel.get("at"):  # the relay answered; it exits RELEASE_GRACE s after a release, so the flag is kept
            self.released_seen = (self.released_seen or now) if rel.get("released") else None
        if self.released_seen and now - self.released_seen > RELEASED_GRACE:
            why = "the run released the GPU (idle, waiting for a usage limit, or finished)"
        elif now - reached > GONE:
            why = f"the relay could not be reached for {GONE // 60} min"
        if why:
            log(f"stopping this runtime: {why}")
            self.report(state=f"stopped: {why}")
            try:
                self.beat()
            except Exception:  # noqa: BLE001 - the shutdown goes ahead
                pass
            if self.server is not None and self.server.poll() is None:
                self.server.terminate()
            self.on_idle(why)

    # --- loop ---

    def step(self):
        self.check_idle()
        with self.lock:
            rel = dict(self.relay)
        if not rel.get("at"):
            self.report(state="waiting for the relay (it starts at the run's next forward pass)")
            return
        if rel.get("released"):
            self.report(state="the run released the GPU")
            return
        if self.session is None or rel.get("session") != self.session["session"]:
            s = self.r.call("GET", "/v1/session", ok=(200, 503)).json()
            if "session" not in s:
                return
            self.adopt(s)
        if self.server is not None and self.server.poll() is not None:
            tail = open(self.p("gpu_server.log"), errors="replace").read()[-1500:]
            self.report(state="gpu_server.py stopped; restarting it", error=tail)
            log("gpu_server.py stopped:\n" + tail)
            for k in self.placed:  # the request it was on fails (no pass is used) instead of running again
                res = self.p(f"requests/fwd_{k}.result.json")
                if not os.path.exists(res):
                    with open(res, "w") as out:
                        json.dump({"ok": False, "error": "the GPU server on Colab stopped during this request: "
                                   + tail[-600:], "wall_seconds": 0}, out)
            self.start_server(self.session)
        for k in sorted(self.placed):
            if os.path.exists(self.p(f"requests/fwd_{k}.result.json")):
                self.send(k)
        if not self.server_ready():
            if not self.placed:
                self.report(state="loading the model")
            return
        if self.status["state"] in ("loading the model", "copying the session's files"):
            self.report(state="ready")
            log("gpu_server.py is ready")
        if not self.placed:
            r = self.r.call("GET", "/v1/next", params={"wait": 20}, ok=(200, 204), timeout=90)
            if r.status_code == 200:
                env = r.json()
                if env.get("session") == self.session["session"]:
                    self.place(env)

    def loop(self):
        threading.Thread(target=self.heartbeat, daemon=True).start()
        log("worker started; waiting for the relay")
        errors = 0
        while True:
            try:
                self.step()
                errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001 - report and go on
                errors += 1
                self.report(error=f"{type(e).__name__}: {e}"[:1500])
                log(f"error: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
                time.sleep(min(15 * errors, 300))
            time.sleep(2)


def stop_runtime(why):
    from google.colab import runtime
    log(f"disconnecting and deleting this Colab runtime: {why}")
    runtime.unassign()


VLLM, TRANSFORMERS = "0.32.0", "5.19.0"  # as on the home server


def bootstrap(url, key, stop_when_idle=True, headers=None):
    """The notebook's steps in one call: checks the GPU, installs vLLM and transformers if needed (vLLM's CUDA 12.9
    build on a driver older than 580), then runs main(). For a one-line Colab cell that has fetched this file."""
    import sys
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    if q.returncode != 0:
        raise SystemExit("no GPU: Runtime -> Change runtime type -> A100 GPU")
    name, mem, driver = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
    log(f"{name}, {mem}, driver {driver}")
    probe = [sys.executable, "-c", "import vllm, transformers, torch; "
             "print(vllm.__version__, transformers.__version__, torch.cuda.is_available())"]
    have = subprocess.run(probe, capture_output=True, text=True).stdout.split()
    if not (len(have) == 3 and have[0].startswith(VLLM) and have[1] == TRANSFORMERS):
        if int(driver.split(".")[0]) >= 580:
            spec = f"vllm=={VLLM} transformers=={TRANSFORMERS}"
        else:
            spec = (f"https://github.com/vllm-project/vllm/releases/download/v{VLLM}/"
                    f"vllm-{VLLM}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl transformers=={TRANSFORMERS} "
                    "--extra-index-url https://download.pytorch.org/whl/cu129")
        log(f"installing vLLM {VLLM} and transformers {TRANSFORMERS} (3 to 6 minutes)")
        r = subprocess.run(f"{sys.executable} -m pip install -q {spec}", shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"pip install failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
        have = subprocess.run(probe, capture_output=True, text=True).stdout.split()
    if not (len(have) == 3 and have[2] == "True"):
        raise SystemExit(f"vLLM or torch does not work here: {' '.join(have) or 'import failed'}")
    log(f"vLLM {have[0]}, transformers {have[1]}, CUDA available")
    main(url, key, stop_when_idle=stop_when_idle, headers=headers)


def main(url, key, stop_when_idle=True, headers=None, **kw):
    """The notebook's entry point: url is the relay's https address, key the relay's key."""
    w = Worker(lambda: Relay(url, key, headers), on_idle=stop_runtime if stop_when_idle else None, **kw)
    try:
        w.loop()
    finally:
        if w.server is not None and w.server.poll() is None:
            w.server.terminate()
