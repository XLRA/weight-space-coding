"""Relay for a run whose GPU server is a Colab notebook (gpus.py "colab").

The driver starts it in place of the GPU container, under the same name (wsp-gpu-<run>), on the Docker network of
the server's Cloudflare tunnel (gpus.py relay_network) with the alias wsp-colab-relay. The tunnel's public hostname
(README, "Colab runner") points at http://wsp-colab-relay:8790. The relay opens no connections of its own: the Colab
worker (colab_worker.py, started by the notebook colab_runner.ipynb) polls it over HTTPS. To the driver nothing
changes: it writes /requests/fwd_<k>.json and waits for /requests/fwd_<k>.result.json.

Every call needs the header "Authorization: Bearer <key>", the key in /colab/key (on the server
/opt/wsp/secrets/colab/key, which the notebook reads from Colab's secrets as WSP_KEY).

GET  /v1/worker                    colab_worker.py, for the notebook
GET  /v1/session                   the session: its id, the presence penalty, vLLM's GPU memory share, the model's
                                   revision and weight hashes, and the md5 of each file the worker copies
                                   (harness/, model/, practice/, start/)
GET  /v1/file/<area>/<name>        one of those files
POST /v1/heartbeat                 the worker's report (JSON), every 15 s; the answer has the session id and whether
                                   the driver released the GPU
GET  /v1/next?wait=20              the oldest request without a result, {k, request, edited: {tensor: md5}}, or 204
                                   if there is none within wait seconds
GET  /v1/tensor/<md5>              an edit tensor of a request (a copy taken when the request was first handed out)
PUT  /v1/out/<k>/<path>?offset=N   a piece (at most 64 MB) of an output file of request k, path relative to /work
POST /v1/done/<k>                  {files: {path: {size, md5}}, result, gpu}: checks the files, moves them into /work
                                   and writes the result; 409 lists files to send again, 410 means the driver dropped
                                   the request

The relay writes /requests/server.ready once a worker serves this session, the worker's last report to
/requests/colab-worker.json and its own phase to /requests/colab-relay.json (both for progress.py). It exits when a
worker that was ready has not reported for STALE seconds, which the driver treats as a crashed GPU server. On
docker stop (the driver released the GPU) it answers heartbeats with "released" for RELEASE_GRACE seconds, so that
the worker can shut its runtime down, then exits.
"""
import glob
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

RUN = os.environ["WSP_RUN"]
PORT = int(os.environ.get("WSP_PORT", "8790"))
REQ, WORK, MODEL = "/requests", "/work", "/model"
TENSORS = f"{REQ}/.colab-tensors"     # edit tensors handed out with a request, by md5
INCOMING = f"{WORK}/.colab-in"        # output files being received, moved into /work when complete
REPO = "Qwen/Qwen3.5-0.8B"
STALE = 600            # seconds without a report from a ready worker before the relay gives up
RELEASE_GRACE = 30     # seconds the relay keeps answering after docker stop
CHUNK_MAX = 64 << 20
KEY = open("/colab/key").read().strip()
NAME = re.compile(r"[A-Za-z0-9_.-]+")

lock = threading.Lock()
state = {"sid": None, "session": None, "files": {}, "worker": None, "seen": 0.0, "ready": False, "released": False,
         "envelopes": {}, "sent": {}}
_md5 = {}


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def write_local(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f)
    os.replace(path + ".tmp", path)


def md5(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _md5:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _md5[key] = h.hexdigest()
    return _md5[key]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def phase(k, what):
    state["phase"] = (k, what)  # a plain assignment, so that it can be called with the lock held
    write_local(f"{REQ}/colab-relay.json", {"k": k, "phase": what, "since": time.time()})


def revision():
    """The Hugging Face commit the server's copy of the model was downloaded at."""
    for meta in glob.glob(f"{MODEL}/.cache/huggingface/download/*.safetensors.metadata"):
        return open(meta).read().split()[0]
    raise SystemExit("cannot tell the model's revision (no .cache/huggingface/download/*.metadata)")


def session_files():
    """{area: {name: path}} of the files the worker copies."""
    files = {"harness": {f: f"/harness/{f}" for f in ("gpu_server.py", "wsp_ext.py")},
             "model": {f: f"{MODEL}/{f}" for f in sorted(os.listdir(MODEL))
                       if not f.startswith(".") and not f.endswith(".safetensors") and os.path.isfile(f"{MODEL}/{f}")},
             "practice": {os.path.basename(p): p for p in sorted(glob.glob("/practice/*.jsonl"))},
             "start": {os.path.basename(p): p for p in sorted(glob.glob("/start/*.npy"))}}
    return files


def start_session():
    phase(None, "preparing the session")
    files = session_files()
    cache = f"{REQ}/.model-sha256.json"
    known = json.load(open(cache)) if os.path.exists(cache) else {}
    weights = {}
    for path in sorted(glob.glob(f"{MODEL}/*.safetensors")):
        st = os.stat(path)
        k = f"{os.path.basename(path)}:{st.st_size}:{st.st_mtime_ns}"
        if k not in known:
            known[k] = sha256(path)
            write_local(cache, known)
        weights[os.path.basename(path)] = known[k]
    sid = uuid.uuid4().hex[:12]
    session = {"session": sid, "run": RUN, "started": time.time(),
               "presence": float(os.environ.get("WSP_PRESENCE", "0")),
               "gpu_memory": float(os.environ.get("WSP_GPU_MEMORY", "0.6")),
               "model": {"repo": REPO, "revision": revision(), "weights": weights},
               "files": {area: {n: md5(p) for n, p in v.items()} for area, v in files.items()}}
    for d in (TENSORS, INCOMING):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    with lock:
        state.update(sid=sid, session=session, files=files)
    phase(None, "waiting for the Colab worker")
    log(f"session {sid}: {sum(len(v) for v in files.values())} files; waiting for the Colab worker")


def pending():
    """Request numbers the driver wrote that have no result yet."""
    out = []
    for p in glob.glob(f"{REQ}/fwd_*.json"):
        m = re.fullmatch(r"fwd_(\d+)\.json", os.path.basename(p))
        if m and not os.path.exists(f"{REQ}/fwd_{m.group(1)}.result.json"):
            out.append(int(m.group(1)))
    return sorted(out)


def envelope(k):
    """Request k as the worker gets it, with copies of its edit tensors taken the first time it is handed out."""
    with lock:
        if k in state["envelopes"]:
            return state["envelopes"][k]
    req = json.load(open(f"{REQ}/fwd_{k}.json"))
    edited = {}
    for p in sorted(glob.glob(f"{WORK}/edited/*.npy")):
        h = md5(p)
        edited[os.path.basename(p)[:-4]] = h
        if not os.path.exists(f"{TENSORS}/{h}.npy"):
            shutil.copyfile(p, f"{TENSORS}/{h}.npy.part")
            os.replace(f"{TENSORS}/{h}.npy.part", f"{TENSORS}/{h}.npy")
    keep = set(edited.values())
    with lock:
        for e in state["envelopes"].values():
            keep |= set(e["edited"].values())
    for p in glob.glob(f"{TENSORS}/*.npy"):
        if os.path.basename(p)[:-4] not in keep:
            os.remove(p)
    env = {"k": k, "session": state["sid"], "request": req, "edited": edited}
    mb = sum(os.path.getsize(f"{TENSORS}/{h}.npy") for h in set(edited.values())) / 1e6
    with lock:
        state["envelopes"][k] = env
        state["sent"][k] = {"tensors": set(), "total": len(set(edited.values())), "mb": mb}
    phase(k, f"sending the request and {len(set(edited.values()))} edit tensors ({mb:.0f} MB)")
    log(f"fwd_{k}: handed out ({req['type']}/{req['model']}, {len(edited)} tensors, {mb:.0f} MB)")
    return env


def output_path(k, rel):
    """rel if it is a path request k may write (gen_<k>*, fwd_<k>/...), else None."""
    parts = rel.split("/")
    if not all(NAME.fullmatch(p) and p not in (".", "..") for p in parts):
        return None
    if len(parts) == 1 and re.fullmatch(rf"gen_{k}([._][A-Za-z0-9_.-]*)?", parts[0]):
        return rel
    if len(parts) > 1 and parts[0] == f"fwd_{k}":
        return rel
    return None


def finish(k, done):
    """Checks request k's received files, moves them into /work and writes the result for the driver."""
    files = done.get("files", {})
    bad = []
    for rel, f in files.items():
        p = f"{INCOMING}/{k}/{rel}"
        if output_path(k, rel) is None:
            return 400, {"error": f"not an output path of request {k}: {rel}"}
        if not os.path.exists(p) or os.path.getsize(p) != f["size"] or md5(p) != f["md5"]:
            bad.append(rel)
    if bad:
        return 409, {"send_again": bad}
    for rel in files:
        dest = f"{WORK}/{rel}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(f"{INCOMING}/{k}/{rel}", dest)
    shutil.rmtree(f"{INCOMING}/{k}", ignore_errors=True)
    res = done.get("result") or {"ok": False, "error": "the worker sent no result"}
    size = sum(f["size"] for f in files.values())
    res["colab"] = {"gpu": done.get("gpu"), "outputs_mb": round(size / 1e6, 1)}
    write_local(f"{REQ}/fwd_{k}.result.json", res)
    with lock:
        state["envelopes"].pop(k, None)
    phase(k, "done")
    log(f"fwd_{k}: result written (ok={res.get('ok')}, {size / 1e6:,.0f} MB of outputs)")
    return 200, {"ok": True}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # one line per call is too much; errors are logged by the handlers
        pass

    def reply(self, code, obj=None, body=None, ctype="application/json"):
        data = body if body is not None else (json.dumps(obj).encode() if obj is not None else b"")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path):
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def body(self, limit):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError(f"body of {n} bytes is over the limit of {limit}")
        return self.rfile.read(n)

    def authorized(self):
        got = self.headers.get("Authorization", "")
        if hmac.compare_digest(got.encode(), f"Bearer {KEY}".encode()):
            return True
        time.sleep(1)
        self.reply(401, {"error": "wsp colab relay: missing or wrong key"})
        return False

    def do_GET(self):
        self.handle_call("GET")

    def do_POST(self):
        self.handle_call("POST")

    def do_PUT(self):
        self.handle_call("PUT")

    def handle_call(self, method):
        try:
            if not self.authorized():
                return
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            q = parse_qs(u.query)
            if parts[:1] != ["v1"]:
                return self.reply(404, {"error": "unknown path"})
            route = (method, parts[1] if len(parts) > 1 else "")
            if route == ("GET", "worker"):
                return self.send_file("/harness/colab_worker.py")
            if route == ("GET", "session"):
                with lock:
                    s = state["session"]
                return self.reply(200, s) if s else self.reply(503, {"error": "the session is being prepared"})
            if route == ("GET", "file") and len(parts) == 4:
                with lock:
                    path = state["files"].get(parts[2], {}).get(parts[3])
                return self.send_file(path) if path else self.reply(404, {"error": "no such file"})
            if route == ("POST", "heartbeat"):
                return self.heartbeat(json.loads(self.body(1 << 20) or b"{}"))
            if route == ("GET", "next"):
                return self.next(min(float(q.get("wait", ["0"])[0]), 50))
            if route == ("GET", "tensor") and len(parts) == 3 and re.fullmatch(r"[0-9a-f]{32}", parts[2]):
                return self.tensor(parts[2])
            if route == ("PUT", "out") and len(parts) >= 4 and parts[2].isdigit():
                return self.put_out(int(parts[2]), "/".join(parts[3:]), int(q.get("offset", ["0"])[0]))
            if route == ("POST", "done") and len(parts) == 3 and parts[2].isdigit():
                k = int(parts[2])
                if k not in pending():
                    return self.reply(410, {"error": f"request {k} has a result or was dropped by the driver"})
                code, obj = finish(k, json.loads(self.body(16 << 20)))
                return self.reply(code, obj)
            return self.reply(404, {"error": "unknown call"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:  # noqa: BLE001 - the worker retries; the relay goes on
            log(f"{method} {self.path.split('?')[0]}: {type(e).__name__}: {e}")
            try:
                self.reply(500, {"error": f"{type(e).__name__}: {e}"[:500]})
            except OSError:
                pass

    def heartbeat(self, w):
        now = time.time()
        w["seen_at"] = now
        write_local(f"{REQ}/colab-worker.json", w)
        with lock:
            state["worker"], state["seen"] = w, now
            sid, released = state["sid"], state["released"]
            became_ready = (not state["ready"] and not released and w.get("session") == sid
                            and w.get("state") in ("ready", "busy"))
            if became_ready:
                state["ready"] = True
        if became_ready:
            write_local(f"{REQ}/server.ready", {"at": now, "gpu": w.get("gpu")})
            phase(None, "ready")
            log(f"the Colab worker is ready ({(w.get('gpu') or {}).get('name', '?')})")
        ph = state.get("phase")
        if w.get("state") == "busy" and ph and ph[0] == w.get("busy") and ph[1].startswith("sending"):
            phase(w["busy"], "the Colab worker runs it")
        self.reply(200, {"session": sid, "released": released})

    def next(self, wait):
        t0 = time.time()
        while True:
            with lock:
                ok = state["ready"] and not state["released"]
            ks = pending() if ok else []
            if ks:
                return self.reply(200, envelope(ks[0]))
            if time.time() - t0 >= wait:
                return self.reply(204)
            time.sleep(1)

    def tensor(self, h):
        path = f"{TENSORS}/{h}.npy"
        if not os.path.exists(path):
            return self.reply(404, {"error": "no such tensor"})
        self.send_file(path)
        with lock:
            for k, s in state["sent"].items():
                if k in state["envelopes"] and h in state["envelopes"][k]["edited"].values():
                    s["tensors"].add(h)
                    if len(s["tensors"]) == s["total"]:
                        phase(k, "the Colab worker runs it")

    def put_out(self, k, rel, offset):
        if k not in pending():
            return self.reply(410, {"error": f"request {k} has a result or was dropped by the driver"})
        if output_path(k, rel) is None:
            return self.reply(400, {"error": f"not an output path of request {k}: {rel}"})
        data = self.body(CHUNK_MAX)
        path = f"{INCOMING}/{k}/{rel}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "r+b" if os.path.exists(path) else "wb") as f:
            f.seek(offset)
            f.write(data)
        got = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(f"{INCOMING}/{k}") for f in fs)
        phase(k, f"receiving the outputs: {got / 1e6:,.0f} MB so far")
        self.reply(200, {"ok": True, "size": os.path.getsize(path)})


def on_stop(signum, frame):
    """docker stop: the driver released the GPU or the run ended. Heartbeats get "released" for a while."""
    with lock:
        state["released"] = True
    phase(None, "released: the GPU is no longer needed")
    log("stopped by the driver: telling the Colab worker the GPU is released")
    threading.Timer(RELEASE_GRACE, lambda: os._exit(0)).start()


def main():
    for f in ("server.ready", "colab-worker.json", "colab-relay.json"):
        if os.path.exists(f"{REQ}/{f}"):
            os.remove(f"{REQ}/{f}")
    signal.signal(signal.SIGTERM, on_stop)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    start_session()
    while True:
        time.sleep(5)
        with lock:
            ready, seen, released = state["ready"], state["seen"], state["released"]
        if ready and not released and time.time() - seen > STALE:
            log(f"the Colab worker has not reported for {STALE // 60} min: stopping (the driver restarts the relay)")
            sys.exit(1)


if __name__ == "__main__":
    main()
