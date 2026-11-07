"""A persistent Python kernel over a unix socket, like one notebook.

Runs inside the kernel container (no network, no GPU). The driver sends one JSON message per
code block ({"code": ...}) and gets back {"output": ..., "seconds": ...}. Variables persist
between blocks. If the last statement is an expression, its value is shown. Time limits are
enforced by the driver, which restarts the container when a block runs too long.
"""
import ast
import io
import json
import os
import socket
import struct
import sys
import time
import traceback

SOCK = "/ctl/kernel.sock"
MAX_CAPTURE = 200_000

namespace = {"__name__": "__main__"}


class Capture(io.TextIOBase):
    def __init__(self):
        self.parts, self.size = [], 0

    def write(self, s):
        if self.size < MAX_CAPTURE:
            self.parts.append(s[:MAX_CAPTURE - self.size])
        self.size += len(s)
        return len(s)

    def value(self):
        out = "".join(self.parts)
        if self.size > MAX_CAPTURE:
            out += f"\n[... {self.size - MAX_CAPTURE:,} more characters of output]"
        return out


def run(code):
    cap = Capture()
    old = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = cap
    t0 = time.time()
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, "<cell>", "exec"), namespace)
        if last is not None:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    except BaseException:  # noqa: BLE001 - report everything, including SystemExit and KeyboardInterrupt
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old
    return {"output": cap.value(), "seconds": round(time.time() - t0, 2)}


def recv(conn):
    head = conn.recv(8, socket.MSG_WAITALL)
    if len(head) < 8:
        return None
    n = struct.unpack("!Q", head)[0]
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf)


def send(conn, obj):
    data = json.dumps(obj).encode()
    conn.sendall(struct.pack("!Q", len(data)) + data)


def main():
    prelude = run(open("/harness/kernel_prelude.py").read())
    if "Traceback" in prelude["output"]:
        print(prelude["output"], flush=True)
        sys.exit(1)
    if os.path.exists(SOCK):
        os.remove(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    print("kernel ready", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn:
            msg = recv(conn)
            if msg is not None:
                send(conn, run(msg["code"]))


if __name__ == "__main__":
    main()
