"""Client side of kernel_server.py: send one code block, get its output (used by the driver and tests)."""
import json
import socket
import struct
import sys


def execute(sock_path, code, timeout):
    """Returns the server's reply dict; raises TimeoutError or ConnectionError."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
        data = json.dumps({"code": code}).encode()
        s.sendall(struct.pack("!Q", len(data)) + data)
        head = s.recv(8, socket.MSG_WAITALL)
        if len(head) < 8:
            raise ConnectionError("kernel closed the connection")
        n = struct.unpack("!Q", head)[0]
        buf = b""
        while len(buf) < n:
            chunk = s.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise ConnectionError("kernel closed the connection")
            buf += chunk
        return json.loads(buf)
    except socket.timeout as e:
        raise TimeoutError from e
    finally:
        s.close()


if __name__ == "__main__":
    print(execute(sys.argv[1], sys.stdin.read(), float(sys.argv[2]) if len(sys.argv) > 2 else 900)["output"])
