"""Offline check of driver.AnthropicBackend against a local mock of the Messages API (no key, no cost).

Checks the request the SDK sends (model, effort, adaptive thinking, auto caching, fallbacks and
their beta header, streaming), that a streamed reply is parsed into text and thinking, that the
assistant turn goes back into the next request with its thinking block and signature unchanged,
and that a refusal is reported without content.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SEEN = []


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def reply(stop="end_turn", text="I will check the kernel first.\n\n```python\nprint(1)\n```\n"):
    usage = {"input_tokens": 100, "output_tokens": 1, "cache_read_input_tokens": 2000,
             "cache_creation_input_tokens": 300}
    ev = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                 "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                                                 "stop_sequence": None, "usage": usage}}]
    if stop != "refusal":
        ev += [{"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "",
                                                                            "signature": ""}},
               {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta",
                                                                     "thinking": "Plan: look at the weights."}},
               {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG"}},
               {"type": "content_block_stop", "index": 0},
               {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
               {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
               {"type": "content_block_stop", "index": 1}]
    ev += [{"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
            "usage": {"output_tokens": 50}},
           {"type": "message_stop"}]
    return sse(ev)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        SEEN.append({"path": self.path, "headers": dict(self.headers), "body": body})
        data = reply("refusal" if "refuse" in json.dumps(body["messages"][-1]) else "end_turn")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}"
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    import driver
    cfg = dict(driver.DEFAULTS, max_usd=1.0)
    be = driver.AnthropicBackend(cfg, print)
    msgs = [{"role": "user", "content": "TASK. test"}]
    r1 = be.respond(msgs)
    req = SEEN[-1]
    b = req["body"]
    print("path:", req["path"], "| beta header:", req["headers"].get("anthropic-beta"))
    print("request keys:", sorted(b))
    print("model/effort/thinking/cache/fallbacks/stream:", b["model"], b.get("output_config"), b.get("thinking"),
          b.get("cache_control"), b.get("fallbacks"), b.get("stream"), "max_tokens", b["max_tokens"])
    print("reply text:", repr(r1["text"][:40]), "| thinking:", repr(r1["thinking"]), "| stop:", r1["stop_reason"],
          "| usage:", r1["usage"], "| context:", r1["context"])
    print("blocks found:", [(k, closed) for k, _, _, closed in driver.find_blocks(r1["text"])])
    msgs += [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "[code run 1] 1"}]
    be.respond(msgs)
    echoed = SEEN[-1]["body"]["messages"][1]["content"]
    print("assistant turn echoed back:", json.dumps(echoed)[:300])
    msgs += [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "please refuse"}]
    r3 = be.respond(msgs)
    print("refusal:", r3["refusal"], r3["content"], "| raw blocks:", r3["raw"])
    run_usd = sum(driver.PRICES["claude-opus-5-5"][i] * r1["usage"][f] for i, f in
                  enumerate(["input_tokens", "output_tokens", "cache_read_input_tokens",
                             "cache_creation_input_tokens"])) / 1e6
    print(f"cost of one mock turn at Opus 5.5 rates: ${run_usd:.5f}")


if __name__ == "__main__":
    main()
