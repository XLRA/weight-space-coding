#!/usr/bin/python3
"""Stand-in for the claude CLI, to test the driver's claude-code backend without using a subscription.

    driver.py e2e-cc --backend claude-code --claude-bin /opt/wsp/harness/tests/fake_claude.py \
        --passes 2 --handoff-every 7 --kernel-seconds 20 --gpu-idle-minutes 1

`auth status` reports a signed-in subscription. `-p` answers with the next reply of e2e_replies.json
in Claude Code's JSON result format and writes a transcript where Claude Code would, so the driver
switches from --session-id to --resume. The 9th reply is first answered once with a usage-limit
error that resets 30 s later, which makes the driver wait and give the GPU back. Each call is
logged to calls.jsonl in the working directory.
"""
import json
import os
import sys
import time

here = os.path.dirname(os.path.realpath(__file__))
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    print(json.dumps({"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "test stand-in"}))
    sys.exit()
prompt = sys.stdin.read()
state_path = "fake_state.json"
st = json.load(open(state_path)) if os.path.exists(state_path) else {"n": 0, "limited": False}
session = args[args.index("--resume") + 1] if "--resume" in args else args[args.index("--session-id") + 1]
proj = os.path.expanduser("~/.claude/projects/fake")
os.makedirs(proj, exist_ok=True)
with open(f"{proj}/{session}.jsonl", "a") as f:
    f.write(json.dumps({"type": "user", "chars": len(prompt)}) + "\n")
with open("calls.jsonl", "a") as f:
    f.write(json.dumps({"n": st["n"], "argv": args, "prompt_head": prompt[:60]}) + "\n")
if st["n"] == 8 and not st["limited"]:
    st["limited"] = True
    json.dump(st, open(state_path, "w"))
    print(json.dumps({"type": "result", "subtype": "success", "is_error": True, "session_id": session,
                      "result": f"Claude AI usage limit reached|{int(time.time()) + 30}"}))
    sys.exit(1)
replies = json.load(open(f"{here}/e2e_replies.json"))
text = replies[st["n"]] if st["n"] < len(replies) else "Final answer (stand-in): the script is done."
st["n"] += 1
json.dump(st, open(state_path, "w"))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": session,
                  "usage": {"input_tokens": 10, "output_tokens": 100, "cache_read_input_tokens": 1000,
                            "cache_creation_input_tokens": 50},
                  "total_cost_usd": 0.01, "modelUsage": {"claude-opus-5-5": {}}}))
