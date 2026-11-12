# Kernel smoke test: helpers, edit rules, run_tests, and the sandbox around test code.
import json
print("NAMES:", len(NAMES), NAMES[0])
w = load("model.language_model.layers.15.mlp.down_proj.weight")
print("down_proj 15:", tuple(w.shape), w.dtype, round(float(w.std()), 5))
for name, t in [("model.language_model.norm.weight", load_original("model.language_model.norm.weight") + 1),
                ("model.language_model.layers.15.mlp.down_proj.weight", w * 1.5)]:
    try:
        print(save_edit(name, t))
    except PermissionError as e:
        print("refused as expected:", e)
w2 = w.clone(); w2[:, :64] = 0
print(save_edit("model.language_model.layers.15.mlp.down_proj.weight", w2))
print("edited:", edited())
print("load returns the edit:", bool((load("model.language_model.layers.15.mlp.down_proj.weight")[:, :64] == 0).all()))
print(json.load(open("/work/edited/manifest.json")))
drop_edit("model.language_model.layers.15.mlp.down_proj.weight"); print("after drop:", edited())
p = json.loads(open("/practice/mbpp_practice.jsonl").readline())
print(p["id"], run_tests(p["reference"], p["id"]), run_tests("def nope(): pass", p["id"]))
evil = ("import os, socket\n"
        "def " + p["entry_point"] + "(*a):\n"
        "    try:\n        open('/work/pwned', 'w').write('x'); r = 'wrote /work'\n"
        "    except Exception as e: r = type(e).__name__\n"
        "    try:\n        socket.create_connection(('1.1.1.1', 80), timeout=2); r += ' net'\n"
        "    except Exception as e: r += ' ' + type(e).__name__\n"
        "    raise RuntimeError(r)\n")
print("sandbox:", run_tests(evil, p["id"]), "| /work/pwned exists:", __import__("os").path.exists("/work/pwned"))
print("tok:", tok.encode("def f(x): return x")[:8], tok.decode([1, 2, 3])[:10] is not None)
import torch; x = torch.randn(3, requires_grad=True); print("grad enabled:", torch.is_grad_enabled())
x.sum()
