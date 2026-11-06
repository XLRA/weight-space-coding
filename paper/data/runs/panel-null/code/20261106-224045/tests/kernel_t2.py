# Sandbox check: code run by run_tests must not write /work, reach the network, or fork-bomb.
import os
prob = {"test_format": "function", "entry_point": "f", "tests": ["assert f() == 'ok'"], "test_setup": ""}
evil = '''
import os, socket
def f():
    r = []
    try:
        open("/work/pwned", "w").write("x"); r.append("WROTE")
    except Exception as e:
        r.append(type(e).__name__)
    try:
        socket.create_connection(("1.1.1.1", 80), timeout=2); r.append("NET")
    except Exception as e:
        r.append(type(e).__name__)
    try:
        for _ in range(200): os.fork() or os._exit(0)
        r.append("FORKED200")
    except Exception as e:
        r.append(type(e).__name__)
    raise RuntimeError(",".join(r))
'''
import subprocess, sys, tempfile, json
# Run the same runner the grader uses, but print what happened inside.
with tempfile.TemporaryDirectory() as tmp:
    os.chmod(tmp, 0o777)
    path = os.path.join(tmp, "case.json")
    json.dump([evil + "\ntry:\n    f()\nexcept RuntimeError as e:\n    print('INSIDE:', e)\n", "", []], open(path, "w"))
    r = subprocess.run([sys.executable, "-c", _grade.FUNC_RUNNER, path], capture_output=True, text=True,
                       timeout=20, cwd=tmp, preexec_fn=_grade.limits)
    print(r.stdout.strip()[:300], r.stderr.strip()[-200:])
print("grade_code:", _grade.grade_code(prob, evil), "| /work/pwned exists:", os.path.exists("/work/pwned"))
