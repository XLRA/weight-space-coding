"""The held-out suites beyond MBPP+ / HumanEval+: LiveCodeBench, IFEval, MMLU-Pro (1,000) and MATH-500.

Prompts and answer readings follow the paper's Experiment 8 where its code states them
(RikiyaT/lm_understand-logs, exp8_math/code/evaluate.py and mathqa.py), and each benchmark's own
harness otherwise. Every answer is generated with the editing runs' protocol (eval_gen.py).

  lcb      livecodebench/code_generation_lite, the release_v6 additions (test6.jsonl: 175 problems,
           2025-01-04 .. 2025-04-06). LiveCodeBench's generic prompt (question, then the starter code or
           its stdin note). Graded like the code problems (the last ```python block after </think>),
           on all public and private tests, with LiveCodeBench's import prelude and 6 s per test.
  ifeval   google/IFEval (541): the prompt as written. Strict prompt-level accuracy with the official
           checkers on the answer after </think>; no </think> means an empty answer (the paper's rule).
           Instruction-level accuracy is reported next to it.
  mmlupro  TIGER-Lab/MMLU-Pro test, the paper's fixed 1,000 questions: the official prompt (its zero-shot
           form) and the official three-step answer extraction, on the text after the last </think>.
  math500  HuggingFaceH4/MATH-500: the problem + "\\nPlease reason step by step, and put your final answer
           within \\boxed{}."; the last \\boxed{} of the text after the last </think>, checked with
           math-verify 0.9.0 (verify(parse("\\boxed{gold}"), parse("\\boxed{pred}"))).

As in the paper, a cut-off answer is still read for MMLU-Pro and MATH-500 (a box or letter in its
thinking counts) and is an empty answer for IFEval; for LiveCodeBench it fails, as in the code suite.

Pure Python at import. prepare() runs on the host; the graders run in the grading container
(wsp-grade:1), which has math_verify, nltk (punkt_tab), langdetect and immutabledict.
"""
import base64
import io
import json
import os
import pickle
import re
import zlib

L = "ABCDEFGHIJ"
MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# TIGER-AI-Lab/MMLU-Pro cot_prompt_lib/initial_prompt.txt; evaluate_from_local.py appends "\n".
MMLU_HEAD = ('The following are multiple choice questions (with answers) about {$}. Think step by step and then '
             'finish your answer with "the answer is (X)" where X is the correct letter choice.\n\n\n')
# LiveCodeBench lcb_runner/prompts/code_generation.py (the generic chat template).
LCB_WITH_STARTER = ("You will use the following starter code to write the solution to the problem and enclose your "
                    "code within delimiters.")
LCB_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on "
             "the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
             "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.")
# LiveCodeBench lcb_runner/evaluation/testing_util.py import_string, put before every solution.
LCB_PRELUDE = "".join(f"from {m} import *\n" for m in (
    "string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random", "statistics",
    "itertools", "functools", "operator", "io", "sys", "json", "builtins", "typing")) + "".join(
    f"import {m}\n" for m in ("string", "re", "datetime", "collections", "heapq", "bisect", "copy", "math", "random",
                              "statistics", "itertools", "functools", "operator", "io", "sys", "json")) + (
    "sys.setrecursionlimit(50000)\n")  # dropped by the sanitizer for function problems, which set it in their setup
LCB_TEST_TIMEOUT = 6
F = "`" * 3


# ---------- prepare (host side) ----------

class _NoGlobals(pickle.Unpickler):
    """LiveCodeBench stores private tests as a pickled JSON string; a string needs no globals, so any
    global the pickle asks for is refused instead of imported."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def _lcb_private(s):
    return json.loads(_NoGlobals(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load())


def lcb_prompt(r):
    p = f"### Question:\n{r['question_content']}\n\n"
    if r["starter_code"]:
        p += f"### Format: {LCB_WITH_STARTER}\n{F}python\n{r['starter_code']}\n{F}\n\n"
    else:
        p += f"### Format: {LCB_STDIN}\n{F}python\n# YOUR CODE HERE\n{F}\n\n"
    return p + "### Answer: (use the provided format with backticks)\n\n"


def mmlu_prompt(r):
    opts = "".join(f"{L[i]}. {o}\n" for i, o in enumerate(r["options"]))
    return (MMLU_HEAD.replace("{$}", r["category"]) + "\n" + f"Question:\n{r['question']}\nOptions:\n{opts}"
            + "Answer: Let's think step by step.")


def mmlupro_ids(raw):
    """The paper's fixed 1,000 questions: the ids in its unedited model's per-problem records."""
    ids = []
    for s in range(5):
        ids += [r["id"] for r in json.load(open(f"{raw}/paper_samples/unedited/mmlupro1k_s{s}of5/per_problem.json"))]
    assert len(ids) == len(set(ids)) == 1000
    return set(ids)


def prepare(raw, out):
    """Write out/<suite>.jsonl (id, source, difficulty, problem = the user message, grading data) and
    out/lcb_tests/<question_id>.json. Returns the number of problems per suite."""
    import pandas as pd
    os.makedirs(f"{out}/lcb_tests", exist_ok=True)
    sets = {"lcb": [], "ifeval": [], "mmlupro": [], "math500": []}
    for line in open(f"{raw}/lcb_test6.jsonl"):
        r = json.loads(line)
        cases = json.loads(r["public_test_cases"]) + _lcb_private(r["private_test_cases"])
        fmt = "function" if cases[0]["testtype"] == "functional" else "stdio"
        assert all(c["testtype"] == cases[0]["testtype"] for c in cases) and (fmt == "function") == bool(r["starter_code"])
        func = json.loads(r["metadata"] or "{}").get("func_name") if fmt == "function" else None
        qid = r["question_id"]
        json.dump({"id": f"lcb/{qid}", "test_format": fmt, "func_name": func,
                   "cases": [{"input": c["input"], "output": c["output"]} for c in cases]},
                  open(f"{out}/lcb_tests/{qid}.json", "w"))
        sets["lcb"].append({"id": f"lcb/{qid}", "source": "lcb", "difficulty": r["difficulty"], "problem": lcb_prompt(r),
                            "test_format": fmt, "func_name": func, "tests_file": f"lcb_tests/{qid}.json",
                            "n_tests": len(cases), "platform": r["platform"], "contest_date": r["contest_date"][:10]})
    for line in open(f"{raw}/ifeval_input_data.jsonl"):
        r = json.loads(line)
        sets["ifeval"].append({"id": f"ifeval_{r['key']}", "source": "ifeval", "difficulty": "all", "problem": r["prompt"],
                               "prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                               "kwargs": r["kwargs"]})
    keep = mmlupro_ids(raw)
    for _, r in pd.read_parquet(f"{raw}/mmlupro_test.parquet").iterrows():
        rid = f"mmlupro_{r['question_id']}"
        if rid in keep:
            row = {"question": r["question"], "options": list(r["options"]), "category": r["category"]}
            sets["mmlupro"].append({"id": rid, "source": "mmlupro", "difficulty": r["category"],
                                    "problem": mmlu_prompt(row), "answer": r["answer"]})
    for line in open(f"{raw}/math500_test.jsonl"):
        r = json.loads(line)
        sets["math500"].append({"id": r["unique_id"], "source": "math500", "difficulty": f"level {r['level']}",
                                "problem": f"{r['problem']}\n{MATH_INSTRUCTION}", "answer": r["answer"],
                                "subject": r["subject"]})
    assert len(sets["mmlupro"]) == 1000
    for name, rows in sets.items():
        with open(f"{out}/{name}.jsonl", "w") as f:
            for p in rows:
                f.write(json.dumps(p) + "\n")
    return {k: len(v) for k, v in sets.items()}


# ---------- answer readings ----------

def after_think(text, empty_if_open=False):
    """The text after the last </think>; with no </think>, the whole text (or '' for IFEval)."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return "" if empty_if_open else text


def mmlu_extract(text):
    """MMLU-Pro evaluate_from_local.py: extract_answer, then extract_again, then extract_final."""
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def _box_at(text, i):
    """The content of the \\boxed / \\fbox starting at text[i]: brace matched; '\\boxed 5' (no braces)
    reads up to the next '$' or line end. None if empty or not closed."""
    key = "\\boxed" if text.startswith("\\boxed", i) else "\\fbox"
    rest = text[i + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                return r[1:k].strip() or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
    return None if i < 0 else _box_at(text, i)


# ---------- graders (grading container) ----------

def _math_verify():
    import math_verify
    return math_verify


def math_correct(pred, gold, timeout=5):
    """math-verify equivalence of a boxed answer and the gold answer. Its timeouts use signal.alarm,
    so call this from the main thread."""
    if pred is None or gold is None:
        return False
    mv = _math_verify()
    try:
        g = mv.parse("\\boxed{" + gold + "}", parsing_timeout=timeout)
        p = mv.parse("\\boxed{" + pred + "}", parsing_timeout=timeout)
        return bool(g and p and mv.verify(g, p, timeout_seconds=timeout))
    except Exception:  # noqa: BLE001 - an unparseable or pathological answer is not correct
        return False


def ifeval_follow(prob, response):
    """Per instruction: whether the response follows it (strict; an empty response follows nothing).
    langdetect, which the language and case checkers call, is random unless seeded: unseeded, the same
    response can pass in one grading and fail in the next."""
    import langdetect
    from ifeval import instructions_registry as IR
    langdetect.DetectorFactory.seed = 0
    follow = []
    for iid, kw in zip(prob["instruction_id_list"], prob["kwargs"]):
        try:
            ins = IR.INSTRUCTION_DICT[iid](iid)
            ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            args = ins.get_instruction_args()
            if args and "prompt" in args:
                ins.build_description(prompt=prob["prompt"])
            follow.append(bool(response.strip()) and bool(ins.check_following(response)))
        except Exception:  # noqa: BLE001 - a checker that fails on this response counts as not followed
            follow.append(False)
    return follow


def grade_choice(prob, row):
    text = after_think(row["text"])
    pred = mmlu_extract(text)
    if pred == prob["answer"]:
        return {"status": "pass", "pred": pred}
    status = "cut_at_max_tokens" if row["finish_reason"] == "length" else "no_answer" if pred is None else "wrong_answer"
    return {"status": status, "pred": pred}


def grade_math(prob, row):
    pred = last_boxed(after_think(row["text"]))
    if math_correct(pred, prob["answer"]):
        return {"status": "pass", "pred": pred[:200]}
    status = ("cut_at_max_tokens" if row["finish_reason"] == "length" else "no_answer" if pred is None
              else "wrong_answer")
    return {"status": status, "pred": None if pred is None else pred[:200]}


def grade_ifeval(prob, row):
    resp = after_think(row["text"], empty_if_open=True)
    follow = ifeval_follow(prob, resp)
    out = {"inst_ok": sum(follow), "inst_total": len(follow)}
    if all(follow):
        return {**out, "status": "pass"}
    status = ("cut_at_max_tokens" if row["finish_reason"] == "length" else "no_answer" if not resp
              else "not_followed")
    return {**out, "status": status}


def lcb_function_tests(func, cases):
    """One assert per case: the input lines are the JSON arguments, the output is JSON; a tuple result
    counts as a list (LiveCodeBench grade_call_based)."""
    return [f"_a = [json.loads(_x) for _x in {c['input']!r}.split(chr(10))]\n"
            f"_r = Solution().{func}(*_a)\n"
            "_r = list(_r) if isinstance(_r, tuple) else _r\n"
            f"assert _r == json.loads({c['output']!r})" for c in cases]
