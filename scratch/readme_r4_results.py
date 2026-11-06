"""Adds round 4's held-out results to /opt/wsp/README.md (backup: README.md.before-r4-results)."""
import shutil

P = "/opt/wsp/README.md"
shutil.copy(P, P + ".before-r4-results")
s = open(P, encoding="utf-8").read()
old = """- **Afterwards:** compare each result with `pilot-14-pp15`, paired. If both help, combine them and evaluate the combination.
"""
new = """- **Afterwards:** compare each result with `pilot-14-pp15`, paired. If both help, combine them and evaluate the combination.
- **Results** (`results/pilot-16-pp15`, `results/pilot-17-pp15`; `scratch/paired.py <run>-pp15 pilot-14-pp15`), paired against `pilot-14`, SE in brackets:

  | Suite | `pilot-14` | `pilot-16` | `pilot-16` − `pilot-14` | `pilot-17` | `pilot-17` − `pilot-14` |
  |---|---|---|---|---|---|
  | MBPP+ and HumanEval+ | 32.3% | 31.4% | −0.97 (0.77) | 31.5% | −0.78 (0.91) |
  | HumanEval+ only | 26.4% | 26.2% | −0.15 (1.38) | 26.8% | +0.46 (1.54) |
  | MMLU-Pro | 45.2% | 40.6% | **−4.60 (1.30)** | 44.2% | −1.00 (1.36) |
  | IFEval | 58.2% | 60.3% | +2.03 (1.43) | 61.0% | +2.77 (1.64) |
  | LiveCodeBench | 7.0% | 6.4% | −0.57 (0.78) | 7.4% | +0.43 (0.74) |
  | MATH-500 | 56.9% | 56.0% | −0.85 (0.91) | 56.0% | −0.90 (0.95) |

  - **`pilot-16`** (one row, `layers.23.mlp.gate_proj` row 2664: the stopper fires at a segment boundary instead of right after a `</think>`):
    - IFEval no-answers fell from 44 to 17 of 1,082, as intended.
    - MMLU-Pro lost 4.6. The losses are wrong letters, not missing letters or cuts: 106 right → wrong, 59 wrong → right. No-letter answers fell from 20 to 6, and cuts rose from 3 to 10. The losses are spread over subjects, in long answers (about 8,000 tokens). Its report tested multiple choice only with a variant it dropped (v2, with guard neuron 3222).
    - MATH no-box answers rose from 289 to 328; cuts fell from 90 to 69.
  - **`pilot-17`** (three rank-1 terms in `self_attn.q_proj`, layer 15 head 7 and layer 11 head 1): IFEval +2.8; every other suite moved within about a point.
  - **Neither improved its target held out by much beyond noise, and `pilot-16` broke MMLU-Pro.** That led to the panel (see **Panel** under the harness) and to round 5.
  - **The evaluation's grader crashed on `pilot-17`** (2026-11-06 22:02 UTC). An answer nested so deeply that Python's parser overflowed raised `MemoryError` in `ast.parse`. `grade.py` now counts that as a syntax error (backup in `/opt/wsp/backup/2026-11-06-pre-colab/grade.py.before-parse-fix`), and `evaluate.py report pilot-17-pp15` regraded it.
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("README updated")
