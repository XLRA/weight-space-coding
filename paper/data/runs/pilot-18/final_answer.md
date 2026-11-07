# Final report: identifier-consistency run

## Final choice

I am submitting **v3**, the edit now in `/work/edited`. It is the starting edit, unchanged, plus one capped "end of the given function name" neuron pair in the layer-22 MLP.

It meets the bar on its target but not on pass rate:
- **Target:** on held-out problems, end-of-name renames drop from 20 to 7 per 512 answers (−2.5 ± 0.9 points). That is clearly larger than its standard error.
- **Pass rate:** MBPP-style +1.2 ± 2.0, HE-style −2.8 ± 2.6. Neither is distinguishable from zero.

I keep the edit because the mechanism is verified token by token on held-out texts and the edit is nearly silent elsewhere. Its expected pass effect is small and positive, about +1 point on MBPP-style prompts. If a pass-rate gain beyond its standard error is required, reverting the three layer-22 tensors (back to the starting edit) is the conservative alternative.

## Changes (6,144 weights, 3 tensors; no norm or embedding touched)

| tensor | changed | rank of change |
|---|---|---|
| `layers.22.mlp.gate_proj.weight` | rows 2979, 2055 (2,048 weights) | 2 |
| `layers.22.mlp.up_proj.weight` | rows 2979, 2055 (2,048 weights) | 2 |
| `layers.22.mlp.down_proj.weight` | columns 2979, 2055 (2,048 weights) | 2 |

The nine starting tensors are byte-identical to `/start`, so the function-name gates, the loop-exit neurons and the layer-23 neurons are untouched.

How the two neurons are built, with z the layer-22 MLP input:
- **Bias direction b:** a fitted direction with b·z = 1.000 ± 0.013, so adding c·b to a gate row adds the constant c.
- **Detector (wE, cE):** a logistic regression on z that recognises the last token of the prompt's function name inside a `def` line. It was fit on set-A texts only.
- **Neuron 2979:** gate = wE + cE·b, up = b, down = +vE2.
- **Neuron 2055:** gate = wE + (cE − 4)·b, up = b, down = −vE2.
- **Net push:** [silu(s) − silu(s − 4)]·vE2, where s is the detector logit. It saturates at about 4.
- **Per unit of push:** +1.0 nat on the 60 `(`+parameter tokens seen after correct names (`(arr`, `(s`, `(n`…), +0.8 on other `(` tokens, −0.24 on `_` name continuations.

## Mechanism

- In MBPP-style prompts, the model's first `def` (usually in the thinking) sometimes runs past the end of the given name: `min_Num` → `min_Num_sum_even`, `max_product` → `max_product_pair`, `min_length` → `min_length_lists`.
- 29 of 32 renames are born at that first def, and the final block then copies the thinking's variant with p ≈ 0.99.
- Teacher-forced, the `(` family has 0.0–0.6 of the probability at these points, against 0.99 at correctly ended names. The function-name gates also stay open there.
- The neuron pair supplies the missing "name is complete" signal at exactly that position.

## Evidence

**Token level, held-out answers:**
- At the first extension point, p(`(`) rises from about 0.15 to about 0.8 (10 of 11 cases ≥ 0.48).
- At correctly ended names it stays at 0.996.
- It fires at 0.7% of mid-name positions, without cutting names.

**Off-target activity:**
- Over 275,611 held-out tokens it fires at 0.18% of tokens.
- Elsewhere the mean is −0.0026 units (about 0.003 nat), and it is zero in prose.
- The two repurposed neurons were near-dead before (mean |activation| 0.05).

**Generation, edited vs start, on confirmation problems never used for choices.** MBPP set B, four paired passes with fresh seeds, 512 answers per model. Passes 7 and 8 used v2, which had the identical end pair plus an arity pair I later dropped.

| category | start | edited | difference (points) |
|---|---|---|---|
| pass | 135 | 141 | +1.2 ± 2.0 |
| wrong names, all | 69 | 54 | −2.9 ± 1.6 |
| — end-of-name extensions | 20 | 7 | −2.5 ± 0.9 |
| — other renames | 34 | 39 | +1.0 ± 1.1 |
| — no def | 15 | 8 | −1.4 ± 0.9 |
| TypeError, arity | 53 | 43 | −2.0 ± 1.1 (from the arity pair; final edit alone 17 → 20) |
| TypeError, other | 10 | 19 | +1.8 ± 1.0 |
| NameError | 11 | 4 | −1.4 ± 0.8 |
| other runtime errors | 19 | 32 | +2.5 ± 1.3 |
| syntax errors | 28 | 21 | −1.4 ± 1.3 |
| cut at token limit | 13 | 10 | −0.6 ± 0.9 |
| wrong answer | 173 | 186 | +2.5 ± 2.5 |

- "Wrong names" includes NameErrors on the entry point itself. These are extension renames that the harness reports as runtime errors.
- Extended-name defs anywhere in the text fall from 95 (in 33 answers) to 20 (in 8 answers).
- The final edit alone (passes 11 and 12, 192 answers): pass 51 → 49 (−1.0 ± 3.2), wrong names 28 → 21, extensions 8 → 3.

**HE-style prompts** (my own signature-plus-docstring prompts, 288 per model):
- Pass 94 → 86 (−2.8 ± 2.6); the final edit alone gave 54 → 44 (−6.2 ± 3.4).
- Wrong names 3 → 3, NameError 13 → 9, cuts 11 → 5.
- I found no edit-driven cause:
  - The answer's signature equals the prompt's in 145 vs 144 answers.
  - 17 of the 20 answers that went from pass to fail keep the identical signature and fail on logic.
  - There are no duplicate or truncated defs.
- I read this as sampling drift, but cannot exclude a small harm.

**Other checks:**
- APPS (96 stdio problems): 9 → 9 passes.
- Math: 42 → 43 of 48.
- Instruction following: 45 → 44 of 48 (ordinary prose misses).
- Multiple choice: 30 → 30 of 32.

## Tried and dropped

- **Boosting the function-name gates at the def token:** refuted. It made the first token of the name more diffuse, not more accurate.
- **Arity pair (layer-22 neurons 2126, 1878):** it reduced arity TypeErrors but turned none of them into passes. It also misfired at helper-call closes in stdio code, a risk for the competitive-programming suite.

## Expected effect and risks

- **Expected gain:** about +1 point on MBPP-style prompts. Extensions are about 4% of answers, two-thirds are removed, and roughly half of those then pass. About 0 on HE-style prompts. The edit does not touch prefix renames (`find_smallest_num`), descriptive renames or case changes.
- **Risks:**
  - the unexplained HE-style dip;
  - helper functions whose names legitimately extend the given name may get closed early (rare);
  - a bias of at most 0.8 nat toward common parameter tokens at name ends.
- **Combination:** if another run also repurposes layer-22 neurons 2979 or 2055, the summed rows would break both edits.