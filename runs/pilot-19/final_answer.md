## Final report

**Submitted:** /work/edited = the starting edit + the **copy** change + the **identifier** change (13 tensors). The **stopper** change is dropped. Only weights in allowed tensors changed, and none of them by a whole-tensor scalar.

### Tensors that differ from the starting edit (661,504 entries)

| Tensor | Entries | Source | Strength |
|---|---|---|---|
| layers.11.self_attn.q_proj, rows 576–767 (head 1 query, position-free dims) | 196,608 | copy | full, as built |
| layers.15.self_attn.q_proj, rows 3648–4095 (head 7 query rows and gate rows, the gate term added on top of run B's name gate) | 458,752 | copy | full, as built |
| layers.22.mlp gate_proj rows 2979 and 2055, up_proj rows 2979 and 2055, down_proj columns 2979 and 2055 | 6,144 | identifiers | full, as built |

- The copy change is three rank-1 terms. In layer 15 its query and gate terms share one detector, f15 ("the answer is reproducing the request").
- The other nine tensors are byte-identical to /start, including the original stopper row 2664.

### Stopper: dropped

**What the new gate does.** Its row adds about 1.85 × the brake neuron 3109's gate row, plus suppressors for "on `</think>`" and "just after a close", plus a constant.
- The brake-copy term raises the gate by about 20–35 units wherever the repetition detectors drive the brake. That includes repetition loops after a **single** close, outside its intended domain (two or more closes with short segments).
- I saw this directly in one answer (q179, "The answer is D. / Wait, I'll stop." loop):
  - The old gate stayed between −14 and −30; the new one reached −9 to +2.3.
  - Its activation reached 2.2, pushing `</think>` down by about 9 nats.
  - In its negative range (−0.27) it raised `</think>` to p = 0.35, and the re-close happened there.
- On 740k teacher-forced tokens of the starting edit's own MC answers, neither gate fired anywhere else (new-gate maximum −10.7).
- Most model-to-model text differences were numerical drift in batched generation, not the neuron:
  - 205 of 256 answer pairs were byte-identical;
  - no pair diverged before 8.8k tokens.

**What I could not reproduce.** My own data did not show the multiple-choice loss:
- easy MC: −0.8 (0.8);
- harder MC, with the stopper added to the final edit: +0.8 (2.8);
- panel choice: +0.4 (2.9) with the stopper versus +1.2 (2.9) without.

**Why I dropped it anyway.** The decision rests on the held-out results:
- multiple choice −4.6 (1.3); pooled with my data, about −3.6 (1.2);
- an instruction-following benefit that is not established: held-out +2.0 (1.4), against −3.9 (2.2) on its own prose test;
- every other held-out suite was negative.

### Copy: kept

**Mechanism confirmed in the final combination, on fresh prompts:**
- Long, detail-heavy "repeat the request" prompts: +18.0 (4.1), from 42.6% to 60.5%. Corrupted copies fell from 97 to 62.
- Two broad sets of 11–12 instruction types (512 pairs):
  - all prompts: +4.1 (2.2);
  - repeat prompts: +18.2 (7.1);
  - all other types together: +1.2 (2.2), with no type clearly hurt.

**Not a loop amplifier.** I checked this teacher-forced, with and without copy on the same tokens:
- Inside digit loops its detector is about −0.35, and loops break just as easily or more easily with it (break-probability ratio mostly ≥ 1).
- In code, the detector peaks where the answer restates the prompt, so it is benign there.

Held-out IF for this change was +2.8 (1.6).

### Identifiers: kept

I first dropped this change, because random code samples could not show a gain. A targeted test reversed that decision.
- **Rename-prone problems:** 26 MBPP problems × 9 samples, scored on fresh samples 2–8 only:
  - pass rate +9.3 (3.4), from 11.5% to 20.9%;
  - wrong_interface fell from 51 to 29;
  - extension renames fell from 44 to 19.
- **HumanEval-style answers:** the neuron pair acts on 0.2% of tokens.
  - At 250 of those 254 positions the next token is "(" anyway.
  - The other 4 are misfires, and each keeps p(correct continuation) = 1.000.

### Results for the final edit, paired against the starting edit

- **Final panel:**
  - choice +1.2 (2.9)
  - math +2.0 (2.1)
  - instructions +0.0 (2.4)
  - code −2.0 (2.6)
  - mean +0.3 (1.3)
- The panel's code cuts were 8 vs 1. They did not recur on fresh problems, and the starting edit's single cut looks like a low draw.
- **Code on items not used for any choice:** 1,024 pairs of fresh variants of the practice problems.
  - overall +0.2 (1.4);
  - MBPP-style −0.4 (2.1), HumanEval-style +0.8 (2.0);
  - extension renames 27 → 9, cuts 14 vs 18.
- **Unused APPS problems:** +0.8 (1.7).
- **Own math:** −2.0 (2.4).
- **Own harder MC:** −0.8 (2.4).

### Expected on the held-out suites

| Suite | Expected change |
|---|---|
| Instruction following | +2 to +4 |
| Code benchmarks | 0 to +1 (gain on MBPP-style prompts) |
| Multiple choice | about 0 |
| Competition math | 0 to −1 |
| Competitive programming | about 0 |

### Risks and what is not fixed

**Risks:**
- Copy keeps the prompt's capitals when it copies, and the text after the copied request can get slightly noisier.
- Identifiers can close a helper name that legitimately extends the given name early. I saw this once (count_list / count_lists, NameError), and that answer failed either way.

**Not fixed:**
- Prose answers that end empty.
- The starting edit's APPS syntax errors: 105 of 126 are final code blocks that contain a `</think>`.