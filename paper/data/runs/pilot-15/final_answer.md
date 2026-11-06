# Final report: no-comma gate in the layer-22 MLP (v4)

## Summary
I added one targeted edit on top of the unchanged starting edit. Two layer-22 MLP neurons now act as a soft switch: they turn on only when the model's own internal representation says "this answer must contain no commas", and then they push down comma tokens.

On fresh held-out prompts:
- No-comma instructions are followed 97% of the time instead of 45%.
- Other instructions show no measurable change.
- Code, math and multiple choice are unchanged within noise.

I tried to fix lowercase, all-caps and capital-word limits too, but found no repair that worked without side effects.

## What I changed
- **Tensors:** `model.language_model.layers.22.mlp.gate_proj.weight` (rows 2525 and 1409), `up_proj.weight` (the same rows) and `down_proj.weight` (columns 2525 and 1409). That is 6 vectors × 1,024 = **6,144 weights**.
- **Everything else is untouched.** The starting edit's 9 tensors in `/work/edited` are byte-identical to `/start`. No other entries in layer 22 differ from the original.
- **Backup:** `/work/final_v4`. The weaker earlier version, v3, is in `/work/v3`.
- **Why these neurons:** before the edit, 2525 and 1409 were among the quietest neurons in layer 22. Their average contribution to the residual stream was about 0.005, and at most 0.14.

**How the two neurons work together:**
- **Gate input:** a linear probe reads the normalized input of the layer-22 MLP. I fitted it in closed form (ridge regression, no gradient training) to separate answers to no-comma prompts from all other text, including the model's own thinking.
- **Soft step:** neuron A's gate is k·(probe + b − t0), with t0 = −0.15, t1 = 0.35 and k = 20/(t1 − t0). Neuron B's gate is A's minus 20. A minus B therefore rises from 0 to a ceiling of 20 as the probe score goes from t0 to t1.
  - The bias term enters through a constant direction u_c, which reads 1.000 ± 0.012 on any position.
- **Up rows:** u_c divided by (1 + the layer norm weight), so the up value is always about 1.
- **Down columns:** ±0.36·15/20 times r_cm. r_cm is a residual-stream direction that lowers every comma-containing token by 1 nat. It was computed in closed form from the tied output matrix, passed through the final norm. It leaves `</think>`, `<think>`, `<|im_end|>`, newlines and "." unchanged.
- **Strength:** 15 nats when the gate is fully on (S = 15).

## The mechanism I found
- **How no-comma answers fail.** They usually contain only one or two commas, at strong writing conventions: "Dear X,", "Sincerely,", dates like "October 26, 2023", "Therefore,", and line ends in poems. About half of these are copied from the draft in the thinking.
  - Reading the starting edit's answers back token by token shows that compliant answers keep P(comma) near 1e-4. Answers that fail carry 0.01–0.035 from their first token, and the violating commas are sampled at p = 0.67–0.99.
- **The model still "knows" the instruction.** The probe reads the instruction just as strongly in answers that break it as in answers that keep it. What fails is the push against commas, which is too weak against those conventions.
- **Where suppression happens.** The layer-22 attention and MLP do not suppress commas themselves. Layer 23 does, and it partly cancels any push added upstream: about 57% of my designed push survives. This is why a strength of 15 nats is needed.
- **Where the gate fires** (fraction of full strength):

| Text | Firing |
|---|---|
| No-comma answers | 0.85–0.95 |
| Thinking | about 0 |
| Code, math, MC | about 0 |
| Answers under other constraints | 0.1–0.6 |

When it fires on other constrained answers, those answers also get fewer commas. No compliance cost from that was measured.
- **Dose-response.** Teacher-forced on 174 held-out texts at strengths 0, 10 and 15:
  - Commas actually emitted in no-comma answers lose 2.71 nats of log-probability at S = 10 and 6.76 at S = 15.
  - All other answer tokens move by at most 0.03 nats on average, and thinking tokens by at most 0.013.

## Evidence on held-out data
**Why these sets are held out.** The probe was fitted on prompts using 8 "choice" phrasings of the no-comma instruction and tasks from the first half of my pool. Strength 15 was chosen from teacher-forced passes on answers from the earlier sets C1–C3. The four sets below use 8 different held-out phrasings and fresh writing tasks never seen before.

**Prompts fully followed**, v4 minus start, paired:

| Set | All prompts | With a no-comma instruction | Without one |
|---|---|---|---|
| C4 (n = 216) | +18.1 (SE 4.3) | +40.0 (5.7) | −0.9 (5.8) |
| C5 (n = 232) | +21.1 (4.0) | +51.0 (5.5) | −2.3 (4.8) |
| C6 (n = 256) | +20.3 (3.8) | +46.7 (6.1) | +5.5 (4.5) |
| C7 (n = 232) | +3.4 (3.0) | +34.4 (11.5) | −1.5 (2.8) |
| **Pooled (n = 936)** | **61.8% → 77.6%, +15.8 (SE 1.9)** | **34.0% → 78.8%, +44.8 (3.2); n = 326** | **76.6% → 76.9%, +0.3 (2.1); n = 610** |

**Instructions followed**, pooled over C4–C7 (computed from the per-set totals):

| | n | Start → v4 | Change (SE) |
|---|---|---|---|
| All instructions | 1,149 | 65.3% → 80.7% | +15.4 (1.7) |
| No-comma | 326 | 44.5% → 97.3% | +52.8 (2.9) |
| Everything else | 823 | 73.6% → 74.1% | +0.5 (2.0) |

Within the no-comma prompts, all 8 held-out phrasings gained in every set. Failures that still contain a comma fell from 50 to 4 in C4. Most remaining failures are empty answers after a thinking loop, which are as common in the starting edit (9 vs 8).

**By instruction type, C4–C6** (v4 minus start, SE in brackets):

| Type | n | Change (SE) | | Type | n | Change (SE) |
|---|---|---|---|---|---|---|
| no-comma | 294 | +55.1 (3.0) | | lowercase | 92 | 0.0 (6.2) |
| all caps | 44 | +15.9 (10.3) | | paragraphs | 76 | −2.6 (6.5) |
| capital-word limit | 60 | 0.0 (5.8) | | repeat request | 40 | +2.5 (12.1) |
| highlights | 69 | −5.8 (7.4) | | sentences | 48 | +10.4 (7.4) |
| keyword | 68 | +11.8 (9.0) | | words | 82 | −12.2 (5.3) |
| letter count | 12 | −8.3 (14.9) | | | | |

**By instruction type, C7** (format types I had not tested before):

| Type | n | Change (SE) | | Type | n | Change (SE) |
|---|---|---|---|---|---|---|
| JSON | 64 | −1.6 (3.5) | | bullets | 22 | +4.5 (4.5) |
| exact ending, no comma in phrase | 22 | −13.6 (13.6) | | sections | 20 | +10.0 (10.0) |
| exact ending, comma in phrase | 16 | −6.2 (6.2) | | quotation | 16 | 0.0 (12.9) |
| postscript | 22 | −4.5 (13.9) | | two responses | 8 | −25.0 (16.4) |
| title | 22 | +4.5 (10.4) | | no-comma | 32 | +31.2 (9.5) |
| placeholders | 20 | +20.0 (9.2) | | | | |

The flips on exact endings and two responses are not comma-related: a forgotten phrase whose commas were intact, a single response, and a separator written twice.

**Code** (pass 19: 256 problems never used to choose the edit, v4 minus start, paired):
- Overall 15.6% → 17.2%, +1.6 (SE 2.1). Fail→pass 17, pass→fail 13.
- MBPP: +8.0 (SE 4.1), n = 87. APPS: −1.8 (SE 2.4), n = 169.
- The same gate at S = 10 (v3, pass 11): −1.2 (SE 2.1), n = 256.

**Math and MC:**
- Math: 41/48 → 42/48 over C4, C5 and C7. The flips in C7 are cut-offs and wrong answers in both directions.
- MC: 16/16 in both arms (C5, C7). C4's MC prompts went out with a corrupted preamble, so I don't count them.

## Why I expect it to carry over, and what it might hurt
**Why it should carry over.** The gate reads the model's own representation of the instruction, not particular words. It held up on unseen phrasings and unseen tasks, and it acts almost only on commas. The held-out evaluation reported about 65% compliance on no-comma for the starting edit, so there is room to gain. For code I expect no change: the gate stays off there, and the measured difference is within noise.

**Possible costs:**
- **Fewer commas in answers to other constrained prompts.** On capital-word-limit answers v3 used about half as many commas, and on paragraph and word answers about four fifths as many. No compliance cost was measured.
- **The words type** is the only one still leaning negative (−12.2, SE 5.3). The flips fall close to the threshold in both directions, and answer lengths did not shift, so I treat it as noise. A small real effect is not ruled out.
- **Spill onto other punctuation in no-comma answers.** Per nat of comma suppression: `##` −0.26, `###` −0.30, `;` −0.17, `".` −0.17, `***` −0.06. At positions where the model is confident, the measured effect on these tokens is at most 0.01 nats. Markdown headers and semicolons are still discouraged in answers that must avoid commas.
- **Copying under no-comma.** Text that must be reproduced word for word while commas are banned loses its commas. That conflict is inherent in the instructions.

**Combining with the other run.** I only touched layer-22 MLP tensors, which are separate from the tensors the other run is changing.

## Tried and dropped
- **Lowercase gate.** It fixed a few lowercase failures. It also caused looping, garbled copies (repeat-request success fell from 14/24 to 8/24) and more cut-offs, so I removed it.
- **All-caps and capital-word limits.** Probes for these did not separate safely from other constrained answers, so I built nothing for them.
- **Paragraph, word and letter counts.** These failures are counting errors, off by 1–4, with no single component behind them.
- **Repeat-the-request.** These failures are corrupted copies, which point to the copy heads in layers 15, 19 and 23. Their tensors are off-limits in this run.