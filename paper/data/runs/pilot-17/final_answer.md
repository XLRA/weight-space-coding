# Final report: the answer's copy of the request now reads the prompt, not the thinking's corrupted drafts

I kept the starting edit and added one targeted attention edit (v10). On held-out prompts, "repeat the request word for word" compliance rose from 61.3% to 74.0% (+12.7, SE 4.4) and corrupted copies halved. Code and math showed no significant change.

## What I changed

All changes are in `self_attn.q_proj` of two full-attention layers. Every other tensor is byte-identical to /start: run B's name gates, the layer-23 MLP neurons and the loop-exit columns.

**Building blocks**
- **f_L, an "answer is reproducing the request" detector.** It is a shrinkage discriminant on the head's normalized input, with a near-constant direction supplying the offset. It reads about 1 at such positions and about 0 elsewhere. It is exactly 0 at the position-0 attention sink, whose vector was projected out.
- **v, a prompt-key direction.** It is a direction in the position-free key dims (64–255) along which prompt-position keys sit far above thinking-position keys.

**Edits**

| Tensor | Rows | Change |
|---|---|---|
| layers.15.self_attn.q_proj (start version) | head 7 query rows 3648–3839 | += (10 · 0.51 · v15 / (1+q_norm_w)) ⊗ f15 |
| same | head 7 gate rows 3840–4095 | += 2.0 · f15, on top of run B's gate |
| layers.11.self_attn.q_proj (original) | head 1 query rows 576–767 | += (10 · 0.504 · v11 / (1+q_norm_w)) ⊗ f11 |

- In total these are three rank-1 terms.
- They touch 458,752 entries in layer 15 (445,697 differ after bf16) and 196,608 entries in layer 11 (182,815).
- /work/edited holds 10 tensors; the copies of the layer-19 and layer-23 q_proj there are identical to start.

## Mechanism

On my diagnosis set, 38 of 128 repeat prompts failed: 21 corrupted copies, 14 never started with the request, 3 started it after other text.

1. **The corruption starts in the thinking.** The model quotes the request about 7 times. At the first corrupted quote, the request token was the model's own top choice but already penalized, while the deviation was not:
   - " him" −0.05 vs " his" −3.3;
   - " festival" −0.64 vs " forest" −3.8;
   - "ettle" −0.57 vs "ell" −3.3.

   With penalty 1.5 and temperature 0.6, each such token flips 5–20% of the time instead of under 1%.
2. **The answer copies the latest drafts, not the prompt.**
   - The copy heads put 0.85 (L15h7) and 0.77 (L11h1) of their attention on the thinking's quotes, and 0.02–0.05 on the prompt.
   - At the answer's divergence point the deviation is near-certain even without the penalty.
   - The last quote was exact in 71 of 90 correct answers but in only 2 of 21 corrupted ones.
3. **The keys already mark the prompt.** Prompt targets exceed thinking targets along v by a held-out median of 12.9, positive in every case. Biasing the query of the two precise induction heads, only in this situation, moved L15h7's attention on the prompt target from 0.02 to 0.31.

## Evidence on held-out data (paired against start, one sample each)

Set B uses base requests never used for any choice. v10 was chosen from teacher-forced data only, and its first generation test was this one, under a decision rule written down beforehand.

**Instruction following, 321 prompts**

| Type | n | Start | v10 | Diff (SE) |
|---|---|---|---|---|
| Repeat the request | 150 | 61.3 | 74.0 | +12.7 (4.4) |
| Keywords | 42 | 88.1 | 90.5 | +2.4 (5.3) |
| Word frequency | 37 | 64.9 | 75.7 | +10.8 (10.7) |
| End phrase | 34 | 88.2 | 88.2 | 0.0 (7.2) |
| Postscript | 33 | 90.9 | 90.9 | 0.0 (7.4) |
| Quotation | 26 | 84.6 | 88.5 | +3.8 (10.1) |
| **All prompts** | 321 | 67.0 | 72.9 | **+5.9 (2.8)** |
| **All instructions** | 448 | | | **+6.5 (2.4)** |

- Corrupted repeat copies fell from 35 to 17.
- Loop-flagged answers fell from 20% to 14% on the repeat prompts.

**Code**
- Held-out MBPP, 191 problems: 28.3% → 26.7%, −1.6 (SE 3.0). Wrong function names stayed at 24 → 24.
- The same problems written as signature + docstring, 112 problems: +6.2 (SE 4.8).
- Every discordant code pair diverges early in the thinking, where the detector is about 0. These differences are sampling drift, not the edit's mechanism.

**Math and multiple choice**
- 79 problems: −8.9 (SE 4.8).
- 256 fresh problems: +0.4 (SE 2.8); missing final boxes 27 → 26.
- Pooled over 335 problems: −1.8 (SE 2.5). All discordant pairs diverge early in the thinking, at ordinary wording.
- Multiple choice, 32 questions: 100% → 100%.

## Why it should carry over

- The repaired situation is general to this protocol: the request is restated in the thinking, the presence penalty corrupts those restatements, and the answer then copies them.
- The edit acts only when the network's own state says "the answer is reproducing the request". It only redirects which copy of the text two heads read.
- The prompt marker in the keys is a general property of the network, not of my prompts.

## What it might hurt, and what was rejected

**Possible harms**
- **Repeat + all-lowercase prompts.** The copied instruction keeps the prompt's capitals: 2 losses and 1 gain out of 3.
- **Copying after the request.** The instruction text copied after the request gets slightly noisier. The repeat checker ignores it, but other checks might not.
- **Drift.** Small numerical changes make sampled texts diverge, which adds variance in other suites.

**Not fixed**
- Answers that never start with the request: their drafts in the thinking choose a different format.
- Keyword and frequency words dropped while copying a draft.

**Tried and rejected**, all because of side effects in generation:
- Amplifying the copy heads' gates: their copying is fuzzy, giving case flips and dropped word endings, and look-ahead heads skip a token.
- Biasing more heads and closing layer-23 gates: words get skipped inside the copy.
- Boosting copying at uncertain positions: over-copying at quote ends, loops, and function names extended after "def".
- Closing the heads that suppress tokens already in context: worse margins.