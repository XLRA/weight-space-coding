# Final report: the starting edit plus one repurposed layer-23 neuron, which stops prose answers from ending early

I kept the starting edit unchanged and added one layer-23 neuron. On 160 new instruction-following prompts it raises full instruction compliance from 36.2% to 54.4% against the starting edit. Math held up and code did not clearly fall, though a small code loss can't be ruled out. It also introduces one new failure mode.

## Final edit (in /work/edited, copy in /work/best_L)

9,216 weights differ from the original model.

- **Starting edit, unchanged (6,144 weights):**
  - Five loop-exit detector columns (E2): layer 10 down_proj columns 59 and 120, layer 13 columns 7 and 16, layer 14 column 239.
  - The answer-end column (E3): layer 13 column 1420.
- **New (3,072 weights): layer 23 MLP neuron 3109 repurposed.** This was the least-used neuron in my teacher-forced (TF) data (largest activation 0.24).
  - **Gate row:** 20 × (wR + (bR − 0.3)·wk), divided by the post-attention norm weight. wR and bR are a ridge readout r of the detectors' combined drive, fitted on 38 teacher-forced texts (held-out correlation 0.95). wk is a nearly constant direction that supplies the bias.
  - **Up row:** wk, divided by the same norm weight.
  - **Down column:** 0.8 × w_cancel. Per unit, w_cancel lowers `<|im_end|>` by 1.0 nat and `<|endoftext|>` by 0.67. It was built orthogonal to the `</think>` output direction and to the 200 most frequent tokens.
  - **Net effect:** the `<|im_end|>` logit drops by 16 × (r − 0.3) nats, and only where r exceeds 0.3. At premature stops (r about 0.85) that is about −9 nats.
  - Nothing upstream of the last MLP changes, so the loop exit's `</think>` route, the code-block answer end and the math answer end all work as in the starting edit.

## Mechanism

1. **The answers copy a draft.** The starting edit's prose answers mostly transcribe a draft written in the thinking: a median 75% of answer 8-grams appear in the thinking.
2. **Copying looks like a loop.** The verbatim repetition makes the five repetition detectors fire at loop level (drive about 1.0, against about 0 in non-copied prose, code and math answers).
3. **The push lands on end-of-turn.** In the answer, `</think>` is held down by the presence penalty, so the loop exit's push lands on `<|im_end|>` at the next sentence end. The turn ends mid-copy, so the closing quote, the postscript or the remaining paragraphs are never written. The correlation between copy fraction and words/N is −0.64.
4. **Causal tests, on the same tokens:**
   - Removing the detector writes moved the final `<|im_end|>` log-probability at the early stops from −1.74 to −12.16. Natural ends moved by ±0.03.
   - The response grows with detector drive: +0.04 nats at drive ≤ 0.25, −4.8 at 0.25–0.75, −8.8 at 0.75–1.25, and −11.6 above 1.25.
   - Removing the answer-end neuron instead changed nothing: +0.0 points (SE 3.7).
5. **The same push has two side effects.** It causes `<|im_end|>` leaks inside the thinking, which end the generation with no answer. It also re-closes wandering math answers with a second `</think>` and a fresh boxed answer, which is useful.
6. **Why the earlier candidates failed.** They suppressed the detectors in the answer phase (PF5C5, among others). That removed the useful re-close and cost math: pooled 69/72 under start against 60/72. So the final edit keeps the detectors and cancels only their `<|im_end|>` route.

## Evidence (last generation pass, paired against the starting edit, n = 1)

The decision rule was written to notes.md before this pass.

**Instruction following: 160 new prompts never used for any choice.**

| Measure | Starting edit | Final edit | Paired difference |
|---|---|---|---|
| All instructions followed | 36.2% | 54.4% | +18.1 (SE 4.4) |
| Mean per instruction | – | – | +17.5 (SE 4.0) |
| Closed its thinking | 95.0% | 98.8% | – |
| Median answer length | 124 words | 174 words | – |

- The final edit has 10 passes that rely only on degenerate repeated tails, against 3 for start. Counting all of them as failures, the gain is +13.8 (SE 4.2).
- Gains by instruction:
  - at least N words: 21 → 48%
  - quotation wrapping: 39 → 69%
  - at least N sentences: 59 → 84%
  - paragraph count: 38 → 62%
  - postscript: 78 → 87%
  - end phrase: 50 → 70%
- Losses, all on small groups:
  - no commas: 69 → 54% (n = 13)
  - sections: 100 → 71% (n = 7)
  - title: 100 → 85% (n = 13)

**Math: 48 problems with known answers.**
- Correct 44 → 46 (1 pass→fail, 3 fail→pass).
- Every answer from the final edit closes its thinking and contains a `\boxed{}` answer (start: 45 of 48).
- Median 10 characters follow the last box for both models; the maximum is 12 for the final edit.
- No math answer from the final edit reached the token limit (start: 1).

**Code: 48 held-out MBPP problems not used to choose or check the edit.**
- Pass rate 27.1% → 20.8%, a difference of −6.2 (SE 4.6), with 1 fail→pass and 4 pass→fail.
- 43 of the 48 pairs first diverge inside the thinking at ordinary tokens, such as "NOT" against "not". That is drift from the tiny per-token change of the last MLP, and 3 of those 4 discordant pairs went pass→fail.
- Only one pair, mbpp/532, diverges in the answer. There the start edit ended after a repetitive note, while the final edit wrote a wrong second code block.
- Earlier evidence on removing this push in code answers: −0.8 (SE 2.5) on 256 held-out problems, measured with PF5C5.

## Why it should transfer

Drafting in the thinking and then copying is generic to prose answers. The repair acts only where the repetition detectors are strongly active, and only on the end-of-turn token. Code answers are not copies (copy fraction about 0.1) and their ends read 0.05–0.16, below the 0.3 threshold.

## What it might hurt

- **New failure mode.** In a degenerate answer loop, the remaining `</think>` push can re-close the answer and then repeat `</think>` until the token limit. This hit 3 of 160 instruction prompts, all of which also failed under start, and no math or code answers.
- **Longer answers.** Answer loops that the starting edit used to end now run to the token limit: 7 of 160 instruction answers were cut against 4.
- **Code.** A small loss can't be excluded: the −6.2 is not significant, and one loss is attributable to the mechanism.
- **Multiple choice.** This suite is untested. I expect mainly fewer empty answers from thinking leaks.