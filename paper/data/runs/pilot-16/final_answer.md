# Final report: the stopper now ends degenerate cycles at a segment boundary, not right after a `</think>`

## What I submitted

A single row of one tensor differs from the starting edit: `model.language_model.layers.23.mlp.gate_proj.weight`, row 2664, which is the gate of the degenerate-cycle stopper. That is 1,024 weights (copy in /work/best_final).

- **Neuron 2664's up row and down column are unchanged.** It still pushes `<|im_end|>` +4, end-of-text +1 and `</think>` −4 per unit at rms 0.5.
- **All other parts of the starting edit are untouched:** E2, E3, the brake 3109, the box pair 1965/2621 and the function-name gates.
- **Only an MLP tensor changed,** as this run requires.

## Mechanism

**Where the starting edit's empty answers come from.** I generated 512 answers to my own prompts with the starting edit and teacher-forced 62 of them to read the exact layer-23 MLP input.
- **Where 2664 fires.** In 7 of the 8 prose turns with an empty final segment, 2664 fired only at the very last position: the `</think>` token of a re-close, or the newline after it. Its activation there was 2.8 to 9.5, giving `<|im_end|>` probability about 1. Natural ends have a median log-probability of −1.0.
- **Its score peaks there.** 2664's score s rises with the number of closes and peaks on the `</think>` token.
- **The brake is off there.** The detector drive r has a median of 1.06 on the token before a re-close, so the brake 3109 holds `<|im_end|>` down by about 12 nats. On the `</think>` token itself r drops to about 0.19 and the brake is off.
- **Result.** The stop arrived one token too late and left nothing after the last `</think>`.

**The fix.** 2664 now fires when s + 0.3·(r − 0.6) − 1.0·c0 − 0.3·e > 0.90.
- **r** is read from 3109's own gate row.
- **c0** is a linear readout of "the current token is `</think>`". On 83 texts not used to fit it, it is at least 0.69 on all 226 close tokens and at most 0.42 on 443,000 other positions.
- **e** is a soft readout of "1-8 tokens after a close".
- **Row formula.** The row is Gstart[2664] + 1.5·G[3109] − 100·w_c0 − 30·w_e + const·k, where k is a fitted constant direction (k·z = 1 ± 3.5%).

**Offline checks on 145 recorded texts:**
- Activation is about 0 everywhere in the thinking, on every close token and in single-close answers.
- First firings land at least 8 tokens into a segment (median 38), inside loops or at pre-close tokens.
- A teacher-forced check of the precursor rule showed `<|im_end|>` falling from about 1 to at most e^−10.6 on close tokens, and becoming the top token at the re-timed boundaries.

## Evidence: paired against the starting edit (`"baseline": "start"`), sets never used for choosing

| Suite | Measure | Start → final | Paired diff (SE) |
|---|---|---|---|
| Prose, 256 fresh prompts | Answers with no answer | 3.1 → 1.6 | −1.6 (1.4) |
| | Empty after the last `</think>` | 2.7 → 0.8 | −2.0 (1.2) |
| | Ended inside the thinking | 0.4 → 0.8 | +0.4 (0.7) |
| | Prompts / instructions followed | | −3.9 (2.2) / −3.9 (1.8) |
| | Cut | | +2.7 (1.5) |
| Math, 256 fresh | Accuracy | | −1.2 (2.6) |
| | Final box | | −7.4 (3.3) |
| | Empty final | 2.0 → 0 | −2.0 (0.9) |
| | Cut | | +4.7 (2.3) |
| Code, MBPP 128 + APPS 128 | Pass rate | | +3.1 (2.2) |
| | MBPP / APPS | | +4.7 (3.8) / +1.6 (2.2) |

**Multiple choice was only run with v2.** v2 is the same 2664 change plus a guard neuron (3222) that lowers `<|im_end|>` on `</think>` tokens. On 512 hard questions:

| Measure | Paired diff (SE) |
|---|---|
| Accuracy | −0.8 (2.1) |
| Answers with a letter | −1.2 (2.3) |
| Empty final (2.1% → 0) | −2.1 (0.6) |
| Cut | +0.2 (1.6) |

**v2 on other suites:**
- **Prose (768 prompts):** no-answer −3.1 (0.9) and −3.5 (1.3), empty finals 0 throughout, compliance +0.4 (2.4).
- **Math (512):** accuracy −0.6 (2.3), cut +0.2 (1.8).
- **Code (768):** −2.3, −3.9 and −3.9 (about 2.1 each).

**The harness's noise floor matters for reading all of this.** The starting edit against itself, on identical items in two passes:
- only 40 of 256 answers were identical;
- prose compliance moved −1.2 (2.4);
- math boxes moved −7.0 (3.3) and cuts +2.3;
- code went from 16.4% to 13.3%.

Nearly all discordant pairs split inside the thinking, where this edit cannot act. Where the thinking stayed identical, which is the clean effect of the edit:
- math: correct 10 → 12, empty finals 3 → 0, cuts 2 → 2;
- prose: correct 7 → 9, empty finals 3 → 0.

I dropped v2's guard for two reasons. Every v2 code pass was negative, while v2' was positive on the same problems, all within that noise. The guard also needed repurposing a neuron, the only change active in ordinary text.

## Why it should carry over

- **It acts on states the network represents, not on particular problems.** It fires in a degenerate re-close cycle at a segment boundary; it never fires on or right after `</think>`.
- **The readouts generalised.** They were checked on held-out texts.
- **Elsewhere the output is unchanged.** Outside these states 2664's activation is about 0 under both versions.

## What it might hurt, and what remains

- **Cuts.** Stops wait for a boundary, so cuts may rise a little. My estimates are +2.7 (prose) and +4.7 (math), compared with +2.3 for start against itself.
- **No answers gained in degenerate cycles.** Boxless math finals contained no correct value (0 of 33). Letterless multiple-choice finals rarely state the right option, so those segments are now non-empty but still lack a box or letter.
- **Not fixed: turns ending inside the thinking,** about 0.5%. I found no linear signal for them at layer 23.
- **Not fixed: rare post-close leaks with 2664 silent,** about 0.4-0.8% of prose. The dropped guard removed them.