# 009 - Tier 1 disagrees with tier 2: C beats B by +3.9 on full HotpotQA. Seeds 2-3 are now essential.

Date: 2026-10-22, 16:20 PT. From: Station controller. Seed 0, final checkpoints.

## Full HotpotQA dev (7,405 questions, T 0.6, 1 sample, paired by prompt sha1)

| | B TreePO | C TreeOPSD |
|---|---|---|
| EM | 42.58 | 46.51 |
| F1 | 0.545 | 0.589 |
| answer rate | 0.992 | 0.983 |
| mean output length | 8.0K chars | 11.2K chars |

- **C - B = +3.93 EM, paired-bootstrap 95% CI [+3.11, +4.75].**
- This CI covers evaluation noise only. It does not cover training-seed variance.

## Why this does not overturn tier 2, and why it changes my 008 question

- **Tier 2 (BC+, headline):** C - B = -0.36 [-2.65, +1.93]. The F1 KILL bar is met.
- **Tier 1 (QA):** C ahead by +3.9, far more than the paper's own MHQA gap (< 1) and the opposite sign to tier 2.
- **M1 at the same time:** the per-update difference between the C and B targets is about 1%.
- **What that implies:** two near-identical update rules over 100 steps diverge into different policies, so B and C at seed 0 are effectively two different training seeds. C also learned a longer, more search-heavy style:
  - outputs 1.4x longer;
  - 14.2 vs 10.6 tool calls per prompt late in training.
- **Seed-to-seed variance in this RL setting is unmeasured.** It may well be several EM points.
- **My 008 question is withdrawn.** Seeds 2-3 of B and C are now the deciding experiment, not optional precision. Without them, tier 1 and tier 2 cannot be separated from seed variance.

## Proposal (needs your ruling)

1. Run seeds 1 and 2 of B and C at E=4, 100 steps.
   - About 4 x 8 GB300-h = 32 GB300-h, on both Stations after D and EXP-1c, from about 03:00 tomorrow.
   - Total compute stays inside the 3 GB300-day cap only if EXP-2 is deferred. Flagging that.
2. Report the B vs C gap per tier as mean ± seed SD, with the eval CI per seed.
3. Keep EXP-1c (drift-dose) running as is: it tests the mechanism directly and is independent of seed variance.

## Other numbers since 008

- C full BC+ 10.12. C full HotpotQA 46.51. The B and C 2Wiki / MuSiQue / Bamboogle full evals follow.
- A (GRPO) finished at step 100. Interim qa500i / bc200i:

  | step | 25 | 50 | 75 | 100 |
  |---|---|---|---|---|
  | A | 29.6 / 8.0 | 34.8 / 7.0 | 34.6 / 10.0 | 37.4 / 9.0 |

- E (TreePO clipped) finished at step 100. Interim qa500i / bc200i:

  | step | 50 | 75 |
  |---|---|---|
  | E | 35.4 / 7.5 | 37.8 / 9.0 |

- D (GRPO G=12) is at step 22, about 290 s/step. EXP-1c treeopsd E=8 is at step 17 of 40, about 275 s/step.
