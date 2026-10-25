# 010 - EXP-1c result: TreePO-literal collapsed at E=16; TreeOPSD stable. Seed 1 C in.

Date: 2026-10-23, 10:55 PT. From: Station controller.

## EXP-1c, step 40, subsets (qa500i / bc200i EM; answer rate in brackets)

| E (blocks/step) | C TreeOPSD | B TreePO literal | C - B (qa / bc) |
|---|---|---|---|
| 4 (EXP-1, step 100 for reference) | 40.8 / 9.5 | 37.4 / 10.0 | +3.4 / -0.5 |
| 8 | 36.0 / 8.0 [0.89 / 0.97] | 36.8 / 7.5 [0.72 / 0.52] | -0.8 / +0.5 |
| 16 | 35.4 / 5.0 [1.00 / 1.00] | **0.0 / 0.0 [0.00 / 0.005]** | +35.4 / +5.0 |

Drift (C runs, blocks 2..E): E8 median 1.23%, last-quarter 1.89%; E16 median 4.79%, last-quarter 7.74%, 34% of calls >10%.

## What happened to B at E=16

A training collapse, not an eval fault. Train reward (critic/score/mean) and response-length clip ratio:

- steps 1-25: reward 0.47-0.60, clip 0.01-0.07 (normal, tracks C).
- step 29: reward 0.18, clip 0.78.
- step 33: reward 0.002, clip 0.99 (nearly every response hits the length cap).
- steps 37-40: partial recovery then relapse; step 40 reward 0.017, clip 0.92.

The step-40 checkpoint emits no answer and no tool call on 499/500 + 199/200 items.

C at E=16 never left the normal band: reward 0.50-0.69, clip 0.00-0.03 throughout.
B at E=8 did not collapse in training (reward 0.68 at step 40), but its eval answer rate was already low (0.72 qa, 0.52 bc vs C's 0.89 / 0.97), consistent with an early form of the same failure.

## Reading (yours to rule on)

1. The predicted signature (C-B gap grows smoothly with drift) is NOT what we see. At E=8 there is no gap. At E=16 the "gap" is entirely B failing.
2. The result is still informative: at high off-policy dose, the unclipped TreePO update was unstable and TreeOPSD's posterior term was not. That is a stability claim, not the paper's accuracy claim. The paper does not, as far as I know, report a B collapse.
3. Caveats: n=1 seed per cell; B collapse could be seed-specific. Arm E (TreePO clipped) was not run in EXP-1c, so we cannot say whether ordinary ratio clipping alone prevents the collapse.

Options, if you want them (each about 40 steps, ~3.5 h on one GB300, after the seed queues end ~17:30):
- (a) treepo_clip at E=16, to separate "teacher term" from "any clipping".
- (b) treepo at E=16 seed 1, to test whether the collapse repeats.
- (c) neither; report as observed with n=1.

## Seed 1 / other full evals so far

- C seed 1 (step 100): bc200i 7.0, qa500i 37.8 (seed 0 was 9.5 / 40.8). **Full BC+ 10.60** (seed 0: 10.12). HotpotQA full running.
- B seed 1: training, step 59 of 100.
- C seed 2: training, step 76 of 100 (right). Then B seed 2.
- A GRPO seed 0 full: BC+ 10.00, HotpotQA 43.44, 2Wiki 48.55. MuSiQue/Bamboogle queued after the seed evals.

Next report: B seed 1 full BC+, and your ruling on (a)/(b)/(c).
