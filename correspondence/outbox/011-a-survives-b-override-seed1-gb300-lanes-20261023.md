# 011 - (a) clipped TreePO survives E=16; (b) run anyway on the Founder's call; seed 1 numbers; GB300 eval lanes

Date: 2026-10-23, 20:15 PT. From: Station controller.

## (a) treepo_clip at E=16 (40 steps, seed 0): NO collapse

- Train reward 0.49-0.64 throughout; length-clip ratio 0.01-0.10, never above 0.10 (B-literal at E16 hit 0.99).
- Step 40 subsets: qa500i **34.2**, bc200i **8.5**; answer rate 0.90 / 0.99. No answer-rate prodrome, unlike B-literal at E8 (0.72 / 0.52).
- For comparison at E16: C TreeOPSD 35.4 / 5.0; B literal 0 / 0.

Under your 010 rule this is the outcome where the stability comes from boundedness in general; TreeOPSD's posterior term adds nothing measurable even at 4x dose.

## (b) treepo E=16 seed 1: RUNNING, Founder override of the 010 condition

The Founder ordered (b) despite (a) surviving. Reason: the B-literal collapse is n=1. The "unclipped collapses, clipping prevents it" finding presumes the collapse is systematic. If seed 1 does not collapse, the stability finding largely dissolves into seed luck. It does not affect the F1 verdict either way.

- Placement: left GB300, after the left lane's current job; seed-package evals are not delayed.
- Expected finish: about 01:00-02:00 Sat. Step-40 subset eval follows.

## Seed 1 full evals (step 100, EM)

| | BC+ | HotpotQA | 2Wiki | MuSiQue | Bamboogle |
|---|---|---|---|---|---|
| B TreePO s1 | 7.23 | 45.97 | 53.78 | 20.56 | 51.2 |
| C TreeOPSD s1 | 10.60 | 45.67 | 50.35 | running | running |
| (s0, for reference) B / C | 10.48 / 10.12 | 42.58 / 46.51 | 49.93 / 51.20 | 18.54 / 22.51 | 45.6 / 54.4 |

- QA: seed 0's C lead does not replicate at seed 1 (HotpotQA tied, 2Wiki B +3.4).
- BC+: the sign flips the other way (C +3.4 at s1, -0.4 at s0).

Seed-1 subsets at step 100 (qa500i / bc200i): B 39.4 / 6.0; C 37.8 / 7.0. Seed 2 C: 35.6 / 13.5; seed 2 B pending.

## GB300 eval lanes (deviation D15, for the record)

Training is over on both GB300s, so each is now a full-eval lane alongside the left RTX PRO 6000.

Assignment:
- right GB300: the seed-2 B/C pair, so each B-vs-C seed pair is scored on one GPU type;
- left GB300: E (treepo_clip) and D (grpo_tok), after (b);
- RTX PRO: seeds 0-1 remainder, plus A.

GB300 evals run flash-attn with remove-padding; the RTX PRO runs SDPA without it. Calibration re-scores (Bamboogle, n=125) on the GB300:
- C s0: 53.6 vs 54.4 on the RTX PRO; 7 of 125 items flip.
- B s0: 43.2 vs 45.6.

The offset is within a few items on a 125-item set, and it cancels inside the seed-2 pair. It could shift seed-2 absolute levels by ~1-2 points; I will report seed means with and without seed 2's hardware noted.

## Next

Seed package (B/C x 3 seeds: per-bench mean +- seed SD, per-seed paired-bootstrap CIs), expected Sat ~06:00-08:00. Then A/D/E full and (b), Sat ~midday.
