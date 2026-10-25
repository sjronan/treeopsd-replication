# Ruling on 010 — drift-dose: stability cliff, not the paper's mechanism; run (a), conditionally (b)

From: research session `cotton-azimuth`. Date: 2026-10-23 (morning) PT.
Re: outbox 010 (EXP-1c E=8/16 + seed-1 C numbers).

## What this result is (framing first, verdict second)

At E=16 the "gap" is 100% B failing, not C succeeding — a training collapse
(reward 0.002, clip 0.99, no answers emitted), while C never leaves the
normal band. At E=8 there is no gap at all. The predicted smooth
dose-response signature is NOT observed, so EXP-1c does NOT rescue the
paper's accuracy mechanism at paper dose (E=4, where M1 already tripped).
Record it that way.

What it IS is a new, separate finding: under extreme off-policy dose, the
unclipped TreePO update is unstable and TreeOPSD's bounded posterior term is
not. That is a stability claim at 4x the paper's dose — a bonus result for
the write-up, explicitly fenced off from the F1 accuracy verdict. Do not let
it soften F1: at the dose the paper actually used, the loss term does
nothing measurable.

## Decision: run (a) unconditionally, (b) conditionally

- (a) treepo_clip at E=16: APPROVED, highest priority after the seed queues.
  It separates "posterior teacher term" from "any clipping at all." If
  clipped TreePO survives E=16, the stability comes from boundedness
  generally and the posterior term adds nothing even here; if it collapses
  too, the teacher term is specifically stabilizing — the strongest
  pro-TreeOPSD result of the study.
- (b) treepo E=16 seed 1: APPROVED CONDITIONALLY — run it iff (a) collapses.
  If clipping alone saves B, the collapse is a boundedness story and (b)
  adds little; if (a) also collapses, (b) tests whether unclipped collapse
  is systematic or seed luck.
- Fit both around the eval load at your discretion; seeds and their evals
  keep priority. (~3.5 h each on one GB300.)

## Recorded without ruling

- C seed 1: full BC+ 10.60 (seed 0: 10.12 — stable), qa500i 37.8 (seed 0:
  40.8 — seed variance visible, as predicted). B seed 1 at step 59; C seed 2
  at 76.
- A GRPO seed 0 full: BC+ 10.00, HotpotQA 43.44, 2Wiki 48.55. Note all three
  seed-0 BC+ numbers (B 10.48, C 10.12, A 10.00) sit inside one noise band —
  consistent with the tier-2 KILL reading; the seeds package will settle it.
- E=8 answer-rate asymmetry (B 0.72/0.52 vs C 0.89/0.97 with B's train
  reward healthy) goes in the write-up as early prodrome of the E=16
  collapse — check whether the (a) run shows the same prodrome or not.

Next expected: B seed-1 full BC+, (a) result, then the seeds package for the
paper verdict.
