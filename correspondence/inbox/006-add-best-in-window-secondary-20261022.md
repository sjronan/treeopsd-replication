# Added analysis — apply the paper's own selection rule to our runs (F3 completeness)

From: research session `cotton-azimuth`. Date: 2026-10-22 (morning) PT.

## Context

Stephen asked whether our study uses the same materials and industry-standard
methods, i.e. whether our methods are defensible. My adjudication: YES, with
one gap I want closed (below). Matched to the paper where specified
(Qwen3-4B-Thinking-2507 identical checkpoint across arms, E=4 blocks of 16,
100 updates, T=1.0 rollouts, thresholded-F1 reward, paper-number bars as
primary); both readings run where the paper is ambiguous (literal vs clipped
TreePO); deviations logged (D9, D12–D14); eval stricter than the paper's
(final checkpoint, frozen data with sha256 manifest, paired-bootstrap CIs).

## The gap and the fix

Because our eval is stricter than the paper's (final checkpoint vs their
best-in-window), a critic could dismiss a null result as "you evaluated
wrong." Preempt it: as a SECONDARY analysis, apply best-in-window checkpoint
selection to OUR OWN runs, arm by arm, and report whether the paper-sized gap
reappears. Both outcomes are informative:

- If best-in-window resurrects a TreeOPSD lead in our data, F3 is confirmed
  causally and the paper's headline is selection artifact — our strongest
  possible result.
- If no selection rule resurrects it, the null result survives the most
  sympathetic possible re-analysis — unassailable.

Use the same interim-eval subsets and step grid for the window. Report it
alongside the primary final-checkpoint verdict, clearly labeled secondary.
This is cheap (no new compute, just scoring saved checkpoints) — do it as
soon as step-100 checkpoints exist for every seed-0 arm.
