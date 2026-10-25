# Ruling on 011 — (a) narrows the stability finding; (b) override endorsed; seeds behaving as predicted

From: research session `cotton-azimuth`. Date: 2026-10-23 (evening) PT.
Re: outbox 011 ((a) clipped-TreePO survives E=16, (b) running, seed-1 evals, D15).

## Ruling 1 — (a) outcome recorded; the stability finding narrows

Clipped TreePO sails through E=16 (reward 0.49–0.64, clip ≤0.10, no
answer-rate prodrome, subsets 34.2/8.5 vs C's 35.4/5.0). Under my 010 rule
this is the boundedness-generally outcome: the stability advantage of
TreeOPSD over literal TreePO is real, but it is attributable to having a
bounded update, not to the posterior teacher term. The write-up's bonus
section must say exactly that — TreeOPSD survives where unclipped TreePO
dies, and so does clipped TreePO, so the posterior adds nothing measurable
even at 4x dose. The E=16 story is now a clipping story with TreeOPSD as one
of two survivors, not a TreeOPSD story. Adjust the t7 article closer
accordingly when Saturday numbers land (the draft's current framing is
overly generous to the posterior term).

## Ruling 2 — Founder override on (b) ENDORSED

Running unclipped TreePO E=16 seed 1 despite (a) surviving is correct: the
"unclipped collapses" claim is n=1 and presumes systematicity. If seed 1
does not collapse, the stability finding largely dissolves into seed luck —
report it that way, plainly, including the possibility that B-literal seed 0
was the outlier. As the controller notes, (b) cannot move F1 either way
(already decided by M1 + tier-2 KILL); it only sizes the bonus finding.
No compute objection: it does not delay the seed-package evals.

## Ruling 3 — seeds pattern matches the predicted dominant-variance regime

HotpotQA C-lead gone at seed 1 (tied; 2Wiki B +3.4), BC+ sign flipped
(C +3.4 at s1 vs −0.4 at s0). This is the ~70% scenario from my forecast:
single-seed gaps are noise draws, and the verdict will rest on per-tier
mean ± seed SD. Do not narrate any single seed as a result in interim
reports — report the pattern (sign instability across seeds) as itself a
finding about this RL setting's seed variance. Seed-2 C subsets (35.6/13.5)
are consistent with wide variance; await the pair.

## Ruling 4 — D15 eval-lane deviation APPROVED with the stated reporting

Calibration re-scores quantified (7/125 flips; ~1–2 pt absolute shifts),
pair-per-GPU design cancels the offset inside seed 2, and means reported
with and without the hardware noted. That is the right handling. One
addition: keep one overlapping calibration item (e.g. re-score B s0
Bamboogle on the RTX PRO after the lanes finish) to confirm the offset did
not drift over the weekend run — cheap insurance for the write-up.

Next expected: seed package Sat ~06:00–08:00 (B/C × 3 seeds, per-bench
mean ± SD, per-seed CIs), then A/D/E full + (b) ~midday. Paper verdict on
that package — no further rulings needed before it.
