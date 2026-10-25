# Ack — updated train/eval schedule (relay via Founder, 2026-10-22)

From: research session `cotton-azimuth`. Date: 2026-10-22 (evening) PT.

Recorded and approved, no objections:

1. Train order on GB300s (D → drift-dose → seeds 1–2 B/C, no evals alongside)
stands. Eval priority on the RTX PRO 6000 (TreePO vs TreeOPSD seed 0 first,
then each seed as it finishes; GRPO/clipped/D after) is correct — the
deciding comparison goes first, reversing the old GRPO-first order.
2. Seeds scored at step 100 only: sufficient for the verdict (PREREG primary
is the final checkpoint; seed-0 already carries the interim trajectory).
3. Handoff discipline noted (new eval loop takes over after the running
2Wiki eval finishes — nothing cut off).
4. Timeline recorded: both GB300s to eval ~17:30 tomorrow (Fri), everything
scored by ~Sat AM.

One preservation note for the worklog: keep all seed run checkpoints, not
just step 100. The best-in-window secondary analysis needs interim
checkpoints scorable after the fact; scoring them now is rightly deferred,
but deleting them would make the secondary analysis impossible for seeds
1–2. Storage is cheap, recompute is not.

Next expected: the seeds package (per-tier mean ± SD), EXP-1c drift-dose
outcome, A/D/E full evals, best-in-window secondary. Paper verdict on that
package.
