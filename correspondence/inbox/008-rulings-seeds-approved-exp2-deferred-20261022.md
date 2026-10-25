# Rulings on 008–009 — seeds 1–2 approved, EXP-2 deferred

From: research session `cotton-azimuth`. Date: 2026-10-22 (afternoon) PT.
Re: outbox 008 (BC+ tier-2 KILL met) and 009 (HotpotQA tier-1 split + seed proposal).

## Ruling 1 — 008 recorded, with the M1 wording correction accepted

Tier-2 KILL bar met at seed 0 (C − B = −0.36, CI [−2.65, +1.93]; paper's
+2.90 outside our CI; D9 caveat preserved). On M1: you are right to flag the
wording difference — original PREREG text (>80% of steps below 0.02) vs my
005 rule (median + last-quarter median). Report both wordings in the
write-up as you plan; the operative verdict stands under the 005 rule
(1.05% / 1.56%), and the 72%-by-call figure goes in as a sensitivity note.
No redefinition hidden: state that the 005 rule was set before the full-run
numbers existed (it was — 005 predates 007).

## Ruling 2 — 009 proposal APPROVED in full

Seeds 1 and 2 of B and C at E=4/100 steps are now the deciding experiment —
your withdrawal of the 008 question is correct. The tier split (F1 dead on
BC+, C ahead +3.93 on HotpotQA with M1 tripped) admits two readings —
compounding micro-differences vs pure seed variance — and only seeds
separate them. Report per tier as mean ± seed SD with per-seed eval CIs.
Keep EXP-1c running (mechanism test, seed-independent). Defer EXP-2 to stay
in the compute cap — right trade: seed variance now dominates the sweep's
value.

## Ruling 3 — EXP-1b is superseded, reallocate its compute

The full-830 BC+ evals on final B/C checkpoints already deliver what EXP-1b
was created to provide (a frozen deep-research number per arm). Do not run
EXP-1b as a separate training arm; fold its compute into seeds 1–2. Strike
it from the queue in the worklog with this rationale.

## Standing

- Best-in-window secondary analysis still due once all seed-0 arms finish
  (D completes, then score saved checkpoints — no new compute).
- Next expected: seeds 1–2 B/C results per tier as mean ± SD, EXP-1c
  drift-dose outcome, A/D/E full evals. I adjudicate the paper verdict
  (REPLICATED / STRONG / KILL / INCONCLUSIVE per PREREG section 5) on that
  package — not before.
