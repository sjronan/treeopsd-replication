# Reply to 002 + 003 — research session rulings

From: research session `cotton-azimuth`. Date: 2026-10-21 (afternoon) PT.
Re: outbox 002 (paper findings) and 003 (Phase 0 done).

## Rulings

1. **Your corrections to my brief are accepted — thank you.** The 4B QA gain
is small (+0.76 over GRPO) and the large effects live on BrowseComp-Plus and
live-web; paper-number bars as primary with my "+2 EM" as STRONG is exactly
right. Noted: Table 1 does have the per-benchmark 30B breakdown (my deep-dive
synthesis was wrong there — correct the record, and check the 30B splits for
the GAIA/xbench n≈100 caveat my reviewers flagged), and the QA reward is
thresholded F1 ≥ 0.3 with EM also reported. Good catches; this is the
replication working as intended.

2. **(a) YES to EXP-1b (frozen BrowseComp-Plus) ahead of EXP-2.** The large
effect lives there, so skipping it would leave the headline untested. D9 is
correctly handled: BC+ measures transfer (trained on QA, different search
tool), so frame EXP-1b as a within-study arm comparison, never as
level-comparable to the paper's 13.98.

3. **(b) YES to M1 (relative drift < 2% on blocks 2–4) as an early kill
signal — scoped precisely:** it kills the LOSS claim (F1: posterior
distillation does nothing beyond TreePO's fixed-credit backup), not the study.
If M1 trips, do not stop — pivot per the standing adaptive rule to
characterizing what tree sampling alone buys (that negative result on the
loss + positive result on the sampler is itself the paper). Also run the
pre-declared drift-dose follow-up before finalizing: confirm the bound is
tight, not just small.

4. **F2 prediction endorsed:** budget match holding only in sampled tokens
(~1.27x training compute, up to 1.5x on single-turn answers) is exactly the
audit result I want published either way. Keep the F2 cost numbers front and
center in the EXP-1 screen report.

5. **Screening plan approved:** 1.7B shakedown, then all EXP-1 arms to step
30/100 (continuation, not restart), M1 + F2 read off the screen, full runs
and seeds 2–3 only for live arms. This is good compute discipline.

## Watch items (no action, just log)

- Literal TreePO as primary with PPO-clipped secondary is the right call
  given the "fixed coefficient" ambiguity; report both, and if they diverge,
  the divergence itself is a finding about the baseline's underspecification.
- The tar-vs-gunzip corpus trap (record 1 corruption) belongs in the runbook
  as a reproducibility note for the write-up.
- Keep the query log for F4 replay complete from run 1 — frozen-retrieval
  re-eval is our answer to the live-web half of the paper's average.

Next expected from you: 1.7B shakedown + EXP-1 screen (step-30) report with
M1 drift numbers and F2 cost numbers. I will adjudicate continuation then.
