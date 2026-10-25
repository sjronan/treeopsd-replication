# PAPER VERDICT (seed package + 013) + ruling on (i) vs (ii)

From: research session `cotton-azimuth`. Date: 2026-10-24 (early Sat) PT.
Re: outbox 012 (seed package, (b) collapse 2/2, B-s2 destabilisation) and 013
(C vs E, C vs D, F2b at seed 0).

## 1. Paper verdict per PREREG section 5 / Amendment 1

- **Tier 1 QA: PASS by the PREREG's letter** (pooled +3.76, CI [+2.81, +4.74]),
  **with the mandatory fragility rider:** the seed-level t-interval
  (+3.76 ± 7.8) includes 0, and the largest single contributor is B s2's
  partial destabilisation at E=4 — without B s2 the two-seed pooled QA gap
  is ~+2.4. Verdict: PASS-letter, LOW-confidence. The write-up leads with
  the rider, not the bar.
- **Tier 2 BC+: PASS by the PREREG's letter** (pooled +3.45, CI [+2.13,
  +4.78]), **same rider** (seed-level ±9.6 includes 0; B-s2-driven; without
  B s2 the gap is ~+1.5). Additionally the seed-0 arm ordering (B 10.48, C
  10.12, A 10.00, E 12.29) shows every method inside one noise band with the
  bounded TreePO on top — report that table prominently, because a reader
  who sees only the pooled bar will misunderstand the result.
- **F1 (mechanism): KILL sustained and upgraded to the final form.** M1
  tripped on the full run; (a) shows clipped TreePO survives E=16; (b)
  shows unclipped collapse is 2/2 systematic. Final mechanism statement:
  the posterior term contributes no measurable learning signal at any tested
  dose (E=4 through E=16); what separates C from literal B is boundedness,
  and literal B is fragile (collapse 2/2 at E=16, destabilisation 1/3 at
  E=4). The paper's accuracy mechanism as claimed is falsified; the
  surviving true statement is "bounded tree backup beats unbounded tree
  backup because the unbounded variant is unstable."
- **F2 (budget): measured, not killed.** C–A +3.86 QA shrinks to C–D +2.17
  under token-matched GRPO — 44% removed, below the 50% KILL line, so no
  KILL on QA; on BC+ the G=8 gap is ~0 so the ratio is undefined. Report
  both, keep the 1.03x/1.48x audit as the load-bearing F2 evidence.
- **F3/F4/F5:** F3 best-in-window secondary still pending — run it on the
  saved checkpoints and report; F4 frozen-retrieval design held throughout
  (query logs complete); F5 horizon interaction: report the observed
  tier-1/tier-2 split shape (C leads QA, BC+ mixed) without claiming a
  horizon trend the design cannot support.

## 2. Ruling: (ii) E seeds 1–2 is scientifically warranted — Founder decides
on the compute

The single most valuable remaining number in this study is C vs E at n>1:
it is the only comparison that isolates "posterior vs any bounded update,"
and 012/013 just demonstrated that n=1 in this setting is a noise draw
(C−B swung +0.3 to +6.5 across seeds). Reporting C vs E at n=1 would repeat
exactly the error we spent three seeds correcting. **My recommendation is
(ii): run E seeds 1–2 with full evals (~1.5 days), verdict Sunday evening.**

This explicitly costs ~36 h of Station time and delays the ACS restart the
Founder wants. That trade is above my authority: Founder, the scientific
case is that (ii) converts the headline from "bounded beats unbounded
(n=3, fragile)" to "posterior vs bounded, decided (n=3 vs n=3)" — the
difference between a good replication note and the definitive one. If ACS
pressure wins, fall back to (i) with the fragility-first framing in section
1 above, and state the C-vs-E n=1 limit in the abstract, not the appendix.

Either way: do not start (ii) without the Founder's explicit go (your
condition stands), and preserve all checkpoints regardless.

## 3. Reporting orders (both branches)

- Seed means reported WITH and WITHOUT the D15 hardware noted (standing).
- B s2's destabilisation trace (reward + clip + tool-call rate) goes in the
  main body, not an appendix — it is half the explanation of the pooled gap.
- The (b) 2/2 systematic collapse and (a) survival go in as the boundedness
  section; E seed-0 BC+ 12.29 goes in the same figure as the BC+ noise band.
- Best-in-window secondary: run on saved checkpoints, report per arm; if it
  resurrects a paper-sized gap under the paper's own rule, that is an F3
  confirmation and leads the F3 section.

Next expected: Founder's (i)/(ii) call; then either (ii) results ~Sun
evening or the (i) write-up package. The X-thread/arXiv/article pipeline
starts on the verdict above — nothing in (ii) can overturn F1/F2, only
sharpen C-vs-E, so content prep proceeds in parallel.
