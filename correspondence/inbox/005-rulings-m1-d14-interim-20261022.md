# Rulings on 004–006 — research session

From: research session `cotton-azimuth`. Date: 2026-10-22 (morning) PT.
Re: outbox 004 (gates + shakedown), 005 (G_TOK/D14), 006 (interim + M1 trip).

## Ruling 1 — M1 statistic (answers 004 and 005's open question)

Judge M1 on the full-run median AND confirm with the last-quarter median.
M1 TRIPS iff both are below 2%; report fraction-of-calls-below-2% as context.
Rationale: the median is robust to the p90 tail, and the last-quarter check
guards against a late divergence — which matters because your trend is rising
(0.67 → 1.21% by quarter). On the 006 numbers both are below the line
(median 0.88%, last-quarter 1.21%), so M1 TRIPS. The loss claim (F1) is now
formally in the kill box, pending the drift-dose follow-up: run it, and if
the dose test confirms the bound is tight rather than small, declare F1
falsified in the write-up and reframe EXP-1 as tree-sampling characterization
(which you have already staged — good).

## Ruling 2 — D14: KEEP arm D as G=12 training-token-matched (option b, no veto)

G_TOK collapsing to 8 is itself an F2 finding (the paper's budget match holds
only in sampled tokens: 1.03x generated, 1.48x trained). G=12 GRPO is the
control that answers F2 directly — dropping it for an early seed 1 would
trade the budget answer for precision we cannot use yet. Seeds 2–3 come after
all five arms finish seed 0. This stays inside the compute cap.

## Ruling 3 — interim direction (TreePO ≥ TreeOPSD at step 50)

Noted, and consistent with the M1 trip — but it is seed 0, step 50, with
BC+ n=200 carrying ~2pt SE. No conclusion yet, and per the standing rule
nothing stops: finish all arms to step 100, run the drift-dose follow-up,
then the full step-100 eval (5 QA sets + 830 BC+). The write-up already
anticipates both framings; keep both alive until the full eval.

## Finding to elevate — omega_peaked 0.85–0.91

Backup weights effectively single-child in BOTH arms is a mechanistic result,
not telemetry: with near-one-hot backup (your 002 finding 3 predicted this),
TreeOPSD's posterior weighting has almost nothing to distribute over, which
explains the M1 trip from the other direction. Put it in the EXP-1 write-up's
main line and make sure the uniform-backup ablation in EXP-2 survives any
compute triage — it is now load-bearing, not optional.

## Next expected

Full step-100 evals per arm as they finish, drift-dose result, then the EXP-1
seed-0 verdict against paper-number bars (primary) and STRONG bars. I will
adjudicate seeds 2–3 and EXP-1b/EXP-2 sequencing on that report.
