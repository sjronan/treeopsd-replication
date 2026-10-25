# 015 - F3 best-in-window (secondary), RTX re-check, (b) step-40 subsets

Date: 2026-10-24, 10:10 PT. From: Station controller.

## F3 best-in-window (secondary)

Subsets qa500i / bc200i (EM). Grid at steps 10/25/50/75/100.
- Coarser than the PREREG's "every 5 updates through step 80": we kept 5 checkpoints per run, not 16.
- The step-100 point is included.
- All points scored on the RTX PRO.

| run | qa500i 10/25/50/75/100 | bc200i 10/25/50/75/100 | best qa | best bc | inflation (best - final) qa / bc |
|---|---|---|---|---|---|
| A s0 | 26.8 29.6 34.8 34.6 37.4 | 6.5 8.0 7.0 10.0 9.0 | 37.4 @100 | 10.0 @75 | +0.0 / +1.0 |
| B s0 | 27.8 30.6 36.2 37.2 37.4 | 7.5 6.5 13.0 9.5 10.0 | 37.4 @100 | 13.0 @50 | +0.0 / +3.0 |
| B s1 | 28.4 33.6 36.4 38.8 39.4 | 7.5 7.5 7.5 7.5 6.0 | 39.4 @100 | 7.5 @10 | +0.0 / +1.5 |
| B s2 | 28.6 34.4 35.6 36.6 32.8 | 5.5 8.5 11.5 8.5 6.5 | 36.6 @75 | 11.5 @50 | +3.8 / +5.0 |
| C s0 | 30.2 32.4 35.6 37.4 40.8 | 7.5 7.0 9.5 8.0 9.5 | 40.8 @100 | 9.5 @50 | +0.0 / +0.0 |
| C s1 | 28.2 30.2 36.6 37.2 37.8 | 9.5 7.0 8.5 10.5 7.0 | 37.8 @100 | 10.5 @75 | +0.0 / +3.5 |
| C s2 | 28.4 34.2 35.2 35.2 35.6 | 8.0 9.0 10.5 12.0 13.5 | 35.6 @100 | 13.5 @100 | +0.0 / +0.0 |
| D s0 | 29.0 33.8 34.6 37.8 37.2 | 6.5 9.0 8.0 8.0 9.5 | 37.8 @75 | 9.5 @100 | +0.6 / +0.0 |
| E s0 | 28.8 31.2 35.4 37.8 36.8 | 8.5 7.5 7.5 9.0 10.5 | 37.8 @75 | 10.5 @100 | +1.0 / +0.0 |

C - B under best-in-window, per seed:

| | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| qa500i | +3.4 | -1.6 | -1.0 | **+0.3** |
| bc200i | -3.5 | +3.0 | +2.0 | **+0.5** |

The same at the final checkpoint (subsets): qa mean +1.5, bc mean +2.5.

Reading:
- Best-in-window selection does NOT resurrect a paper-sized gap. It shrinks the C - B gap toward 0.
  - The main reason: it rescues B s2's pre-destabilisation peak (qa 36.6 @75 against 32.8 @100). The destabilisation is visible in eval, not only in the training trace.
- The PREREG F3 KILL pattern (CI includes 0 at final, excludes 0 at best-in-window) is not met; the shift goes the other way.
- B's inflation exceeds C's on both tiers.
- The subsets are small (500 / 200), so per-cell noise is about +-2 / +-2 EM. Treat as secondary.

## RTX re-check (your 011 ruling 4)

B s0 Bamboogle re-scored on the RTX PRO at 09:57 Sat: **45.6, identical** to the original 22 Oct score. There is no eval drift on that GPU across the run.

## (b) step-40 subsets

qa500i 17.8 / bc200i 0.0; answer rate 0.95 / 0.67. B-literal E16 seed 0 was 0.0 / 0.0. The collapse is confirmed in eval for both seeds.

## E seeds 1-2

Step 45 / 46 at 10:00. Training is done about 14:15; full evals, then the C rescore on GB300 (C s0 left, C s1 right), done about 16:30.
