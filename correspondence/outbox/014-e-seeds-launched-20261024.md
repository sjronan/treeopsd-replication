# 014 - Founder chose (ii): arm E seeds 1-2 launched

Date: 2026-10-24, 06:25 PT. From: Station controller.

- Founder: "go, run both seeds".
- exp1-treepo_clip-s1 runs on the left GB300, exp1-treepo_clip-s2 on the right GB300. Both started 06:07-06:08 with the same config as E s0 (100 steps, save every 5, keep 2).
- Each run's full 5-bench eval follows on its own GB300.
- Correction to the cost in 012: a full 5-bench eval takes about 1 h on a GB300 (measured on D and E overnight), not 7-8 h. Expected:
  - training done about 13:45-14:15;
  - full evals done about 15:00-15:30 Sat;
  - C vs E at n = 3 vs 3 the same afternoon, not Sunday evening.
- Hardware note (D15): E s1 and s2 are scored on GB300s; E s0's QA was also scored on a GB300 (BC+ left GB300). So all E evals are on GB300; C s0/s1 were on the RTX PRO, C s2 on the right GB300. I will report C vs E with this noted, plus the calibration offsets (about 1-2 points absolute on Bamboogle).
- All checkpoints preserved: interim HF at 10/25/50/75/100 for E s1/s2 is kept.
- Disk: the old Kimi K3 weights were removed from the Stations, after verifying the NAS copy.
- Still finishing: the seed-1/2 interim grid on the RTX PRO; the (b) step-40 subsets; the RTX re-check.
