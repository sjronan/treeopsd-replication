# 013 - Addendum to 012: E and D full evals in; C vs clipped TreePO at seed 0

Date: 2026-10-24, 02:25 PT. From: Station controller. Seed 0 only for A, D, E.

| run (s0) | BC+ | HotpotQA | 2Wiki | MuSiQue | Bamboogle | QA avg |
|---|---|---|---|---|---|---|
| A GRPO G=8 | 10.00 | 43.44 | 48.55 | 18.41 | 48.80 | 39.80 |
| B TreePO literal | 10.48 | 42.58 | 49.93 | 18.57 | 45.60 | 39.17 |
| C TreeOPSD | 10.12 | 46.51 | 51.20 | 22.55 | 54.40 | 43.67 |
| D GRPO G=12 (grpo_tok) † | 7.83 | 44.09 | 51.19 | 20.32 | 50.40 | 41.50 |
| E TreePO clipped † | 12.29 | 44.74 | 47.73 | 20.36 | 46.40 | 39.81 |

† QA sets scored on a GB300 (D15). E's BC+ was scored on the left GB300 and D's on the left GB300.

Paired bootstrap (10,000 resamples), seed 0:

| pair | tier 1 QA avg | tier 2 BC+ |
|---|---|---|
| **C - E (posterior vs bounded TreePO)** | **+3.86 [+2.15, +5.68]** | **-2.17 [-4.34, +0.00]** |
| E - B (clipping vs none) | +0.64 [-1.52, +2.77] | +1.81 [-0.48, +4.10] |
| C - A (vs GRPO G=8) | +3.86 [+2.23, +5.52] | +0.12 [-1.93, +2.17] |
| C - D (vs token-matched GRPO) | +2.17 [+0.59, +3.74] | +2.29 [+0.36, +4.22] |
| D - A | +1.70 [+0.24, +3.23] | -2.17 [-4.22, -0.12] |

Reading:
- C vs E at seed 0 has the same split-tier shape as C vs B at seed 0. C leads on QA (+3.9); E leads on BC+ (-2.2, CI touching 0).
- The 012 seed package showed that a single seed is a noise draw in this setting (C - B swung from +0.3 to +6.5 on QA).
- So C vs E is undecided at n = 1. This strengthens the case for option (ii), E seeds 1-2, if the decisive question is "posterior vs any bounded update".

F2b (GRPO budget matching), seed 0:
- the C - A gap is +3.86 QA / +0.12 BC+;
- the C - D gap is +2.17 / +2.29.

On QA the matched GRPO removes 44% of the gap, below the PREREG's >50% kill line, so KILL-F2b does NOT trigger on QA. On BC+ the G=8 gap is near zero, so the ratio is undefined. Seed 0 only.

Still running: (b) step-40 subsets and the seed-1/2 interim grid (RTX PRO, until about 08:30), then the RTX re-check.
