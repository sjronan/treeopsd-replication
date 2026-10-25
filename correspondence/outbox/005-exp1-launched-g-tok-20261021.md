# 005 - EXP-1 seed 0 launched; G_TOK collapses to 8, so arm D runs training-token-matched (veto window open)

Date: 2026-10-21, 23:35 PT. From: Station controller.

## 4B smoke (S1b, B=64, 10 steps each)

| per prompt | TreeOPSD (4,2) | GRPO G=8 |
|---|---|---|
| generated tokens | 19,857 | 19,322 (2,415 per trajectory) |
| trained tokens | 41,064 | 27,779 |
| tool calls | 4.9 | 8.1 |
| step wall time | ~345 s | ~250 s |

- **The tree:** traj_eq 8.72, n_nodes 16.9, suffix fraction 0.59.
- **G_TOK** = round(19,857 / 2,415) = round(8.22) = 8. Arm D as pre-registered (generated-token match) is therefore identical to arm A.
- **F2 headline, so far:** the paper's budget match holds in sampled tokens (1.03x). The tree trains on 1.48x the tokens and makes 40% fewer tool calls.

## Deviation D14: needs your ruling

So that arm D is not a duplicate run, I queued it as training-token-matched GRPO: G = round(41,064 / (27,779/8)) = 12. That is the F2-relevant control.

- It is queued last on the left Station and starts in about 17 h. Veto or redefine it before then and I will swap it.
- The alternatives I see:
  - (a) drop D and spend the compute on seed 1 of A/C;
  - (b) keep G=12.

## Running

- **Left Station:** treeopsd -> grpo -> grpo_tok(12).
- **Right Station:** treepo (literal, primary F1 control) -> treepo_clip.
- **Reporting:**
  - Each arm runs to step 100 without pause.
  - Interim evals run on a side GPU at steps 10/25/50/75: fixed 500 QA-dev + 200 BC+ subsets, seed 0, the same ids every time. A step-0 base-model eval on the same subsets is the shared starting point.
  - The step-10 reports land around 01:00-02:00 PT.
- **Compute:** about 50 GB300-hours for the five arms, inside the 3 GB300-day cap.

Still open from 004: which statistic M1 is judged on.
