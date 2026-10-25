# 008 - B vs C on the full BC+ slice: F1 tier-2 KILL bar met (seed 0)

Date: 2026-10-22, 11:45 PT. From: Station controller. Seed 0, final checkpoint (step 100), primary protocol.

## Full BC+ (830 queries, greedy, 20 turns, 16K budget, frozen corpus)

| arm | EM | correct |
|---|---|---|
| B TreePO literal | 10.48 | 87 / 830 |
| C TreeOPSD | 10.12 | 84 / 830 |

- **Paired:** both arms right on 41; B only, 46; C only, 43.
  - Pairing is by question text (sha1 of the prompt), not row order.
  - All 830 paired.
- **C - B = -0.36 EM, paired-bootstrap 95% CI [-2.65, +1.93]** (5,000 resamples, seed 0).
- **Against PREREG amendment 1 (tier 2):** the KILL bar is C - B < 1.0 EM. It is **met**.
  - The paper's BC+ gap (+2.90, 13.98 vs 11.08) lies outside our CI.
  - Caveat D9: our BC+ result measures transfer from QA training, so levels are not comparable. The arm comparison holds.

## Combined with M1

M1 trips under your 005 rule (median 1.05%, last quarter 1.56%). For the record, the original PREREG wording of M1 was "drift < 0.02 in more than 80% of steps". By call fraction we have 72%, which does not meet that wording. Your 005 rule (median + last-quarter) is the operative one. I will report both in the write-up so the redefinition is visible.

## Still running (no decision needed)

- Tier 1 (QA) for C and the rest of B's QA sets are on the side GPU. B HotpotQA full: EM 42.58.
- A (GRPO) is at step 56. E (TreePO clipped) is at step 59.
- EXP-1c starts after E, about 17:00.
- Best-in-window secondary follows once all arms finish.

## Interim grid additions

- C step 100: qa500i 40.8 / bc200i 9.5.
- A step 10: 26.8 / 6.5.
- E steps 10 / 25: 28.8 / 8.5 and 31.2 / 7.5.

Question: with F1 meeting its tier-2 KILL bar at seed 0 and M1 tripped, do you still want seeds 2-3 of B and C? Or should that compute go to EXP-1c plus the tree-sampling characterisation (4,2)-TreePO vs (8,0)-TreePO vs GRPO?
