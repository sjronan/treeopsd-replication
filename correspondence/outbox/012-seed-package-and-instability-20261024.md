# 012 - Seed package (B/C x 3 seeds) + instability evidence: (b) collapsed, B s2 destabilised at E=4

Date: 2026-10-24, 00:55 PT. From: Station controller.

## Seed package: final checkpoint (step 100), full sets, EM

| run | BC+ | HotpotQA | 2Wiki | MuSiQue | Bamboogle | QA avg |
|---|---|---|---|---|---|---|
| B s0 | 10.48 | 42.58 | 49.93 | 18.57 | 45.60 | 39.17 |
| B s1 | 7.23 | 45.97 | 53.78 | 20.61 | 51.20 | 42.89 |
| B s2 † | 6.27 | 39.54 | 41.57 | 14.05 | 44.80 | 34.99 |
| C s0 | 10.12 | 46.51 | 51.20 | 22.55 | 54.40 | 43.67 |
| C s1 | 10.60 | 45.67 | 50.35 | 20.77 | 56.00 | 43.20 |
| C s2 † | 13.61 | 45.43 | 48.74 | 21.27 | 50.40 | 41.46 |
| A s0 (GRPO G=8) | 10.00 | 43.44 | 48.55 | 18.41 | 48.80 | 39.80 |

† Scored on the right GB300 (D15); B s2 and C s2 share the GPU.

Notes:
- MuSiQue values here pair by prompt sha1 and average duplicate prompts, so B s0 reads 18.57 against the 18.54 raw row mean. Pairing is unaffected.
- Script: scripts/seed_package.py; output in interim/seed_package/seed_package.json.

### C - B, paired bootstrap over questions (10,000 resamples, stratified by benchmark)

| seed | tier 1 QA avg | tier 2 BC+ |
|---|---|---|
| s0 | +4.50 [+2.91, +6.14] | -0.36 [-2.65, +1.81] |
| s1 | +0.31 [-1.03, +1.75] | +3.37 [+1.20, +5.54] |
| s2 | +6.47 [+4.47, +8.53] | +7.35 [+5.06, +9.76] |
| pooled (per-question diff averaged over seeds) | **+3.76 [+2.81, +4.74]** | **+3.45 [+2.13, +4.78]** |

Seed level, mean +- sd (n = 3):

| | QA avg | BC+ |
|---|---|---|
| B | 39.02 +- 3.95 | 7.99 +- 2.21 |
| C | 42.77 +- 1.16 | 11.45 +- 1.89 |
| C - B | +3.76 +- 3.14 | +3.45 +- 3.86 |

### Against the PREREG bars

The script applies section 5 / Amendment 1 to the pooled question-bootstrap CI: **tier 1 PASS, tier 2 PASS (confirmatory, 3 seeds).**

The controller flags two things you need to rule on:
1. The question bootstrap carries eval noise only. A seed-level t interval (n = 3, t(2) = 4.30) is about +3.76 +- 7.8 on QA and +3.45 +- 9.6 on BC+. Both include 0. The PREREG CI is the question bootstrap; the seed SD is reported as required.
2. M1 tripped. The PREREG says M1 kills F1 "whatever the EM shows". The EM now shows a confirmatory gap. Your 010/011 framing (F1 decided by M1 + seed-0 tier-2 KILL) predates this package.

## Where the gap comes from: B (literal TreePO) is unstable, C is not

**1. Run (b), treepo E=16 seed 1: collapsed, the same way as seed 0.**
- Reward was normal (0.39-0.65) through step 31.
- Step 32: clip 0.34.
- Steps 33-38: reward 0.03-0.26, length-clip ratio 0.975-1.000.
- Seed 0 collapsed at steps 29-33.
- **The unclipped E=16 collapse is 2/2 seeds: systematic, not seed luck.**
- Step 40 lands about 01:20; the step-40 subset eval follows.

**2. B seed 2 destabilised at the paper's dose (E=4), in the last 10 steps.**
- Reward: steps 84-91 0.54-0.70, steps 92-95 0.22-0.40, steps 97-100 0.47-0.58.
- Length-clip ratio: 0.000 through step 92, then 0.12 at 96, 0.23 at 97-98, 0.27 at 100.
- The step-100 checkpoint shows it in evaluation: tool calls per BC+ question 3.52 (B s0/s1: 1.97 / 1.81; C s2: 2.30).
- C s2's clip ratio never exceeded 0.09 over 100 steps. B s0 and s1 stayed below 0.07.

B s2 is the largest single contributor to the pooled gap (QA +6.47, BC+ +7.35). Without B s2 the two-seed pooled QA gap is about +2.4 and BC+ about +1.5.

**3. Arm E (clipped TreePO, seed 0): full BC+ = 12.29.** That is above C at every seed except s2 (13.61), and above the C seed mean of 11.45. E's four QA sets are scoring now on the right GB300 (done about 05:00-06:00).

## Controller reading (yours to rule on)

- The tier-1/tier-2 PASS is real by the PREREG's letter.
- The mechanism evidence says the pass is not the posterior teacher signal working as the paper claims.
  - M1: the C and B updates differ by about 1-2% of the update norm.
  - Literal TreePO, B, has no update bound. It collapsed at E=16 on 2/2 seeds and began to destabilise at E=4 on 1 of 3 seeds.
  - TreeOPSD, C, never did.
  - Clipped TreePO, E, survived E=16. On BC+ it matches or beats C at seed 0.
- The most parsimonious reading: C beats **literal** TreePO because literal TreePO is fragile, not because the posterior adds learning signal. The decisive comparison is C vs E, the bounded TreePO. E has seed 0 only.
- Options:
  - (i) report as-is: F1 bars pass versus literal TreePO; mechanism attributed to boundedness; C vs E at seed 0 only;
  - (ii) run E seeds 1-2 (two 100-step runs, about 7.5 h each, one per GB300 in parallel, then full evals about 8 h). The verdict would land about Sun evening.

The Founder wants ACS restarted on the Stations as soon as this study completes. Option (ii) delays that by about a day and a half. Please rule on (i) vs (ii) explicitly. I will not start (ii) without the Founder's go either way.

## Still running

- E QA full (right GB300).
- D (grpo_tok) full (left GB300, after (b) ends about 01:20).
- (b) step-40 subsets, then the seed-1/2 interim grid at 10/25/50/75 for best-in-window (RTX PRO).
- RTX re-check of B s0 Bamboogle after the lanes (your 011 ruling 4).
- Expected done about Sat 10:00-12:00.
