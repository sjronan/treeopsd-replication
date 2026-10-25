# TreeOPSD replication: final analysis (2026-10-24, PT)

Study: preregistered replication of TreeOPSD, *Tree-Aware Posterior Distillation
for Long-Horizon Agentic Reinforcement Learning* (arXiv:2610.23181, Tsinghua;
lead author Heming Zou, PhD student), on Qwen3-4B-Thinking-2507, two DGX
Station GB300s plus an RTX PRO 6000 eval sidecar. Five arms, frozen
retrieval, fixed step-100 checkpoint, greedy single-sample eval.

Arms: A = GRPO G=8; B = TreePO literal (the paper's control, unclipped);
C = TreeOPSD; D = GRPO G=12 (token-matched GRPO); E = TreePO with ordinary
ratio clipping (our added arm; isolates "posterior" from "any bound").
B, C, E at seeds 0-2; A, D at seed 0. Tier 1 = mean EM over HotpotQA
(7,405 Q), 2WikiMultihopQA, MuSiQue, Bamboogle (125 Q). Tier 2 = BC+
(830 Q, long-horizon deep-research track), the preregistered headline.

Raw packages: results/seed_package_CB.json (C vs B pooling),
results/seed_package_CE_orig.json (C vs E, as scored),
results/seed_package_CE_gb300.json (C vs E, C s0/s1 re-scored on GB300 so
all of C and E share hardware). Script: scripts/seed_package.py. Paired
bootstrap: 10,000 resamples over questions, rows paired by prompt sha1,
stratified by benchmark. Seed-level t-intervals below are n=3, t(2)=4.30,
computed in this analysis; they carry seed noise only and are wide by
construction.

## 1. The deciding comparison: posterior (C) vs any bounded update (E)

Hardware-matched (GB300) package, 3 seeds per arm:

| C - E | s0 | s1 | s2 | seed mean +- sd | pooled question-bootstrap |
|---|---|---|---|---|---|
| Tier 1 QA | +3.78 | +2.10 | +0.05 | +1.98 +- 1.87 | +1.98 [+1.03, +2.94] |
| Tier 2 BC+ | -1.81 | -3.01 | +5.06 | +0.08 +- 4.35 | +0.08 [-1.24, +1.41] |

As-scored (C s0/s1 on RTX PRO) the same table reads QA pooled +2.17
[+1.29, +3.06], BC+ +0.48 [-0.84, +1.81]. The GB300 re-score moved C by
-0.08/-0.49 QA points and +0.36/-1.57 BC+ points. Hardware mixing did not
drive the answer.

Read, stated exactly as far as the data goes:

- On tier-1 QA there is a pooled +2 EM edge for C over E whose
  question-level CI excludes 0. Per-seed it declines 3.78, 2.10, 0.05 and
  the seed-level t-interval is [-2.66, +6.61]. At n=3 seeds that is
  suggestive, not established. The script's PREREG bar returns tier-1 PASS.
- On tier-2 BC+, the track where the paper claims its large effect
  (TreeOPSD - TreePO +2.90), C - E is 0.08 with both intervals centred on
  zero. The script's bar returns tier-2 KILL. There is no posterior-specific
  effect on the long-horizon benchmark at any seed count we have: seeds 0
  and 1 favour E, seed 2 favours C, carried by C s2's BC+ 13.61.
- Seed stability differs by arm on QA: E sd 0.80 (39.81, 40.61, 41.41),
  C sd 1.07 (43.58, 42.71, 41.46, GB300-scored), B sd 3.95. The clipped
  arm is not merely stable against collapse; its outcomes vary least.

## 2. Against the paper's own control: C vs literal TreePO (B)

Pooled C - B: tier-1 QA +3.76 [+2.81, +4.74]; tier-2 BC+ +3.45 [+2.13,
+4.78]. Both pass by the preregistration's letter. Both carry the same
rider: seed-level t-intervals ([-4.06, +11.57] QA; [-6.13, +13.03] BC+)
include 0 at n=3, and the largest single contributor is B seed 2, which
began to destabilise in its last 10 steps at the paper's own dose (reward
0.54-0.70 through step 91, then 0.22-0.40; length-clip ratio 0.00 through
step 92, then 0.27 at step 100; 3.52 tool calls per BC+ question against
1.81-1.97 for B s0/s1). Without B s2 the two-seed C - B means are +2.40 QA
and +1.51 BC+. The paper reports 5-seed MHQA means of GRPO 43.37 +- 0.19,
TreePO 43.40 +- 0.17, TreeOPSD 44.13 +- 0.16 (gap +0.73). Our seed sds on
the same model class are 0.80-3.95 EM points depending on arm. A reader
should not compare our absolute MHQA average to theirs without matching
benchmark composition; our sanity anchor agrees with theirs (untrained
base, frozen HotpotQA dev: 30.8 EM here, paper ReAct 31.61).

## 3. Mechanism: M1 and the drift-dose experiment (EXP-1c)

M1 measures how far TreeOPSD's per-update target moves from TreePO's,
drift_rel = mean|Gamma - V| / mean|V| on blocks 2-4. Full run at the
paper's dose (E=4 blocks/step): median 1.05%, last-quarter 1.56%, against
the preregistered 2% inactivity line. The preregistered response to a trip
was a dose sweep. Raising the dose raised the drift (E=8: median 1.23%,
last-quarter 1.89%; E=16: 4.79% / 7.74%, 34% of calls above 10%), and the
posterior-signal prediction is a B-vs-C accuracy gap that grows with dose.
Measured instead, step-40 subsets (qa500i/bc200i):

| dose | C TreeOPSD | B TreePO literal | C - B (qa/bc) |
|---|---|---|---|
| E=8 | 36.0 / 8.0 | 36.8 / 7.5 | -0.8 / +0.5 |
| E=16 | 35.4 / 5.0 | 0.0 / 0.0 | +35.4 / +5.0 |

At E=8 there is no gap at twice the drift. At E=16 the gap is B collapsing
in training (reward 0.002, length-clip 0.99 at step 33; the checkpoint
answers almost nothing), not C improving. Arm E at E=16 (ordinary clipping,
no posterior term): reward 0.49-0.64 throughout, clip ratio never above
0.10, subsets 34.2 / 8.5, against C's 35.4 / 5.0. The unclipped collapse
repeated in a second seed (training collapse steps 32-38; step-40 subsets
17.8 / 0.0 with QA answer rate 0.95: same instability, milder expression).
Unclipped collapse is 2/2 seeds at E=16, partial destabilisation 1/3 seeds
at E=4. Neither C (0/3) nor E (0/3 at E=4; survives E=16) destabilised.

So the boundedness story is a complete causal chain in three panels, and
it does not need the posterior: unclipped collapses, clipped survives,
posterior-bounded survives. What the posterior term cannot be credited
with, at 1-2% measured update drift and a flat dose response, is the
paper's claimed turn-level credit-assignment accuracy mechanism. The
remaining +2 pooled tier-1 C - E edge (section 1) is real as a number and
unexplained by that mechanism; candidates consistent with our telemetry
are second-order training-dynamics effects (C kept trained tokens near
43K/prompt while B's fell to 33K; C learned a longer, more search-heavy
style: 14.2 vs 10.6 tool calls per prompt late in training, outputs 1.4x
longer on HotpotQA) and ordinary seed noise. We do not adjudicate between
those at n=3. Backup weights were near-one-hot in both tree arms
(omega_peaked 0.85-0.91), so the tree backup is effectively single-child
in this setting either way.

## 4. Budget (F2)

Measured in the S1b smoke runs and carried through EXP-1: tree vs GRPO G=8
is 1.03x in sampled (generated) tokens per prompt and 1.48x in trained
tokens per prompt (41,064 vs 27,779), because shared prefixes are re-scored
per branch in training. The paper's budget match holds only in the sampled
accounting. Token-matched GRPO (D, G=12, seed 0) removes 44% of the tier-1
C - A gap (+3.86 to C - D +2.17 [+0.59, +3.74] as first scored; +2.08 on
the GB300 package), below the preregistered 50% F2b kill line; on BC+ the
G=8 gap is near zero so the ratio is undefined. Report the 1.03x/1.48x
audit as the F2 evidence and the 44% as measured-not-killed.

## 5. Selection (F3), retrieval (F4), horizon (F5)

F3: best-in-window scoring (checkpoints 10/25/50/75/100, subsets) does not
resurrect a paper-sized gap; it shrinks C - B toward zero (per-seed mean
+0.3 QA / +0.5 BC+ under best-in-window vs +1.5 / +2.5 at the final
checkpoint). The selection rule helps B more than C because it rescues
B s2's pre-destabilisation peak (qa500i 36.6 at step 75 vs 32.8 at 100).
The preregistered F3 kill pattern is not met and the shift runs the other
way. Coarser grid than the preregistration's every-5-updates stays a
stated limitation; it is conservative against this conclusion.

F4: retrieval was frozen by construction; query logs complete. The
eval-drift control closed: B s0 Bamboogle re-scored on the RTX PRO after
the run returned 45.6, identical to its 22 Oct score.

F5: report the tier split as observed (C leads pooled QA against every
control; BC+ is mixed and seed-carried). The design caps MHQA at 5 turns
and cannot support a horizon-trend claim either way. State that.

## 6. Verdict, per preregistered bars, with riders first

- F1 (posterior loss does the work): KILLED as claimed. M1 tripped at the
  paper's dose, the dose response is flat where the mechanism predicts
  growth, and against a bounded control the effect is zero on the
  headline track and a seed-fragile +2 on QA. The tier PASSes against
  literal TreePO are real by the letter of section 5 and are explained
  by the control's fragility, not by posterior credit assignment.
- F2 (budget matched): FAILS in trained tokens (1.48x), passes in sampled
  tokens (1.03x); F2b measured at 44% removal, no kill.
- F3 (selection): confirmed negative; selection favours the control.
- F4: held by construction; drift control clean.
- F5: not decidable in this design; tier split reported without a trend.
- The durable positive finding is ours, not the paper's: literal
  (unclipped) tree backup is unstable (collapse 2/2 at 4x dose,
  destabilisation 1/3 at 1x), and any bound - the paper's posterior
  construction or ordinary ratio clipping - prevents it. Bounded tree
  backup trains a 4B search agent from 22.8 EM (base, qa500i) to
  40.6-42.6 mean QA EM across arms E/C at 3 seeds.

Per-arm step-100 means (EM; C from the GB300 package):

| arm | tier-1 QA (mean +- seed sd) | tier-2 BC+ |
|---|---|---|
| A GRPO G=8 (s0) | 39.80 | 10.00 |
| B TreePO literal (3 seeds) | 39.02 +- 3.95 | 7.99 +- 2.21 |
| C TreeOPSD (3 seeds) | 42.58 +- 1.07 | 11.04 +- 2.34 |
| D GRPO G=12 (s0) | 41.50 | 7.83 |
| E TreePO clipped (3 seeds) | 40.61 +- 0.80 | 10.96 +- 2.09 |

Limitations to print in the paper, not bury: 3 seeds (seed-level intervals
include 0 for every gap); one model at one scale (Qwen3-4B); BC+ at 830
questions has +-2 EM class noise per arm; best-in-window on a 5-point
grid and small subsets; single-sample greedy eval; A and D at one seed;
our absolute MHQA composition differs from the paper's average.
