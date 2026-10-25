# Pre-registration: EXP-1 (TreeOPSD vs TreePO vs GRPO, Qwen3-4B-Thinking-2507, frozen MHQA)

- Written **2026-10-21, before any training or evaluation run.**
- Committed on branch `autoresearch/treeopsd-20261021`. The commit hash of this file is the
  timestamp of record.
- Any later edit must be a new commit with a dated "AMENDMENT" section that says what changed and
  why. The original text stays.
- Changes made after seeing results are labelled exploratory.

## 1. Question

At 4B scale, in the paper's own Multi-Hop QA setting:
- the HotpotQA training set;
- Search-R1 wiki-18 with an E5 flat index, fully frozen;
- at most 5 turns;

does TreeOPSD's forward-KL posterior loss improve on
- **(a)** the same tree trained with the fixed tree-credit (TreePO) loss, and
- **(b)** GRPO with G = 8,

when evaluation uses a fixed final checkpoint and budget is measured rather than assumed?

## 2. Arms

All arms share the image digest, data hashes, prompt template, retriever, LR 1e-6, B = 64 prompts
(E = 4 blocks of 16), 100 updates, T = 1.0 rollouts, and seed list. DESIGN.md sections 2 and 5 give
the details.

| Arm | Rollout | Loss | Note |
|---|---|---|---|
| A. GRPO | G = 8 independent | verl vanilla PPO-clip ε = 0.2, token-mean | paper baseline |
| B. TreePO | (M, N) = (4, 2) tree, likelihood-ω backup | `treepo_fixed`: −V_i log π, prompt-normalised, no ratio/clip | paper's control; **F1** |
| C. TreeOPSD | the same tree, same V_i tensor | `treeopsd_fkl`, ρ = 0.25, τ = 12.7017 | method |
| D. GRPO-tok (conditional) | G = G_tok | as A | runs only if S1b's measured G_tok ≠ 8; **F2** |
| E. TreePO-clip (secondary) | as B | PPO-clip with A = V_i | resolves PAPER.md A2 |

Seeds:
- **Preliminary:** seed 0 for A, B and C.
- **Confirmatory:** seeds 0, 1, 2 for A, B and C.

Seed s fixes the data order, the anchor RNG and the vLLM sampling seed. Arms with the same seed see
the same prompts in the same order.

## 3. The paper's own numbers that set our expectations

PAPER.md section 9 gives these Qwen3-4B-Thinking-2507 values (five-seed means ± sd).

| | MHQA Avg |
|---|---|
| GRPO | 43.37 ± 0.19 |
| TreePO | 43.40 ± 0.17 |
| TreeOPSD | 44.13 ± 0.16 |

Implied gaps: TreeOPSD − GRPO = **+0.76**, TreeOPSD − TreePO = **+0.73**. On HotpotQA alone the gaps
are +1.23 and +0.91. The paper's large effects (+2.9 to +4.0) appear only on the Deep Research
track (EXP-1b).

The study brief's EXP-1 success bar, "+2 EM at ≤ 80% rollout tokens", is about 2.6× the paper's own
MHQA claim, and it sits at a budget below the paper's claim (100%). We keep it below as **STRONG**,
but it is not the replication criterion.

## 4. Our predictions (made now; probabilities are our honest priors)

| # | Quantity (final checkpoint, step 100) | Point prediction | Probability statement |
|---|---|---|---|
| P1 | generated tokens per prompt, tree / GRPO | 1.00 to 1.10 | P(ratio in [0.85, 1.15]) = 0.6 |
| P2 | fraction of root rollouts with T = 1 (no search) | 10 to 30% | P(> 10%) = 0.6 |
| P3 | tool calls per prompt, tree / GRPO | 0.85 to 1.0 | P(≤ 1.15) = 0.85 |
| P4 | train GPU-hours per step, tree / GRPO | 1.20 to 1.35 | **P(> 1.15) = 0.8** (12 full rows vs 8; PAPER.md A13) |
| P5 | M1 `drift_rel` = mean\|Γ−V\|/mean\|V\| on blocks 2 to 4 | 0.5 to 5% | P(< 0.02 for > 80% of steps) = 0.5 |
| P6 | TreeOPSD − TreePO, MHQA avg | +0.2 | 80% interval [−0.8, +1.2] |
| P7 | TreeOPSD − GRPO, MHQA avg | +0.4 | 80% interval [−1.0, +1.5] |
| P8 | best-in-window minus final, any arm | +0.3 to +1.0 | P(> 0.5 for TreeOPSD) = 0.5 |
| P9 | TreeOPSD − GRPO gain on MuSiQue 3/4-hop minus 2-hop | ≈ 0 | P(> 0) = 0.45 |
| P10 | STRONG: TreeOPSD ≥ GRPO + 2.0 at ≤ 0.8× generated tokens | — | P = 0.05 |

Reasoning for P5 and P6:
- TreeOPSD and TreePO produce identical gradients on block 1 (Theorem 2(i)).
- They differ on blocks 2 to 4 by at most τ/4 · |s| per turn (Eq. 40), with τ = 12.7.
- At LR 1e-6 and B_e = 16 prompts per block, we expect per-token-mean log-ratios |s| of order
  1e-3 to 1e-2 on other prompts' turns. That gives a relative difference of a few percent.
- So we expect the two updates to be numerically close and the outcome gap to be small.
- Our B_e of 16 against the paper's 64 should make drift *smaller* (D2). That tilts the test against
  TreeOPSD. We state it here so a null cannot later be dismissed on these grounds without a direct
  check (EXP-1c below).

## 5. Pass and kill bars, mapped to F1 to F5

**Primary metric:** the unweighted MHQA EM average (HotpotQA dev 7,405; 2Wiki dev 12,576; MuSiQue dev
2,417; Bamboogle 125). It is measured at the **fixed final checkpoint, step 100**, with one sample per
question at T = 0.6 and top-p = 0.95, on the full sets.

**CI:** a paired bootstrap over questions, stratified by benchmark, 10,000 resamples. With ≥ 3 seeds,
the seed-level mean ± sd is also reported. Every verdict below is labelled **preliminary** at one seed
and **confirmatory** at three.

### F1: is the posterior loss doing the work (TreePO, same tree)?

- **Mechanism check M1 (decisive at any seed count).** If `drift_rel` on blocks 2 to 4 is below 0.02
  in more than 80% of training steps, the TreeOPSD and TreePO updates are numerically near-identical
  in this setting. Any EM gap is then attributed to noise, and **F1 = KILLED (mechanism inactive at
  this scale)**, whatever the EM shows.
- **KILL:** |TreeOPSD − TreePO| < 0.25, and the 95% CI excludes +0.73 (the paper's gap).
- **PASS:** TreeOPSD − TreePO ≥ +0.5 with the 95% CI excluding 0 (confirmatory), and M1 drift_rel ≥ 0.05.
- **Otherwise:** INCONCLUSIVE at this power.
- **Note on the brief's threshold.** The brief's "within ~1 point" threshold applies to the DR track
  (paper gap +2.90 on BC+, used in EXP-1b). On MHQA the paper's own gap is below 1 point, so a 1-point
  bar would kill the paper's MHQA claim by construction.

### F2: is the budget actually matched?

Measured per prompt over all 100 steps.

- **KILL-F2a:** tree / GRPO ratio > 1.15 in any of generated tokens, tool calls, or train GPU-hours.
  We report which unit failed. We expect GPU-hours to fail (P4). A pass in generated tokens alone
  supports only the narrow claim "matched in sampled tokens".
- **KILL-F2b** (if arm D runs, or a GPU-hour-matched GRPO arm runs in EXP-2): matching GRPO's budget
  removes more than half of the TreeOPSD − GRPO gap, i.e. the matched-GRPO gap is less than 0.5 × the
  G = 8 gap.
- **PASS:** all three ratios ≤ 1.15, or TreeOPSD still beats the GPU-hour-matched GRPO with the CI
  excluding 0.

### F3: fixed checkpoint and seeds vs best-in-window

- **Primary:** the final checkpoint only.
- **Secondary:** the paper's protocol. Evaluate every 5 updates through step 80 on a fixed 500-per-
  benchmark subset (Bamboogle full) and take the best.
- **KILL (confirmatory):** the TreeOPSD − TreePO gap has a 95% CI including 0 at the final checkpoint
  but excluding 0 under best-in-window selection. This would mean the reported gap is a selection
  effect.
- **Also reported:** inflation = best-in-window − final, per arm. If TreeOPSD's inflation exceeds
  TreePO's by more than half the measured final gap, that is flagged.

### F4: retrieval drift

Our retrieval is frozen by construction, so F4 is tested here only as a control.
- Replay 1,000 logged queries on a later day. **PASS** if 100% of top-3 ids are identical.
- Score one checkpoint twice with different eval sampling seeds. If the spread is at least the
  between-arm gap, report the arms as indistinguishable with single-sample eval.
- The paper's live-web numbers (BC / GAIA / xbench) cannot be re-evaluated without the authors'
  checkpoints (none are released). F4 is recorded as **untestable** for those numbers. EXP-1b
  measures the closed-corpus (BC+) gap only.

### F5: horizon

Strata are fixed now and frozen into `data/mhqa/strata.json` before training:
- MuSiQue hops (2 / 3 / 4, from the id prefix);
- 2Wiki `type`;
- HotpotQA `type` (bridge / comparison).

- **KILL (MHQA):** the TreeOPSD − GRPO gain on MuSiQue 3/4-hop is no larger than on 2-hop (point
  estimate; paired-bootstrap CI on the difference reported), **and** TreeOPSD's mean tool calls per
  question is no larger than GRPO's. The paper (Fig. 8) claims TreeOPSD elicits more turns.
- **PASS:** the gain increases with hop count and the CI on the 4-hop vs 2-hop difference excludes 0.
- MHQA caps the horizon at 5 turns, so a F5 kill here is weak evidence. The decisive F5 test is
  EXP-1b (up to 60 turns).

## 6. Overall verdict classes for EXP-1

- **REPLICATED (paper-consistent, MHQA):** at ≥ 3 seeds,
  - TreeOPSD − TreePO ≥ +0.5 and TreeOPSD − GRPO ≥ +0.5, both with 95% CI excluding 0;
  - M1 drift_rel ≥ 0.05;
  - generated-token ratio ≤ 1.15.
- **STRONG (brief's bar):** TreeOPSD ≥ GRPO + 2.0 at ≤ 0.8× generated tokens.
- **KILLED on F_k:** any KILL above, by name.
- **INCONCLUSIVE:** anything else. We then report the measured effect with its CI and the seed
  count needed for 80% power at the paper's effect size.

Either outcome is published, including the telemetry.

## 7. Pre-declared next steps

Following brief section 6, we take the cheapest discriminator first.

- **If M1 shows inactive drift (drift_rel < 0.02):** run **EXP-1c, a "drift-dose" test.** Use the same
  arms B and C with E ∈ {4, 8, 16} sequential blocks at fixed B = 64, or LR ∈ {1e-6, 3e-6}, for 40
  steps each (about 6 GB300-h per pair).
  - TreeOPSD's claimed mechanism predicts that the B-vs-C gap grows with the drift dose. If it does
    not grow even when drift_rel exceeds 0.1, the mechanism claim is killed independently of the
    benchmark.
- **If EXP-1 is null but drift is active:** run the gated EXP-1b (BC+, up to 60 turns, the paper's
  headline setting) before EXP-2. The paper places its effect there.
- **If C > B > A clearly:** run EXP-2, the allocation sweep at fixed measured budget.
  - Iso-budget-8 line: (8, 0), (4, 2), (2, 6).
  - Iso-budget-12 line: (12, 0), (8, 1), (4, 4).
  - Two more seeds of the TreePO control.
  - These replace the brief's {4x1, 4x2, 4x4, 8x1, 2x2}, which mixes nominal budgets of 6, 8, 12,
    12 and 4.
- **If the TreePO control matches TreeOPSD:** pivot to characterising what tree sampling alone buys.
  Compare (4, 2)-TreePO with (8, 0)-TreePO against GRPO.

## 8. Exclusions and stopping

- A run is excluded only for an infrastructure failure: a crash, a retriever outage longer than 1%
  of calls, or a data-hash mismatch. Exclusion is logged with its cause.
- A run is never excluded for its metric value.
- A crashed run restarts from its last checkpoint with the same seed. The restart is logged.
- No early stopping. Every arm trains exactly 100 updates.

## AMENDMENT 1 (2026-10-21, ~09:45 PT; before any training or evaluation run)

**Source:** binding decisions from the research session (inbox 001), relayed by the controller. No
results had been seen when this was written.

1. **Headline metric = the frozen BrowseComp-Plus slice (tier 2).**
   - The slice is all 830 BC+ test queries, decrypted with the official canary scheme. File:
     `prepared/bcplus/eval_bcplus.parquet`, sha256 `ecb2bd6a…3f2c`, query-id list sha256 `74e81cb2…2b1a`.
   - Retrieval is frozen: the fixed 100,195-document BC+ corpus, with Qwen3-Embedding-8B released
     document vectors and query encoder (`Qwen/Qwen3-Embedding-8B@1d8ad4ca…`).
   - The agent interface is the same `search` tool as training (query_list → top-3 documents, each
     truncated to 2,500 characters). It is not the paper's `browser.search/open/find` tools.
   - Decoding is greedy (as in the paper's BC+ eval), with at most 20 assistant turns and a
     16,384-token response budget.
   - Every arm is evaluated identically at the fixed final checkpoint.
   - Paper anchor: 4B BC+ is TreeOPSD 13.98, TreePO 11.08, GRPO 10.00. Those models were trained on
     OpenResearcher with browser tools. Ours are trained on HotpotQA, so tier 2 measures *transfer*,
     and absolute levels are not comparable to the paper.
2. **The QA average (tier 1) becomes the secondary metric.** All section 5 bars for F1/F3/F5 are
   evaluated on **both** tiers. A verdict holds for the headline only if it holds on tier 2.
   - On tier 2 the F1 bar is: KILL if TreeOPSD − TreePO < 1.0 EM (the brief's threshold, which is
     appropriate where the paper claims +2.90). PASS if it is ≥ +1.5 with the paired-bootstrap 95%
     CI excluding 0.
   - BC+ has 830 questions, so a single-sample paired comparison has an EM standard error of about
     1.0 to 1.5 points at the expected accuracy levels.
3. **F2 budget telemetry is logged from run 1.** This covers generated tokens per prompt, tool calls
   per prompt, trainer tokens per prompt, step wall time and GPU-hours
   (`treeopsd/verl_ext/advantage.py` → `telemetry/steps.jsonl` → `summary.json` → `METRIC` lines). It
   applies to every arm, GRPO included.
4. **One model for all arms:** `Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea`,
   from the identical checkpoint. Any substitution must be logged prominently.
5. **Execution order (controller):** seed 0 of arms A, B and C first, then seeds 1 and 2 of the arms
   that matter. The decision rule stays as in section 7.
