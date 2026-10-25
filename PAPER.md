# TreeOPSD as specified in the paper

Source: Heming Zou, Qi Wang, Yixiu Mao, Yun Qu, Yuyi Guo, Zeyuan Liu, Chenglin Li,
Weijie Liu, Saiyong Yang, Xiangyang Ji. *Tree-Aware Posterior Distillation for
Long-Horizon Agentic Reinforcement Learning.* arXiv:2610.23181v1 [cs.LG], submitted
16 Oct 2026 (PDF dated 19 Oct 2026). Tsinghua University and Tencent.

Read from the arXiv PDF and HTML on 2026-10-21. Equation numbers below follow the
paper. Section 9 lists the ambiguities, labelled A1 to A20. DESIGN.md says how we
resolve each one.

---

## 1. Problem setting (Sec. 2.1)

The setting is a ReAct agent. At each turn `t < T` the policy emits an actor span `a_t`
(a thought plus a tool call). The environment returns an observation `o_t`, and the
history becomes `H_t = (H_{t-1}, a_t, o_t)`. The final turn `a_T` is the answer.

    P_θ(H_T | x) = π_θ(a_T | H_{T-1}) · Π_{t=1}^{T-1} π_θ(a_t | H_{t-1}) P_env(o_t | H_{t-1}, a_t)      (1)

Observation tokens are fixed context and masked out of the loss. The reward `R(H_T)` is in
[0, 1] and rule-based. There is no reference-KL penalty (`β_KL = 0`).

## 2. Initialize-then-expand tree (Sec. 2.2, Eq. 3; App. E.4)

For each prompt `x`, the frozen behavior policy `π_θold`:

1. Samples `M` complete trajectories `H_{m,T_m}`.
2. For each trajectory `m`, draws a **multiset** `A_m = {t_{m,1}, ..., t_{m,N}}` of `N`
   nonterminal anchors, uniform on `1 ≤ t < T_m`. If `T_m = 1` the anchor is `t = 0`.
3. For each anchor, samples **one fresh terminal suffix** `S_{m,n} ~ P_θold(· | H_{m,t_{m,n}})`.
   The prefix includes `a_t` and `o_t`, so the suffix starts at turn `t+1`.

Keeping the roots gives `K := M(N+1)` terminal trajectories. The defaults are `M=4, N=2`, so `K=12`.

Budget claim, quoted: "Because a uniformly branched continuation starts mid-trajectory,
its expected generation cost in tokens or tool calls is roughly half that of a full
rollout ... an (M, N) tree matches the expected rollout budget of G full trajectories
in GRPO when **M(1 + N/2) ≈ G**". The comparison is (4, 2) against G = 8.

Shared prefixes are merged into a tree with node set `T_x`. Node `i` is one actor span `a_i` of
`L_i` tokens with pre-turn history `h_i` and child set `C_i`. Each leaf `i_k` has reward `R_k`.

## 3. Reward backup (Sec. 2.3, Eqs. 4-6)

    A_k := (R_k − mean({R_k'})) / std({R_k'}),  k = 1..K        (4)

The std is the **sample** standard deviation. `A_k := 0` when all rewards are equal.

    ω_{i,c} := π_θold(a_c | h_c) / Σ_{c'∈C_i} π_θold(a_{c'} | h_{c'}),
    V_i := A_k              if i = i_k (leaf)
    V_i := Σ_{c∈C_i} ω_{i,c} V_c   otherwise                    (5)

Here "`π_θold(a_c | h_c)` is the behavior-policy likelihood of child span `a_c` given the shared
post-observation history `h_c = (h_i, a_i, o_i)`". Each unique turn enters the gradient once.
`L_x := Σ_{i∈T_x} L_i` is the number of unique actor tokens:

    g_x^tree := (1/L_x) Σ_{i∈T_x} Σ_{j=1}^{L_i} V_i ∇_θ log π_θ(a_{i,j} | h_{i,j}) |_{θ=θold}   (6)

## 4. Behavior-relative turn posterior (Sec. 3.1, Eqs. 8-9)

    s_{θ,i} := (1/L_i) Σ_j log[ π_θ(a_{i,j}|h_{i,j}) / π_θold(a_{i,j}|h_{i,j}) ],
    r_{θ,i} := e^{s_{θ,i}} = (π_θ(a_i|h_i)/π_θold(a_i|h_i))^{1/L_i}            (8)

    p_{θ,i} := r_{θ,i} / (1 + r_{θ,i}) = σ(s_{θ,i}),   P_{θ,i} := Ber(p_{θ,i})   (9)

The paper says it uses "only the logged probabilities of the sampled tokens rather than
vocabulary-wide behavior logits".

## 5. Closed-form bounded teacher (Sec. 3.2, Eqs. 10-12; App. B.1, Eqs. 18-25)

Proximal objective in Bernoulli coordinates:

    J_i(Ber(q)) := V_i log(q/(1−q)) − τ D_KL(Ber(1/2) ‖ Ber(q))               (10)

In log-odds, `J_i(s) = V_i s − τ log cosh(s/2)`. Its unique maximiser is

    q_i := 1/2 + V_i/τ,     r_i* := q_i/(1−q_i) = (τ + 2V_i)/(τ − 2V_i)        (11)

**Temperature calibration (Samuelson).** The standardised advantages satisfy
`|A_k| ≤ B_K := (K−1)/√K`, and convex backup preserves this bound. Setting

    τ = (K − 1)/(ρ √K) = B_K / ρ,    ρ ∈ (0, 1/2)                              (Alg. 1, Eq. 18)

confines `q_i` to `[1/2 − ρ, 1/2 + ρ]` and

    r_i* ∈ [ (1−2ρ)/(1+2ρ), (1+2ρ)/(1−2ρ) ]                                    (12)

The default is ρ = 0.25. With K = 12 this gives τ = 12.7017, q ∈ [0.25, 0.75] and r* ∈ [1/3, 3].
App. C.4 adds that the RMS of `q_{i_k} − 1/2` over leaves is `ρ/√(K−1)`, about 0.075 at the
defaults. Only one dissenting reward reaches the full radius. App. B.1 shows that the affine
map is the unique symmetric, radius-calibrated map that commutes with convex backup (Eq. 22).
The alternative "exponential-tilting" map `q = σ(V/τ')` does not commute with backup.

## 6. Forward-KL distillation loss (Sec. 3.3, Eqs. 13-17; App. B.3)

    L_x(θ) := (τ/L_x) Σ_{i∈T_x} L_i D_KL(Q*_i ‖ P_{θ,i})
            = (1/L_x) Σ_i L_i [ −V_i s_{θ,i} + τ D_KL(Ber(1/2) ‖ P_{θ,i}) ] + C_x     (13)

Here `Q*_i = Ber(q_i)` is frozen at rollout time, and `D_KL(Ber(1/2)‖Ber(σ(s))) = log cosh(s/2)`.

Theorem 1 (Eqs. 14-15): the forward KL to the backed-up parent teacher has the same gradient
as the backup-weighted sum of the descendant-leaf KLs. Also `q_i = Σ_c ω_{i,c} q_c`.

Theorem 2 (Eqs. 16-17), per-token gradient:

    ∂L_x/∂ log π_θ(a_{i,j}|h_{i,j}) = (τ/L_x)(p_{θ,i} − q_i) = −(1/L_x)[V_i − τ(p_{θ,i} − 1/2)]   (16)
    Γ_i(s) := V_i − τ(σ(s) − 1/2) = τ (r_i* − r_{θ,i}) / ((1 + r_i*)(1 + r_{θ,i}))           (17)

The theorem's three properties are:

- (i) At `θ = θold`, `−∇L_x = g_x^tree` exactly.
- (ii) `|Γ_i(s) − V_i| ≤ (τ/4)|s|` and `|Γ_i| ≤ (1/2 + ρ)τ` (Eq. 36).
- (iii) `Γ_i` vanishes at `r_i*` and changes sign past it.

Eq. 40: `‖g^OPSD − g^fixed‖ ≤ τ/(4L_x) Σ_i |s_{θ,i}| ‖d_i(θ)‖`. TreeOPSD and the
fixed-credit control (TreePO) are therefore **identical on the first optimizer block**. They
can differ only in proportion to the policy drift `|s|` built up over blocks 2..E.

## 7. Algorithm and practical optimisation (Alg. 1; Sec. 3.4)

Each iteration:

1. Freeze `π_θold`.
2. Sample `B` prompts and split them into `E` disjoint mini-batches.
3. Set `K = M(N+1)` and `τ = (K−1)/(ρ√K)`.
4. For each prompt: build the (M, N) tree, standardise rewards once, back up `V_i`, and set
   `q_i = 1/2 + V_i/τ`.
5. For `e = 1..E`, take one optimizer step per block. For every unique turn, recompute `s_{θ,i}`
   and `p_{θ,i}` under the current `π_θ`, give each actor token the loss-gradient weight
   `τ(p_{θ,i} − q_i)/L_x`, and backpropagate "the prompt-averaged token gradients".

## 8. Experimental protocol (Sec. 4.1; App. E)

### 8.1 Models (App. E.2)

| role | HF id | revision (HEAD as of 2026-10-21) |
|---|---|---|
| main | Qwen/Qwen3-4B-Thinking-2507 | 768f209d9ea81521153ed38c47d515654e938aea |
| main | Qwen/Qwen3-30B-A3B-Thinking-2507 | 144afc2f379b542fdd4e85a1fcd5e1f79112d95d |
| App. D.1 | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| App. D.1 | meta-llama/Llama-3.2-3B-Instruct (gated) | 0cb88a4f764b7a12671c53f0838cd831a0843b95 |

The paper does not give revisions. The hashes above are today's HEADs, which we will pin.

### 8.2 Multi-Hop QA (App. E.1.1). Follows Search-R1

- **Train:** all 90,447 questions of the HotpotQA *distractor* train split.
- **Environment:** a local E5 retrieval server over the Wikipedia dump of Karpukhin et al. (2020),
  i.e. Search-R1's "wiki-18" (App. F.1: "fixed Wikipedia-2018 index").
  - One `search` tool: `query_list` of 1 to 4 queries, each at most 512 characters, top-3 passages
    per query.
  - At most 5 assistant turns including the answer, so at most 4 tool-calling turns.
  - Prompt: the Search-R1 instruction ("Answer the given question. You must conduct reasoning inside
    <think> ... <answer> Beijing </answer>"). Tool calls use the Hermes `<tool_call>{json}</tool_call>`
    format of the Qwen3 chat template (App. F.1 gives the full system prompt and tool schema).
- **Eval:** HotpotQA validation (7,405), 2WikiMultiHopQA dev (12,576), MuSiQue dev (2,417) and
  Bamboogle test (125).
- **Reward:** training uses *thresholded token F1*: the F1 if it is ≥ 0.3, else 0. Evaluation
  reports EM.
- **Lengths:** prompt ≤ 2,048, response ≤ 6,144, retrieved observations ≤ 16,384 characters.
- **Batching:** 256 prompts per collection, split into E = 4 blocks of 64.
- **Eval decoding:** T = 0.6, top-p = 0.95, one sample per question, every 5 updates.

### 8.3 Deep Research (App. E.1.2). Follows OpenResearcher

- **Train:** 6,102 unique questions from OpenResearcher (MiroVerse questions). One prompt per
  question, with no teacher trajectories in the input.
- **Environment:** closed-corpus retrieval over the OpenResearcher corpus, through the tools
  `browser.search` (top-N, default 10), `browser.open` and `browser.find`.
- **Eval:** BrowseComp-Plus, all 830 test questions, against its fixed corpus (closed). BrowseComp
  (1,266), GAIA text-only dev (103) and xbench-DeepSearch (100) run **live through a Serper web
  backend**. "To limit temporal variation in live search, we interleave the open-web methods within
  the same evaluation window and archive the raw search responses with each generation."
- **Reward:** EM, plus a 0.1 bonus for a well-formed final-answer span. Tables report EM without
  the bonus.
- **Lengths:** prompt 4,096, response 16,384 and tool observation 2,048 tokens. 60 assistant turns in
  training, 80 in BC+ eval.
- **Batching:** 128 prompts per collection, split into 4 blocks of 32.
- **Final DR eval:** greedy, one rollout per question, at most 200 assistant turns and a 131,072-token
  response budget.

### 8.4 Optimisation and systems (App. E.3)

- bfloat16 throughout, with **SGLang** rollout on one 8-GPU node (96 GB H20).
- 4B uses FSDP. 30B-A3B uses FSDP for MHQA and FSDP2 for DR, with rollout TP = 4.
- 100 policy updates. AdamW with constant LR 1e-6 and weight decay 0.01. Betas are (0.9, 0.999) for
  MHQA and (0.9, 0.98) for DR.
- No KL penalty and no entropy bonus. Rollouts use T = 1.0 and top-p = 1.0.
- TreeOPSD settings: M = 4, N = 2, likelihood-weighted backup, ρ = 0.25, E = 4 sequential
  forward-KL blocks, unique-turn reconstruction.
- "Every trainable method applies four actor optimizer blocks per collected prompt batch."

### 8.5 Checkpoint selection (App. E.3)

"We evaluate every five policy updates through step 80 on Multi-Hop QA and step 70 on
BrowseComp-Plus, and **report one checkpoint per run in that window**."

### 8.6 Baselines (Sec. 4.1; App. E.4)

- **ReAct:** the untrained initialisation in the same scaffold.
- **GRPO:** G = 8, rewards standardised over complete rollouts.
- **GSPO:** G = 8, sequence-level (geometric-mean) importance ratio, group-relative advantages.
- **TreePO:** "uses the same initialize-then-expand trees, likelihood-weighted credits `V_i`,
  unique-turn reconstruction, prompt normalization, and optimizer blocks as TreeOPSD, but applies the
  fixed ascent coefficient `V_i/L_x` throughout optimization instead of matching a finite posterior
  target."
  - This is the paper's own fixed-credit control. It is **not** the TreePO of Li et al. 2025
    (arXiv:2508.17445, "heuristic tree-based modeling"), which shares the name.

## 9. Reported numbers

### Table 1 (EM)

| | Hotpot | 2Wiki | MuSiQue | Bamb. | **MHQA Avg** | BC+ | BC | GAIA | xbench | **DR Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| *Qwen3-4B-Thinking-2507* | | | | | | | | | | |
| ReAct | 31.61 | 29.63 | 9.68 | 25.60 | 24.13 | 4.70 | 1.97 | 15.53 | 13.00 | 8.80 |
| GRPO | 46.93 | 48.62 | 21.93 | 56.00 | 43.37 | 10.00 | 2.29 | 18.45 | 18.00 | 12.18 |
| GSPO | 46.55 | 49.10 | 21.76 | 55.20 | 43.15 | 9.88 | 2.37 | 18.45 | 18.00 | 12.17 |
| TreePO | 47.25 | 49.90 | 22.05 | 54.40 | 43.40 | 11.08 | 2.76 | 19.42 | 19.00 | 13.07 |
| TreeOPSD | 48.16 | 52.00 | 22.76 | 53.60 | **44.13** | 13.98 | 3.48 | 21.36 | 23.00 | **15.45** |
| *Qwen3-30B-A3B-Thinking-2507* | | | | | | | | | | |
| ReAct | 41.49 | 42.89 | 17.87 | 46.40 | 37.16 | 7.23 | 3.63 | 30.10 | 18.00 | 14.74 |
| GRPO | 47.25 | 52.13 | 23.83 | 56.00 | 44.80 | 10.36 | 4.50 | 31.07 | 19.00 | 16.23 |
| GSPO | 47.70 | 52.50 | 24.00 | 55.20 | 44.85 | 11.08 | 4.74 | 32.04 | 18.00 | 16.47 |
| TreePO | 48.10 | 51.80 | 24.29 | 56.00 | 45.05 | 12.29 | 5.06 | 33.01 | 20.00 | 17.59 |
| TreeOPSD | 48.89 | 50.88 | 24.74 | 59.20 | **45.93** | 17.47 | 6.40 | 34.95 | 26.00 | **21.20** |

**The 30B table does report a per-benchmark Deep Research breakdown.** The study brief said it did
not, and that was wrong. The 4B rows of Table 1 equal the five-seed means in Table 4, so the 4B
numbers are averages over seeds. Each seed contributes its own selected checkpoint.

### Effect sizes the paper itself predicts at 4B

| Comparison | MHQA Avg | HotpotQA | BC+ (closed corpus) | DR Avg |
|---|---|---|---|---|
| TreeOPSD − GRPO | +0.76 | +1.23 | +3.98 | +3.27 |
| TreeOPSD − TreePO | +0.73 | +0.91 | +2.90 | +2.38 |

These numbers set the bars in PREREG-EXP1.md. Two points follow from them:

- **On MHQA at 4B the paper claims less than one point.** A +2 EM bar on HotpotQA asks for more than
  the paper claims.
- Bamboogle (125 questions) goes the wrong way at 4B (TreeOPSD 53.60 vs GRPO 56.00), and 2Wiki goes
  the wrong way at 30B.

### Table 4 (Qwen3-4B, five training seeds, mean ± std)

| | MHQA | DR |
|---|---|---|
| ReAct | 24.13 ± 0.05 | 8.80 ± 0.26 |
| GRPO | 43.37 ± 0.19 | 12.18 ± 0.27 |
| GSPO | 43.15 ± 0.19 | 12.17 ± 0.32 |
| TreePO | 43.40 ± 0.17 | 13.07 ± 0.37 |
| TreeOPSD | 44.13 ± 0.16 | 15.45 ± 0.39 |

The paper says: "the remaining variation comes from which rollouts were collected and **which
checkpoint in the evaluation window was reported**."

### Table 2 (Qwen3-4B, averages)

**(a) Allocation at ρ = 0.25.** The first three rows are at full budget (8); the last two are at
half budget (4).

| (M, N) | MHQA | DR |
|---|---|---|
| (8, 0) | 43.66 | 11.28 |
| (4, 2) | 44.13 | 15.45 |
| (2, 6) | 44.07 | 14.97 |
| (4, 0) | 42.98 | 10.05 |
| (2, 2) | 43.25 | 11.33 |

**(b) ρ sweep at (4, 2).**

| ρ | MHQA | DR |
|---|---|---|
| 0.10 | 43.79 | 14.62 |
| 0.20 | 44.01 | 15.06 |
| 0.25 | 44.13 | 15.45 |
| 0.30 | 43.95 | 14.88 |
| 0.40 | 43.81 | 14.69 |

### Table 3 (MHQA, App. D.1)

| | Hotpot | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---|---|---|---|---|
| *Qwen3-1.7B* | | | | | |
| ReAct | 23.04 | 26.11 | 6.37 | 19.20 | 18.68 |
| GRPO | 29.10 | 31.48 | 8.61 | 33.60 | 25.70 |
| GSPO | 29.09 | 31.85 | 8.52 | 32.80 | 25.57 |
| TreePO | 29.86 | 32.18 | 8.90 | 33.60 | 26.13 |
| TreeOPSD | 31.30 | 33.81 | 9.60 | 32.80 | 26.88 |
| *Llama-3.2-3B-Instruct* | | | | | |
| GRPO | 36.80 | 36.35 | 13.78 | 40.00 | 31.73 |
| TreePO | 37.56 | 37.43 | 14.23 | 40.00 | 32.30 |
| TreeOPSD | 39.59 | 41.15 | 15.97 | 45.60 | 35.58 |

**Fig. 8 (30B, BC+).** Average assistant turns are about 2 to 7 at eval and about 1.5 to 3.5 in
training, after dividing the serialised count by two. So the "long-horizon" DR setting averages a
handful of turns. Panel (c) claims TreeOPSD stays ahead of GRPO at equal training *time*. That
comparison is wall-clock and is not a token audit.

## 10. Ambiguities and gaps that change the experiment

**A1. Backup weights `ω`.** Eq. 5 normalises the *full-span* likelihoods `π_θold(a_c|h_c)` (the
product over tokens) across siblings.
- For spans of hundreds of thinking tokens, sibling log-likelihoods differ by tens of nats, so `ω`
  is almost one-hot on the more likely sibling. The backup then mostly copies one child's value.
  The test `test_likelihood_weights_are_extremely_peaked_for_long_spans` shows this.
- It is unclear whether the authors use summed or length-normalised log-probs.
- *Changes the experiment:* this decides how much credit is actually shared. We implement the
  literal version, log the entropy of `ω`, and keep `geomean` and `uniform` as ablations.

**A2. TreePO's update rule.** "Applies the fixed ascent coefficient `V_i/L_x` throughout
optimization."
- Read literally, this is a REINFORCE-style `−V_i log π_θ` on stale data, with no importance ratio
  and no clipping.
- A PPO-clipped surrogate with `V_i` as the advantage is an equally natural reading, and it would
  be much more stable on blocks 2 to 4.
- *Changes F1:* if the control is unclipped and has no ratio, part of TreeOPSD's margin could be
  "anything bounded beats something unbounded". We run the literal version as primary and
  PPO-clip-with-V as a secondary control.

**A3. GRPO and GSPO details are unspecified.** The paper does not give:
- the clip ε (and whether clip-higher is used);
- loss aggregation (token-mean, seq-mean-token-mean or prompt-normalised);
- gradient clipping;
- whether GRPO uses std-normalisation with ddof = 1.

**A4. Source of `log π_θold`.** "Logged probabilities of the sampled tokens" could mean SGLang's
sampling log-probs or a trainer recompute.
- With engine log-probs, the engine–trainer mismatch makes `s ≠ 0` on block 1, so Theorem 2(i)
  ("exact tangent") does not hold numerically.
- We recompute with the trainer (verl's `old_log_prob`) and log the mismatch against the vLLM
  log-probs.

**A5. Checkpoint selection rule.** "Report one checkpoint per run in that window" (every 5 updates,
through step 80 for MHQA and step 70 for BC+). The rule is not stated (best? last?) and the
selection split is not stated (the test set?). This is F3.

**A6. 30B seed count.** It is not stated. Table 4 covers only 4B, so the 30B numbers look like a
single run.

**A7. "Deep Research Avg" mixes closed-corpus and live-web benchmarks.**
- Only BC+ is reproducible with frozen retrieval. BC, GAIA and xbench go through live Serper.
- The archived search responses are not released.
- Because the average is unweighted, GAIA (103 q) and xbench (100 q) carry half its weight. On
  those sets one question moves the score by about 1 point (the paper says so in App. D.4).

**A8. DR retriever.** The paper does not say which retriever backs `browser.search` for the
OpenResearcher corpus and for BC+ (BM25? Qwen3-Embedding-8B, as in OpenResearcher's release?). Nor
does it give the viewport size of `browser.open`.

**A9. Answer normalisation.** EM and F1 normalisation are not given. We use the SQuAD/Search-R1
normaliser and take the max over gold aliases (FlashRAG `golden_answers`).

**A10. Thinking mode for Qwen3-1.7B.** Qwen3-1.7B is the hybrid-thinking checkpoint and the paper
does not say whether `enable_thinking` is on. The 4B and 30B models are Thinking-2507 (always
thinking).

**A11. Single-turn rollouts.** "`t_{m,n} = 0` if `T_m = 1`." A continuation from anchor 0 is a
full new rollout.
- If many roots answer without searching, the tree costs up to `M(1+N) = 12` trajectory equivalents,
  not 8. See `test_budget_breaks_when_rollouts_are_single_turn`.
- The 2507 Thinking models often answer HotpotQA comparison questions from memory, so this matters
  for F2.

**A12. "Half the cost" holds only for flat per-turn cost profiles.**
- When turn 1 is front-loaded (long first thought), continuations cost less than half.
- When the answer turn is long, they cost more.
- Measured in tool calls, a suffix costs `T/2 − 1` against `T − 1`, so the claim does not hold in
  that unit either.
- See `test_budget_front_loaded_and_tool_call_units`.

**A13. Trainer cost is not budget-matched.** Unique-turn reconstruction removes duplicate *loss*
terms, but the paper never says it shares prefix *compute*.
- Packed one row per leaf, the trainer runs forward/backward on 12 full-length rows against GRPO's 8.
  That is about 1.5× the actor-update FLOPs.
- The paper's equal-time plot (Fig. 8c) covers only 30B on BC+.

**A14. Response length cap.** It is unclear whether the 6,144 (MHQA) / 16,384 (DR) response cap
counts observation tokens or only generated tokens.

**A15. Truncation.** The reward for rollouts that hit the turn or length cap without an
`<answer>` is not stated. We assume 0.

**A16. Anchor RNG and continuation sampling.** The paper does not say whether continuations use
the same sampling temperature (1.0, presumably) or whether duplicate anchors in the multiset are
sampled independently (presumably yes).

**A17. Realised K.** If a rollout errors or is dropped, the realised K differs from M(N+1). The
paper fixes `τ` from the nominal K.

**A18. Evaluation set sizes for curves.** Evaluating "every five updates" on the full 22,523
MHQA questions is costly. Whether the curves use the full sets or subsets is not stated.

**A19. Framework.** The paper says SGLang plus FSDP/FSDP2 and "we follow Search-R1", which suggests
a verl-family codebase. No code is released (Sec. 11).

**A20. GRPO's "four optimizer blocks".** This presumably means `ppo_mini_batch = B/4` prompts and
`ppo_epochs = 1`. GRPO then sees 8 trajectories per prompt per block, and the tree methods see 12
leaves.

## 11. Code availability (checked 2026-10-21)

- **No official code.**
  - The arXiv abstract and HTML pages contain no code link.
  - The HF paper page (`huggingface.co/papers/2610.23181`, org Tencent-Hunyuan, 46 upvotes) lists
    0 linked models, datasets or spaces.
  - GitHub repository search for "TreeOPSD", "tree-aware posterior distillation" and
    "posterior self-distillation agentic" returns 0 results.
  - The first author's GitHub (`gfyddha`) has FlyLoRA, Fly-CL and UDS but no TreeOPSD.
  - HF model and dataset search for "TreeOPSD" returns nothing.
- **Closest public code, pinned for reference:**

| repo | what | commit |
|---|---|---|
| verl-project/verl | RL framework (GRPO/GSPO losses, `ToolAgentLoop`, vLLM/SGLang rollout, FSDP2) | v0.9.1 = `1876b06d0a3e4e71e06230be10af14492ca8a75b` (2026-09-20); main HEAD `f2923cbf154731918b1e36db8c567433f349fede` |
| PeterGriffinJin/Search-R1 | MHQA env, wiki-18 E5 retriever server, data scripts | `598e61bd1d36895726d28a8d06b3a15bed19f5d3` |
| AMAP-ML/Tree-GRPO | "Ji et al., 2026" tree rollouts for agent RL on Search-R1 (old verl fork, vLLM 0.8.5); reference for tree sampling | `19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59` |
| multimodal-art-projection/TreePO | the *other* TreePO (Li et al. 2025) | `f05552394e734ea2838cf9e3b8191f5093319dcf` |
| TIGER-AI-Lab/OpenResearcher | DR data, browser tools, BC+ eval | `785fd6ba5fcbc068daa4a2f07bbe0964f2983c86` |
| texttron/BrowseComp-Plus | BC+ benchmark code | `046949032b0328319cc9a02663a759ec601d9402` |
| OpenRLHF/OpenRLHF | alternative framework | v0.11.2; HEAD `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53` |
| THUDM/slime | alternative framework | v0.4.0; HEAD `a4202e0595467d9374d9a68d45a34f45c3f8d230` |

TRACE (Zou et al. 2026a), which the paper cites for tree acquisition, has no public repository
that we could find.
