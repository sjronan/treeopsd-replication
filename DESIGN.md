# Implementation plan

This document says how we turn PAPER.md into runs on two DGX Station GB300s and three DGX Sparks
without changing what is being tested. Every deviation from the paper is listed in section 10.

## 0. What the paper lets us test, and where

| Track | Paper setting | Frozen? | Paper's own 4B effect (TreeOPSD − GRPO / − TreePO) | Cost | Role here |
|---|---|---|---|---|---|
| **MHQA** | HotpotQA train; E5 over wiki-18; ≤5 turns | fully frozen, identical to the paper | +0.76 / +0.73 EM avg | cheap | **EXP-1**: mechanism, budget audit, F1/F2/F3/F5 |
| **BC+** | OpenResearcher questions; closed corpora; ≤60 turns | frozen (corpus fixed) | +3.98 / +2.90 EM on BC+ | 2 to 4× MHQA per step | **EXP-1b** (gated): where the headline lives |
| open-web DR | BC / GAIA / xbench via live Serper | **not frozen** | part of the 15.45 vs 13.07 avg | — | out of scope (F4) |

The study brief framed EXP-1 as "Deep Research proxy EM on HotpotQA" with success = "+2 EM at ≤80%
rollout tokens". On the paper's own numbers that bar is about 2.6× the claimed MHQA gain, at a
budget the paper does not claim. PREREG-EXP1.md therefore keeps the brief's bar as a "strong"
outcome but sets pass/kill bars from the paper's stated effects.

## 1. Framework: verl v0.9.1

**Choice: verl, commit `1876b06d0a3e4e71e06230be10af14492ca8a75b` (tag v0.9.1, 2026-09-20).**

Reasons:
- **Closest to the paper's stack.** The paper "follows Search-R1" (a verl codebase), uses FSDP/FSDP2,
  and evaluates "every five updates". The only public tree-rollout agent-RL code it cites, Tree-GRPO
  (Ji et al.), is also a verl fork on Search-R1.
- **The baselines are built in.**
  - GRPO is `AdvantageEstimator.GRPO`.
  - GSPO is `@register_policy_loss("gspo")`.
  - Both the policy-loss and advantage-estimator registries are extension points. TreeOPSD and TreePO
    become two registered losses over the same advantage tensor.
- **Multi-turn agent rollout is token-in/token-out.** `verl/experimental/agent_loop/tool_agent_loop.py`
  appends tokens without re-rendering the chat template. A continuation can therefore resume from an
  exact token prefix, which the tree needs. The loop already masks observations
  (`response_mask = 0`) and returns `response_logprobs`.
- **aarch64 is a documented build target.** `docker/README.md` covers GB200/aarch64, built natively.
  It uses vLLM ≥ 0.18 on CUDA 13 / torch 2.11.

| Alternative | Why not the primary |
|---|---|
| TRL (`huggingface/trl`) | Its GRPO trainer centres on single-turn generation. Tree rollouts plus tool loops would mean writing the whole rollout layer, and nothing in it is closer to Search-R1. |
| OpenRLHF v0.11.2 (`dc2a7ad3`) | Capable and supports agents, but it has no Search-R1 lineage and no shipped GSPO loss. It is no better on aarch64. |
| slime v0.4.0 (`a4202e05`) | Megatron + SGLang. Strong for the 30B-A3B MoE stage (EXP-7). Megatron-on-aarch64 is more build risk than this study needs at 4B. Revisit for EXP-7. |

**Rollout engine.** We use vLLM; the paper uses SGLang. vLLM is the engine with a known-good arm64
image on this hardware. The engine only affects sampling numerics, so this deviation is accepted
(D1 in section 10).

**Versions.** Fixed by `env/Dockerfile`:
- base `vllm/vllm-openai@sha256:befccf0e…` (v0.32.0; arm64 manifest `sha256:6ff5ff0d…`; CUDA 13.0.2;
  Python 3.12);
- verl at the commit above, installed `--no-deps` under constraints frozen from the base image;
- `faiss-cpu==1.12.0`; `liger-kernel>=0.8.2`;
- flash-attn optional (`BUILD_FLASH_ATTN=1`).

verl v0.9.1's own lockfile was tested with vLLM 0.24.0 (x86) and 0.20.2 (aarch64). Running it on
vLLM 0.32.0 is gate **G1**. The fallbacks are the nightly vLLM 0.30.1rc1 image, then verl's
`docker/Dockerfile.stable.vllm` built natively.

## 2. Code layout and integration plan

This repo is mounted at `/workspace/treeopsd` with `PYTHONPATH` set. We do not fork verl.
Everything plugs in through registries. One small, reviewed patch file lives in `env/patches/`
(applied with `git apply --check` at build time).

```
treeopsd/core.py              # DONE: reference math (Eqs. 4-17, budget, ownership). Single source of truth.
treeopsd/metrics.py           # DONE: answer extraction, EM, thresholded-F1 reward.
treeopsd/verl_ext/
  tree_agent_loop.py          # @register("tree_tool_agent"): subclass of ToolAgentLoop
  tree_rollout.py             # two-phase (roots -> anchors -> continuations) driver
  tree_advantage.py           # @register_adv_est("tree_backup"): V_i per token, owner mask, L_x
  losses.py                   # @register_policy_loss("treeopsd_fkl"), ("treepo_fixed"), ("treepo_clip")
  telemetry.py                # drift, Gamma-V, omega entropy, budget counters -> summary.json
  search_tool.py              # BaseTool: query_list (<=4, <=512 chars) -> top-3 passages
  main_tree.py                # entry point: RayPPOTrainer subclass wiring the above
retriever/server.py           # FastAPI exact-IP server (torch.mm over reconstructed e5 vectors)
retriever/check_vs_faiss.py   # gate G3
scripts/prepare_mhqa.py       # FlashRAG jsonl -> verl parquet (prompt per App. F.1) + strata.json
scripts/eval_mhqa.py          # vLLM server + same tool loop, writes runs/<id>/summary.json
configs/{grpo,treepo,treeopsd,grpo_tokmatch}_mhqa_4b.yaml
env/patches/verl-v0.9.1-treeopsd.patch   # see 2.4
```

### 2.1 Tree rollout (`tree_agent_loop.py`, `tree_rollout.py`)

**Phase A.** Generate `M` root trajectories per prompt with the standard tool loop
(`rollout.n = M`). For each assistant turn, record:
- its `[start, end)` offsets in the response;
- whether it emitted a tool call;
- its token count.

These go into `extra_fields`.

**Anchors.** `treeopsd.core.sample_anchors(T_m, N, rng)` with
`rng = np.random.default_rng(hash(seed, global_step, prompt_uid, m))`. Draws are uniform on
`{1..T_m−1}` with replacement, or `0` when `T_m = 1` (A11). Every anchor is logged.

**Phase B.** For each anchor, submit a request whose non-tensor field `tree_prefix` carries:
- `prompt_ids` up to and including `o_t` and its chat-template wrapper;
- `response_mask` and `response_logprobs` for that prefix;
- `assistant_turns = t` and `user_turns`.

`TreeToolAgentLoop.run` then starts in `GENERATING` instead of `PENDING`. It uses the same sampling
params (T = 1.0, top-p = 1.0), the same `max_assistant_turns = 5` counted from the root, and the same
response cap. vLLM prefix caching is on (`enable_prefix_caching=True`), so the shared prefix is not
re-prefilled when the cache still holds it.

**Merge.** Build one row per leaf (`K = M(N+1)` rows per prompt). Each row is the full trajectory:
the prefix followed by the suffix. Each row carries:
- `tree_uid` (the prompt);
- `turn_id` per response token (−1 on observation tokens);
- `node_id` per turn, where a shared turn has the same id in every row containing it.

**Ownership.** `treeopsd.core.owner_rows` assigns each unique node to one row. In every other row
the node's tokens get loss-mask 0. This is the paper's "unique-turn reconstruction": each turn is
scored once, but every row keeps its full context for the forward pass.

### 2.2 Credit (`tree_advantage.py`)

This runs after verl computes `old_log_prob` with the actor (FSDP recompute). That is our resolution
of A4: at block 1, `s ≡ 0` up to kernel nondeterminism. We also log the gap
`|old_log_prob − rollout_log_probs|` against vLLM.

1. Sum `old_log_prob` over each node's tokens to get `log π_old(a_i|h_i)`.
2. Rebuild the `Tree`.
3. `V = treeopsd.core.backup(tree, weighting=cfg.omega)`. The weighting is `likelihood` (literal
   Eq. 5) for primary runs, with `geomean` and `uniform` as ablations.
4. Write the `advantages` tensor = `V_i` broadcast on each owned token and 0 elsewhere.

The **same tensor** feeds TreePO and TreeOPSD. `τ = tau_from_K(M(N+1), ρ)` is fixed from the nominal
K (A17). Rows from dropped or failed rollouts are logged and keep their slot with reward 0.

### 2.3 Losses (`losses.py`)

Signature: verl's `PolicyLossFn(old_log_prob, log_prob, advantages, response_mask, loss_agg_mode,
config, rollout_is_weights)`.

- **`treeopsd_fkl`** (Eq. 13):
  - Segment the row's owned tokens by `turn_id`.
  - `s_i = mean_j(log_prob − old_log_prob)` over the turn.
  - `q_i = 1/2 + V_i/τ`.
  - Per-token loss `τ · KL(Ber(q_i) ‖ Ber(σ(s_i))) · w_tok`, with `w_tok = 1/(L_x · B_e)`
    ("prompt-averaged token gradients"); aggregate with `loss_agg_mode="token-sum"`.
  - Per-token gradient = `τ(p_i − q_i)/L_x`, the property `test_token_gradient_matches_eq16` checks
    for the reference code.
- **`treepo_fixed`** (A2, primary control): per-token loss `−V_i · log_prob · w_tok`. Its gradient is
  `−V_i/L_x` on every block, with no ratio and no clip. This is the literal reading.
- **`treepo_clip`** (A2, secondary control): the PPO-clip surrogate with `A = V_i`, ε = 0.2, and the
  same `w_tok`. It shows whether TreeOPSD only beats an *unbounded* fixed-credit update.
- **GRPO**: verl's `vanilla` loss with `adv_estimator=grpo`, clip ε = 0.2, `token-mean`. That is
  verl's default (A3), and it is recorded.

By construction, TreeOPSD and TreePO differ **only** in `actor.policy_loss.loss_mode`. That makes the
F1 control exact.

### 2.4 The one verl patch

`PolicyLossFn` does not receive extra batch tensors. `env/patches/verl-v0.9.1-treeopsd.patch`
(≈30 lines, to be written and reviewed in Phase 0) forwards optional per-token tensors from the actor
micro-batch to the loss function as kwargs:
- `tree_turn_id`;
- `tree_token_weight` (= `1/(L_x·B_e)`).

The patch targets `verl/workers/actor/dp_actor.py` and the loss call site. If the patch does not apply
cleanly, the build fails.

Without the patch, `w_tok` and `turn_id` would have to be smuggled through `rollout_is_weights`, which
we refuse to do.

Each turn lives entirely in one row, so segment means stay inside a micro-batch even with dynamic
batch packing. A CPU unit test is required: compare the patched verl loss on a packed toy batch
against `treeopsd.core.treeopsd_loss` (gate G4).

### 2.5 Optimizer blocks

The paper's E = 4 "sequential optimizer blocks" map to:
- `data.train_batch_size = B` prompts;
- `actor.ppo_mini_batch_size = B/4` prompts;
- `actor.ppo_epochs = 1`;
- `shuffle = False` within the step.

All arms use these settings (App. E.4: "every trainable method applies four actor optimizer
blocks"). The optimizer is AdamW with LR 1e-6 constant, wd 0.01, betas (0.9, 0.999) and grad-clip
1.0. Grad-clip is verl's default; the paper does not state it. KL loss, KL in reward and entropy
coefficient are all off.

### 2.6 Token-matched GRPO (F2)

After S1b measures the generated tokens per prompt for both GRPO and the tree, set
`G_tok = round(tree_gen_tokens_per_prompt / grpo_gen_tokens_per_trajectory)`.
- If `G_tok ≠ 8`, EXP-1 adds a GRPO arm with `rollout.n = G_tok`.
- Separately, a **GPU-hour-matched** GRPO arm uses the G that equalises measured train GPU-hours. A13
  predicts G ≈ 10 to 12 because the trainer packs 12 full rows per prompt.

Both arms are logged; neither is tuned.

## 3. Frozen retrieval

**MHQA:** Search-R1 wiki-18 with E5-base-v2 and exact inner product (DATA.md section 1).
- **Server:** `retriever/server.py` holds the reconstructed 21.0 M × 768 fp32 matrix (64.6 GB) on the
  **RTX PRO 6000** of the Station that is training.
- **Encoder:** E5-base-v2 query encoder on the same GPU, with the `query: ` prefix.
- **API:** `POST /retrieve {queries: [...≤4], topk: 3}` returns passages formatted as Search-R1 does
  (`Doc i(Title: …) text`).
- **Determinism:** fixed fp32 matmul with `torch.backends.cuda.matmul.allow_tf32 = False`. Tied scores
  break by doc id.
- **Response cache:** a response hash is logged per call. Identical queries must return identical ids
  across days; this is the F4 control for our own runs.
- **Eval replicas** run on Sparks (128 GB unified).

**BC+ (EXP-1b):** a fixed corpus with the Qwen3-Embedding-8B dense index (BC+ release) for eval,
plus BM25 over the OpenResearcher corpus for training (DATA.md section 3, A8). The `browser.*`
tools are taken from the OpenResearcher repo at `785fd6ba`.

## 4. Models

| Role | HF id | Revision |
|---|---|---|
| Shakedown (S1a) | `Qwen/Qwen3-1.7B` with `enable_thinking=True` (A10) | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| **Main** | `Qwen/Qwen3-4B-Thinking-2507` | `768f209d9ea81521153ed38c47d515654e938aea` |
| EXP-7 only | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` |

Full-parameter training in bf16 with fp32 master weights and Adam state needs about 70 GB for 4B. With
vLLM colocated (verl hybrid engine, sleep mode between phases) it fits in the GB300's 252 GB HBM with no
FSDP sharding and no offload. No LoRA is used, because the paper trains full parameters.

## 5. Device allocation

| Device | Job |
|---|---|
| Station A GB300 | trainer + colocated vLLM rollout for one arm |
| Station A RTX PRO 6000 | E5 exact-search retriever for Station A's rollouts |
| Station B GB300 + RTX PRO 6000 | a second arm in parallel (same image digest, same data hashes) |
| Spark 1 to 3 | final-checkpoint full evaluation (vLLM 4B + a local retriever replica) and the S1a 1.7B shakedown |

The 400G/200G RoCE link between Stations is **not** used. No arm spans two Stations, so there is no
cross-node NCCL to debug and no variance from it.

## 6. Phase 0 (staging) gates

Each gate is logged in `experiments/worklog.md` and must pass before any EXP-1 run.

| Gate | Check | Pass |
|---|---|---|
| G0 | `env/smoke/run.sh` on the Alien (x86, CPU) and inside the image | 51/51 pass |
| G1 | verl v0.9.1 imports on vLLM 0.32.0. verl's CPU tests `tests/trainer/ppo/test_core_algos_on_cpu.py`, `tests/experimental/agent_loop/test_qwen3_tool_parser_on_cpu.py` pass. A 2-step GSM8K GRPO run with Qwen3-1.7B on the GB300 completes. | all green, else use the fallback base |
| G2 | `fetch_data.sh STAGE=exp1` verifies every sha256. FlashRAG row counts are 90,447 / 7,405 / 12,576 / 2,417 / 125. | exact |
| G3 | Retriever: 1,000 fixed queries, top-3 ids from torch exact search equal `faiss-cpu` `IndexFlatIP`. Latency p95 < 100 ms per 4-query call. | ≥ 999/1000 identical (ties only) |
| G4 | Patched verl loss on a packed toy batch equals `treeopsd.core.treeopsd_loss` / `treepo_loss` (values and grads, fp64). | max abs err < 1e-10 |
| G5 | Tree builder on 32 prompts: K = 12 rows per prompt. Each node owned once. Prefix tokens of continuations byte-identical to their root's prefix. | exact |
| G6 | ReAct baseline (no training) of Qwen3-4B-Thinking-2507 on a fixed 500-question HotpotQA dev subset, plus measured tokens/turns/tool calls per trajectory. These replace the assumptions in `scripts/compute_estimate.py`. | EM within ±5 of the paper's ReAct 31.61, else investigate the prompt/tool format |
| G7 | First-block identity: on one batch, the TreeOPSD and TreePO gradients after block 1 are identical (cosine > 0.9999), as Eq. 40 predicts. | yes |

Recorded in the run header:
- GPU topology (`nvidia-smi topo -m`), driver and CUDA versions, image digest;
- vLLM `enable_prefix_caching`, `gpu_memory_utilization`, `max_num_seqs`;
- batch sizes, LR, KL coefficient (0), token caps (prompt 2,048; response 6,144 including
  observations, A14; observation ≤ 16,384 chars), max turns (5).

## 7. Compute estimates (one GB300 per arm)

These come from `scripts/compute_estimate.py`, and every input is an assumption until G6 measures it:
- GB300 2.25 PF dense BF16 at MFU 0.30;
- 4B effective agentic decode 12k tok/s;
- MHQA 2,500 generated / 7,000 total tokens per trajectory;
- DR 12,000 / 18,000;
- training FLOPs = 8·N·tokens·1.15, which covers fwd/bwd, the old-logprob recompute and attention.

| item | note | estimate |
|---|---|---|
| 4B MHQA paper-scale step (GRPO, B=256) | 5.1M gen / 14.3M train tok | 20.5 min/step |
| 4B MHQA paper-scale step (tree, B=256) | 5.1M gen / 21.5M train tok | 27.0 min/step |
| S1a shakedown 1.7B TreeOPSD, B=64, 30 steps | 1 arm (Spark or GB300) | 2.0 GB300-h |
| S1b 4B smoke, B=64, 10 steps (+ token/turn measurement) | 1 arm | 1.4 GB300-h |
| EXP-1 4B MHQA, B=64 (4×16), 100 steps, per tree arm | TreeOPSD or TreePO | 12.7 GB300-h |
| EXP-1 4B MHQA, B=64, 100 steps, GRPO arm | GRPO G=8 | 10.0 GB300-h |
| **EXP-1 total, 3 arms × 1 seed** | | **35.4 GB300-h** (about 18 h wall on 2 Stations) |
| EXP-1 total, 3 arms × 3 seeds | | 106.1 GB300-h |
| final full MHQA eval per checkpoint (22,523 q) | Spark / RTX PRO 6000 | 6.3 h / 2.0 h |
| paper-scale 4B MHQA arm (B=256, 100 steps) | for reference | 46.0 GB300-h |
| EXP-1b 4B BC+ track, B=32, 70 steps, per tree arm | scaled from B=128 | 13.1 GB300-h |
| EXP-1b total, 3 arms × 1 seed | | 37.0 GB300-h |
| paper-scale 4B DR arm (B=128, 70 steps) | for reference | 50.1 GB300-h |
| EXP-2 allocation sweep, 5 configs × 60 steps, B=64 | TreeOPSD | 38.3 GB300-h |
| EXP-2 TreePO-same-tree control at (4,2), 2 extra seeds | F1 | 25.4 GB300-h |

With decode at half the assumed speed (`--decode-scale 0.5`), EXP-1 for 3 arms × 1 seed is 47 GB300-h.
Training, not rollout, dominates. The tree arms cost about 1.27× GRPO per step because the trainer
processes 12 full-length rows per prompt against 8 (A13). We predict that this is where F2 trips.

**Against the brief.** The brief planned EXP-1 as "~1 GB300 day + 1 Spark eval". That budget covers
three arms at B = 64 × 100 steps for one seed only if both Stations run in parallel. It does not cover
the paper's B = 256. Seeds 2 and 3 are another 1.5 days on two Stations. The plan:
1. Run EXP-1 single-seed and label it "preliminary".
2. Use its mechanism telemetry (PREREG M1) to decide whether seeds 2 and 3, or EXP-1b, are the better
   use of compute.

## 8. Telemetry per optimizer block (written to `runs/<id>/telemetry.jsonl`)

- `|s_{θ,i}|`: mean, p50, p90 over owned turns, per block e = 1..4.
- `drift_rel = mean|Γ_i − V_i| / mean|V_i|` on blocks 2 to 4. This is the exact size of the TreeOPSD vs
  TreePO difference (Eq. 40), and the mechanism test M1 in the prereg.
- Fraction of turns whose `Γ_i` has reversed sign relative to `V_i` (overshoot events).
- Entropy of `ω` at internal nodes and the fraction of nodes with `max ω > 0.99` (A1).
- Budget, per prompt:
  - generated tokens (roots vs continuations), tool calls, prefill tokens, prefix-cache hit rate;
  - trainer row tokens, unique tokens and `L_x`;
  - fraction of roots with T = 1 (A11);
  - realised suffix fraction vs the nominal 0.5.
- Wall-clock split (rollout / old-logprob / update / weight sync) and GPU-hours.
- vLLM-vs-trainer log-prob gap on behavior tokens (A4).

`scripts/eval_mhqa.py` writes `runs/<id>/summary.json` with the schema in autoresearch.md.
`autoresearch.sh` turns it into `METRIC` lines.

## 9. Statistics

- **Primary:** the unweighted 4-benchmark EM average at the **fixed final checkpoint** (step 100), on
  the full dev/test sets.
- **Paired comparison:** compare arms on the same questions with a paired bootstrap over questions
  (10,000 resamples, stratified by benchmark). Bamboogle (125 q) alone has an EM standard error of about
  4.4 points, which is about 1.1 points on the average. That is the same size as the paper's whole MHQA
  effect, so we also report the 3-benchmark average without Bamboogle as a pre-declared secondary.
- **Seeds:** with ≥ 3 seeds, report mean ± sd and a Welch t-test across seeds as well.
- **Best-in-window:** reported as a secondary using the paper's protocol (every 5 steps to step 80, on a
  500-per-benchmark subset), to measure checkpoint-selection inflation (F3).

## 10. Deviations from the paper

| ID | Deviation | Why | Expected effect |
|---|---|---|---|
| D1 | vLLM instead of SGLang | known-good arm64 image on GB300 | sampling numerics only |
| D2 | B = 64 prompts (4 blocks × 16) instead of 256 for EXP-1 | compute (section 7) | noisier updates. Drift per block is probably *smaller* with fewer prompts per block, which favours TreeOPSD ≈ TreePO; noted in the prereg. |
| D3 | 1 GB300 per arm instead of 8 × H20 | hardware | none on the algorithm |
| D4 | in-training eval on fixed 500-question subsets (Bamboogle full) instead of full sets | compute | none on the primary (full final eval) |
| D5 | trainer-recomputed `π_old` | resolves A4 | makes Theorem 2(i) hold exactly |
| D6 | `treepo_clip` as an added secondary control | resolves A2 | — |
| D7 | Deep Research restricted to BC+ (closed) | F4; live-web not frozen | cannot reproduce the "DR Avg" number, only its BC+ column |
| D8 | verl's tool-schema model would drop `items/minItems/maxItems/additionalProperties` from the rendered tool block. The patch keeps them, so the prompt matches App. F.1. | prompt fidelity | none (restores the paper's prompt) |
| D9 | Tier-2 BC+ is evaluated with the training `search` tool (top-3 documents of at most 2,500 chars), not `browser.search/open/find`, and models are trained on HotpotQA, not OpenResearcher | research-session decision; one scaffold across tiers | BC+ numbers measure transfer; not comparable in level to the paper's 13.98 |
| D10 | legacy `RayPPOTrainer` (`trainer.use_v1=false`) | the tree hooks sit at its synchronous `generate_sequences` / `compute_advantage` call sites | none expected; the V1 trainer is still available for GRPO-only reruns |
| D11 | `multi_turn.max_tool_response_length=8192` tokens (verl default 256) | the paper caps observations at 16,384 characters, which our tool enforces | none (avoids double truncation) |
