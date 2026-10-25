# Worklog

Each entry gives a date and time (PT), the device, what was done, the result, and anything odd. Runs
also append one line to `autoresearch.jsonl`.

## 2026-10-21 (Alien workstation, x86, CPU only; no GPU, Station or Spark touched)

- Read arXiv:2610.23181v1 in full (PDF + HTML). The method, equations, protocol, all tables and 20
  ambiguities are in PAPER.md.
- **Findings that change the study brief:**
  1. The paper's main models are **Qwen3-4B-Thinking-2507** and Qwen3-30B-A3B-Thinking-2507.
     Qwen3-1.7B (hybrid) appears in App. D.1.
  2. The Multi-Hop QA track is exactly Search-R1's wiki-18 with E5 and flat retrieval. It is fully
     frozen and reproducible as published. On this track the paper's own 4B gains are only +0.76 over
     GRPO and +0.73 over TreePO. The large gains (+2.9 to +4.0) are on BrowseComp-Plus (closed corpus)
     and on the live-web DR benchmarks.
  3. The 30B results *do* include a per-benchmark DR breakdown (Table 1). The brief said they did not.
  4. Training reward for MHQA is thresholded F1 (≥ 0.3), not EM.
  5. The ω backup weights use full-span likelihoods. These are near one-hot for long thinking spans
     (A1).
  6. TreeOPSD and TreePO produce identical first-block gradients. Their difference is bounded by
     policy drift on blocks 2 to 4 (Eq. 40). This yields a cheap mechanism test (PREREG M1).
  7. The trainer packs 12 full rows per prompt against GRPO's 8, so the "budget match" holds only in
     sampled tokens (A13).
- Code search found no official TreeOPSD code (arXiv, HF paper page, GitHub, HF hub). Related repos
  are pinned in PAPER.md section 11.
- Wrote `treeopsd/core.py` and `treeopsd/metrics.py`. `env/smoke/run.sh` → **51 passed**. A mutation
  check showed the tests catch:
  - a wrong τ (18 failures);
  - a wrong suffix cost (5 failures);
  - a halved loss (3 failures).
- Generated `scripts/manifest.tsv` (224 files, all pinned to revisions, with sha256 for LFS files).
  `fetch.py` was tested end to end on two small files (Bamboogle: 125 rows, golden_answers format
  confirmed).
- Wrote `env/Dockerfile`. The base is vLLM v0.32.0 arm64, pinned by digest; its metadata was read from
  the registry (CUDA 13.0.2, Py3.12). verl is pinned at `1876b06d` (v0.9.1). The image was **not built**:
  that needs an arm64 host (gate G1).
- Next: Phase 0 gates G1 to G7 (DESIGN.md section 6) on the Stations, once GPU claims are granted.

## 2026-10-21 09:15 to 09:50 PT (left Station `STATION_IP`, CPU and network only; no docker, no GPU)

- Created `/models/treeopsd`. The fetch ran detached; log at `/models/treeopsd/fetch.log`.
  - EXP-1 set + BC+ tier = 107 GB, done 09:19 to 09:29.
  - Every LFS sha256 matched the manifest.
- **Gotcha: `wiki-18.jsonl.gz` is a gzipped TAR.** `gzip -d` (as the Search-R1 README says) leaves a
  512-byte tar header in front of the first record. Re-extracted with `tar -xzOf`: 21,015,324 lines,
  sha256 `43d7d3f5…752d`. The scripts and DATA.md are fixed.
- `e5_Flat.index` sha256 is `69c98463…d166`. Reconstructed vectors went to `e5_vectors.f32.npy`
  (bit-identical on spot checks; unit norm).
- **Prepared data** (Station venv with pandas/pyarrow/faiss-cpu only, at `/models/treeopsd/venv`):
  - MHQA parquets: row counts exact. MuSiQue hops 2/3/4 = 1,252 / 760 / 405. MuSiQue ids in
    FlashRAG are renamed (`dev_k`), so hops come from `question_decomposition`.
  - BC+ slice: 830 rows, decrypt OK.
  - BC+ corpus: 100,195 docs.
- **Code.** verl v0.9.1 extension:
  - tree rollout manager;
  - `tree_tool_agent` loop;
  - tree-credit hook with budget telemetry;
  - 3 losses;
  - patch with 2 hunks (losses forwarding, tool-schema extras);
  - retriever server, G3/G3b checks, data prep, eval scorer, run scripts.
- **Tests:**
  - local CPU venv with verl@1876b06 (patched): **86 passed**;
  - system python (no verl): 82 passed, 4 skipped.
- Wrote `RUNBOOK-PHASE0.md`.
- **Research-session decisions (inbox 001)** are folded into PREREG amendment 1: BC+ slice is the
  headline, QA is secondary.

## 2026-10-21 ~10:00 PT - run protocol (Founder decision)
- Shakedown: Qwen3-1.7B plumbing test before any 4B run.
- EXP-1 arms run continuously to step 100 (no pause). Checkpoints saved at steps 10, 25, 50, 75, 100.
- Interim reports at 10/25/50/75: fixed eval subset (500 QA dev + 200 BrowseComp-Plus, same ids every time), on side devices (RTX PRO 6000s / Sparks), training not paused. Report: per-arm accuracy vs step, M1 drift, F2 cost (tokens, tool calls, GPU-hours), health.
- Step 100: full eval (5 QA sets + all 830 BC+), the fixed checkpoint that is reported.
- Manual stop only on a clear kill signal (M1, broken run) or Founder/research-session call; never on early accuracy alone.

## 2026-10-21 13:20 PT - Phase 0 start (right Station)
- Stations released by the other workload at 13:20. Its CPU tail still calls the left Station's daily model, so Phase 0 starts on the **right** Station (STATION_IP) instead of the left; the left joins when free. Deviation from RUNBOOK section 1 ("left Station first"): host only, commands unchanged.
- 13:21 GPU reservation `treeopsd-phase0-right-20261021` (right GB300 + RTX PRO 6000). Right daily stopped via its own launcher (`launch-dsv41-daily-A.sh stop`, rc 0); GPUs then 22 MiB / 47 MiB used.
- 13:23 data copy left -> right over the Station link (STATION_LEFT_RAIL_IP -> .2), ~1.46 GB/s, excluding `wiki18-index/part_a*` (split download parts; e5_Flat.index is the assembled file).
- 13:25 repo (local tree at cbace88, no .git/data/runs) synced to right `/models/treeopsd/repl-build`; `docker build` started, log `/models/treeopsd/build-right.log`. Base image befccf0e already present on the right.
- 13:26 build 1 FAILED at step [4/10] (verl requirements): pip ResolutionImpossible. Base image ships nvidia-nccl-cu13 2.30.7, but its own torch 2.13.0+cu130 metadata pins 2.29.7.
- 13:35 data copy done (177 GB on the right).
- 13:36 build 2 (uv, base pins as --override; Dockerfile edit uncommitted) FAILED, same step: torch metadata pins nvidia-cusparselt-cu13 0.9.1 vs 0.8.1 installed, and the base ships **transformers 5.19.0**, outside verl 1876b06's `>=5.5.3,!=5.6.0,<5.11`. That is the runbook's STOP condition. The listed fallback base (3af6a45b, nightly 0.30.1rc1) ships transformers 5.17.0, so it fails the same pin.
- **STOPPED** after two consecutive failures of step 1 (controller rule). Escalated to the Founder with options. Right Station claimed and idle; data and base images in place.
- 16:45 Founder: "1, go ahead and build". **Deviation D12:** keep the base's transformers 5.19.0 (vLLM 0.32 requirement), outside verl 1876b06's `<5.11` pin. Dockerfile step 2 split: 2a non-torch deps resolved by uv under base pins; 2b torch-dependent packages (accelerate, peft, torchdata, tensordict, liger-kernel) `--no-deps`; 2c import check; 5b asserts that no base pin moved. Safety net: G1 (2-step GRPO + tree), CPU G7 test, S1a shakedown.
- 16:47 build 3 started (`build-right-3.log`).
- 17:0x Research-session rulings (inbox 003/004): corrections accepted; EXP-1b (frozen BC+) approved ahead of EXP-2 as a within-study arm comparison only; M1 (<2% relative drift, blocks 2-4) is a kill signal for the LOSS claim only, so on a trip pivot to tree-sampling characterization and run the drift-dose follow-up; F2 cost goes up front in reports. The research session asked for a "step-30 screen". Founder protocol (continuous to 100, reports at 10/25/50/75) governs, so the step-25 report serves as the screen. Watch items logged: literal vs clipped TreePO divergence is a finding; the tar-vs-gunzip trap is a runbook note; keep the full query log for F4 replay from run 1.
- 17:1x build 4 PASSED: image `treeopsd-repl:verl0.9.1-vllm0.32.0` sha256:3aca73405a0fdc964b5eab850226cc6b571c2b938f6abe0703016d484f6b5db8 (right Station). 2c imports ok (accelerate 1.15.0, peft 0.21.2, tensordict 0.10.0); base pins unchanged; 86 passed; `verl 0.9.1 vllm 0.32.0 torch 2.13.0+cu130 cuda 13.0 transformers 5.19.0`. Build 3 had failed on a missing pyvers (hand-listed deps), so build 4 derives the --no-deps packages' requirements from metadata (env/nodeps_reqs.py).
- GPU map, right Station: GPU0 = RTX PRO 6000, GPU1 = GB300 (SYS between them).
- 17:2x G1a PASS (GB300 (10, 3); verl CPU tests 28 passed). Retriever wiki18 up on GPU0: n_vectors 21,015,324, 65 GB. BC+ conversion: 4 shards, vectors (100195, 4096), norms 1.0000. G3 running (1000 q, 64 threads).
- G1b try 1 FAILED: verl loads the actor with flash_attention_2, and flash-attn is not in the image (BUILD_FLASH_ATTN=0). **Deviation D13:** actor uses `attn_implementation=sdpa` with `use_remove_padding=false`. Packed (remove-padding) sequences are only isolated under varlen flash attention; under SDPA they would attend across sequences silently. Cost: slower padded training, to be measured in G6. Retry launched with these two overrides; flash-attn source build is an optional later speed-up.
- 16:49 G1b try 2 (SDPA) FAILED at the first weight sync, and the job HUNG instead of exiting; I found it at 19:38 (about 3 GB300-hours idle; my watcher waited for container exit, not for progress). Cause: vLLM 0.32's `_check_skipped_aliases` runs per `load_weights` call, and verl's bucketed transfer calls `load_weights` per 2048 MB bucket. Qwen3-1.7B and 4B both tie `lm_head` to `embed_tokens`; when they land in different buckets vLLM raises "lm_head.weight was skipped ... embed_tokens.weight not found". Proposed fix (config only): `actor_rollout_ref.rollout.update_weights_bucket_megabytes` large enough for one bucket (1.7B fp32 ~7 GB, 4B ~16 GB, so 20480). **STOPPED**: second consecutive G1b failure; escalated to the Founder.
- G3 run 1 printed FAIL: 951/1000 identical, 49 mismatches, 19 "tie-only". Checker bug: it counted a tie only at the rank k/k+1 boundary and missed in-list swaps of equal scores. All 5 printed mismatches are exact ties (|ds| <= 2e-7; wiki-18 has duplicate passages with identical vectors). Fixed the classifier (every rank's score equal within 1e-6) and print every mismatch; rerun started. p95 latency 0.0755 s (pass).
- 19:43 Founder: "yes, restart with the larger bucket and use the tie rule". G1b try 3 launched with `update_weights_bucket_megabytes=20480`; the watcher now also stops on an error line or 45 min without step 2. G3 gate amended: PASS = every mismatch is a score tie (the ≥999 bar is dropped); checker and runbook updated.
- 19:44 G1b try 3 failed at Hydra start (wrong key path, mine); the correct key is `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes`.
- 19:50 **G1b PASS** (try 4, 5 min): step 1 score 0.197, response 1,422 tok, 4.0 turns, gen 17.3 s, update_actor 40.8 s, update_weights 0.86 s, step 86 s; step 2 score 0.219, 1,475 tok, 4.28 turns, step 44 s. telemetry/steps.jsonl 2 lines. Standard overrides from now on: sdpa, use_remove_padding=false, bucket 20480.
- G3 run 2: 48/49 tie-only. The non-tie had an identical id at rank 1 with |ds| 1.1e-6 (fp32 noise) and an exact tie at rank 3. Classifier compares only ranks whose ids differ (applies the Founder's tie rule as written). Offline regrade 49/49; run 3 launched for the record.
- 19:51 G1c (tree arm, 2 steps) launched.
- 19:58 **G3 PASS** (run 3): 951 identical, 49 mismatches, all 49 tie-only; e2e 381/400 (informational); latency p50 0.077 s, p95 0.078 s.
- 19:55 **G1c PASS** (tree arm, 2 steps, 5 min): step 1 score 0.265, step 112 s; step 2 0.146, 63 s. Tree: n_nodes 16.6/17.6 (>=12), empty_leaves 0, V_abs_max_over_bound 0.46/0.08 (<=1), traj_equivalents 8.25/8.12, suffix_fraction 0.53/0.51, single_turn_root_frac 0.03, tool_calls/prompt 4.6/5.6. F2 note: train_tokens_per_prompt 23.5K vs gen 6.8K. loss_*.jsonl: treeopsd/drift_rel ~4e-7 and s_abs_mean 0.0 on the first block, >0 from call 3 (G7 GPU form preview). Shutdown traceback = DataLoader worker killed at exit, after step 2 (benign).
- 19:56 S1a shakedown (1.7B, 30 steps) launched, same three overrides.
- 20:13 S1a at step 2. Step 1: 571 s = gen 24 + old_log_prob 127 + update_actor 418 (B=64, 1.7B). Initial val: musique 0.120, bamboogle 0.315. Tree: n_nodes 18.0/17.5, V/bound 0.36/0.40, traj_eq 8.47/8.10, tool calls 6.0/5.5. About 95% of the step is padded actor passes (D13: SDPA, no remove-padding, sequences padded to 8,192). That projects about 5 h for S1a and makes 4B EXP-1 too slow.
- 20:15 flash-attn 2.8.3 layer build started on the right (env/Dockerfile.fa, sm_100 SASS, MAX_JOBS 24, nice 10) as `-fa2`. If it imports and passes the G1 2-step check, D13 is reverted for 4B runs (remove-padding on).
- 20:21 flash-attn layer built (6 min): flash_attn 2.8.3 imports, base pins unchanged. Kernel test on the GB300: varlen FA vs fp32 SDPA per sequence, max abs err 0.009, PASS.
- 20:22 left Station claimed (`treeopsd-left-20261021`); glmf-daily + nemotron-super-rtx stopped (docker stop -t 120; state recorded). Retriever on the right reachable from the left over rail 1 (STATION_RIGHT_RAIL_IP:8000).
- 20:23 image push right -> left FAILED instantly (no right->left SSH key), and my waiter missed the "Permission denied" line, so about 50 min were lost (found 21:16). 21:17 re-run as a pull from the left; the waiter now keys on an explicit EXIT= line.
- 21:15 S1a at step 7: scores 0.380 0.385 0.470 0.446 0.429 0.398 0.435.
- 21:21 G1fa (left, flash-attn image `-fa2`, remove-padding ON, sdpa override dropped) **PASS**. Step 1: score 0.245, old_log_prob 13.0 s, update_actor 19.3 s, step 58 s. Step 2: 0.147, old_log_prob 1.2 s, update_actor 3.9 s, step 19 s. SDPA G1b step 2 had update_actor 28.3 s and step 44 s. **D13 reverted** for all further runs: flash-attn + remove-padding, as in the paper's verl config.
- 21:26 SDPA shakedown stopped at step 8 (scores 0.380 0.385 0.470 0.446 0.429 0.398 0.435 0.455) and archived as `s1a-shakedown17-s0.sdpa`; it was validating a path we will not use.
- Weight-sync bucket 20480 is now the default in run_train.sh and run_eval.sh (run_eval also passes "$@" through). Runs execute from the bind-mounted repo copy `$D/repl-build` (REPO env) so script fixes need no image rebuild.
- 21:27 S1a shakedown (fa2) launched on the LEFT; G6 4B ReAct baseline (hotpotqa500, fa2) launched on the RIGHT.
- 21:28/21:29 G6 FAILED twice at config validation (run_eval.sh had never run on a GPU): (1) no actor micro-batch / dynamic bsz; (2) no `algorithm.adv_estimator`, so it defaulted to GAE and tried to build a critic from verl's placeholder path. **STOPPED** (second consecutive failure of G6).
- Fix: eval now carries every non-arm setting from run_train.sh COMMON (dynamic bsz, adv_estimator=grpo, use_kl_in_reward=false, remove-padding, TP1, prefix caching, max_parallel_calls 1). Validated on CPU: resolved Hydra configs of eval vs the train config that ran in G1b differ in exactly 3 keys outside data/trainer/optim/val_kwargs/model.path, all intended: ppo_mini_batch_size 16 vs 2, gpu_memory_utilization 0.8 vs 0.45, rollout.n 1 vs 8. Awaiting Founder go to relaunch G6.
- 22:21 Founder: "yes, relaunch the 4B baseline". G6 try 3 launched (right).
- 22:26 **G6 PASS** (5 min): untrained Qwen3-4B-Thinking-2507, hotpotqa500, T 0.6 / top-p 0.95, 1 sample: EM 30.8 (paper ReAct 31.61, bar +-5), F1 0.389, reward 0.385, answer rate 89.6% (expected >90%, stop <80%), 3.92 turns. Many "Failed to decode tool call" (malformed tool-call JSON from the base model) account for the missing answers.
- 22:27 S1b TreeOPSD smoke (4B, B=64, 10 steps) launched on the RIGHT. GRPO smoke follows on the left after S1a (step 27/30).
- 22:33 **S1a PASS** (left, fa2): 30/30 steps, 0 errors; ckpts 10/20/30; mean step 112 s. Reward steps 1-5 mean 0.41, steps 26-30 mean 0.45. Val MuSiQue 0.114 -> 0.151, Bamboogle 0.335 -> 0.374. n_nodes min 17.6, V/bound max 0.38, traj_eq mean 7.96, tool calls/prompt 6.3 -> 10.4. G7 GPU: block-1 s_abs 0 (447 calls, drift max 2.5e-6). Blocks 2-4 drift_rel median 1.73% (IQR 1.19-2.78%, p90 4.39%, 60% < 2%), no trend, sign reversals 0. M1 preview: at the line.
- 22:34 S1b GRPO smoke (4B, B=64, 10 steps) launched on the LEFT. 22:40 outbox 004 to the research session (gates, deviations, S1a, M1 preview + question on the M1 statistic, F2 note).
- 23:2x **S1b (4B, B=64, 10 steps each)**. TreeOPSD (right): mean step about 345 s; gen tokens/prompt 19,857; train tokens/prompt 41,064; tool calls/prompt 4.88; traj_eq 8.72; n_nodes 16.9; suffix_fraction 0.59. GRPO G=8 (left): about 250 s/step; gen tokens/prompt 19,322 (2,415 per trajectory); train tokens/prompt 27,779; tool calls/prompt 8.06. **G_TOK = round(19,857 / 2,415) = round(8.22) = 8**, so prereg arm D (generated-token match) is identical to arm A. F2: the tree is budget-matched in sampled tokens (1.03x) but trains on **1.48x** the tokens and makes fewer tool calls.
- **Deviation D14 (pending a research-session ruling):** arm D runs as **training-token-matched GRPO, G = round(41,064 / (27,779/8)) = 12**, not the degenerate G=8. It is queued LAST on the left (starts in about 17 h), so the research session can veto it first. RUN_ID exp1-grpo_tok-s0 with G_TOK=12.
- Compute check: 3 tree arms at about 9.6 GB300-h, GRPO about 7, GRPO G=12 about 10.5: about 50 GB300-h (2.1 GB300-days) < 3. No amendment needed.
- Founder 23:1x: "launch EXP-1 when ready, if you need to do fixes, do them, don't stop for permission, keep the test going." Overnight the two-failure stop is waived for fixes; HARD rules still apply.
- 23:28 **EXP-1 seed 0 launched.** LEFT queue `scripts/exp1_queue.sh`: treeopsd -> grpo -> grpo_tok (G_TOK=12, D14), retriever over rail 1. RIGHT queue: treepo -> treepo_clip (starts when the S1b tree smoke exits). Each arm: 100 steps, B=64, save every 5 / keep 2, no in-training val, auto-resume up to 4 tries.
- Interim eval pipeline (no training pause): fixed subsets seed 0: `mhqa/interim_qa500.parquet` (500 of val_dev: 147 HotpotQA, 167 2Wiki, 149 MuSiQue, 37 Bamboogle; sha256 e1d53dc2...72fe) and `bcplus/interim_bc200.parquet` (200 of 830; sha256 8e4e95f9...ac77). BC+ retriever on the LEFT RTX PRO 6000 (:8001). `scripts/exp1_sidecar.sh` MODE=merge (FSDP -> HF at steps 10/25/50/75/100 on the holding Station, pulled to the left) and MODE=eval (qa500i + bc200i on the left PRO 6000, SDPA, vLLM 0.5; results in $D/interim/<run>/step<N>/). Merge tested on S1a step 30 (3.3 GB bf16 + tokenizer). Base-model (step 0) eval started 23:30.
- run_eval.sh: WIKI_URL / BC_URL env, benches qa500i and bc200i.

## 2026-10-22 05:15 PT - overnight EXP-1 status (no anomalies; the monitor ran 23:32-05:14)
- treeopsd (left) at step 64, about 300 s/step; treepo (right) at step 76, about 230 s/step. No restarts. Disk 1.3 T / 0.9 T free.
- Interim EM, qa500i / bc200i: base 22.8 / 6.5 | TreeOPSD s10 30.2/7.5, s25 32.4/7.0, s50 35.6/9.5 | TreePO s10 27.8/7.5, s25 30.6/6.5, s50 36.2/13.0 (s75 running). bc200i SE about 2 pts per arm; no difference is significant yet.
- **M1 (4B TreeOPSD, blocks 2-4, 3,912 calls):** drift_rel median 0.88% (IQR 0.52-1.80%, p90 4.4%), 78% < 2%; rising by quarter 0.67 -> 0.80 -> 0.97 -> 1.21%; sign reversals mean 0.05%. On the median statistic **M1 trips** (statistic ruling pending, outbox 004/006). Per the research-session ruling this kills the LOSS claim only: no stop, pivot to tree-sampling characterization plus the drift-dose follow-up.
- F2: TreeOPSD train tokens/prompt flat at about 43K while gen tokens fell 21K -> 11.6K; TreePO train tokens fell 43K -> 33K. Tool calls/prompt rose 4.7 -> 14.2 (TreeOPSD) and 4.7 -> 10.6 (TreePO). omega_peaked_frac about 0.85-0.91 (backup is near single-child).
- 05:2x Research-session rulings (inbox 005): **M1 judged on full-run median AND last-quarter median; trips iff both < 2%; fraction < 2% as context. On 006 numbers M1 TRIPS** (0.88% / 1.21%). F1 is in the kill box pending EXP-1c (drift-dose). **D14 kept** (arm D = GRPO G=12, training-token matched). Interim direction noted, no conclusion; finish all arms, then EXP-1c, then the full step-100 eval. **omega_peaked 0.85-0.91 elevated to the main line**; the EXP-2 uniform-backup ablation is now load-bearing.
- EXP-1c per PREREG section 7: arms B (treepo) and C (treeopsd) at E in {8, 16} blocks (BE 8, 4; B=64 fixed), 40 steps each, 4 runs, about 12 GB300-h. E=4 is EXP-1 itself.
- 05:35 interim TreePO s75: qa500i 37.2, bc200i 9.5 (s50 was 13.0; bc200i noise is visible).
- 05:52 EXP-1c queue (`scripts/exp1c_queue.sh`) armed on the RIGHT: waits for the EXP-1 right queue to end (~16:30), then treeopsd/treepo at E=8 and E=16 (BE 8 / 4), 40 steps, save every 10.
- 05:53 eval loop replaced by `scripts/exp1_sidecar2.sh` (MODE=eval; the merge loop v1 is unchanged): interims keep priority, then one full step-100 bench job at a time (bcplus 830, then hotpotqa, 2wikimultihopqa, musique, bamboogle) on the left RTX PRO 6000. Right PRO 6000 has no room (wiki retriever 81 of 97 GB).
- 06:0x Research session (inbox 006): add a SECONDARY best-in-window analysis (the paper's selection rule applied to our runs, on the same subsets and the 10/25/50/75/100 grid), reported beside the primary final-checkpoint verdict. Eval loop swapped to `scripts/exp1_sidecar3.sh` (adds the subset eval at step 100).
- 11:4x **B vs C full BC+ (830, final ckpt):** B 10.48 (87), C 10.12 (84); paired by prompt sha1: both 41, B-only 46, C-only 43. **C - B = -0.36, 95% CI [-2.65, +1.93]** (paired bootstrap, 5,000). PREREG tier-2 F1 KILL bar (C - B < 1.0) MET at seed 0; the paper's +2.90 is outside the CI. Note: the original PREREG M1 wording (>80% of steps < 0.02) is not met by call fraction (72%); the research-session 005 rule (median + last quarter) trips. Both reported. B HotpotQA full EM 42.58. C s100 interim 40.8 / 9.5. Outbox 008.
- 14:01 A (grpo) step 100; 14:31 E (treepo_clip) step 100; right EXP-1 queue end. 14:32 EXP-1c started (treeopsd E8 first). 14:01 D (grpo_tok G=12) launched on the left.
- 16:1x **Full HotpotQA (7,405): B 42.58, C 46.51; paired C - B = +3.93 [+3.11, +4.75]** (eval noise only). F1 0.545 / 0.589; C outputs 1.4x longer. Tier 1 disagrees with tier 2 (BC+ -0.36). With M1 about 1%, B and C at seed 0 behave like different training seeds. Outbox 009: seeds 1-2 of B and C proposed as the deciding experiment (needs a research-session ruling; compute cap implies deferring EXP-2).
- 16:3x Research-session rulings (inbox 008): 008 recorded; M1 reported under both wordings (005 rule operative, set before the full-run numbers; 72% by call as a sensitivity note). **009 approved in full: seeds 1-2 of B and C (E=4, 100 steps) are the deciding experiment; report per tier as mean +- seed SD with per-seed eval CIs; EXP-1c continues; EXP-2 deferred.** **EXP-1b struck** (superseded by the full-830 BC+ evals on final checkpoints; its compute folds into seeds 1-2).
- Seed queues armed (`scripts/exp1_seed_queue.sh` -> exp1_queue.sh): LEFT seed 1 (treeopsd, treepo) after D; RIGHT seed 2 (treeopsd, treepo) after EXP-1c.
- 16:5x Eval capacity: GB300 training peak 164.5 GB of 256.7 (10-min sampling, both Stations; mean about 110 GB). Co-locating evals on the training GB300s was rejected (Founder: "safe plan"): about 25 GB of margin against the rollout engine's wake, and an OOM would force an unplanned restart mid-seed. The left PRO 6000 keeps the eval queue (B/C seed 0 first, then A, E, D). Both GB300s become eval lanes when the seed queues end (about 17:30 Fri), coordinated from the left so no checkpoint is evaluated twice. All evals expected about Sat morning.
- 17:0x Sidecar v4 (`scripts/exp1_sidecar4.sh`): seed runs s1 (left) / s2 (right) merged and full-evaluated at step 100 only; full-eval priority B/C (s0, s1, s2) then A, E, D. Merge loop swapped at once; the eval loop restarts as v4 when the running C-s0 2wikimultihopqa full eval container exits (v3 killed; its container keeps running and writes its summary).
- 18:xx Research-session ack (inbox 009): schedule and eval priority approved; keep seed-run interim checkpoints scorable. Merge loop -> `scripts/exp1_sidecar5.sh`: seed runs' 10/25/50/75 merged to HF (about 7.6 GB each, kept, not evaluated); FSDP optimizer state still rotates (keep 2).
- 17:43 EXP-1c treeopsd E=8 done (40 steps, about 3.2 h). Drift (blocks 2+, 2,717 calls): median 1.23%, last-quarter 1.89%, p90 9.5%, 67% < 2%, 9.5% > 10%. The dose rises only modestly over E=4 (1.05 / 1.56%). treepo E=8 running.
- C full 2WikiMultihopQA (12,576): EM 51.20.
- 19:0x Sidecar v6 (`scripts/exp1_sidecar6.sh`): EXP-1c runs merged and subset-evaluated (qa500i + bc200i) at step 40, so the B-vs-C gap can be read per dose. Merge loop swapped; the eval loop restarts as v6 when the running eval exits.

## 2026-10-23 05:45 PT - overnight status
- D (grpo_tok G=12) step 100 at 21:55. Seed 1 (left): treeopsd-s1 at step 90 (launched 21:56). EXP-1c done 04:19. Seed 2 (right): treeopsd-s2 at step 14 (launched 04:20). No failures.
- **Full evals, seed 0 (EM):** B TreePO: BC+ 10.48 | HotpotQA 42.58 | 2Wiki 49.93 | MuSiQue 18.54 | Bamboogle 45.6 | QA avg 39.16. C TreeOPSD: BC+ 10.12 | 46.51 | 51.20 | 22.51 | 54.4 | QA avg 43.66. A GRPO: BC+ 10.00 | HotpotQA 43.44 (rest pending). At seed 0, C leads B on all four QA sets (+3.9, +1.3, +4.0, +8.8; Bamboogle n=125) and ties on BC+.
- **EXP-1c drift by dose (C, blocks 2+):** E=4 median 1.05% / lastQ 1.56%; E=8 1.23% / 1.89% (p90 9.5%, 9.5% of calls > 10%); E=16 4.79% / 7.74% (34% of calls > 10%; p90 is non-finite-large: drift_rel blows up on near-zero denominators, to investigate). Step-40 subset EM (qa500i / bc200i): E=8 C 36.0/8.0 vs B 36.8/7.5; E=16 C 35.4/5.0 vs B pending. So far no growth of a C advantage with dose.
- Monitor v3: RUN_ID taken from queue-log launch lines (v2 mapped seed runs to s0 because the container names are reused).

## 2026-10-23 10:55 PT - EXP-1c treepo-E16 collapsed; outbox 010

- exp1c-treepo-E16-s0 step 40 subsets: qa500i 0.0 / bc200i 0.0, answer rate 0.00 / 0.005. Training collapse: reward 0.55 @25 -> 0.002 @33 (clip ratio 0.99), 0.017 @40 (clip 0.92). treeopsd-E16 stayed 0.50-0.69 reward, clip <=0.03.
- Dose table (C/B, step 40): E8 36.0/8.0 vs 36.8/7.5; E16 35.4/5.0 vs 0/0. No smooth gap growth; E16 gap = B instability.
- Seed 1: treeopsd-s1 step 100 subsets 37.8/7.0; full BC+ 10.60. treepo-s1 at step 59. treeopsd-s2 at step 76.
- GRPO s0 full 2Wiki 48.55.
- Outbox 010 sent with options (a) treepo_clip E16, (b) treepo E16 seed 1, (c) none.

## 2026-10-23 11:10 PT - inbox 010 ruling; (a) queued

- Ruling: EXP-1c does not rescue the accuracy mechanism; E16 B collapse recorded as a separate stability finding, fenced off from F1. (a) treepo_clip E=16 approved; (b) treepo E=16 seed 1 only if (a) collapses. Check (a) for the E8 answer-rate prodrome.
- (a) queued on the LEFT GB300 after the left seed queue ends (exp1c_queue.sh RUNS_1C=treepo_clip:16, wiki over rail 1). Run id exp1c-treepo_clip-E16-s0.
- Sidecar v7: C1C_LEFT_RUNS (a)/(b) merged locally and subset-evaluated at step 40. Merge process swapped; eval process swaps when the running HotpotQA full eval (treeopsd-s1) container exits.

## 2026-10-23 20:15 PT - training done; (a) survives; (b) by Founder override; GB300 eval lanes; outbox 011

- All training finished: treepo-s1 14:17, (a) exp1c-treepo_clip-E16-s0 17:53, treepo-s2 19:32.
- (a): reward 0.49-0.64, clip <=0.10; step-40 subsets 34.2 / 8.5, answer rate 0.90 / 0.99. No collapse.
- Founder: "run it and continue in logical order" -> (b) exp1c-treepo-E16-s1 chained on the left GB300 (exp1_chain_b.sh) after the lane's running job; then the left lane resumes.
- Seed 1 full: B BC+ 7.23, HQA 45.97, 2Wiki 53.78, MuSiQue 20.56, Bamboogle 51.2; C BC+ 10.60, HQA 45.67, 2Wiki 50.35.
- D15: GB300 eval lanes (exp1_evallane.sh). Right = seed-2 pair; left = E, D; RTX PRO = rest (sidecar v8 trims FULL_ORDER). Results are pulled right->left by exp1_pull_right.sh. Calibration Bamboogle: C s0 53.6 (RTX 54.4, 7/125 flips), B s0 43.2 (RTX 45.6).
- Slip: the lanes were planned for 17:30 but not built; the left GB300 sat idle 17:53-19:45.

## 2026-10-24 00:55 PT - seed package; (b) collapsed; B s2 late destabilisation; outbox 012

- Seed package (scripts/seed_package.py, 10k paired bootstrap): pooled C-B QA +3.76 [+2.81, +4.74], BC+ +3.45 [+2.13, +4.78]. PREREG tier 1 PASS, tier 2 PASS (confirmatory). Seed level: C-B QA +3.76 +- 3.14, BC+ +3.45 +- 3.86 (n=3).
- Per seed C-B: s0 +4.50 / -0.36; s1 +0.31 / +3.37; s2 +6.47 / +7.35.
- B s2: reward fell 0.55 -> 0.22 at steps 92-95; length-clip ratio 0 -> 0.27 by step 100; BC+ tool calls 3.52/question. C s2 clip <= 0.09 throughout.
- (b) exp1c-treepo-E16-s1: collapse at steps 32-33 (clip 0.975-1.0), like seed 0 (steps 29-33). 2/2 seeds.
- E (treepo_clip-s0) full BC+ 12.29. E's QA moved to the right GB300 (lane right-gb300-e); D after (b) on the left GB300 (exp1_chain_d.sh; chain_b killed by PID, its exp1c_queue child kept running). Sidecar v9 eval adds the seed interim grid.
- Outbox 012 asks the research session to rule (i) report vs (ii) E seeds 1-2 (~1.5 day; needs the Founder's go, conflicts with the ACS restart).

## 2026-10-24 02:25 PT - E and D full; outbox 013

- E s0: BC+ 12.29, HQA 44.74, 2Wiki 47.73, MuSiQue 20.36, Bamboogle 46.40 (QA 39.81). D s0: BC+ 7.83, HQA 44.09, 2Wiki 51.19, MuSiQue 20.32, Bamboogle 50.40 (QA 41.50). Both on GB300 lanes; each took about 1 h there.
- C - E s0: QA +3.86 [+2.15, +5.68], BC+ -2.17 [-4.34, +0.00]. C - D: +2.17 / +2.29. F2b not triggered on QA (matched GRPO removes 44% of the gap).
- Correction: in chat I first gave D's BC+ as 11.08 before reading the log; the actual value is 7.83.

## 2026-10-24 06:20 PT - paper verdict (inbox 012); arm E seeds 1-2 launched (Founder "go, run both seeds")

- Verdict (inbox 012):
  - Tier 1 and tier 2 PASS by the letter, LOW confidence (fragility rider leads).
  - F1 mechanism KILL sustained: boundedness, not the posterior.
  - F2 measured, not killed.
  - (ii) recommended; the Founder decides on compute.
- Founder: go. exp1-treepo_clip-s1 on the left GB300 and -s2 on the right GB300 (launched 06:07 / 06:08). exp1_chain_e.sh runs the full eval on the same GB300 after each ends. Merge sidecar v10 adds both runs.
- Disk (Founder: drop the old Kimi, keep it on the NAS):
  - vellum-ai__Kimi-K3-W3A16-g64 was already on the NAS hf-archive. Verified: 880 files, every size matches, 3 shard sha256 match.
  - The 3 local-only small files are kept at ~/model-archive-extras on the Alien (the NAS refuses non-root writes).
  - Deleted from both Stations.
  - exp1-treepo_clip-s0/ckpt (93 GB): copied to the left runs-archive, checksum-verified, removed on the right.
  - Free now: left 986 GB, right 1.4 TB.
- Right GB300 s2 subset lane failed at 02:16 (no interim parquet on the right); the RTX PRO lane covers it.

## 2026-10-24 10:10 PT - F3 best-in-window; RTX re-check; (b) subsets; outbox 015

- Best-in-window C - B (subset means over 3 seeds): qa +0.3, bc +0.5; final-checkpoint subsets +1.5 / +2.5. Selection shrinks the gap, mainly by rescuing B s2's step-75 peak (36.6 vs 32.8 at step 100). F3 kill pattern not met.
- RTX re-check B s0 Bamboogle 45.6 = original.
- (b) step 40: qa 17.8, bc 0.0.
- E s1 / s2 at steps 45 / 46 (10:00).
