# Phase 0 runbook (left Station first)

Run these steps in order, **only after the controller says the Stations are free.**
Each step lists the command, the expected output, and a STOP condition. Log every step in
`experiments/worklog.md`, with times in PT and the stdout lines it asks for.

## House rules for every command

- Wrap remote commands as `timeout <N>m ssh -n -o BatchMode=yes user@STATION_LEFT '<cmd>'`.
- Start long jobs detached: `setsid -f nohup <cmd> > <log> 2>&1 < /dev/null`.
- Kill a process only by PID, never with `pkill -f`.
- Before touching a GPU, reserve it with the hardware owner and release it afterwards.
- Do not run station-mode. Do not stop or start any container that is not named `treeopsd-*`.
- Shell variables used below:

```bash
S=user@STATION_LEFT          # left Station
D=/models/treeopsd        # data root (already staged, see section 0)
R=$D/repl                 # repo copy on the Station (rsync from the local repo; no git remote)
IMG=treeopsd-repl:verl0.9.1-vllm0.32.0
```

## 0. Already done (2026-10-21, CPU only, no GPU or docker)

All files were fetched with sha256 verified against `scripts/manifest.tsv` (`$D/fetch.log`).

| What | Detail |
|---|---|
| Downloads | EXP-1 set + BC+ tier, 107 GB in about 10 min |
| `wiki18/wiki-18.jsonl` | 21,015,324 lines, 14,393,573,105 B, sha256 `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d`. Extracted with `tar` because the `.gz` is a tar archive (see DATA.md). |
| `wiki18-index/e5_Flat.index` | 64,559,075,373 B, sha256 `69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166` |
| `wiki18-index/e5_vectors.f32.npy` | [21,015,324 × 768] fp32. Bit-identical to `faiss.reconstruct` on spot checks; norms ≈ 1. |
| `prepared/mhqa/*` | see `prepared/mhqa/manifest.json`. train 90,447; eval 7,405 / 12,576 / 2,417 / 125; val_dev 1,625; MuSiQue hops 2/3/4 = 1,252 / 760 / 405. |
| `prepared/mhqa/g6_hotpot500.parquet` | sha256 `c2fd623a3c7d5d2938d4b8bbe78f9be5c37958b6dd5c00e915cb47d6213e389d` |
| `prepared/bcplus/eval_bcplus.parquet` | 830 rows, sha256 `ecb2bd6a40fa090ad941ec4c15f03113f74314c1d1c60834e25730fa7a853f2c`, query-id list sha256 `74e81cb2…2b1a`. Holds decrypted benchmark text: never publish. |
| `bcplus-corpus/corpus.jsonl` | 100,195 docs |

Still to do on CPU:
- **BC+ index pickle conversion.** It must run inside the container with no network (step 2b).
- Optionally delete `wiki18-index/part_aa` and `part_ab` (64.6 GB). Run `stat` first, and only after G3 passes.

## 1. Build the image (left Station, aarch64 native)

```bash
rsync -a --delete --exclude .git --exclude data --exclude runs ./ $S:$R/      # from the local repo
timeout 5m ssh -n -o BatchMode=yes $S "cd $R && setsid -f nohup docker build -f env/Dockerfile -t $IMG . > $D/build.log 2>&1 < /dev/null"
# poll:
timeout 2m ssh -n -o BatchMode=yes $S "tail -5 $D/build.log; docker images $IMG --digests"
```

**Expected:**
- `build.log` shows the base constraints list (torch / vllm / transformers versions).
- `pytest` inside the build ends with `86 passed` (or more), 0 failed, 0 skipped. Inside the image verl is importable, so no tests are skipped.
- The final line prints `verl 0.9.1 vllm 0.32.0 torch 2.x cuda 13.0 transformers 5.x`.

Record the image ID / digest in the worklog and in the `autoresearch.jsonl` header.

**STOP if:**
- pip reports a constraint conflict (typically transformers outside verl's `>=5.5.3,!=5.6.0,<5.11`);
- `git apply --check` fails on the patch;
- any test fails.

**Fallback:** set `BASE_IMAGE=vllm/vllm-openai@sha256:3af6a45b…` (nightly 0.30.1rc1) and rebuild. If that also fails, use verl's `docker/Dockerfile.stable.vllm` at commit `1876b06d` plus our patch. Log which one was used.

## 2. GPU discovery (read-only)

```bash
timeout 2m ssh -n -o BatchMode=yes $S "nvidia-smi -L; nvidia-smi topo -m; nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv"
```

Set `GB=<index of the GB300>` and `PRO=<index of the RTX PRO 6000>`. Record the topology in the worklog.

## 2b. BC+ pickle conversion (CPU, container, no network)

```bash
timeout 30m ssh -n -o BatchMode=yes $S "docker run --rm --network none -v $D:$D $IMG \
  python3 retriever/build_vectors.py bcplus-index $D/bcplus-index-q3e8b/qwen3-embedding-8b $D/bcplus-index-q3e8b"
```

**Expected:** 4 shards, `vectors (100195, 4096)`, norm mean ≈ 1.

**STOP if** the row count is not 100,195. The shards' sha256 were already verified by the fetch.

## 3. G1: verl + vLLM 0.32 on the GB300

**3a. Imports and verl's own CPU tests, inside the image:**

```bash
timeout 20m ssh -n -o BatchMode=yes $S "docker run --rm --gpus '\"device=$GB\"' -v $D:$D $IMG bash -lc '
  python3 -c \"import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))\";
  cd /opt/verl && python3 -m pytest -q tests/trainer/ppo/test_core_algos_on_cpu.py tests/experimental/agent_loop/test_qwen3_tool_parser_on_cpu.py'"
```

**Expected:** device name contains GB300 and the capability is (10, 3). Both verl test files pass.

**3b. A 2-step GRPO run of the real pipeline on Qwen3-1.7B.** This needs the retriever from step 4, so run step 4 first and come back.

```bash
timeout 5m ssh -n -o BatchMode=yes $S "setsid -f nohup docker run --rm --name treeopsd-g1 --gpus '\"device=$GB\"' --network host --ipc host --shm-size 64g \
  -v $D:$D -e D=$D $IMG bash -lc 'ARM=grpo SEED=0 RUN_ID=g1-grpo-2step STEPS=2 B=8 BE=2 MODEL_PATH=$D/model-qwen3-1.7b \
  TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE=false bash scripts/run_train.sh' \
  > $D/runs/g1.log 2>&1 < /dev/null"
```

**Expected:**
- Two `step:1` / `step:2` metric lines in `$D/runs/g1-grpo-2step/train.log`, with non-zero `response_length/mean` and `reward/mean` ≥ 0.
- `telemetry/steps.jsonl` has 2 lines.

**STOP if** vLLM fails to start on sm_103, the weight sync errors, or the agent loop raises. In that case switch to the fallback image (step 1).

**3c. The same 2-step check for a tree arm.** Use `ARM=treeopsd RUN_ID=g1-tree-2step STEPS=2 B=8 BE=2 MODEL_PATH=$D/model-qwen3-1.7b TEST_FREQ=-1 SAVE_FREQ=-1 VAL_BEFORE=false`.

**Expected:**
- `telemetry/steps.jsonl` contains `tree/L_x`, `budget/suffix_fraction` and `budget/traj_equivalents`.
- `telemetry/loss_*.jsonl` exists with `treeopsd/drift_rel`.

## 4. Retriever up on the RTX PRO 6000 (wiki-18, port 8000)

```bash
timeout 5m ssh -n -o BatchMode=yes $S "setsid -f nohup docker run --rm --name treeopsd-retriever-wiki --gpus '\"device=$PRO\"' --network host -v $D:$D $IMG \
  python3 retriever/server.py --backend wiki18 --device cuda:0 --port 8000 \
    --vectors $D/wiki18-index/e5_vectors.f32.npy --corpus $D/wiki18/wiki-18.jsonl --encoder $D/model-e5-base-v2 \
    --log $D/runs/retriever-wiki-$(date +%Y%m%d).jsonl > $D/runs/retriever-wiki.log 2>&1 < /dev/null"
timeout 2m ssh -n -o BatchMode=yes $S "curl -s localhost:8000/health"
```

**Expected:**
- The log shows `loaded torch.Size([21015324, 768]) on cuda:0`. The first start also builds `wiki-18.jsonl.offsets.npy` (about 1 to 2 min).
- `/health` returns `n_vectors 21015324`.
- GPU memory in use is about 66 GB.

## 5. G3: exact search equals faiss-cpu

```bash
timeout 30m ssh -n -o BatchMode=yes $S "docker run --rm --network host -v $D:$D $IMG \
  python3 retriever/check_vs_faiss.py --server http://127.0.0.1:8000/retrieve --index $D/wiki18-index/e5_Flat.index \
  --encoder $D/model-e5-base-v2 --questions $D/mhqa/hotpotqa/dev.jsonl --corpus $D/wiki18/wiki-18.jsonl --n 1000 --threads 64"
```

**Expected:**
- every mismatch tie-only (equal score at every rank, |ds| < 1e-6). Amended 2026-10-21 by the Founder: the old "≥ 999 identical" bar ignored wiki-18's duplicate passages (~5% of lists tie).
- `lat_p95_s` < 0.1.
- The last line is `G3 PASS`. Also record `e2e_identical_of_400`, which is informational.

**STOP if** the check fails. A mismatch that is not a tie means the vector row order or the matmul precision is wrong.

**BC+ retriever (port 8001)** is needed only for tier-2 evals. It can share the PRO 6000: about 18 GB on top of the 66 GB.

```bash
timeout 5m ssh -n -o BatchMode=yes $S "setsid -f nohup docker run --rm --name treeopsd-retriever-bcplus --gpus '\"device=$PRO\"' --network host -v $D:$D $IMG \
  python3 retriever/server.py --backend bcplus --device cuda:0 --port 8001 --vectors $D/bcplus-index-q3e8b/vectors.f32.npy \
  --docids $D/bcplus-index-q3e8b/docids.json --corpus $D/bcplus-corpus/corpus.jsonl --encoder $D/model-qwen3-embedding-8b \
  > $D/runs/retriever-bcplus.log 2>&1 < /dev/null"
timeout 20m ssh -n -o BatchMode=yes $S "docker run --rm --network host -v $D:$D $IMG python3 retriever/check_bcplus.py \
  --server http://127.0.0.1:8001/retrieve --parquet $D/prepared/bcplus/eval_bcplus.parquet"     # G3b
```

**G3b:** compare the recalls with the BrowseComp-Plus paper's Qwen3-Embedding-8B retriever-only numbers. **STOP if** recall is far below them; that points to a query-prompt mismatch.

## 6. G4 / G5 / G7 inside the image (CPU unit tests)

These already ran in step 1's build. To re-run on demand:

```bash
timeout 10m ssh -n -o BatchMode=yes $S "docker run --rm -v $D:$D $IMG python3 -m pytest -q env/smoke -k 'patched_ppo or rollout_manager or first_block or reference_loss'"
```

**Expected:** all pass.
- **G4:** `test_patched_ppo_loss_forwards_tree_inputs`.
- **G5:** `test_tree_rollout_manager_order_prefix_and_meta`.
- **G7 (CPU form):** `test_first_block_treeopsd_equals_treepo_gradient`.

The GPU form of G7 is step 7's check that `treeopsd/drift_rel` ≈ 0 on block 1.

## 7. S1a: 1.7B TreeOPSD shakedown, 30 steps (GB300; about 2 GB300-h)

```bash
timeout 5m ssh -n -o BatchMode=yes $S "setsid -f nohup docker run --rm --name treeopsd-s1a --gpus '\"device=$GB\"' --network host --ipc host --shm-size 64g \
  -v $D:$D -e D=$D $IMG bash -lc 'ARM=shakedown17 SEED=0 RUN_ID=s1a-shakedown17-s0 bash scripts/run_train.sh' > $D/runs/s1a.log 2>&1 < /dev/null"
```

**Expected:**
- 30 steps complete.
- `reward/mean` is not flat at 0.
- In `telemetry/steps.jsonl`, `tree/V_abs_max_over_bound` ≤ 1 and `budget/traj_equivalents` is about 8 (record it).
- `telemetry/loss_*.jsonl`: per step, the first optimizer block has `treeopsd/s_abs_mean` ≈ 0 (< 1e-3, from bf16 kernel noise), and blocks 2 to 4 are > 0. This is the GPU form of G7.
- Checkpoints exist at steps 10, 20 and 30.

**STOP if any of these happens:**
- reward is NaN;
- `tree/n_nodes` is less than 12 per prompt;
- the `prefix length mismatch` error appears;
- the agent loop raises tokenization sanity errors.

For a tokenization sanity error, rerun with `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable` and log it.

## 8. G6: ReAct baseline and throughput (4B)

**8a. ReAct baseline:** the untrained Qwen3-4B-Thinking-2507 on the fixed 500-question HotpotQA dev subset.

```bash
timeout 5m ssh -n -o BatchMode=yes $S "setsid -f nohup docker run --rm --name treeopsd-g6 --gpus '\"device=$GB\"' --network host --ipc host --shm-size 64g \
  -v $D:$D -e D=$D $IMG bash -lc 'RUN_ID=g6-react-4b CKPT_HF=$D/model-qwen3-4b-thinking BENCHES=hotpotqa500 \
  bash scripts/run_eval.sh' > $D/runs/g6.log 2>&1 < /dev/null"
```

`BENCHES=hotpotqa500` selects the fixed 500-question subset `prepared/mhqa/g6_hotpot500.parquet`, with the same scaffold (T = 0.6, top-p 0.95, one sample).

**Expected:** EM within ±5 of the paper's ReAct 31.61 (HotpotQA). Answer rate > 90%.

**STOP if** EM < 20 or the answer rate < 80%. Either one means a prompt or tool-format problem.

**8b. S1b smoke:** 4B TreeOPSD and GRPO, 10 steps each, B = 64.

```bash
... ARM=smoke4b SEED=0 RUN_ID=s1b-smoke4b-tree ...   then   ARM=grpo STEPS=10 RUN_ID=s1b-smoke4b-grpo ...
```

From the two `telemetry/steps.jsonl` files, compute:
- generated tokens per trajectory for GRPO, and tokens per prompt for the tree;
- tool calls;
- step wall time split (`timing_s/gen`, `timing_s/update_actor`, `timing_s/old_log_prob` in train.log);
- `budget/single_turn_root_frac` and `budget/suffix_fraction`.

Then:
1. Put the measured values into `scripts/compute_estimate.py` (replacing `DECODE_TPS`, `MHQA.gen_tok`, `MHQA.ctx_tok`) and re-run it.
2. Set `G_TOK = round(tree_gen_tokens_per_prompt / grpo_gen_tokens_per_trajectory)` for arm D (PREREG section 2).
3. Commit the updated estimate to the repo.

**STOP and message the research session if** the measured per-step cost implies that EXP-1 (3 arms × 100 steps) exceeds 3 GB300-days on two Stations. Propose B=32 or 60 steps as a dated PREREG amendment.

## 9. Exit criteria for Phase 0 (then start EXP-1)

Phase 0 is complete when all of the following hold:
- G0 to G7 pass.
- S1a completed.
- G6 recorded.
- G_TOK fixed.
- Image digest, model revisions and corpus/index hashes are in the `autoresearch.jsonl` header line.

The completion report to the research session (outbox) must include:
- the exact checkpoint revisions:
  - `Qwen/Qwen3-4B-Thinking-2507@768f209d9ea81521153ed38c47d515654e938aea`
  - `Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
  - `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd`
  - `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- the corpus/index sha256s (section 0);
- the prepared-set manifests.

**EXP-1 launch, one arm per Station, seed 0 first.** Right Station: rsync `$D` first over the Station link, then rebuild the image or `docker save | ssh docker load`.

```bash
ARM=treeopsd SEED=0 RUN_ID=exp1-treeopsd-s0 bash scripts/run_train.sh     # left
ARM=treepo   SEED=0 RUN_ID=exp1-treepo-s0   bash scripts/run_train.sh     # right
ARM=grpo     SEED=0 RUN_ID=exp1-grpo-s0     bash scripts/run_train.sh     # whichever frees first
# then, per run:  RUN_ID=... ARM=... SEED=0 STEP=100 bash scripts/run_eval.sh   (needs retrievers :8000 and :8001)
```
