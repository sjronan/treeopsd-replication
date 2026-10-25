#!/usr/bin/env bash
# Fixed-checkpoint evaluation of one run on both tiers, inside the replication container.
#
#   RUN_ID=exp1-treeopsd-s0 STEP=100 bash scripts/run_eval.sh            # all benchmarks
#   RUN_ID=react-base CKPT_HF=$D/model-qwen3-4b-thinking BENCHES="hotpotqa" bash scripts/run_eval.sh
#
# Extra hydra overrides pass through ("$@"). Weight-sync bucket 20480 MB: vLLM 0.32 needs tied
# lm_head/embed_tokens in one load_weights call (G1b, 2026-10-21).
# 1) merges the FSDP checkpoint of global_step_$STEP to HF format (skipped if CKPT_HF given)
# 2) for each benchmark runs verl in val_only mode with the SAME agent loop / tool scaffold
#    as training, dumping generations to $OUT/eval/<bench>/
# 3) scripts/eval_qa.py rescoring + telemetry -> $OUT/summary.json -> autoresearch.sh
# Tier 1 (QA): T=0.6, top-p 0.95, 1 sample (paper App. E.3), wiki-18 retriever on :8000
# Tier 2 (headline, BC+ slice): greedy, 1 sample, <= $BC_TURNS assistant turns,
#   16,384-token response budget, BC+ retriever on :8001 (research-session decision 2026-10-21)
set -euo pipefail
: "${RUN_ID:?}"; : "${D:=/models/treeopsd}"; : "${REPO:=/workspace/treeopsd}"; : "${RUNS:=$D/runs}"
: "${STEP:=100}"; : "${BENCHES:=hotpotqa 2wikimultihopqa musique bamboogle bcplus}"
: "${BC_TURNS:=20}"; : "${GPU_MEM_UTIL:=0.8}"; : "${EVAL_SEED:=0}"
: "${WIKI_URL:=http://127.0.0.1:8000/retrieve}"; : "${BC_URL:=http://127.0.0.1:8001/retrieve}"
# Interim subsets (Founder protocol 2026-10-21, fixed seed 0): qa500i = 500 of val_dev, bc200i = 200 of the BC+ slice.
OUT="$RUNS/$RUN_ID"; mkdir -p "$OUT/eval"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" VERL_EXTRA_MODULES=treeopsd.verl_ext.losses
if [[ -z "${CKPT_HF:-}" ]]; then
  CKPT_HF="$OUT/hf_step$STEP"
  [[ -d "$CKPT_HF" ]] || python3 -m verl.model_merger merge --backend fsdp \
      --local_dir "$OUT/ckpt/global_step_$STEP/actor" --target_dir "$CKPT_HF"
fi
for BENCH in $BENCHES; do
  if [[ "$BENCH" == bcplus || "$BENCH" == bc200i ]]; then
    if [[ "$BENCH" == bc200i ]]; then VAL="$D/prepared/bcplus/interim_bc200.parquet"
    else VAL="$D/prepared/bcplus/eval_bcplus.parquet"; fi; URL=$BC_URL
    TOOLS="$REPO/configs/tools/search_bcplus.yaml"; TURNS=$BC_TURNS; RESP=16384
    VK=(actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.do_sample=false)
  else
    if [[ "$BENCH" == hotpotqa500 ]]; then VAL="$D/prepared/mhqa/g6_hotpot500.parquet"   # G6 subset
    elif [[ "$BENCH" == qa500i ]]; then VAL="$D/prepared/mhqa/interim_qa500.parquet"
    else VAL="$D/prepared/mhqa/eval_$BENCH.parquet"; fi
    URL=$WIKI_URL
    TOOLS="$REPO/configs/tools/search_wiki18.yaml"; TURNS=5; RESP=6144
    VK=(actor_rollout_ref.rollout.val_kwargs.temperature=0.6 actor_rollout_ref.rollout.val_kwargs.top_p=0.95
        actor_rollout_ref.rollout.val_kwargs.do_sample=true)
  fi
  export TREEOPSD_RETRIEVER_URL=$URL
  ( cd "$REPO" && python3 -m treeopsd.verl_ext.main_tree \
      data.train_files="$D/prepared/mhqa/val_dev.parquet" data.val_files="$VAL" \
      data.max_prompt_length=2048 data.max_response_length=$RESP data.return_raw_chat=true \
      data.train_batch_size=16 data.val_batch_size=null \
      actor_rollout_ref.model.path="$CKPT_HF" actor_rollout_ref.actor.ppo_mini_batch_size=16 \
      actor_rollout_ref.actor.shuffle=false actor_rollout_ref.actor.ppo_epochs=1 \
      actor_rollout_ref.actor.use_dynamic_bsz=true actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
      algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false actor_rollout_ref.model.use_remove_padding=true \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.enable_prefix_caching=true \
      actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
      actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
      actor_rollout_ref.rollout.prompt_length=2048 actor_rollout_ref.rollout.response_length=$RESP \
      actor_rollout_ref.rollout.seed=$EVAL_SEED actor_rollout_ref.rollout.val_kwargs.n=1 "${VK[@]}" \
      actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$TURNS \
      actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 actor_rollout_ref.rollout.multi_turn.format=hermes \
      actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOLS" \
      actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO/configs/agent_loops.yaml" \
      actor_rollout_ref.rollout.agent.default_agent_loop=tree_tool_agent \
      reward.custom_reward_function.path="$REPO/treeopsd/verl_ext/reward.py" reward.custom_reward_function.name=compute_score \
      trainer.use_v1=false trainer.balance_batch=false trainer.n_gpus_per_node=1 trainer.nnodes=1 \
      trainer.val_only=true trainer.val_before_train=true trainer.validation_data_dir="$OUT/eval/$BENCH" \
      trainer.project_name=treeopsd-replication trainer.experiment_name="$RUN_ID-eval-$BENCH" "trainer.logger=[console]" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$REPO" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_EXTRA_MODULES=treeopsd.verl_ext.losses" \
      "+ray_kwargs.ray_init.runtime_env.env_vars.TREEOPSD_RETRIEVER_URL=$URL" \
      +treeopsd.tree=false +treeopsd.prompts_per_block=16 \
      actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=20480 "$@" ) 2>&1 | tee "$OUT/eval/$BENCH.log"
done
DUMPS=(); for BENCH in $BENCHES; do DUMPS+=(--dump "$BENCH=$OUT/eval/$BENCH"); done
python3 "$REPO/scripts/eval_qa.py" --run-dir "$OUT" "${DUMPS[@]}" --prepared "$D/prepared" \
  --telemetry "$OUT/telemetry" --arm "${ARM:-}" --seed "${SEED:-0}" --model "$CKPT_HF" --image "${IMAGE_DIGEST:-}"
bash "$REPO/autoresearch.sh" "$OUT"
