#!/usr/bin/env bash
# One training run of one arm, inside the replication container on ONE GB300.
#
#   ARM=treeopsd SEED=0 RUN_ID=exp1-treeopsd-s0 bash scripts/run_train.sh [extra hydra overrides]
#
# ARM presets (PREREG-EXP1.md section 2; configs/arms.md has the table):
#   shakedown17  Qwen3-1.7B, TreeOPSD, B=64, 30 steps               (S1a)
#   smoke4b      Qwen3-4B-Thinking-2507, TreeOPSD, B=64, 10 steps   (S1b, measures tokens/turns)
#   grpo         GRPO G=8                                            (arm A)
#   grpo_tok     GRPO G=$G_TOK (set G_TOK from S1b measurement)      (arm D, F2)
#   treepo       tree (4,2), fixed credit, no ratio/clip             (arm B, F1 control)
#   treepo_clip  tree (4,2), PPO-clip with A=V                       (arm E, secondary)
#   treeopsd     tree (4,2), forward-KL posterior loss, rho=0.25     (arm C)
# Tree shape overrides for EXP-2: TREE_M, TREE_N (rollout.n becomes M(N+1)).
set -euo pipefail
: "${ARM:?set ARM}"; : "${SEED:=0}"; : "${RUN_ID:=${ARM}-s${SEED}}"
: "${D:=/models/treeopsd}"                          # data root (DATA.md)
: "${REPO:=/workspace/treeopsd}"
: "${RUNS:=$D/runs}"
: "${RETRIEVER_URL:=http://127.0.0.1:8000/retrieve}"
: "${GPU_MEM_UTIL:=0.45}"
: "${STEPS:=100}"; : "${B:=64}"; : "${BE:=16}"     # B prompts per collection, E = B/BE = 4 blocks
: "${TREE_M:=4}"; : "${TREE_N:=2}"; : "${G_TOK:=8}"
MODEL="$D/model-qwen3-4b-thinking"; MODEL_REV=768f209d9ea81521153ed38c47d515654e938aea
: "${TEST_FREQ:=10}"; : "${SAVE_FREQ:=10}"; : "${VAL_BEFORE:=true}"
case "$ARM" in
  shakedown17) MODEL="$D/model-qwen3-1.7b"; MODEL_REV=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; STEPS=${STEPS_OVERRIDE:-30}; KIND=tree; LOSS=treeopsd_fkl ;;
  smoke4b)     STEPS=${STEPS_OVERRIDE:-10}; KIND=tree; LOSS=treeopsd_fkl ;;
  grpo)        KIND=flat; G=8 ;;
  grpo_tok)    KIND=flat; G=$G_TOK ;;
  treepo)      KIND=tree; LOSS=treepo_fixed ;;
  treepo_clip) KIND=tree; LOSS=treepo_clip ;;
  treeopsd)    KIND=tree; LOSS=treeopsd_fkl ;;
  *) echo "unknown ARM $ARM" >&2; exit 2 ;;
esac
if [[ -n "${MODEL_PATH:-}" ]]; then MODEL="$MODEL_PATH"; MODEL_REV="${MODEL_REV_OVERRIDE:-see-path}"; fi
OUT="$RUNS/$RUN_ID"; mkdir -p "$OUT/telemetry"
export TREEOPSD_TELEMETRY_DIR="$OUT/telemetry" TREEOPSD_RETRIEVER_URL="$RETRIEVER_URL"
export VERL_EXTRA_MODULES=treeopsd.verl_ext.losses PYTHONPATH="$REPO:${PYTHONPATH:-}"

COMMON=(
  data.train_files="$D/prepared/mhqa/train.parquet"
  data.val_files="$D/prepared/mhqa/val_dev.parquet"
  data.train_batch_size=$B
  data.max_prompt_length=2048
  data.max_response_length=6144
  data.return_raw_chat=true
  data.shuffle=true
  data.seed=$SEED
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.01
  "actor_rollout_ref.actor.optim.betas=[0.9,0.999]"
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.actor.optim.clip_grad=1.0
  actor_rollout_ref.actor.ppo_mini_batch_size=$BE
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.use_dynamic_bsz=true
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.clip_ratio=0.2
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL
  actor_rollout_ref.rollout.enable_prefix_caching=true
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.prompt_length=2048
  actor_rollout_ref.rollout.response_length=6144
  actor_rollout_ref.rollout.seed=$SEED
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$REPO/configs/tools/search_wiki18.yaml"
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO/configs/agent_loops.yaml"
  actor_rollout_ref.rollout.agent.default_agent_loop=tree_tool_agent
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=true
  reward.custom_reward_function.path="$REPO/treeopsd/verl_ext/reward.py"
  reward.custom_reward_function.name=compute_score
  algorithm.use_kl_in_reward=false
  trainer.use_v1=false
  trainer.balance_batch=false
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.total_training_steps=$STEPS
  trainer.test_freq=$TEST_FREQ
  trainer.save_freq=$SAVE_FREQ
  trainer.val_before_train=$VAL_BEFORE
  trainer.validation_data_dir="$OUT/val_dumps"
  trainer.rollout_data_dir="$OUT/rollout_dumps"
  trainer.default_local_dir="$OUT/ckpt"
  trainer.project_name=treeopsd-replication
  trainer.experiment_name=$RUN_ID
  "trainer.logger=[console,file]"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_EXTRA_MODULES=treeopsd.verl_ext.losses"
  "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=$REPO"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TREEOPSD_TELEMETRY_DIR=$OUT/telemetry"
  "+ray_kwargs.ray_init.runtime_env.env_vars.TREEOPSD_RETRIEVER_URL=$RETRIEVER_URL"
)
if [[ "$KIND" == tree ]]; then
  K=$(( TREE_M * (TREE_N + 1) ))
  ARMCFG=(
    actor_rollout_ref.rollout.n=$K
    algorithm.adv_estimator=tree_backup
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS
    actor_rollout_ref.actor.loss_agg_mode=token-sum
    +treeopsd.tree=true +treeopsd.M=$TREE_M +treeopsd.N=$TREE_N +treeopsd.rho=${RHO:-0.25}
    +treeopsd.weighting=${WEIGHTING:-likelihood} +treeopsd.prompts_per_block=$BE +treeopsd.seed=$SEED
  )
else
  ARMCFG=(
    actor_rollout_ref.rollout.n=$G
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=true
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    +treeopsd.tree=false +treeopsd.prompts_per_block=$BE +treeopsd.seed=$SEED
  )
fi
{
  echo "run_id=$RUN_ID arm=$ARM seed=$SEED model=$MODEL@$MODEL_REV steps=$STEPS B=$B BE=$BE"
  echo "date=$(date -Is) host=$(hostname)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
  python3 -c "import verl,vllm,torch;print('verl',verl.__version__,'vllm',vllm.__version__,'torch',torch.__version__)"
  printf '%s\n' "${COMMON[@]}" "${ARMCFG[@]}" actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=20480 "$@"
} > "$OUT/run_header.txt"
cd "$REPO"; set -o pipefail
python3 -m treeopsd.verl_ext.main_tree "${COMMON[@]}" "${ARMCFG[@]}" actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=20480 "$@" 2>&1 | tee -a "$OUT/train.log"
