#!/usr/bin/env bash
# EXP-1 interim-eval sidecar. Runs on the LEFT Station host, detached; never touches a GB300.
#
#   MODE=merge setsid -f nohup bash scripts/exp1_sidecar.sh > /dev/null 2>&1 < /dev/null
#   MODE=eval  setsid -f nohup bash scripts/exp1_sidecar.sh > /dev/null 2>&1 < /dev/null
# Two processes, so a long eval can never delay a merge past checkpoint rotation
# (save every 5, keep 2: global_step_N is deleted once N+10 is saved, about 40-60 min later).
#
# Every POLL seconds, for each EXP-1 run on either Station:
#   1. merge: when ckpt/global_step_N (N in STEPS) is complete (latest_checkpointed_iteration.txt >= N),
#      convert it to HF on the Station that holds it (CPU container), before checkpoint rotation removes it;
#   2. pull: copy a right-Station HF dir to the left over rail 1;
#   3. eval: on the left RTX PRO 6000 (GPU 0), score the fixed interim subsets (qa500i + bc200i) for
#      N < 100. Step 100 is reported with the full eval, run separately. Results land in
#      $D/interim/<run>/step<N>/summary.json. Training is never paused (Founder protocol 2026-10-21).
# Step 0 (the shared base model) is evaluated once, first.
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${POLL:=120}"; : "${STEPS:=10 25 50 75 100}"
: "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"; : "${RIGHT:=user@STATION_RIGHT}"
: "${WIKI_URL:=http://STATION_RIGHT_RAIL_IP:8000/retrieve}"; : "${BC_URL:=http://127.0.0.1:8001/retrieve}"
: "${BENCHES:=qa500i bc200i}"; : "${EVAL_GPU:=0}"
LEFT_RUNS="exp1-treeopsd-s0 exp1-grpo-s0 exp1-grpo_tok-s0"; RIGHT_RUNS="exp1-treepo-s0 exp1-treepo_clip-s0"
I=$D/interim; mkdir -p "$I"; LOG=$I/sidecar.log
ts(){ TZ=America/Los_Angeles date '+%F %T %Z'; }
log(){ echo "$(ts) $*" >> "$LOG"; }
MERGE='python3 -m verl.model_merger merge --backend fsdp --local_dir %s/ckpt/global_step_%s/actor --target_dir %s/hf_step%s'
merge_cmd(){ printf "docker run --rm --network none -v $D:$D --entrypoint bash $IMG -lc '$MERGE && chmod -R a+rX %s/hf_step%s && touch %s/hf_step%s/.merged'" "$1" "$2" "$1" "$2" "$1" "$2" "$1" "$2"; }
ready(){ # $1 = local|right, $2 run dir, $3 step
  local T="[ -d $2/ckpt/global_step_$3/actor ] && [ \$(cat $2/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0) -ge $3 ] && [ ! -f $2/hf_step$3/.merged ]"
  if [ "$1" = local ]; then bash -c "$T"; else ssh -n -o BatchMode=yes "$RIGHT" "$T"; fi; }
eval_one(){ # $1 = label dir, $2 = hf dir, $3 = step
  local out=$I/$1/step$3; mkdir -p "$out"
  [ -f "$out/summary.json" ] && return 0
  log "eval $1 step $3 from $2"
  docker run --rm --name treeopsd-interim-eval --gpus "\"device=$EVAL_GPU\"" --network host --ipc host --shm-size 32g \
    -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" "$IMG" bash -lc \
    "cd $D/repl-build && RUNS=$I/$1 RUN_ID=step$3 CKPT_HF=$2 BENCHES='$BENCHES' GPU_MEM_UTIL=0.5 WIKI_URL=$WIKI_URL BC_URL=$BC_URL \
     bash scripts/run_eval.sh +actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.model.use_remove_padding=false" \
    > "$out.log" 2>&1 < /dev/null
  log "eval $1 step $3 rc=$? $(grep -E '^METRIC' "$out.log" | tr '\n' ' ')"
}
: "${MODE:?MODE=merge|eval}"
log "sidecar $MODE start: steps [$STEPS] benches [$BENCHES] wiki $WIKI_URL bc $BC_URL"
if [ "$MODE" = eval ]; then
  eval_one base-4b "$D/model-qwen3-4b-thinking" 0
  while true; do
    for r in $LEFT_RUNS $RIGHT_RUNS; do for n in $STEPS; do
      [ "$n" = 100 ] && continue
      [ -f "$D/runs/$r/hf_step$n/.merged" ] && eval_one "$r" "$D/runs/$r/hf_step$n" "$n"
    done; done
    sleep "$POLL"
  done
fi
while true; do
  for r in $LEFT_RUNS; do for n in $STEPS; do
    if ready local "$D/runs/$r" "$n"; then log "merge $r step $n (left)"; bash -c "$(merge_cmd $D/runs/$r $n)" >> "$LOG" 2>&1; log "merge $r step $n rc=$?"; fi
  done; done
  for r in $RIGHT_RUNS; do for n in $STEPS; do
    if ready right "$D/runs/$r" "$n"; then
      log "merge $r step $n (right)"; ssh -n -o BatchMode=yes "$RIGHT" "$(merge_cmd $D/runs/$r $n)" >> "$LOG" 2>&1; log "merge $r step $n rc=$?"; fi
    if ssh -n -o BatchMode=yes "$RIGHT" "[ -f $D/runs/$r/hf_step$n/.merged ]" && [ ! -f "$D/runs/$r/hf_step$n/.merged" ]; then
      log "pull $r hf_step$n"; mkdir -p "$D/runs/$r"
      rsync -a -e 'ssh -o BatchMode=yes' "$RIGHT:$D/runs/$r/hf_step$n" "$D/runs/$r/" >> "$LOG" 2>&1 && log "pulled $r hf_step$n"; fi
  done; done
  sleep "$POLL"
done
