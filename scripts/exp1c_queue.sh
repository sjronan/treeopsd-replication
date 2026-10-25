#!/usr/bin/env bash
# EXP-1c drift-dose queue (PREREG-EXP1 section 7; research-session ruling 2026-10-22).
# Arms B (treepo) and C (treeopsd) at E in {8, 16} sequential blocks (BE = 64/E), B = 64, 40 steps.
# E = 4 is EXP-1 itself. Runs on ONE Station host, after that Station's EXP-1 queue ends.
#
#   AFTER_LOG=/models/treeopsd/runs/exp1-queue-<host>.log bash exp1c_queue.sh
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${SEED:=0}"; : "${MAX_TRIES:=4}"; : "${GB:=1}"
: "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"; : "${RETRIEVER_URL:=http://127.0.0.1:8000/retrieve}"
: "${AFTER_LOG:?}"; : "${RUNS_1C:=treeopsd:8 treepo:8 treeopsd:16 treepo:16}"
LOG=$D/runs/exp1c-queue-$(hostname -s).log
ts(){ TZ=America/Los_Angeles date '+%F %T %Z'; }
log(){ echo "$(ts) $*" | tee -a "$LOG"; }
log "exp1c queue waiting for 'queue end' in $AFTER_LOG; runs [$RUNS_1C]"
until tail -1 "$AFTER_LOG" 2>/dev/null | grep -q "queue end"; do sleep 120; done
log "exp1c queue start"
for spec in $RUNS_1C; do
  ARM=${spec%%:*}; E=${spec##*:}; BE=$(( 64 / E )); RUN_ID=exp1c-$ARM-E$E-s$SEED; C=treeopsd-exp1c-$ARM-E$E
  for try in $(seq 1 "$MAX_TRIES"); do
    [ -d "$D/runs/$RUN_ID/ckpt/global_step_40" ] && break
    log "launch $RUN_ID (BE=$BE) try $try"
    docker run --rm --name "$C" --gpus "\"device=$GB\"" --network host --ipc host --shm-size 64g \
      -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" -e RETRIEVER_URL="$RETRIEVER_URL" \
      "$IMG" bash -lc "cd $D/repl-build && ARM=$ARM SEED=$SEED RUN_ID=$RUN_ID STEPS=40 BE=$BE TEST_FREQ=-1 SAVE_FREQ=10 VAL_BEFORE=false \
        bash scripts/run_train.sh trainer.max_actor_ckpt_to_keep=2" >> "$D/runs/$RUN_ID.log" 2>&1 < /dev/null
    log "$RUN_ID exited rc=$? at $(grep -oE 'step:[0-9]+ ' "$D/runs/$RUN_ID.log" | tail -1)"
    sleep 30
  done
done
log "exp1c queue end"
