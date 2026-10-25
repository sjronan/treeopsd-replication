#!/usr/bin/env bash
# EXP-1 arm queue for ONE Station (runs on the Station host, detached).
#
#   QUEUE="treeopsd grpo grpo_tok" G_TOK=9 RETRIEVER_URL=... bash exp1_queue.sh
#
# Each arm runs to step 100 in its own container (treeopsd-exp1-<arm>) on the GB300.
# Founder protocol (2026-10-21): no pause, checkpoints at 10/25/50/75/100 (save every 5,
# keep the last 2; the eval sidecar converts 10/25/50/75/100 to HF before they rotate out),
# no in-training validation (interim evals run on side devices).
# A run that exits without reaching step 100 is relaunched with the same RUN_ID
# (verl resume_mode=auto continues from the latest checkpoint), up to MAX_TRIES times.
set -uo pipefail
: "${QUEUE:?}"; : "${SEED:=0}"; : "${D:=/models/treeopsd}"; : "${MAX_TRIES:=4}"
: "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"; : "${GB:=1}"
: "${RETRIEVER_URL:=http://127.0.0.1:8000/retrieve}"; : "${G_TOK:=8}"
LOG=$D/runs/exp1-queue-$(hostname -s).log
ts(){ TZ=America/Los_Angeles date '+%F %T %Z'; }
log(){ echo "$(ts) $*" | tee -a "$LOG"; }
done_100(){ [ -d "$D/runs/$1/ckpt/global_step_100" ] || grep -qE 'step:100 ' "$D/runs/$1.log" 2>/dev/null; }
log "queue start: [$QUEUE] seed $SEED img $IMG retriever $RETRIEVER_URL G_TOK $G_TOK"
for ARM in $QUEUE; do
  RUN_ID=exp1-$ARM-s$SEED; C=treeopsd-exp1-$ARM
  for try in $(seq 1 "$MAX_TRIES"); do
    if done_100 "$RUN_ID"; then log "$RUN_ID already at step 100"; break; fi
    if docker ps -a --format '{{.Names}}' | grep -qx "$C"; then log "$C exists; waiting for it"; docker wait "$C" >/dev/null 2>&1; continue; fi
    log "launch $RUN_ID try $try"
    docker run --rm --name "$C" --gpus "\"device=$GB\"" --network host --ipc host --shm-size 64g \
      -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" -e RETRIEVER_URL="$RETRIEVER_URL" \
      "$IMG" bash -lc "cd $D/repl-build && ARM=$ARM SEED=$SEED RUN_ID=$RUN_ID G_TOK=$G_TOK TEST_FREQ=-1 SAVE_FREQ=5 VAL_BEFORE=false \
        bash scripts/run_train.sh trainer.max_actor_ckpt_to_keep=2" >> "$D/runs/$RUN_ID.log" 2>&1 < /dev/null
    rc=$?; last=$(grep -oE 'step:[0-9]+ ' "$D/runs/$RUN_ID.log" | tail -1)
    log "$RUN_ID exited rc=$rc at ${last:-no step}"
    if done_100 "$RUN_ID"; then log "$RUN_ID DONE"; break; fi
    sleep 30
  done
  done_100 "$RUN_ID" || log "$RUN_ID FAILED after $MAX_TRIES tries; moving to next arm"
done
log "queue end"
