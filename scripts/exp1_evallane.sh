#!/usr/bin/env bash
# EXP-1 full-eval lane on a GB300 (2026-10-23 19:50). Training is over, so each GB300 becomes an eval lane
# (Founder "safe plan": evals only once training on that GPU has ended). Static partition, no shared locks:
#   left GB300:  secondary arms (exp1-treepo_clip-s0 exp1-grpo_tok-s0)
#   right GB300: seed-2 pair (exp1-treepo-s2 exp1-treeopsd-s2), so each B-vs-C seed pair is scored on one GPU type
#   left RTX PRO 6000: everything else (sidecar v8 eval, FULL_ORDER trimmed to match)
# First job is a CALIBRATION re-score of one bench already scored on the RTX PRO (sdpa, no remove-padding), to measure
# the GB300 (flash-attn, remove-padding) eval offset; written to <run>/calib-gb300-<bench>/, never used as a result.
#   LANE=left-gb300 RUNS="..." CALIB=run:bench WIKI_URL=... BC_URL=... setsid -f nohup bash scripts/exp1_evallane.sh ...
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${GPU:=1}"; : "${GMU:=0.85}"; : "${LANE:?}"; : "${RUNS:?}"; : "${CALIB:=}"
: "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"; : "${WIKI_URL:?}"; : "${BC_URL:?}"
: "${BENCHES_FULL:=bcplus hotpotqa 2wikimultihopqa musique bamboogle}"
I=$D/interim; LOG=$I/lane-$LANE.log
ts(){ TZ=America/Los_Angeles date '+%F %T %Z'; }
log(){ echo "$(ts) $*" >> "$LOG"; }
job(){ # $1 run, $2 bench, $3 out label
  local out=$I/$1/$3; mkdir -p "$I/$1"
  [ -f "$out/summary.json" ] && return 0
  [ -f "$out.failed2" ] && return 0
  log "full eval $1 $2 -> $3"
  docker run --rm --name "treeopsd-lane-$LANE" --gpus "\"device=$GPU\"" --network host --ipc host --shm-size 32g \
    -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" "$IMG" bash -lc \
    "cd $D/repl-build && RUNS=$I/$1 RUN_ID=$3 CKPT_HF=$D/runs/$1/hf_step100 BENCHES=$2 GPU_MEM_UTIL=$GMU WIKI_URL=$WIKI_URL BC_URL=$BC_URL \
     bash scripts/run_eval.sh" > "$out.log" 2>&1 < /dev/null
  log "full eval $1 $2 -> $3 rc=$? $(grep -E '^METRIC' "$out.log" | tr '\n' ' ')"
  if [ ! -f "$out/summary.json" ]; then [ -f "$out.failed1" ] && touch "$out.failed2" || { touch "$out.failed1"; job "$@"; }; fi
}
log "lane $LANE start: gpu $GPU runs [$RUNS] calib [$CALIB]"
[ -n "$CALIB" ] && job "${CALIB%%:*}" "${CALIB##*:}" "calib-gb300-${CALIB##*:}"
for r in $RUNS; do
  until [ -f "$D/runs/$r/hf_step100/.merged" ]; do sleep 120; done
  for b in $BENCHES_FULL; do job "$r" "$b" "step100-$b"; done
done
log "lane end"
