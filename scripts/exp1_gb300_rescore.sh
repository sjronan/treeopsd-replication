#!/usr/bin/env bash
# 2026-10-24 06:35: after this host's arm-E lane ends, re-score a C run's step-100 checkpoint on this GB300, so that C vs E
# is compared on one GPU type at every seed (E s0-s2 and C s2 are GB300-scored; C s0/s1 were RTX PRO). Output goes to
# <run>/gb300-step100-<bench>/ (the RTX results are kept untouched as the primary record).
#   AFTER_LANE=left-gb300-e1 RUN=exp1-treeopsd-s0 LANE=left-gb300-c WIKI_URL=... BC_URL=... bash exp1_gb300_rescore.sh
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${AFTER_LANE:?}"; : "${RUN:?}"; : "${LANE:?}"; : "${WIKI_URL:?}"; : "${BC_URL:?}"
: "${GPU:=1}"; : "${GMU:=0.85}"; : "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"
I=$D/interim; LOG=$I/lane-$LANE.log; cd "$D/repl-build"
log(){ echo "$(TZ=America/Los_Angeles date '+%F %T %Z') $*" >> "$LOG"; }
until tail -1 "$I/lane-$AFTER_LANE.log" 2>/dev/null | grep -q "lane end"; do sleep 120; done
[ -f "$D/runs/$RUN/hf_step100/config.json" ] || { log "missing $D/runs/$RUN/hf_step100"; log "lane end"; exit 1; }
log "lane $LANE start: rescore $RUN on gpu $GPU"
for b in bcplus hotpotqa 2wikimultihopqa musique bamboogle; do
  out=$I/$RUN/gb300-step100-$b; [ -f "$out/summary.json" ] && continue; mkdir -p "$I/$RUN"
  log "rescore $RUN $b"
  docker run --rm --name "treeopsd-lane-$LANE" --gpus "\"device=$GPU\"" --network host --ipc host --shm-size 32g \
    -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" "$IMG" bash -lc \
    "cd $D/repl-build && RUNS=$I/$RUN RUN_ID=gb300-step100-$b CKPT_HF=$D/runs/$RUN/hf_step100 BENCHES=$b GPU_MEM_UTIL=$GMU WIKI_URL=$WIKI_URL BC_URL=$BC_URL \
     bash scripts/run_eval.sh" > "$out.log" 2>&1 < /dev/null
  log "rescore $RUN $b rc=$? $(grep -E '^METRIC' "$out.log" | tr '\n' ' ')"
done
log "lane end"
