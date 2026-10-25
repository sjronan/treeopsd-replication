#!/usr/bin/env bash
# Right GB300 (2026-10-24 02:30): seed-2 B/C interim subsets (qa500i + bc200i at 10/25/50/75; best-in-window grid, PREREG F3
# secondary). Both arms on one GPU type (D15 pairing). Results go to the right interim tree; exp1_pull_right2.sh copies them left.
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${GPU:=1}"; : "${GMU:=0.85}"; : "${LANE:?}"; : "${RUNS:?}"; : "${STEPS:=10 25 50 75}"
: "${IMG:=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2}"; : "${WIKI_URL:?}"; : "${BC_URL:?}"
I=$D/interim; LOG=$I/lane-$LANE.log
log(){ echo "$(TZ=America/Los_Angeles date '+%F %T %Z') $*" >> "$LOG"; }
log "lane $LANE start: runs [$RUNS] steps [$STEPS]"
for r in $RUNS; do for n in $STEPS; do
  out=$I/$r/step$n; [ -f "$out/summary.json" ] && continue
  mkdir -p "$out"; log "eval $r step $n"
  docker run --rm --name "treeopsd-lane-$LANE" --gpus "\"device=$GPU\"" --network host --ipc host --shm-size 32g \
    -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" "$IMG" bash -lc \
    "cd $D/repl-build && RUNS=$I/$r RUN_ID=step$n CKPT_HF=$D/runs/$r/hf_step$n BENCHES='qa500i bc200i' GPU_MEM_UTIL=$GMU WIKI_URL=$WIKI_URL BC_URL=$BC_URL \
     bash scripts/run_eval.sh" > "$out.log" 2>&1 < /dev/null
  log "eval $r step $n rc=$? $(grep -E '^METRIC' "$out.log" | tr '\n' ' ')"
done; done
log "lane end"
