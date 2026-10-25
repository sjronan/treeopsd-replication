#!/usr/bin/env bash
# Inbox 011 ruling 4: after both GB300 lanes end and the RTX PRO lane is idle, re-score B s0 Bamboogle on the RTX PRO
# (same settings as the original step100 score) to confirm the eval did not drift over the weekend run.
set -uo pipefail
: "${D:=/models/treeopsd}"; I=$D/interim; IMG=treeopsd-repl:verl0.9.1-vllm0.32.0-fa2
until tail -1 $I/lane-left-gb300.log 2>/dev/null | grep -q "lane end" && tail -1 $I/lane-right-gb300.log 2>/dev/null | grep -q "lane end" \
      && [ -z "$(docker ps -q -f name=^treeopsd-interim-eval$)" ]; do sleep 300; done
out=$I/exp1-treepo-s0/recheck-rtx-bamboogle
echo "$(TZ=America/Los_Angeles date '+%F %T %Z') recheck start" >> $I/sidecar.log
docker run --rm --name treeopsd-rtx-recheck --gpus '"device=0"' --network host --ipc host --shm-size 32g \
  -v "$D:$D" -e D="$D" -e REPO="$D/repl-build" -e PYTHONPATH="$D/repl-build" "$IMG" bash -lc \
  "cd $D/repl-build && RUNS=$I/exp1-treepo-s0 RUN_ID=recheck-rtx-bamboogle CKPT_HF=$D/runs/exp1-treepo-s0/hf_step100 BENCHES=bamboogle GPU_MEM_UTIL=0.5 \
   WIKI_URL=http://STATION_RIGHT_RAIL_IP:8000/retrieve BC_URL=http://127.0.0.1:8001/retrieve \
   bash scripts/run_eval.sh +actor_rollout_ref.model.override_config.attn_implementation=sdpa actor_rollout_ref.model.use_remove_padding=false" \
  > "$out.log" 2>&1 < /dev/null
echo "$(TZ=America/Los_Angeles date '+%F %T %Z') recheck rc=$? $(grep -E '^METRIC' "$out.log" | tr '\n' ' ')" >> $I/sidecar.log
