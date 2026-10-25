#!/usr/bin/env bash
# Left GB300, 2026-10-23 20:05 (Founder: "run it and continue in logical order"): after the lane's calibration job,
# train (b) = treepo E=16 seed 1 (inbox 010 condition overridden by the Founder), then resume the left eval lane
# (secondary arms). The lane process was stopped by PID; its running calibration container finishes first.
set -uo pipefail
: "${D:=/models/treeopsd}"; cd "$D/repl-build"
while [ -n "$(docker ps -q -f name=^treeopsd-lane-left-gb300$)" ]; do sleep 30; done
AFTER_LOG=$D/runs/exp1c-queue-$(hostname -s).log RUNS_1C="treepo:16" SEED=1 RETRIEVER_URL=http://STATION_RIGHT_RAIL_IP:8000/retrieve \
  bash scripts/exp1c_queue.sh
LANE=left-gb300 GPU=1 RUNS="exp1-treepo_clip-s0 exp1-grpo_tok-s0" CALIB=exp1-treeopsd-s0:bamboogle \
  WIKI_URL=http://STATION_RIGHT_RAIL_IP:8000/retrieve BC_URL=http://127.0.0.1:8001/retrieve exec bash scripts/exp1_evallane.sh
