#!/usr/bin/env bash
# Left GB300 (2026-10-24 00:50): replaces exp1_chain_b.sh's lane step. E's QA full evals moved to the idle right GB300
# (lane right-gb300-e), so after run (b) ends the left GB300 scores D (grpo_tok-s0) only. LANE stays left-gb300 so the
# RTX re-check waiter sees "lane end".
set -uo pipefail
: "${D:=/models/treeopsd}"; cd "$D/repl-build"
while [ -n "$(docker ps -q -f name=^treeopsd-exp1c-treepo-E16$)" ]; do sleep 60; done
LANE=left-gb300 GPU=1 RUNS="exp1-grpo_tok-s0" WIKI_URL=http://STATION_RIGHT_RAIL_IP:8000/retrieve BC_URL=http://127.0.0.1:8001/retrieve \
  exec bash scripts/exp1_evallane.sh
