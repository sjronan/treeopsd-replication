#!/usr/bin/env bash
# Arm E seeds 1-2 (2026-10-24 06:15, Founder "go, run both seeds"): on THIS host's GB300, wait for the E training
# container to exit and its step-100 HF to be merged, then run the full eval lane on the same GB300.
#   RUN=exp1-treepo_clip-s1 LANE=left-gb300-e1 WIKI_URL=... BC_URL=... bash exp1_chain_e.sh
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${RUN:?}"; : "${LANE:?}"; cd "$D/repl-build"
sleep 600
until [ -z "$(docker ps -q -f name=^treeopsd-exp1-treepo_clip$)" ] && [ -f "$D/runs/$RUN/hf_step100/.merged" ]; do sleep 120; done
LANE=$LANE GPU=1 RUNS="$RUN" exec bash scripts/exp1_evallane.sh
