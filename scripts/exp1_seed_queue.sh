#!/usr/bin/env bash
# Seeds 1-2 of B and C (research-session ruling inbox 008, 2026-10-22): waits for AFTER_LOG to end
# ("queue end" as its last line), then runs scripts/exp1_queue.sh with QUEUE and SEED.
#   AFTER_LOG=<queue log> QUEUE="treeopsd treepo" SEED=1 [RETRIEVER_URL=...] bash exp1_seed_queue.sh
set -uo pipefail
: "${AFTER_LOG:?}"; : "${QUEUE:?}"; : "${SEED:?}"
until tail -1 "$AFTER_LOG" 2>/dev/null | grep -q "queue end"; do sleep 120; done
exec bash "$(dirname "$0")/exp1_queue.sh"
