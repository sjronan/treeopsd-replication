#!/usr/bin/env bash
# Left host (2026-10-24 06:40): pull the right GB300 C-rescore lane (gb300-step100-* dirs) into the left interim tree.
# left interim tree every 5 min until $LANE's log on the right says "lane end", then one last pass.
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${RIGHT:=user@STATION_RIGHT}"; : "${LANE:?}"; : "${RUNS:?}"; I=$D/interim
while true; do
  done_=$(ssh -n -o BatchMode=yes "$RIGHT" "tail -1 $I/lane-$LANE.log 2>/dev/null")
  for r in $RUNS; do
    rsync -a --include='gb300-step100-*/***' --include='*.log' --include='*.failed*' --exclude='*' \
      -e 'ssh -o BatchMode=yes' "$RIGHT:$I/$r/" "$I/$r/" 2>/dev/null
  done
  rsync -a -e 'ssh -o BatchMode=yes' "$RIGHT:$I/lane-$LANE.log" "$I/" 2>/dev/null
  case "$done_" in *"lane end"*) exit 0;; esac
  sleep 300
done
