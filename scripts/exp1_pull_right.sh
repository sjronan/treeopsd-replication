#!/usr/bin/env bash
# Left host: copy the right GB300 lane's results (seed-2 full evals, calibration, lane log) into the left interim tree
# every 5 min, until the right lane log says "lane end" and one last pass has run. Right->left has no SSH.
set -uo pipefail
: "${D:=/models/treeopsd}"; : "${RIGHT:=user@STATION_RIGHT}"; I=$D/interim
while true; do
  done_=$(ssh -n -o BatchMode=yes "$RIGHT" "tail -1 $I/lane-right-gb300.log 2>/dev/null")
  for r in exp1-treepo-s2 exp1-treeopsd-s2 exp1-treepo-s0; do
    rsync -a --include='step100-*/***' --include='calib-gb300-*/***' --include='*.log' --include='*.failed*' --exclude='*' \
      -e 'ssh -o BatchMode=yes' "$RIGHT:$I/$r/" "$I/$r/" 2>/dev/null
  done
  rsync -a -e 'ssh -o BatchMode=yes' "$RIGHT:$I/lane-right-gb300.log" "$I/" 2>/dev/null
  case "$done_" in *"lane end"*) exit 0;; esac
  sleep 300
done
