#!/usr/bin/env bash
# Emit `METRIC name=number` lines for one run directory.
#
#   ./autoresearch.sh runs/<run-id>
#
# Contract: the eval harness writes <run-dir>/summary.json (schema in
# autoresearch.md). Exits non-zero if the summary is missing or malformed, so a
# broken run can never be logged as a result.
set -euo pipefail
RUN_DIR="${1:?usage: autoresearch.sh <run-dir>}"
SUMMARY="$RUN_DIR/summary.json"
[[ -f "$SUMMARY" ]] || { echo "no $SUMMARY" >&2; exit 2; }
python3 - "$SUMMARY" <<'PY'
import json, math, sys
s = json.load(open(sys.argv[1]))

def fmt(x):
    x = float(x)
    if not math.isfinite(x):
        raise SystemExit(f"non-finite metric {x}")
    return f"{x:.6g}"

em = s.get("em") or {}
# primary: unweighted mean over the evaluated benchmarks (paper's "Avg")
mhqa = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
if not em:
    raise SystemExit("summary.json has no 'em' block")
# headline (research-session decision 2026-10-21): frozen BrowseComp-Plus slice
if "bcplus" in em:
    print(f"METRIC em_headline_bcplus={fmt(em['bcplus'])}")
if all(k in em for k in mhqa):
    qa = sum(em[k] for k in mhqa) / 4
    print(f"METRIC em_qa_avg={fmt(qa)}")
    print(f"METRIC em_avg={fmt(qa)}")  # paper's "Avg" over the 4 QA sets (kept for continuity)
    print(f"METRIC em_qa_avg_no_bamboogle={fmt(sum(em[k] for k in mhqa[:3]) / 3)}")
elif "bcplus" not in em:
    print(f"METRIC em_avg={fmt(sum(em.values()) / len(em))}")
for k, v in sorted(em.items()):
    print(f"METRIC em_{k}={fmt(v)}")
for k in ("rollout_gen_tokens_per_prompt", "tool_calls_per_prompt",
          "train_tokens_per_prompt", "train_gpu_hours", "drift_rel_blocks234",
          "single_turn_root_frac", "em_best_in_window", "drift_rel_lt_0.02_frac"):
    if k in s:
        print(f"METRIC {k}={fmt(s[k])}")
PY
