#!/usr/bin/env python3
"""Score fixed-checkpoint evaluations and write runs/<id>/summary.json (autoresearch.sh
schema). Both tiers are covered:

  tier 1 (QA, secondary):     hotpotqa, 2wikimultihopqa, musique, bamboogle  (wiki-18)
  tier 2 (headline):          bcplus  (frozen BrowseComp-Plus slice, same agent interface)

Generations come from verl's validation dump (one val-only run per benchmark, see
RUNBOOK-PHASE0.md / scripts/run_eval.sh): <dump_dir>/<step>.jsonl with fields
input, output, gts, score, em, f1, ...  EM/F1 are RECOMPUTED here from the output text
(last <answer> span, tool responses stripped), not trusted from the dump.

    python3 scripts/eval_qa.py --run-dir runs/exp1-treeopsd-s0 \
        --dump hotpotqa=$E/hotpotqa --dump 2wikimultihopqa=$E/2wiki --dump musique=$E/musique \
        --dump bamboogle=$E/bamboogle --dump bcplus=$E/bcplus \
        --prepared $D/prepared --telemetry runs/exp1-treeopsd-s0/telemetry \
        --arm treeopsd --seed 0 --model Qwen/Qwen3-4B-Thinking-2507@768f209d --image sha256:...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from treeopsd.metrics import exact_match, extract_answer, max_f1  # noqa: E402

_TOOL_RESP = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)
_Q = re.compile(r"Question:\s*(.*?)\s*(?:assistant\s*)?$", re.DOTALL)
QA = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]


def last_dump(d: str) -> str:
    files = glob.glob(os.path.join(d, "*.jsonl"))
    if not files:
        raise SystemExit(f"no dump in {d}")
    return max(files, key=lambda p: int(os.path.basename(p).split(".")[0]))


def score_rows(rows):
    out = []
    for r in rows:
        text = r.get("output", "")
        pred = extract_answer(_TOOL_RESP.sub(" ", text))
        gts = r.get("gts")
        if isinstance(gts, dict):
            gts = gts.get("ground_truth", gts.get("target", []))
        if isinstance(gts, str):
            try:
                gts = json.loads(gts)
            except Exception:
                gts = [gts]
        gts = [str(g) for g in (gts or [])]
        out.append({
            "em": exact_match(pred, gts),
            "f1": max_f1(pred, gts),
            "has_answer": float(pred is not None),
            "tool_calls": float(text.count("<tool_response>")),
            "question": (_Q.search(r.get("input", "")) or [None, ""])[1] if r.get("input") else "",
        })
    return out


def strata_breakdown(bench, scored, prepared):
    """Join scored rows to strata by question text (order-preserving for duplicates)."""
    try:
        import pandas as pd
    except Exception:
        return {}
    p = os.path.join(prepared, "mhqa", f"eval_{bench}.parquet")
    sp = os.path.join(prepared, "mhqa", "strata.json")
    if not (os.path.exists(p) and os.path.exists(sp)):
        return {}
    strata = json.load(open(sp))
    df = pd.read_parquet(p, columns=["extra_info"])
    q2ids = defaultdict(list)
    for ei in df["extra_info"]:
        q2ids[ei["question"].strip()].append(ei["id"])
    groups = defaultdict(list)
    for s in scored:
        ids = q2ids.get(s["question"].strip())
        if not ids:
            continue
        st = strata.get(f"{bench}/{ids.pop(0)}", {})
        key = st.get("hops", st.get("type"))
        if key is not None:
            groups[str(key)].append(s["em"])
    return {k: {"n": len(v), "em": 100 * float(np.mean(v))} for k, v in sorted(groups.items())}


def telemetry_summary(tdir):
    out = {}
    steps = []
    p = os.path.join(tdir, "steps.jsonl")
    if os.path.exists(p):
        steps = [json.loads(l) for l in open(p)]
    if steps:
        def mean(k):
            v = [s[k] for s in steps if k in s]
            return float(np.mean(v)) if v else None
        out["rollout_gen_tokens_per_prompt"] = mean("budget/gen_tokens_per_prompt")
        out["tool_calls_per_prompt"] = mean("budget/tool_calls_per_prompt")
        out["train_tokens_per_prompt"] = mean("budget/train_tokens_per_prompt")
        if mean("budget/single_turn_root_frac") is not None:
            out["single_turn_root_frac"] = mean("budget/single_turn_root_frac")
        walls = [s["time/step_wall_s_prev"] for s in steps if "time/step_wall_s_prev" in s]
        if walls:
            # first step's wall time is not observed; extrapolate with the mean
            out["train_gpu_hours"] = float(np.mean(walls)) * len(steps) / 3600.0
        out["n_steps_logged"] = len(steps)
    loss = []
    for f in glob.glob(os.path.join(tdir, "loss_*.jsonl")):
        loss += [json.loads(l) for l in open(f)]
    if loss:
        # blocks 2-4 = calls with non-trivial drift; block 1 has s == 0 by construction
        dr = [r["treeopsd/drift_rel"] for r in loss if "treeopsd/drift_rel" in r]
        s_abs = [r["treeopsd/s_abs_mean"] for r in loss if "treeopsd/s_abs_mean" in r]
        if dr:
            later = [d for d, s in zip(dr, s_abs) if s > 1e-6]
            out["drift_rel_blocks234"] = float(np.mean(later)) if later else 0.0
            out["drift_rel_lt_0.02_frac"] = float(np.mean([d < 0.02 for d in later])) if later else 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dump", action="append", default=[], help="bench=dir")
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--telemetry", default=None)
    ap.add_argument("--arm", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--best-in-window", type=float, default=None)
    a = ap.parse_args()

    summary = {"run_id": os.path.basename(a.run_dir.rstrip("/")), "arm": a.arm, "seed": a.seed,
               "model": a.model, "image": a.image, "em": {}, "f1": {}, "n": {}, "strata": {}}
    for spec in a.dump:
        bench, d = spec.split("=", 1)
        rows = [json.loads(l) for l in open(last_dump(d))]
        sc = score_rows(rows)
        summary["em"][bench] = 100 * float(np.mean([s["em"] for s in sc]))
        summary["f1"][bench] = 100 * float(np.mean([s["f1"] for s in sc]))
        summary["n"][bench] = len(sc)
        summary.setdefault("eval_tool_calls", {})[bench] = float(np.mean([s["tool_calls"] for s in sc]))
        summary.setdefault("answer_rate", {})[bench] = float(np.mean([s["has_answer"] for s in sc]))
        if a.prepared and bench in QA:
            summary["strata"][bench] = strata_breakdown(bench, sc, a.prepared)
        with open(os.path.join(a.run_dir, f"scored_{bench}.jsonl"), "w") as f:
            for s in sc:
                f.write(json.dumps({k: v for k, v in s.items() if k != "question"}) + "\n")
    if all(b in summary["em"] for b in QA):
        summary["em_qa_avg"] = float(np.mean([summary["em"][b] for b in QA]))
        summary["em_qa_avg_no_bamboogle"] = float(np.mean([summary["em"][b] for b in QA[:3]]))
    if "bcplus" in summary["em"]:
        summary["em_headline_bcplus"] = summary["em"]["bcplus"]
    if a.telemetry:
        summary.update({k: v for k, v in telemetry_summary(a.telemetry).items() if v is not None})
    if a.best_in_window is not None:
        summary["em_best_in_window"] = a.best_in_window
    os.makedirs(a.run_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(a.run_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("em", "n")}, indent=1))


if __name__ == "__main__":
    main()
