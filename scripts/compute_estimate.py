#!/usr/bin/env python3
"""Back-of-envelope compute model for the TreeOPSD replication.

Every number below is an ASSUMPTION to be replaced by Phase-0 measurements
(see DESIGN.md section 6). The point of the script is that the estimate is
reproducible and its sensitivity is explicit, not that it is right.

Usage:  python3 scripts/compute_estimate.py            # markdown table
        python3 scripts/compute_estimate.py --decode-scale 0.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ----------------------------------------------------------------- hardware
GB300_BF16_DENSE = 2.25e15  # FLOP/s, Blackwell-Ultra-class dense BF16 (spec-sheet order; MEASURE)
TRAIN_MFU = 0.30  # FSDP single-GPU, long packed sequences, activation ckpt (MEASURE)
ATTN_OVERHEAD = 1.15  # attention FLOPs on top of 6ND at 7-20k context
WEIGHT_SYNC_S = 20.0  # per step: trainer -> vLLM weight refit + sleep/wake (MEASURE)

# effective agentic decode throughput on ONE GB300 with vLLM, many concurrent
# multi-turn requests, prefix caching on, including tool-wait and straggler tails
DECODE_TPS = {"Qwen3-1.7B": 20_000, "Qwen3-4B-Thinking-2507": 12_000, "Qwen3-30B-A3B-Thinking-2507": 8_000}
PARAMS = {"Qwen3-1.7B": 1.7e9, "Qwen3-4B-Thinking-2507": 4.0e9, "Qwen3-30B-A3B-Thinking-2507": 3.3e9}  # active
# eval decode on a DGX Spark (GB10, 273 GB/s) and on the RTX PRO 6000 (1.8 TB/s)
SPARK_EVAL_TPS = {"Qwen3-1.7B": 4_000, "Qwen3-4B-Thinking-2507": 2_500}
PRO6000_EVAL_TPS = {"Qwen3-1.7B": 12_000, "Qwen3-4B-Thinking-2507": 8_000}


@dataclass
class Task:
    name: str
    gen_tok: float  # generated (actor) tokens per full trajectory
    ctx_tok: float  # total sequence tokens per trajectory (prompt + obs + generated)


MHQA = Task("multi-hop QA (wiki-18, <=5 turns)", gen_tok=2_500, ctx_tok=7_000)
DR = Task("deep research (BC+/OpenResearcher, <=60 turns)", gen_tok=12_000, ctx_tok=18_000)


def step_seconds(model: str, task: Task, B: int, arm: str, decode_scale: float) -> dict:
    """One policy-update iteration (rollout + old-logprob + E=4 optimizer blocks)."""
    traj_equiv = 8.0  # GRPO G=8 ; tree (4,2) = 4 + 8*0.5 in expectation
    rows = 8 if arm == "grpo" else 12  # trajectories the trainer packs as rows
    gen = B * traj_equiv * task.gen_tok
    t_roll = gen / (DECODE_TPS[model] * decode_scale)
    train_tokens = B * rows * task.ctx_tok  # no prefix-shared attention in the trainer
    flops = 8 * PARAMS[model] * train_tokens * ATTN_OVERHEAD  # 6ND fwd/bwd + 2ND old-logprob
    t_train = flops / (GB300_BF16_DENSE * TRAIN_MFU)
    return {"gen_tokens": gen, "train_tokens": train_tokens, "rollout_s": t_roll,
            "train_s": t_train, "total_s": t_roll + t_train + WEIGHT_SYNC_S}


def run_hours(model, task, B, steps, arm, decode_scale=1.0, eval_q=0, eval_every=10):
    st = step_seconds(model, task, B, arm, decode_scale)
    h = steps * st["total_s"] / 3600
    # in-loop subsampled eval on the GB300 (greedy/T=0.6, 1 sample per question)
    n_eval = steps // eval_every + 1
    h += n_eval * eval_q * task.gen_tok / (DECODE_TPS[model] * decode_scale) / 3600
    return h, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-scale", type=float, default=1.0)
    a = ap.parse_args()
    ds = a.decode_scale
    m4, m17 = "Qwen3-4B-Thinking-2507", "Qwen3-1.7B"
    rows = []

    # per-arm step anatomy at paper scale
    for arm in ("grpo", "tree"):
        st = step_seconds(m4, MHQA, 256, arm, ds)
        rows.append((f"4B MHQA paper-scale step ({arm}, B=256)",
                     f"{st['gen_tokens']/1e6:.1f}M gen / {st['train_tokens']/1e6:.1f}M train tok",
                     f"{st['total_s']/60:.1f} min/step"))

    h, _ = run_hours(m17, MHQA, 64, 30, "tree", ds, eval_q=1_625)
    rows.append(("S1a shakedown 1.7B TreeOPSD, B=64, 30 steps", "1 arm", f"{h:.1f} GB300-h"))
    h, _ = run_hours(m4, MHQA, 64, 10, "tree", ds, eval_q=1_625)
    rows.append(("S1b 4B smoke, B=64, 10 steps (+ token/turn measurement)", "1 arm", f"{h:.1f} GB300-h"))

    per_arm, _ = run_hours(m4, MHQA, 64, 100, "tree", ds, eval_q=1_625)
    per_grpo, _ = run_hours(m4, MHQA, 64, 100, "grpo", ds, eval_q=1_625)
    rows.append(("EXP-1 4B MHQA, B=64 (4x16), 100 steps, per tree arm", "TreeOPSD or TreePO", f"{per_arm:.1f} GB300-h"))
    rows.append(("EXP-1 4B MHQA, B=64, 100 steps, GRPO arm", "GRPO G=8", f"{per_grpo:.1f} GB300-h"))
    tot1 = 2 * per_arm + per_grpo
    rows.append(("EXP-1 total, 3 arms x 1 seed", "", f"{tot1:.1f} GB300-h"))
    rows.append(("EXP-1 total, 3 arms x 3 seeds", "", f"{3*tot1:.1f} GB300-h"))
    full_eval_q = 7_405 + 12_576 + 2_417 + 125
    ev_spark = full_eval_q * MHQA.gen_tok / SPARK_EVAL_TPS[m4] / 3600
    ev_pro = full_eval_q * MHQA.gen_tok / PRO6000_EVAL_TPS[m4] / 3600
    rows.append(("final full MHQA eval per checkpoint (22,523 q)", "Spark / RTX PRO 6000",
                 f"{ev_spark:.1f} h / {ev_pro:.1f} h"))

    per_paper, _ = run_hours(m4, MHQA, 256, 100, "tree", ds, eval_q=1_625)
    rows.append(("paper-scale 4B MHQA arm (B=256, 100 steps)", "for reference", f"{per_paper:.1f} GB300-h"))

    dr_arm, _ = run_hours(m4, DR, 32, 70, "tree", ds, eval_q=200)
    dr_grpo, _ = run_hours(m4, DR, 32, 70, "grpo", ds, eval_q=200)
    rows.append(("EXP-1b 4B BC+ track, B=32, 70 steps, per tree arm", "scaled from B=128", f"{dr_arm:.1f} GB300-h"))
    rows.append(("EXP-1b total, 3 arms x 1 seed", "", f"{2*dr_arm+dr_grpo:.1f} GB300-h"))
    dr_paper, _ = run_hours(m4, DR, 128, 70, "tree", ds, eval_q=200)
    rows.append(("paper-scale 4B DR arm (B=128, 70 steps)", "for reference", f"{dr_paper:.1f} GB300-h"))

    sweep_cfgs = 5  # (4,1) (4,2) (4,4) (8,1) (2,2) -- different budgets, see DESIGN.md
    h_sweep, _ = run_hours(m4, MHQA, 64, 60, "tree", ds, eval_q=1_625)
    rows.append(("EXP-2 branching sweep, 5 configs x 60 steps, B=64", "TreeOPSD", f"{sweep_cfgs*h_sweep:.1f} GB300-h"))
    rows.append(("EXP-2 TreePO-same-tree control at (4,2), 2 extra seeds", "F1", f"{2*per_arm:.1f} GB300-h"))

    print(f"Assumptions: GB300 {GB300_BF16_DENSE/1e15:.2f} PF dense BF16, MFU {TRAIN_MFU}, "
          f"decode scale x{ds}, 4B decode {DECODE_TPS[m4]*ds:.0f} tok/s, "
          f"MHQA {MHQA.gen_tok:.0f} gen / {MHQA.ctx_tok:.0f} ctx tok per traj, "
          f"DR {DR.gen_tok:.0f} gen / {DR.ctx_tok:.0f} ctx tok per traj\n")
    print("| item | note | estimate |\n|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")


if __name__ == "__main__":
    main()
