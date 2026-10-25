"""Tree credit hook + per-step budget telemetry for verl's legacy RayPPOTrainer.

verl's registered advantage estimators receive neither old_log_probs nor the non-tensor
tree metadata, so the tree arms wrap `verl.trainer.ppo.ray_trainer.compute_advantage`
(called once per step, after old_log_prob recompute, before the actor update).

For `algorithm.adv_estimator == "tree_backup"` the wrapper writes
  advantages, tree_turn_id, tree_token_weight, tree_tau      (consumed by losses.py)
For every arm (GRPO included) it appends one JSON line per step to
  $TREEOPSD_TELEMETRY_DIR/steps.jsonl
with measured generated tokens, tool calls, trainer tokens and wall time (F2, logged
from run 1 as the research session required).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

from treeopsd.verl_ext.tree_build import batch_credit, budget_stats, turn_segments

SETTINGS: Dict[str, Any] = {
    "M": 4, "N": 2, "rho": 0.25, "weighting": "likelihood", "prompts_per_block": 16,
}
_STATE = {"last_t": None, "step": 0}


def _anchors_from_meta(meta: np.ndarray, M: int, N: int) -> List[List[List[int]]]:
    K = M * (N + 1)
    P = len(meta) // K
    out = []
    for g in range(P):
        a = [[0] * N for _ in range(M)]
        for j in range(M, K):
            d = meta[g * K + j]
            m, n = divmod(j - M, N)
            assert d["root"] == m, (d, m)
            a[m][n] = int(d["anchor"])
        out.append(a)
    return out


def _count_tool_calls(non_tensor: Dict[str, np.ndarray], response_mask: torch.Tensor) -> np.ndarray:
    """Tool calls per row = assistant turns followed by an observation (0-masked tokens)
    before the end of the response (max_parallel_calls = 1, so one call per such turn)."""
    out = []
    for r in range(response_mask.shape[0]):
        row = response_mask[r]
        if not row.any():
            out.append(0)
            continue
        end = int(row.nonzero().max().item()) + 1  # last actor token + 1
        out.append(sum(1 for (_, e) in turn_segments(row.tolist()) if e < end))
    return np.array(out)


def tree_compute_advantage(data, adv_estimator, *args, _orig=None, **kwargs):
    """Drop-in replacement for ray_trainer.compute_advantage."""
    t0 = time.time()
    mask = data.batch["response_mask"]
    est = str(getattr(adv_estimator, "value", adv_estimator))
    rec: Dict[str, Any] = {}
    if est == "tree_backup":
        M, N = SETTINGS["M"], SETTINGS["N"]
        anchors = _anchors_from_meta(data.non_tensor_batch["tree_meta"], M, N)
        rewards = data.batch["token_level_rewards"].sum(-1).float()
        adv, tid, w, tau, stats = batch_credit(
            mask.cpu(), data.batch["old_log_probs"].float().cpu(), rewards.cpu(), M, N, anchors,
            rho=SETTINGS["rho"], prompts_per_block=SETTINGS["prompts_per_block"],
            weighting=SETTINGS["weighting"],
        )
        dev = data.batch["old_log_probs"].device
        data.batch["advantages"] = adv.float().to(dev)
        data.batch["returns"] = adv.float().to(dev)
        data.batch["tree_turn_id"] = tid.to(dev)
        data.batch["tree_token_weight"] = w.float().to(dev)
        data.batch["tree_tau"] = tau.float().to(dev)
        rec.update(stats)
        rec.update(budget_stats(mask.cpu(), M, N, anchors))
        rows_per_prompt = M * (N + 1)
    else:
        data = _orig(data, adv_estimator, *args, **kwargs)
        rows_per_prompt = None
    # ---- budget telemetry (all arms) -------------------------------------------------
    n_rows = mask.shape[0]
    uid = data.non_tensor_batch.get("uid")
    n_prompts = len(set(uid.tolist())) if uid is not None else n_rows
    gen_tok = float(mask.sum().item())
    if est != "tree_backup":
        rec["budget/gen_tokens_per_prompt"] = gen_tok / n_prompts
    tool_calls = _count_tool_calls(data.non_tensor_batch, mask.cpu())
    if est == "tree_backup":
        # continuation rows re-contain the root's prefix calls; count only NEW calls
        M, N = SETTINGS["M"], SETTINGS["N"]
        K = M * (N + 1)
        anchors = _anchors_from_meta(data.non_tensor_batch["tree_meta"], M, N)
        new_calls = 0
        for g in range(n_rows // K):
            for j in range(K):
                c = tool_calls[g * K + j]
                if j >= M:
                    m, n = divmod(j - M, N)
                    c -= anchors[g][m][n]
                new_calls += max(c, 0)
        rec["budget/tool_calls_per_prompt"] = new_calls / n_prompts
    else:
        rec["budget/tool_calls_per_prompt"] = float(tool_calls.sum()) / n_prompts
    am = data.batch.get("attention_mask")
    if am is not None:
        rec["budget/train_tokens_per_prompt"] = float(am.sum().item()) / n_prompts
    acc = data.batch["token_level_rewards"].sum(-1).float()
    rec["reward/mean"] = float(acc.mean().item())
    now = time.time()
    if _STATE["last_t"] is not None:
        rec["time/step_wall_s_prev"] = now - _STATE["last_t"]
    _STATE["last_t"] = now
    _STATE["step"] += 1
    rec["step_call"] = _STATE["step"]
    rec["adv_hook_s"] = time.time() - t0
    d = os.environ.get("TREEOPSD_TELEMETRY_DIR")
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "steps.jsonl"), "a") as f:
            f.write(json.dumps({"t": now, "estimator": est, "n_prompts": n_prompts, **rec}) + "\n")
    return data


def register_placeholder() -> None:
    """Make "tree_backup" a known estimator name (config validation runs in the driver)."""
    try:
        from verl.trainer.ppo import core_algos
    except Exception:  # pragma: no cover
        return
    if "tree_backup" not in core_algos.ADV_ESTIMATOR_REGISTRY:
        @core_algos.register_adv_est("tree_backup")
        def _placeholder(*a, **k):  # never called: the wrapper intercepts first
            raise RuntimeError("tree_backup must be computed by treeopsd.verl_ext.advantage")


register_placeholder()


def install(settings: Dict[str, Any]) -> None:
    """Patch the trainer module in the CURRENT process (the TaskRunner actor)."""
    from functools import partial

    import verl.trainer.ppo.ray_trainer as rt

    SETTINGS.update(settings)
    register_placeholder()
    if not getattr(rt.compute_advantage, "_treeopsd", False):
        orig = rt.compute_advantage
        wrapped = partial(tree_compute_advantage, _orig=orig)
        wrapped._treeopsd = True  # type: ignore[attr-defined]
        rt.compute_advantage = wrapped
