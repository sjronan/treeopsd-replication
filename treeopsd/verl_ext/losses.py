"""TreeOPSD and TreePO policy losses registered in verl's policy-loss registry.

All three consume the SAME `advantages` tensor (V_i broadcast on owned actor tokens),
so the TreeOPSD vs TreePO comparison differs only in `actor.policy_loss.loss_mode`.

Extra per-token inputs (forwarded by env/patches/verl-v0.9.1-treeopsd.patch):
    tree_turn_id      [B, Lr] long   node id on owned actor tokens, -1 elsewhere
    tree_token_weight [B, Lr] float  1/(L_x * B_e) on owned actor tokens, 0 elsewhere
    tree_tau          [B, Lr] float  tau = (K-1)/(rho sqrt(K)), constant

Aggregation: the loss is the weighted token SUM times dp_size (verl sums micro-batch
losses and averages gradients over DP ranks), so the mini-batch gradient is exactly
sum_prompts (1/B_e) * grad L_x  -- the paper's "prompt-averaged token gradients".
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:  # verl is present in the training image; tests may import the pure functions only
    from verl.trainer.ppo.core_algos import register_policy_loss
except Exception:  # pragma: no cover
    def register_policy_loss(name):  # type: ignore
        def deco(f):
            return f
        return deco


def _dp_size(config) -> int:
    try:
        return int(config.global_batch_info.get("dp_size", 1) or 1)
    except Exception:
        return 1


def _turn_scores(log_prob, old_log_prob, turn_id):
    """Per-token s_i (segment mean over the turn's owned tokens) and the per-turn index."""
    owned = turn_id >= 0
    B, L = turn_id.shape
    rows = torch.arange(B, device=turn_id.device).unsqueeze(1).expand(B, L)
    # turn ids are unique per batch already; include the row to be safe under packing
    key = torch.where(owned, rows * (int(turn_id.max().item()) + 2) + turn_id, torch.full_like(turn_id, -1))
    flat = key[owned]
    uniq, inv = torch.unique(flat, return_inverse=True)
    diff = (log_prob - old_log_prob)[owned].float()
    num = torch.zeros(uniq.numel(), device=diff.device, dtype=diff.dtype).index_add(0, inv, diff)
    cnt = torch.zeros_like(num).index_add(0, inv, torch.ones_like(diff))
    s_turn = num / cnt.clamp_min(1)
    return owned, inv, s_turn


def _bern_kl(q, s):
    return torch.xlogy(q, q) + torch.xlogy(1 - q, 1 - q) - q * F.logsigmoid(s) - (1 - q) * F.logsigmoid(-s)


_TELEM = {"pid": os.getpid(), "calls": 0}


def _telemetry(rec: dict):
    d = os.environ.get("TREEOPSD_TELEMETRY_DIR")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    _TELEM["calls"] += 1
    rec = {"t": time.time(), "pid": _TELEM["pid"], "call": _TELEM["calls"], **rec}
    with open(os.path.join(d, f"loss_{_TELEM['pid']}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def treeopsd_fkl_core(log_prob, old_log_prob, advantages, tree_turn_id, tree_token_weight, tree_tau):
    """Returns (loss_sum, metrics). Pure torch; unit-tested against treeopsd.core."""
    owned, inv, s_turn = _turn_scores(log_prob, old_log_prob, tree_turn_id)
    tau = tree_tau[owned].float()
    V = advantages[owned].float()
    q = 0.5 + V / tau
    s_tok = s_turn[inv]
    per_tok = tau * _bern_kl(q, s_tok)
    loss = (per_tok * tree_token_weight[owned].float()).sum()
    with torch.no_grad():
        p = torch.sigmoid(s_tok)
        gamma = tau * (q - p)  # Eq. 17 ascent weight
        v_abs = V.abs().mean().clamp_min(1e-12)
        metrics = {
            "treeopsd/s_abs_mean": s_turn.abs().mean().item() if s_turn.numel() else 0.0,
            "treeopsd/drift_rel": ((gamma - V).abs().mean() / v_abs).item() if V.numel() else 0.0,
            "treeopsd/sign_reversed_frac": ((gamma * V) < 0).float().mean().item() if V.numel() else 0.0,
            "treeopsd/q_minus_half_abs": (q - 0.5).abs().mean().item() if V.numel() else 0.0,
        }
    return loss, metrics


def treepo_fixed_core(log_prob, advantages, tree_turn_id, tree_token_weight):
    owned = tree_turn_id >= 0
    loss = -(advantages[owned].float() * log_prob[owned].float() * tree_token_weight[owned].float()).sum()
    return loss


def treepo_clip_core(log_prob, old_log_prob, advantages, tree_turn_id, tree_token_weight, clip=0.2):
    owned = tree_turn_id >= 0
    ratio = torch.exp((log_prob - old_log_prob)[owned].float().clamp(-20, 20))
    A = advantages[owned].float()
    pg = torch.maximum(-A * ratio, -A * ratio.clamp(1 - clip, 1 + clip))
    return (pg * tree_token_weight[owned].float()).sum(), ((ratio - 1).abs() > clip).float().mean().item()


def _require(kw, name):
    if kw.get(name) is None:
        raise RuntimeError(
            f"{name} missing: the verl extra-loss-inputs patch is not applied or the tree "
            "advantage hook did not run (see env/patches/ and treeopsd/verl_ext/advantage.py)"
        )
    return kw[name]


@register_policy_loss("treeopsd_fkl")
def compute_policy_loss_treeopsd_fkl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-sum",
    config: Optional[Any] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    **kw,
):
    tid, w, tau = _require(kw, "tree_turn_id"), _require(kw, "tree_token_weight"), _require(kw, "tree_tau")
    loss, metrics = treeopsd_fkl_core(log_prob, old_log_prob, advantages, tid.long(), w, tau)
    _telemetry({"loss_mode": "treeopsd_fkl", **metrics})
    return loss * _dp_size(config), metrics


@register_policy_loss("treepo_fixed")
def compute_policy_loss_treepo_fixed(
    old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-sum", config=None,
    rollout_is_weights=None, **kw,
):
    tid, w = _require(kw, "tree_turn_id"), _require(kw, "tree_token_weight")
    loss = treepo_fixed_core(log_prob, advantages, tid.long(), w)
    # same drift telemetry as TreeOPSD (what TreeOPSD WOULD have changed), for PREREG M1
    tau = kw.get("tree_tau")
    metrics = {}
    if tau is not None:
        with torch.no_grad():
            _, metrics = treeopsd_fkl_core(log_prob.detach(), old_log_prob, advantages, tid.long(), w, tau)
    _telemetry({"loss_mode": "treepo_fixed", **metrics})
    return loss * _dp_size(config), metrics


@register_policy_loss("treepo_clip")
def compute_policy_loss_treepo_clip(
    old_log_prob, log_prob, advantages, response_mask, loss_agg_mode="token-sum", config=None,
    rollout_is_weights=None, **kw,
):
    tid, w = _require(kw, "tree_turn_id"), _require(kw, "tree_token_weight")
    clip = 0.2
    try:
        clip = float(config.clip_ratio)
    except Exception:
        pass
    loss, clipfrac = treepo_clip_core(log_prob, old_log_prob, advantages, tid.long(), w, clip)
    metrics = {"treepo/clipfrac": clipfrac}
    _telemetry({"loss_mode": "treepo_clip", **metrics})
    return loss * _dp_size(config), metrics
