"""Two-phase initialize-then-expand rollout (paper Sec. 2.2 / Alg. 1) around verl's
AgentLoopManager.

The trainer is run with rollout.n = K = M(N+1), so it hands us K interleaved copies of
each prompt and expects K rows back in the same order (see tree_build.py for the
layout). Phase A samples the M roots; phase B samples one continuation per
(root, anchor) from the root's exact token prefix. Rows keep verl's fixed
prompt_length / response_length padding, so the two phases concatenate directly.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List

import numpy as np
import torch

from treeopsd.verl_ext.tree_build import anchor_rng, plan_anchors

AGENT_NAME = "tree_tool_agent"


def _resolve(x):
    if inspect.isawaitable(x):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():  # pragma: no cover - trainer calls us synchronously
            raise RuntimeError("TreeRolloutManager must be called from synchronous code")
        return asyncio.run(x) if loop is None else loop.run_until_complete(x)
    if hasattr(x, "get") and type(x).__name__ in ("ObjectRef", "DataProtoFuture"):
        return x.get()
    return x


def build_prefix(out_a, row: int, t: int) -> Dict[str, Any]:
    """Prefix of root row `row` up to and including observation o_t (snapshot t-1)."""
    snaps = out_a.non_tensor_batch["tree_snapshots"][row]
    snap = snaps[t - 1]
    prompts = out_a.batch["prompts"][row]
    plen = prompts.shape[0]
    am = out_a.batch["attention_mask"][row]
    prompt_ids = prompts[am[:plen].bool()].tolist()
    n_resp = int(snap["n_resp"])
    resp_ids = out_a.batch["responses"][row][:n_resp].tolist()
    full = prompt_ids + resp_ids
    if len(full) != int(snap["n_ids"]):
        raise ValueError(f"prefix length mismatch row={row} t={t}: {len(full)} vs {snap['n_ids']}")
    resp_mask = out_a.batch["response_mask"][row][:n_resp].tolist()
    lp = None
    if "rollout_log_probs" in out_a.batch.keys():
        lp = out_a.batch["rollout_log_probs"][row][:n_resp].tolist()
    msgs = out_a.non_tensor_batch["tree_messages"][row][: int(snap["n_messages"])]
    return {
        "prompt_ids": full,
        "response_mask": resp_mask,
        "response_logprobs": lp,
        "messages": [dict(m) for m in msgs],
        "assistant_turns": int(snap["assistant_turns"]),
        "user_turns": int(snap["user_turns"]),
        "snapshots": [dict(s) for s in snaps[:t]],
    }


def n_anchorable(out_a, row: int) -> int:
    """Number of completed (action, observation) pairs before the terminal turn."""
    snaps = out_a.non_tensor_batch["tree_snapshots"][row] or []
    n_turns = int(out_a.batch["response_mask"][row].diff(prepend=torch.zeros(1, dtype=torch.long)
                                                         .to(out_a.batch["response_mask"].device)).eq(1).sum())
    return int(min(len(snaps), max(n_turns - 1, 0)))


class TreeRolloutManager:
    def __init__(self, inner, M: int, N: int, seed: int):
        self.inner = inner
        self.M, self.N, self.K = M, N, M * (N + 1)
        self.seed = seed

    def __getattr__(self, name):  # delegate everything else (wake/sleep, validation, ...)
        return getattr(self.inner, name)

    def _gen(self, batch):
        return _resolve(self.inner.generate_sequences(batch))

    def generate_sequences(self, prompts):
        from verl import DataProto

        K, M, N = self.K, self.M, self.N
        R = len(prompts)
        if prompts.meta_info.get("validate", False) or R % K != 0:
            # validation batches are not trees: plain loop, no prefix
            prompts.non_tensor_batch["tree_prefix"] = np.array([None] * R, dtype=object)
            return self._gen(prompts)
        P = R // K
        step = int(prompts.meta_info.get("global_steps", 0))
        idx_a = np.array([g * K + j for g in range(P) for j in range(M)])
        batch_a = prompts.select_idxs(idx_a)
        batch_a.non_tensor_batch["agent_name"] = np.array([AGENT_NAME] * len(batch_a), dtype=object)
        batch_a.non_tensor_batch["tree_prefix"] = np.array([None] * len(batch_a), dtype=object)
        out_a = self._gen(batch_a)

        # ---- anchors (deterministic per seed/step/prompt) -----------------------------
        uids = prompts.non_tensor_batch["uid"]
        anchors: List[List[List[int]]] = []
        for g in range(P):
            rng = anchor_rng(self.seed, step, str(uids[g * K]))
            anchors.append(plan_anchors([n_anchorable(out_a, g * M + m) for m in range(M)], N, rng))

        idx_b, prefixes = [], []
        for g in range(P):
            for m in range(M):
                for n in range(N):
                    idx_b.append(g * K + M + m * N + n)
                    t = anchors[g][m][n]
                    prefixes.append(build_prefix(out_a, g * M + m, t) if t > 0 else None)
        batch_b = prompts.select_idxs(np.array(idx_b))
        batch_b.non_tensor_batch["agent_name"] = np.array([AGENT_NAME] * len(batch_b), dtype=object)
        pref = np.empty(len(prefixes), dtype=object)
        pref[:] = prefixes
        batch_b.non_tensor_batch["tree_prefix"] = pref
        out_b = self._gen(batch_b)

        # ---- reassemble in the trainer's row order ------------------------------------
        for o in (out_a, out_b):
            for k in ("tree_prefix",):
                o.non_tensor_batch.pop(k, None)
        keys_a, keys_b = set(out_a.non_tensor_batch), set(out_b.non_tensor_batch)
        for k in keys_a ^ keys_b:  # keep the concat well-formed
            (out_a if k not in keys_a else out_b).non_tensor_batch[k] = np.array(
                [None] * len(out_a if k not in keys_a else out_b), dtype=object)
        timing_a = out_a.meta_info.pop("timing", {})
        timing_b = out_b.meta_info.pop("timing", {})
        both = DataProto.concat([out_a, out_b])
        order = np.empty(R, dtype=np.int64)
        order[np.concatenate([idx_a, np.array(idx_b)])] = np.arange(R)
        out = both.select_idxs(order)
        meta = np.empty(R, dtype=object)
        for g in range(P):
            for j in range(K):
                if j < M:
                    meta[g * K + j] = {"role": "root", "root": j, "anchor": -1}
                else:
                    m, n = divmod(j - M, N)
                    meta[g * K + j] = {"role": "cont", "root": m, "anchor": anchors[g][m][n]}
        out.non_tensor_batch["tree_meta"] = meta
        out.meta_info["timing"] = {**{f"phaseA/{k}": v for k, v in timing_a.items()},
                                   **{f"phaseB/{k}": v for k, v in timing_b.items()},
                                   **timing_a}
        return out
