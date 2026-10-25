"""Framework-free tree assembly for verl batches (CPU-testable).

Row layout of one prompt group (K = M(N+1) consecutive rows, as produced by verl's
`repeat(n=K, interleave=True)`):

    j = 0 .. M-1                 root rollout m = j
    j = M + m*N + n              continuation n of root m (anchor t_{m,n})

Every row's *response* holds the full trajectory after the prompt. A continuation row
from anchor t >= 1 therefore starts with the root's first t assistant turns (and their
observations) verbatim; those turns are SHARED nodes owned by the root row. An anchor
of 0 (root answered in one turn) makes the continuation an independent rollout.

Assistant turns are the maximal runs of 1s in `response_mask` (observation tokens are 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from treeopsd.core import Node, Tree, backup, tau_from_K, samuelson_bound


def turn_segments(mask_row: Sequence[int]) -> List[Tuple[int, int]]:
    """[start, end) of every maximal run of 1s (one assistant turn each)."""
    m = np.asarray(mask_row).astype(np.int8)
    if m.size == 0:
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def plan_anchors(n_anchorable: Sequence[int], N: int, rng: np.random.Generator) -> List[List[int]]:
    """Anchors for each root: uniform on {1..A_m} with replacement, A_m = #complete
    (action, observation) pairs before the terminal turn; 0 if A_m == 0 (paper Eq. 3)."""
    out = []
    for a in n_anchorable:
        if a <= 0:
            out.append([0] * N)
        else:
            out.append(sorted(int(t) for t in rng.integers(1, a + 1, size=N)))
    return out


def anchor_rng(seed: int, global_step: int, uid: str) -> np.random.Generator:
    """Deterministic per-(seed, step, prompt) RNG, independent of PYTHONHASHSEED."""
    h = np.frombuffer(uid.encode("utf-8"), dtype=np.uint8).astype(np.uint64)
    mix = int((h * np.arange(1, h.size + 1, dtype=np.uint64)).sum() % np.uint64(2**61 - 1))
    return np.random.default_rng([int(seed), int(global_step), mix])


@dataclass
class GroupCredit:
    advantages: torch.Tensor  # [K, Lr] V_i on owned actor tokens, 0 elsewhere
    turn_id: torch.Tensor  # [K, Lr] node id (>=0) on owned actor tokens, -1 elsewhere
    token_weight: torch.Tensor  # [K, Lr] 1/L_x on owned actor tokens, 0 elsewhere
    stats: Dict[str, float]


def group_credit(
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: Sequence[float],
    M: int,
    N: int,
    anchors: Sequence[Sequence[int]],
    weighting: str = "likelihood",
) -> GroupCredit:
    """Build the rollout tree of one prompt from its K rows and return per-token credit.

    response_mask, old_log_probs: [K, Lr]; rewards: K terminal rewards (one per row/leaf).
    anchors[m][n]: anchor of continuation n of root m (0 = independent rollout).
    """
    K = M * (N + 1)
    assert response_mask.shape[0] == K == len(rewards), (response_mask.shape, K, len(rewards))
    segs = [turn_segments(response_mask[r].tolist()) for r in range(K)]

    # node key -> (owner_row, seg_index); row -> list of node keys per turn
    row_nodes: List[List[Tuple]] = []
    node_owner: Dict[Tuple, Tuple[int, int]] = {}
    for r in range(K):
        keys = []
        if r < M:
            for k in range(len(segs[r])):
                keys.append(("root", r, k))
        else:
            m, n = divmod(r - M, N)
            t = anchors[m][n]
            n_root = len(segs[m])
            if t > n_root:
                raise ValueError(f"anchor {t} beyond root {m} turns {n_root}")
            if t > len(segs[r]):
                raise ValueError(f"continuation row {r} shorter than its prefix")
            for k in range(len(segs[r])):
                keys.append(("root", m, k) if k < t else ("cont", r, k))
        for k, key in enumerate(keys):
            node_owner.setdefault(key, (r, k))
        # a row with no turn of its own (empty generation / failed continuation) gets a
        # zero-length virtual leaf, so K stays fixed and shared prefix turns never become leaves
        if not keys or keys[-1][0] == "root" and r >= M:
            key = ("empty", r, 0)
            node_owner[key] = (r, -1)
            keys.append(key)
        row_nodes.append(keys)

    # every row must END in its own (unshared) terminal turn; an empty row (no turn at
    # all, e.g. generation failure) is given a zero-length virtual leaf so K stays fixed
    ids: Dict[Tuple, int] = {}
    nodes: List[Node] = []
    lp = old_log_probs.double()
    n_empty = 0
    for r in range(K):
        parent = None
        for k, key in enumerate(row_nodes[r]):
            if key not in ids:
                orow, ok = node_owner[key]
                is_last = k == len(row_nodes[r]) - 1
                ids[key] = len(nodes)
                if ok < 0:  # virtual leaf: zero backup weight, zero tokens
                    n_empty += 1
                    nodes.append(Node(ids[key], parent, -1e30, 0, reward=float(rewards[r])))
                else:
                    s, e = segs[orow][ok]
                    nodes.append(
                        Node(
                            id=ids[key],
                            parent=parent,
                            logp_old_sum=float(lp[orow, s:e].sum()),
                            n_tokens=e - s,
                            reward=float(rewards[r]) if is_last else None,
                        )
                    )
            parent = ids[key]
    # a shared node must not be a leaf of a different row (prefix turns are nonterminal)
    tree = Tree(nodes)
    if tree.K != K:
        raise ValueError(f"tree has {tree.K} leaves, expected {K}")
    V = backup(tree, weighting)
    L_x = max(tree.L_x, 1)

    shape = response_mask.shape
    adv = torch.zeros(shape, dtype=torch.float64)
    tid = torch.full(shape, -1, dtype=torch.long)
    w = torch.zeros(shape, dtype=torch.float64)
    for key, nid in ids.items():
        orow, ok = node_owner[key]
        if ok < 0:
            continue
        s, e = segs[orow][ok]
        adv[orow, s:e] = V[nid]
        tid[orow, s:e] = nid
        w[orow, s:e] = 1.0 / L_x

    ent = []
    peaked = 0
    internal = 0
    from treeopsd.core import child_weights

    for i, n in tree.nodes.items():
        if len(n.children) > 1:
            internal += 1
            ws = np.array(list(child_weights(tree, i, weighting).values()))
            ent.append(float(-(ws * np.log(ws + 1e-300)).sum()))
            peaked += int(ws.max() > 0.99)
    stats = {
        "tree/L_x": float(L_x),
        "tree/n_nodes": float(len(nodes)),
        "tree/n_branch_nodes": float(internal),
        "tree/omega_entropy_mean": float(np.mean(ent)) if ent else 0.0,
        "tree/omega_peaked_frac": float(peaked / internal) if internal else 0.0,
        "tree/V_abs_mean": float(np.mean([abs(v) for v in V.values()])),
        "tree/empty_leaves": float(n_empty),
        "tree/V_abs_max_over_bound": float(max(abs(v) for v in V.values()) / samuelson_bound(K)),
    }
    return GroupCredit(adv, tid, w, stats)


def batch_credit(
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    M: int,
    N: int,
    anchors_per_group: Sequence[Sequence[Sequence[int]]],
    rho: float,
    prompts_per_block: int,
    weighting: str = "likelihood",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Credit for a whole batch of P prompt groups (rows grouped contiguously).

    Returns advantages, tree_turn_id (made unique across the batch), tree_token_weight
    (= 1/(L_x * B_e): the paper's prompt-averaged token gradient within a block of B_e
    prompts), tree_tau (constant per token) and averaged stats.
    """
    K = M * (N + 1)
    R = response_mask.shape[0]
    assert R % K == 0
    P = R // K
    adv = torch.zeros(response_mask.shape, dtype=torch.float64)
    tid = torch.full(response_mask.shape, -1, dtype=torch.long)
    w = torch.zeros_like(adv)
    stats_acc: Dict[str, List[float]] = {}
    offset = 0
    for g in range(P):
        sl = slice(g * K, (g + 1) * K)
        gc = group_credit(response_mask[sl], old_log_probs[sl], rewards[sl].tolist(), M, N,
                          anchors_per_group[g], weighting)
        adv[sl] = gc.advantages
        t = gc.turn_id.clone()
        t[t >= 0] += offset
        offset = int(t.max().item()) + 1 if (t >= 0).any() else offset
        tid[sl] = t
        w[sl] = gc.token_weight / prompts_per_block
        for k, v in gc.stats.items():
            stats_acc.setdefault(k, []).append(v)
    tau = tau_from_K(K, rho)
    tau_t = torch.full_like(adv, tau)
    stats = {k: float(np.mean(v)) for k, v in stats_acc.items()}
    return adv, tid, w, tau_t, stats


def budget_stats(response_mask: torch.Tensor, M: int, N: int, anchors_per_group, num_tool_calls=None):
    """Generated-token accounting per prompt: root tokens, continuation NEW tokens (suffix
    only), and the realised suffix fraction vs the nominal 0.5."""
    K = M * (N + 1)
    P = response_mask.shape[0] // K
    root_tok, cont_new_tok, single_turn_roots = 0, 0, 0
    for g in range(P):
        rows = response_mask[g * K:(g + 1) * K]
        segs = [turn_segments(rows[r].tolist()) for r in range(K)]
        for m in range(M):
            root_tok += sum(e - s for s, e in segs[m])
            single_turn_roots += int(len(segs[m]) <= 1)
        for m in range(M):
            for n in range(N):
                r = M + m * N + n
                t = anchors_per_group[g][m][n]
                cont_new_tok += sum(e - s for s, e in segs[r][t:])
    mean_root = root_tok / max(P * M, 1)
    mean_cont = cont_new_tok / max(P * M * N, 1)
    return {
        "budget/gen_tokens_per_prompt": (root_tok + cont_new_tok) / max(P, 1),
        "budget/root_tokens_mean": mean_root,
        "budget/cont_new_tokens_mean": mean_cont,
        "budget/suffix_fraction": mean_cont / mean_root if mean_root else 0.0,
        "budget/traj_equivalents": (root_tok + cont_new_tok) / max(P, 1) / mean_root if mean_root else 0.0,
        "budget/single_turn_root_frac": single_turn_roots / max(P * M, 1),
    }
