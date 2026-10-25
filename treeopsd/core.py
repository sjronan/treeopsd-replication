"""Reference (CPU, framework-free) implementation of the TreeOPSD math.

Everything here follows arXiv:2610.23181v1 (Zou et al., 2026) equation by
equation. Equation numbers in comments refer to that paper. This module is the
single source of truth for the math; the verl integration (see DESIGN.md)
must import these functions or be tested against them.

Conventions
-----------
* A "turn" (tree node) is one policy-generated actor span a_i of L_i tokens.
  Observation tokens are never part of a node and never receive gradient.
* `logp_old_sum[i]` is log pi_old(a_i | h_i): the SUM of behavior log-probs of
  the span's tokens (Eq. 5 uses the full-span likelihood; see PAPER.md A1).
* Rewards R_k live on the K terminal (leaf) turns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One actor turn in a rollout tree."""

    id: int
    parent: Optional[int]  # parent turn id; None for the first turn of a root rollout
    logp_old_sum: float  # log pi_old(a_i | h_i), summed over the span's tokens
    n_tokens: int  # L_i (actor tokens only)
    reward: Optional[float] = None  # set iff the node is a terminal (answer) turn
    tool_calls: int = 0  # tool calls emitted by this turn (0 for the answer turn)
    children: List[int] = field(default_factory=list)


class Tree:
    """Rollout tree T_x for one prompt (Sec. 2.2)."""

    def __init__(self, nodes: Iterable[Node]):
        self.nodes: Dict[int, Node] = {}
        for n in nodes:
            if n.id in self.nodes:
                raise ValueError(f"duplicate node id {n.id}")
            self.nodes[n.id] = n
        for n in self.nodes.values():
            n.children = []
        for n in self.nodes.values():
            if n.parent is not None:
                self.nodes[n.parent].children.append(n.id)
        for n in self.nodes.values():
            is_leaf = len(n.children) == 0
            if is_leaf and n.reward is None:
                raise ValueError(f"leaf node {n.id} has no reward")
            if (not is_leaf) and n.reward is not None:
                raise ValueError(f"internal node {n.id} carries a reward")

    @property
    def leaves(self) -> List[int]:
        return sorted(i for i, n in self.nodes.items() if not n.children)

    @property
    def K(self) -> int:
        return len(self.leaves)

    @property
    def L_x(self) -> int:
        """Total number of UNIQUE actor tokens in the tree (Eq. 6)."""
        return sum(n.n_tokens for n in self.nodes.values())

    def postorder(self) -> List[int]:
        order: List[int] = []
        roots = [i for i, n in self.nodes.items() if n.parent is None]
        stack = [(r, False) for r in sorted(roots)]
        while stack:
            i, done = stack.pop()
            if done:
                order.append(i)
                continue
            stack.append((i, True))
            for c in sorted(self.nodes[i].children, reverse=True):
                stack.append((c, False))
        return order

    def path_to_root(self, i: int) -> List[int]:
        path = []
        while i is not None:
            path.append(i)
            i = self.nodes[i].parent
        return path[::-1]


# ---------------------------------------------------------------------------
# Eq. 4: group standardisation of terminal rewards
# ---------------------------------------------------------------------------


def standardize(rewards: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    """A_k = (R_k - mean) / std with the SAMPLE std (ddof=1); 0 if all equal."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.size < 2:
        return np.zeros_like(r)
    sd = r.std(ddof=1)
    if sd <= eps:
        return np.zeros_like(r)
    return (r - r.mean()) / sd


def samuelson_bound(K: int) -> float:
    """B_K = (K-1)/sqrt(K): max |A_k| for sample-std standardisation (Lemma B.1)."""
    return (K - 1) / math.sqrt(K)


# ---------------------------------------------------------------------------
# Eq. 5: likelihood-weighted recursive backup
# ---------------------------------------------------------------------------


def child_weights(tree: Tree, i: int, weighting: str = "likelihood") -> Dict[int, float]:
    """omega_{i,c} over the children of node i.

    weighting:
      "likelihood" -- paper Eq. 5: pi_old(a_c|h_c) normalised over siblings,
                      computed as a softmax of full-span summed log-probs.
      "geomean"    -- ablation: softmax of per-token MEAN log-probs.
      "uniform"    -- ablation: 1/|C_i|.
    """
    ch = tree.nodes[i].children
    if not ch:
        return {}
    if weighting == "uniform":
        return {c: 1.0 / len(ch) for c in ch}
    if weighting == "likelihood":
        logits = np.array([tree.nodes[c].logp_old_sum for c in ch], dtype=np.float64)
    elif weighting == "geomean":
        logits = np.array(
            [tree.nodes[c].logp_old_sum / max(tree.nodes[c].n_tokens, 1) for c in ch],
            dtype=np.float64,
        )
    else:
        raise ValueError(weighting)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return dict(zip(ch, w.tolist()))


def backup(tree: Tree, weighting: str = "likelihood") -> Dict[int, float]:
    """V_i for every node (Eq. 4 then Eq. 5)."""
    leaves = tree.leaves
    A = standardize([tree.nodes[k].reward for k in leaves])
    V: Dict[int, float] = {k: float(a) for k, a in zip(leaves, A)}
    for i in tree.postorder():
        if i in V:
            continue
        w = child_weights(tree, i, weighting)
        V[i] = float(sum(w[c] * V[c] for c in w))
    return V


# ---------------------------------------------------------------------------
# Eqs. 11-12, 18: closed-form bounded teacher
# ---------------------------------------------------------------------------


def tau_from_K(K: int, rho: float) -> float:
    """tau = (K-1)/(rho*sqrt(K)) = B_K/rho (Eq. 18; Alg. 1)."""
    if not (0.0 < rho < 0.5):
        raise ValueError("rho must lie in (0, 1/2)")
    if K < 2:
        raise ValueError("K must be >= 2")
    return samuelson_bound(K) / rho


def teacher_q(V, tau: float):
    """q_i = 1/2 + V_i / tau (Eq. 11)."""
    return 0.5 + np.asarray(V, dtype=np.float64) / tau


def teacher_ratio(V, tau: float):
    """r*_i = q/(1-q) = (tau + 2V)/(tau - 2V) (Eq. 11/25)."""
    V = np.asarray(V, dtype=np.float64)
    return (tau + 2 * V) / (tau - 2 * V)


def ratio_interval(rho: float):
    """Depth-independent interval for r* (Eq. 12)."""
    return (1 - 2 * rho) / (1 + 2 * rho), (1 + 2 * rho) / (1 - 2 * rho)


def proximal_objective(s, V: float, tau: float):
    """J_i(s) = V s - tau * log cosh(s/2) (Prop. B.2, Eq. 10 in log-odds coords)."""
    s = np.asarray(s, dtype=np.float64)
    return V * s - tau * np.log(np.cosh(s / 2.0))


# ---------------------------------------------------------------------------
# Eqs. 8-9, 13, 16-17: turn scores, Bernoulli posteriors, forward KL loss
# ---------------------------------------------------------------------------


def bernoulli_kl(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """D_KL(Ber(q) || Ber(sigmoid(s))), numerically stable in the logit s."""
    q = q.to(s.dtype)
    log_p = F.logsigmoid(s)
    log_1mp = F.logsigmoid(-s)
    ent_terms = torch.xlogy(q, q) + torch.xlogy(1 - q, 1 - q)
    return ent_terms - q * log_p - (1 - q) * log_1mp


def turn_scores(
    logp: torch.Tensor, logp_old: torch.Tensor, turn_index: torch.Tensor, n_turns: int
) -> torch.Tensor:
    """s_{theta,i} = mean over the turn's tokens of (log pi_theta - log pi_old) (Eq. 8).

    logp, logp_old: [T] token log-probs of actor tokens (flattened over the tree,
    each unique turn's tokens appearing ONCE). turn_index: [T] long in [0, n_turns).
    """
    diff = logp - logp_old
    num = torch.zeros(n_turns, dtype=diff.dtype).index_add(0, turn_index, diff)
    cnt = torch.zeros(n_turns, dtype=diff.dtype).index_add(
        0, turn_index, torch.ones_like(diff)
    )
    return num / cnt.clamp_min(1)


def treeopsd_loss(
    logp: torch.Tensor,
    logp_old: torch.Tensor,
    turn_index: torch.Tensor,
    q: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """L_x(theta) = tau/L_x * sum_i L_i * D_KL(Ber(q_i) || Ber(sigmoid(s_i))) (Eq. 13).

    Implemented as a per-token sum: every token of turn i carries
    tau/L_x * D_KL(q_i || sigma(s_i)); summing L_i identical terms gives the
    L_i weight, and d/d log pi(a_ij) = tau/L_x * (p_i - q_i) (Eq. 16).
    """
    n_turns = q.shape[0]
    s = turn_scores(logp, logp_old, turn_index, n_turns)
    L_x = logp.shape[0]
    per_tok = bernoulli_kl(q[turn_index], s[turn_index])
    return tau * per_tok.sum() / L_x


def treeopsd_loss_linear_form(
    logp, logp_old, turn_index, V: torch.Tensor, tau: float
) -> torch.Tensor:
    """Second expression of Eq. 13 without the constant C_x:
    1/L_x * sum_i L_i [ -V_i s_i + tau * psi(s_i) ],  psi(s) = log cosh(s/2) (Eq. 23)."""
    n_turns = V.shape[0]
    s = turn_scores(logp, logp_old, turn_index, n_turns)
    L_i = torch.zeros(n_turns, dtype=logp.dtype).index_add(
        0, turn_index, torch.ones_like(logp)
    )
    psi = torch.log(torch.cosh(s / 2))
    return (L_i * (-V.to(s.dtype) * s + tau * psi)).sum() / logp.shape[0]


def treepo_loss(logp: torch.Tensor, turn_index: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Fixed-credit control (Sec. 4.1 'TreePO'): gradient -V_i/L_x on every token,
    on every optimizer block, with no importance ratio (see PAPER.md A2)."""
    L_x = logp.shape[0]
    return -(V.to(logp.dtype)[turn_index] * logp).sum() / L_x


def ascent_weight(s, V: float, tau: float):
    """Gamma_i(s) = V_i - tau*(sigmoid(s) - 1/2) = tau*(q_i - sigmoid(s)) (Eqs. 17, 33)."""
    s = np.asarray(s, dtype=np.float64)
    return V - tau * (1.0 / (1.0 + np.exp(-s)) - 0.5)


# ---------------------------------------------------------------------------
# GRPO advantage (baseline), for completeness
# ---------------------------------------------------------------------------


def grpo_advantages(rewards: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    """GRPO outcome advantage as in verl's compute_grpo_outcome_advantage:
    (R - mean)/(std + eps) with torch's default unbiased std."""
    r = torch.tensor(rewards, dtype=torch.float64)
    if r.numel() < 2:
        return np.zeros(r.numel())
    return ((r - r.mean()) / (r.std() + eps)).numpy()


# ---------------------------------------------------------------------------
# Initialize-then-expand sampling and budget accounting (Sec. 2.2, App. E.4)
# ---------------------------------------------------------------------------


def sample_anchors(T: int, N: int, rng: np.random.Generator) -> List[int]:
    """Multiset of N anchors uniform on {1..T-1} (with replacement); all 0 if T == 1.

    Anchor t means the continuation reuses the prefix H_t = (x, a_1, o_1, ..., a_t, o_t)
    and regenerates turns t+1..T'.
    """
    if T < 1:
        raise ValueError("T >= 1")
    if T == 1:
        return [0] * N
    return sorted(int(t) for t in rng.integers(1, T, size=N))


def nominal_tree_budget(M: int, N: int) -> float:
    """M (1 + N/2): the paper's full-trajectory-equivalent budget (Sec. 2.2)."""
    return M * (1 + N / 2)


def expected_suffix_cost(turn_costs: Sequence[float]) -> float:
    """Expected cost of ONE continuation drawn from a root trajectory whose turns
    cost turn_costs[0..T-1] (any unit: generated tokens, tool calls, seconds),
    assuming the continuation reproduces the same per-turn cost profile.
    Anchor t ~ U{1..T-1}; the suffix regenerates turns t+1..T. If T == 1 the
    anchor is 0 and the continuation is a full rollout."""
    c = np.asarray(turn_costs, dtype=np.float64)
    T = c.size
    if T == 1:
        return float(c.sum())
    suffix = [c[t:].sum() for t in range(1, T)]  # anchor t -> turns t+1..T == c[t:]
    return float(np.mean(suffix))


def expected_tree_cost(root_turn_costs: Sequence[Sequence[float]], N: int) -> float:
    """Expected generation cost of an (M, N) tree given the M roots' per-turn costs."""
    return float(
        sum(np.sum(c) + N * expected_suffix_cost(c) for c in root_turn_costs)
    )


def budget_ratio_vs_grpo(root_turn_costs: Sequence[Sequence[float]], N: int, G: int) -> float:
    """(expected tree cost) / (G * mean full-trajectory cost). 1.0 == budget matched."""
    full = np.mean([np.sum(c) for c in root_turn_costs])
    return expected_tree_cost(root_turn_costs, N) / (G * full)


# ---------------------------------------------------------------------------
# Unique-turn reconstruction: packing tree turns into trajectory rows
# ---------------------------------------------------------------------------


def owner_rows(tree: Tree) -> Dict[int, int]:
    """Assign each unique turn to exactly one trajectory row (the row of the
    lowest-id leaf below it). A trainer that packs one row per terminal
    trajectory must mask a shared turn out of the loss in every non-owner row,
    otherwise shared prefixes are counted once per descendant leaf."""
    owner: Dict[int, int] = {}
    for row, leaf in enumerate(tree.leaves):
        for i in tree.path_to_root(leaf):
            owner.setdefault(i, row)
    return owner


def trainer_tokens(tree: Tree, context_tokens: Dict[int, int]) -> Dict[str, int]:
    """Token counts the TRAINER sees when the tree is packed as one row per leaf.

    context_tokens[i] = non-actor tokens (prompt / observations) that precede turn
    i's span and follow its parent's span. Returns unique actor tokens (L_x),
    total row tokens (what FSDP forward/backward actually processes without
    prefix-shared attention), and unique tokens (lower bound with prefix sharing).
    """
    rows_total = 0
    for leaf in tree.leaves:
        for i in tree.path_to_root(leaf):
            rows_total += tree.nodes[i].n_tokens + context_tokens.get(i, 0)
    unique_total = sum(n.n_tokens + context_tokens.get(i, 0) for i, n in tree.nodes.items())
    return {"L_x": tree.L_x, "row_tokens": rows_total, "unique_tokens": unique_total}
