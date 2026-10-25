"""CPU smoke tests for the TreeOPSD math (arXiv:2610.23181v1).

Run:  python -m pytest -q env/smoke   (from the repo root)
Each test names the equation / statement it checks.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from treeopsd.core import (  # noqa: E402
    Node,
    Tree,
    ascent_weight,
    backup,
    bernoulli_kl,
    budget_ratio_vs_grpo,
    child_weights,
    expected_suffix_cost,
    expected_tree_cost,
    grpo_advantages,
    nominal_tree_budget,
    owner_rows,
    proximal_objective,
    ratio_interval,
    sample_anchors,
    samuelson_bound,
    standardize,
    tau_from_K,
    teacher_q,
    teacher_ratio,
    trainer_tokens,
    treeopsd_loss,
    treeopsd_loss_linear_form,
    treepo_loss,
    turn_scores,
)

torch.set_default_dtype(torch.float64)
RHO = 0.25


# ---------------------------------------------------------------------------
# Helpers: build an (M, N) initialize-then-expand tree from synthetic rollouts
# ---------------------------------------------------------------------------


def build_tree(M=4, N=2, seed=0, T_choices=(2, 3, 4, 5), p_success=0.4):
    rng = np.random.default_rng(seed)
    nodes = []
    nid = [0]

    def new(parent, reward=None):
        i = nid[0]
        nid[0] += 1
        nodes.append(
            Node(
                id=i,
                parent=parent,
                logp_old_sum=float(rng.uniform(-60, -5)),
                n_tokens=int(rng.integers(5, 40)),
                reward=reward,
                tool_calls=0 if reward is not None else 1,
            )
        )
        return i

    def rollout(parent, n_turns):
        ids = []
        p = parent
        for t in range(n_turns):
            last = t == n_turns - 1
            r = float(rng.random() < p_success) if last else None
            p = new(p, r)
            ids.append(p)
        return ids

    for _ in range(M):
        T = int(rng.choice(T_choices))
        root_ids = rollout(None, T)
        for t in sample_anchors(T, N, rng):
            # anchor t: reuse turns 1..t (root_ids[:t]), regenerate T' - t turns
            parent = root_ids[t - 1] if t >= 1 else None
            T_new = int(rng.integers(t + 1, t + 4))  # continuation length may differ
            rollout(parent, T_new - t)
    return Tree(nodes)


# ---------------------------------------------------------------------------
# Eq. 4 / Lemma B.1
# ---------------------------------------------------------------------------


def test_standardize_sample_std_and_all_equal():
    A = standardize([1, 0, 0, 1, 1])
    assert abs(A.mean()) < 1e-12
    assert abs(A.std(ddof=1) - 1) < 1e-12
    assert np.all(standardize([1, 1, 1]) == 0)
    assert np.all(standardize([0, 0, 0, 0]) == 0)


@pytest.mark.parametrize("K", [2, 3, 8, 12, 24])
def test_samuelson_bound_is_tight(K):
    # one dissenting reward attains |A| = (K-1)/sqrt(K)
    r = [1.0] + [0.0] * (K - 1)
    A = standardize(r)
    assert abs(abs(A).max() - samuelson_bound(K)) < 1e-12
    rng = np.random.default_rng(K)
    for _ in range(200):
        A = standardize(rng.random(K))
        assert abs(A).max() <= samuelson_bound(K) + 1e-12


def test_grpo_advantage_matches_standardize():
    r = [1, 0, 0, 1, 0, 0, 0, 1]
    assert np.allclose(grpo_advantages(r), standardize(r), atol=1e-5)


# ---------------------------------------------------------------------------
# Eq. 5: tree reward backup
# ---------------------------------------------------------------------------


def hand_tree():
    # root turn 0 -> children 1 (leaf, R=1) and 2 ; 2 -> children 3 (R=0), 4 (R=1)
    # plus an independent root 5 (leaf, R=0)
    nodes = [
        Node(0, None, -10.0, 10),
        Node(1, 0, math.log(0.6), 5, reward=1.0),
        Node(2, 0, math.log(0.2), 5),
        Node(3, 2, math.log(0.1), 5, reward=0.0),
        Node(4, 2, math.log(0.3), 5, reward=1.0),
        Node(5, None, -3.0, 4, reward=0.0),
    ]
    return Tree(nodes)


def test_backup_by_hand():
    t = hand_tree()
    assert t.K == 4 and t.leaves == [1, 3, 4, 5]
    A = standardize([1.0, 0.0, 1.0, 0.0])  # leaves 1,3,4,5
    V = backup(t)
    assert np.isclose(V[1], A[0]) and np.isclose(V[3], A[1]) and np.isclose(V[4], A[2])
    # node 2: weights 0.1/(0.1+0.3), 0.3/(0.4)
    assert np.isclose(V[2], 0.25 * A[1] + 0.75 * A[2])
    # node 0: children 1 (p=.6) and 2 (p=.2) -> weights .75, .25
    assert np.isclose(V[0], 0.75 * A[0] + 0.25 * V[2])
    assert np.isclose(V[5], A[3])


def test_backup_is_convex_and_bounded():
    for seed in range(50):
        t = build_tree(seed=seed)
        V = backup(t)
        B = samuelson_bound(t.K)
        assert all(abs(v) <= B + 1e-12 for v in V.values())
        for i, n in t.nodes.items():
            if n.children:
                lo = min(V[c] for c in n.children) - 1e-12
                hi = max(V[c] for c in n.children) + 1e-12
                assert lo <= V[i] <= hi


def test_M4_N2_gives_K12():
    for seed in range(20):
        t = build_tree(M=4, N=2, seed=seed)
        assert t.K == 12  # K = M (N + 1)


def test_likelihood_weights_are_extremely_peaked_for_long_spans():
    """Documents PAPER.md ambiguity A1: with full-span likelihoods, siblings whose
    summed log-probs differ by 20 nats get weights ~ (1, 2e-9)."""
    nodes = [
        Node(0, None, -1.0, 3),
        Node(1, 0, -500.0, 400, reward=1.0),
        Node(2, 0, -520.0, 400, reward=0.0),
    ]
    w = child_weights(Tree(nodes), 0, "likelihood")
    assert w[1] > 1 - 1e-8 and w[2] < 1e-8
    wg = child_weights(Tree(nodes), 0, "geomean")
    assert 0.4 < wg[1] < 0.6  # per-token normalisation keeps both children


# ---------------------------------------------------------------------------
# Eqs. 11-12, 18, 22, 25: the bounded teacher
# ---------------------------------------------------------------------------


def test_tau_value_for_paper_default():
    tau = tau_from_K(12, RHO)
    assert np.isclose(tau, 11 / (0.25 * math.sqrt(12)))
    assert np.isclose(tau, 12.70170592, atol=1e-6)


@pytest.mark.parametrize("K", [4, 8, 12, 24])
@pytest.mark.parametrize("rho", [0.1, 0.25, 0.4])
def test_teacher_in_radius_and_ratio_interval(K, rho):
    tau = tau_from_K(K, rho)
    B = samuelson_bound(K)
    V = np.linspace(-B, B, 101)
    q = teacher_q(V, tau)
    assert np.all(q >= 0.5 - rho - 1e-12) and np.all(q <= 0.5 + rho + 1e-12)
    assert np.isclose(q[0], 0.5 - rho) and np.isclose(q[-1], 0.5 + rho)
    r = teacher_ratio(V, tau)
    assert np.allclose(r, q / (1 - q))
    lo, hi = ratio_interval(rho)
    assert np.all(r >= lo - 1e-12) and np.all(r <= hi + 1e-12)


def test_teacher_is_maximiser_of_proximal_objective():
    """Prop. B.2: argmax_s V s - tau log cosh(s/2) = logit(1/2 + V/tau)."""
    tau = tau_from_K(12, RHO)
    s = np.linspace(-6, 6, 600001)
    for V in [-3.0, -1.0, -0.2, 0.0, 0.5, 2.0, samuelson_bound(12)]:
        s_hat = s[np.argmax(proximal_objective(s, V, tau))]
        q = teacher_q(V, tau)
        assert abs(s_hat - math.log(q / (1 - q))) < 1e-4


def test_affine_teacher_commutes_with_backup_and_logistic_does_not():
    """Theorem 1 / Eq. 15 and Sec. 4.3 negative control."""
    tau = tau_from_K(12, RHO)
    for seed in range(20):
        t = build_tree(seed=seed)
        V = backup(t)
        for i, n in t.nodes.items():
            if not n.children:
                continue
            w = child_weights(t, i)
            lhs = teacher_q(V[i], tau)
            rhs = sum(w[c] * teacher_q(V[c], tau) for c in w)
            assert np.isclose(lhs, rhs, atol=1e-12)
    # logistic map q = sigmoid(V/tau') breaks commutation for a non-trivial mix
    sig = lambda v: 1 / (1 + math.exp(-v))  # noqa: E731
    tau_p = tau / 4
    Vc = [2.5, -1.0]
    w = [0.5, 0.5]
    parent = sig(sum(a * b for a, b in zip(w, Vc)) / tau_p)
    mixed = sum(a * sig(v / tau_p) for a, v in zip(w, Vc))
    assert abs(parent - mixed) > 1e-3


def test_rms_teacher_offset_matches_paper_appendix_C4():
    """App. C.4: RMS of q_k - 1/2 over leaves = rho/sqrt(K-1) (~0.075 at K=12)."""
    K = 12
    tau = tau_from_K(K, RHO)
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = standardize(rng.integers(0, 2, K).astype(float) + 0 * rng.random(K))
        if np.all(A == 0):
            continue
        rms = np.sqrt(np.mean((teacher_q(A, tau) - 0.5) ** 2))
        assert np.isclose(rms, RHO / math.sqrt(K - 1))
    assert abs(RHO / math.sqrt(11) - 0.0754) < 1e-3


# ---------------------------------------------------------------------------
# Eqs. 13-17, 27, 30, 36-38: forward-KL loss and its gradients
# ---------------------------------------------------------------------------


def test_bernoulli_kl_against_closed_form():
    q = torch.tensor([0.25, 0.5, 0.7, 0.75])
    s = torch.tensor([-2.0, 0.0, 0.3, 1.5])
    p = torch.sigmoid(s)
    ref = q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))
    assert torch.allclose(bernoulli_kl(q, s), ref)
    # psi(s) = KL(Ber(1/2) || Ber(sigmoid(s))) = log cosh(s/2)  (Eq. 23)
    assert torch.allclose(bernoulli_kl(torch.full_like(s, 0.5), s), torch.log(torch.cosh(s / 2)))


def _random_turns(seed, n_turns=6):
    g = torch.Generator().manual_seed(seed)
    L = torch.randint(3, 12, (n_turns,), generator=g)
    turn_index = torch.repeat_interleave(torch.arange(n_turns), L)
    T = int(L.sum())
    logp_old = -torch.rand(T, generator=g) * 3
    logp = (logp_old + 0.3 * torch.randn(T, generator=g)).clone().requires_grad_(True)
    tau = tau_from_K(12, RHO)
    B = samuelson_bound(12)
    V = (torch.rand(n_turns, generator=g) * 2 - 1) * B
    q = torch.tensor(teacher_q(V.numpy(), tau))
    return logp, logp_old, turn_index, V, q, tau, L


def test_token_gradient_matches_eq16():
    for seed in range(10):
        logp, logp_old, ti, V, q, tau, L = _random_turns(seed)
        loss = treeopsd_loss(logp, logp_old, ti, q, tau)
        (g,) = torch.autograd.grad(loss, logp)
        s = turn_scores(logp.detach(), logp_old, ti, len(L))
        p = torch.sigmoid(s)
        expected = tau / logp.shape[0] * (p - q)[ti]
        assert torch.allclose(g, expected, atol=1e-12)
        # and equals -(1/L_x) * Gamma_i(s)  (Eq. 16-17)
        gam = torch.tensor(ascent_weight(s.numpy(), V.numpy(), tau))
        assert torch.allclose(g, -gam[ti] / logp.shape[0], atol=1e-12)


def test_on_policy_tangent_equals_tree_policy_gradient():
    """Theorem 2(i): at pi_theta = pi_old the TreeOPSD gradient equals the TreePO /
    tree-PG gradient -V_i/L_x on every token (Eq. 35)."""
    for seed in range(5):
        logp, logp_old, ti, V, q, tau, L = _random_turns(seed)
        on = logp_old.clone().requires_grad_(True)
        (g_opsd,) = torch.autograd.grad(treeopsd_loss(on, logp_old, ti, q, tau), on)
        on2 = logp_old.clone().requires_grad_(True)
        (g_po,) = torch.autograd.grad(treepo_loss(on2, ti, V), on2)
        assert torch.allclose(g_opsd, g_po, atol=1e-12)
        assert torch.allclose(g_po, -V[ti] / on.shape[0])


def test_two_forms_of_eq13_differ_by_a_constant():
    """Eq. 13: tau/L_x sum L_i KL(Q*||P) = 1/L_x sum L_i[-V s + tau psi(s)] + C_x."""
    logp, logp_old, ti, V, q, tau, L = _random_turns(3)
    diffs = []
    g = torch.Generator().manual_seed(7)
    for _ in range(5):
        lp = logp_old + 0.5 * torch.randn(logp_old.shape, generator=g)
        a = treeopsd_loss(lp, logp_old, ti, q, tau)
        b = treeopsd_loss_linear_form(lp, logp_old, ti, V, tau)
        diffs.append(float(a - b))
    assert np.ptp(diffs) < 1e-10


def test_fenchel_young_identity_eq30():
    """tau * KL(q(V)||sigmoid(s)) = J(s*) - J(s) with J(s) = V s - tau log cosh(s/2)."""
    tau = tau_from_K(12, RHO)
    for V in [-2.0, -0.3, 0.0, 1.1, 3.0]:
        q = teacher_q(V, tau)
        s_star = math.log(q / (1 - q))
        for s in [-3.0, -0.5, 0.0, 0.2, 2.0]:
            lhs = tau * float(bernoulli_kl(torch.tensor(q), torch.tensor(s)))
            rhs = float(proximal_objective(s_star, V, tau) - proximal_objective(s, V, tau))
            assert abs(lhs - rhs) < 1e-10


def test_theorem1_descendant_aggregation_gradient():
    """Eq. 27-28: sum_u alpha_u KL(q_u || p) = sum_u alpha_u KL(q_u||q_i) + KL(q_i||p);
    the gradient wrt the parent's score is identical."""
    rng = np.random.default_rng(1)
    alpha = rng.dirichlet(np.ones(5))
    q_u = torch.tensor(rng.uniform(0.3, 0.7, 5))
    q_i = float((torch.tensor(alpha) * q_u).sum())
    for s0 in [-1.0, 0.0, 0.4]:
        s = torch.tensor(s0, requires_grad=True)
        agg = (torch.tensor(alpha) * bernoulli_kl(q_u, s.expand(5))).sum()
        (g1,) = torch.autograd.grad(agg, s)
        s2 = torch.tensor(s0, requires_grad=True)
        par = bernoulli_kl(torch.tensor(q_i), s2)
        (g2,) = torch.autograd.grad(par, s2)
        assert torch.allclose(g1, g2, atol=1e-12)
        const = (torch.tensor(alpha) * bernoulli_kl(q_u, torch.logit(torch.tensor(q_i)).expand(5))).sum()
        assert torch.allclose(agg.detach(), const + par.detach(), atol=1e-12)


def test_ascent_weight_sign_reversal_and_drift_bound():
    """Theorem 2 (ii)-(iii), Eqs. 36-37."""
    tau = tau_from_K(12, RHO)
    for V in [-2.5, -0.4, 0.7, 3.0]:
        q = teacher_q(V, tau)
        phi = math.log(q / (1 - q))
        assert np.isclose(ascent_weight(0.0, V, tau), V)
        assert abs(ascent_weight(phi, V, tau)) < 1e-12
        for s in np.linspace(-5, 5, 41):
            g = ascent_weight(s, V, tau)
            if abs(s - phi) > 1e-9:
                assert np.sign(g) == np.sign(phi - s)
            assert abs(g - V) <= tau / 4 * abs(s) + 1e-12
            assert abs(g) <= (0.5 + RHO) * tau + 1e-12


def test_scalar_contraction_vs_fixed_credit_overshoot():
    """Eq. 38: s <- s + eta (L_i/L_x) Gamma(s) contracts to phi without overshoot
    when eta <= 4 L_x/(tau L_i); the fixed-credit update s <- s + eta (L_i/L_x) V
    keeps moving past phi."""
    tau = tau_from_K(12, RHO)
    V = 2.0
    q = teacher_q(V, tau)
    phi = math.log(q / (1 - q))
    L_i, L_x = 10, 40
    eta = 4 * L_x / (tau * L_i)
    s_opsd, s_fixed = 0.0, 0.0
    prev_gap = abs(phi)
    for _ in range(200):
        s_opsd += eta * L_i / L_x * ascent_weight(s_opsd, V, tau)
        s_fixed += eta * L_i / L_x * V
        assert s_opsd <= phi + 1e-12  # never overshoots
        assert abs(phi - s_opsd) <= prev_gap + 1e-12
        prev_gap = abs(phi - s_opsd)
    assert abs(s_opsd - phi) < 1e-6
    assert s_fixed > 10 * phi


def test_first_block_identity_and_later_block_divergence_bound():
    """Eq. 40: ||g_OPSD - g_fixed|| <= tau/(4 L_x) * sum_i |s_i| ||d_i|| ; at the
    first block (s=0) the two are identical. In score space with d_i = L_i * e_i."""
    tau = tau_from_K(12, RHO)
    rng = np.random.default_rng(5)
    V = rng.uniform(-3, 3, 8)
    L = rng.integers(3, 30, 8)
    L_x = L.sum()
    for scale in [0.0, 0.01, 0.1, 1.0]:
        s = rng.normal(0, scale, 8)
        g_opsd = np.array([ascent_weight(si, vi, tau) for si, vi in zip(s, V)]) * L / L_x
        g_fixed = V * L / L_x
        bound = tau / (4 * L_x) * np.sum(np.abs(s) * L)
        assert np.linalg.norm(g_opsd - g_fixed) <= bound + 1e-12
        if scale == 0:
            assert np.allclose(g_opsd, g_fixed)


# ---------------------------------------------------------------------------
# Budget accounting: M (1 + N/2), anchors, unique-turn reconstruction
# ---------------------------------------------------------------------------


def test_nominal_budget():
    assert nominal_tree_budget(4, 2) == 8 == 8  # == GRPO G=8
    assert nominal_tree_budget(8, 0) == 8
    assert nominal_tree_budget(2, 6) == 8
    assert nominal_tree_budget(4, 0) == 4 and nominal_tree_budget(2, 2) == 4


def test_anchor_sampling_support_and_multiset():
    rng = np.random.default_rng(0)
    counts = np.zeros(6)
    for _ in range(20000):
        a = sample_anchors(6, 2, rng)
        assert len(a) == 2 and all(1 <= t <= 5 for t in a)
        for t in a:
            counts[t] += 1
    freq = counts[1:] / counts[1:].sum()
    assert np.allclose(freq, 0.2, atol=0.01)  # uniform on {1..T-1}
    assert sample_anchors(1, 2, rng) == [0, 0]


@pytest.mark.parametrize("T", [2, 3, 5, 8])
def test_equal_turns_give_exactly_half(T):
    c = [100.0] * T
    assert np.isclose(expected_suffix_cost(c), sum(c) / 2)
    roots = [c] * 4
    assert np.isclose(expected_tree_cost(roots, 2), 8 * sum(c))
    assert np.isclose(budget_ratio_vs_grpo(roots, 2, 8), 1.0)


def test_budget_breaks_when_rollouts_are_single_turn():
    """T = 1 (answer without searching) => anchor 0 => each continuation is a FULL
    rollout, so an (4, 2) tree costs 12 full trajectories, 1.5x GRPO G=8."""
    roots = [[500.0]] * 4
    assert np.isclose(budget_ratio_vs_grpo(roots, 2, 8), 1.5)


def test_budget_front_loaded_and_tool_call_units():
    # long thinking in turn 1, short later turns: continuations are much cheaper
    roots = [[2000.0, 300.0, 300.0, 200.0]] * 4
    assert budget_ratio_vs_grpo(roots, 2, 8) < 0.8
    # back-loaded (long final answer turn): continuations cost MORE than half
    roots = [[100.0, 100.0, 100.0, 2000.0]] * 4
    assert budget_ratio_vs_grpo(roots, 2, 8) > 1.3
    # tool calls: answer turn has 0 calls -> suffix calls = T - t - 1
    calls = [1, 1, 1, 0]
    assert np.isclose(expected_suffix_cost(calls), np.mean([2, 1, 0]))


def test_unique_turn_ownership_and_trainer_token_inflation():
    for seed in range(10):
        t = build_tree(seed=seed)
        own = owner_rows(t)
        assert set(own) == set(t.nodes)  # every turn owned exactly once
        ctx = {i: 50 for i in t.nodes}  # 50 observation/prompt tokens per turn
        tok = trainer_tokens(t, ctx)
        assert tok["L_x"] == sum(n.n_tokens for n in t.nodes.values())
        assert tok["row_tokens"] >= tok["unique_tokens"]


def test_flat_tree_has_no_shared_prefix():
    """(M, N) = (8, 0) is a flat group: 8 leaves, V = standardized rewards."""
    t = build_tree(M=8, N=0, seed=3)
    assert t.K == 8
    V = backup(t)
    leaves = t.leaves
    A = standardize([t.nodes[k].reward for k in leaves])
    for k, a in zip(leaves, A):
        for i in t.path_to_root(k):
            assert np.isclose(V[i], a)
