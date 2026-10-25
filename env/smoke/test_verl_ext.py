"""CPU tests for the verl integration pieces.

Pure tests (tree assembly, loss cores vs the reference math, retriever search, BC+
decryption, eval scoring) always run. Tests that need verl itself (registry, DataProto,
the patched ppo_loss, the tree rollout manager) are skipped when verl is not importable;
run them in the local CPU venv or inside the image (gate G4/G5).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "retriever"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from treeopsd import core  # noqa: E402
from treeopsd.verl_ext.tree_build import (  # noqa: E402
    batch_credit, budget_stats, group_credit, plan_anchors, turn_segments, anchor_rng,
)
from treeopsd.verl_ext.losses import treeopsd_fkl_core, treepo_fixed_core, treepo_clip_core  # noqa: E402

try:
    import verl  # noqa: F401
    HAVE_VERL = True
except Exception:
    HAVE_VERL = False
needs_verl = pytest.mark.skipif(not HAVE_VERL, reason="verl not importable here")

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------- helpers
def make_group(M=4, N=2, Lr=120, seed=0, empty_cont=None):
    """Synthesise K rows: roots with T_m turns (turn = 6-10 tokens, obs = 5 tokens),
    continuations copy the root prefix up to anchor t then add new turns."""
    rng = np.random.default_rng(seed)
    K = M * (N + 1)
    mask = torch.zeros(K, Lr, dtype=torch.long)
    lp = torch.zeros(K, Lr)
    roots = []
    for m in range(M):
        T = int(rng.integers(1, 5))
        pos, segs = 0, []
        for k in range(T):
            L = int(rng.integers(6, 11))
            segs.append((pos, pos + L))
            pos += L + (5 if k < T - 1 else 0)
        for s, e in segs:
            mask[m, s:e] = 1
            lp[m, s:e] = torch.tensor(-rng.uniform(0.05, 2.0, e - s))
        roots.append(segs)
    anchors = plan_anchors([len(s) - 1 for s in roots], N, rng)
    for m in range(M):
        for n in range(N):
            r = M + m * N + n
            t = anchors[m][n]
            if t > 0:
                end_prefix = roots[m][t - 1][1] + 5
                mask[r, :end_prefix] = mask[m, :end_prefix]
                lp[r, :end_prefix] = lp[m, :end_prefix]
                pos = end_prefix
            else:
                pos = 0
            if empty_cont == (m, n):
                continue
            for k in range(int(rng.integers(1, 3))):
                L = int(rng.integers(6, 11))
                mask[r, pos:pos + L] = 1
                lp[r, pos:pos + L] = torch.tensor(-rng.uniform(0.05, 2.0, L))
                pos += L + 5
    rewards = rng.integers(0, 2, K).astype(float)
    if rewards.std() == 0:
        rewards[0] = 1 - rewards[0]
    return mask, lp, rewards, anchors, roots


def reference_tree(mask, lp, rewards, anchors, M, N):
    """Independent re-derivation of the tree with treeopsd.core.Node, for comparison."""
    K = M * (N + 1)
    segs = [turn_segments(mask[r].tolist()) for r in range(K)]
    nodes, key2id = [], {}

    def add(key, parent, s, e, row, reward):
        key2id[key] = len(nodes)
        nodes.append(core.Node(len(nodes), parent, float(lp[row, s:e].sum()), e - s, reward=reward))
        return key2id[key]

    for r in range(K):
        if r < M:
            p = None
            for k, (s, e) in enumerate(segs[r]):
                p = add(("root", r, k), p, s, e, r, rewards[r] if k == len(segs[r]) - 1 else None)
        else:
            m, n = divmod(r - M, N)
            t = anchors[m][n]
            p = key2id[("root", m, t - 1)] if t > 0 else None
            own = segs[r][t:]
            if not own:
                key2id[("empty", r)] = len(nodes)
                nodes.append(core.Node(len(nodes), p, -1e30, 0, reward=rewards[r]))
            for k, (s, e) in enumerate(own):
                p = add(("cont", r, k), p, s, e, r, rewards[r] if k == len(own) - 1 else None)
    return core.Tree(nodes)


# ----------------------------------------------------------------------------- tree build
def test_turn_segments():
    assert turn_segments([0, 1, 1, 0, 0, 1, 0, 1, 1]) == [(1, 3), (5, 6), (7, 9)]
    assert turn_segments([]) == [] and turn_segments([0, 0]) == []


@pytest.mark.parametrize("seed", range(12))
def test_group_credit_matches_reference_backup(seed):
    M, N = 4, 2
    mask, lp, rewards, anchors, _ = make_group(seed=seed)
    gc = group_credit(mask, lp, rewards.tolist(), M, N, anchors)
    ref = reference_tree(mask, lp, rewards, anchors, M, N)
    Vref = core.backup(ref)
    # every owned token carries the V of its node; the multiset of (V, L_i) must match
    got = {}
    for nid in gc.turn_id.unique().tolist():
        if nid < 0:
            continue
        sel = gc.turn_id == nid
        got[nid] = (round(float(gc.advantages[sel][0]), 9), int(sel.sum()))
    exp = sorted((round(Vref[i], 9), n.n_tokens) for i, n in ref.nodes.items() if n.n_tokens > 0)
    assert sorted(got.values()) == exp
    # unique-turn reconstruction: weights sum to 1 per prompt (sum_i L_i / L_x)
    assert abs(float(gc.token_weight.sum()) - 1.0) < 1e-6
    assert ref.K == M * (N + 1)


def test_shared_prefix_tokens_are_not_owned_by_continuations():
    M, N = 4, 2
    mask, lp, rewards, anchors, roots = make_group(seed=3)
    gc = group_credit(mask, lp, rewards.tolist(), M, N, anchors)
    for m in range(M):
        for n in range(N):
            r = M + m * N + n
            t = anchors[m][n]
            if t > 0:
                end_prefix = roots[m][t - 1][1]
                assert (gc.turn_id[r, :end_prefix] == -1).all()
                assert (gc.token_weight[r, :end_prefix] == 0).all()


def test_failed_continuation_gets_virtual_leaf():
    M, N = 4, 2
    for seed in range(30):
        mask, lp, rewards, anchors, _ = make_group(seed=seed, empty_cont=(0, 0))
        if anchors[0][0] > 0:
            break
    gc = group_credit(mask, lp, rewards.tolist(), M, N, anchors)
    assert gc.stats["tree/empty_leaves"] == 1.0


def test_batch_credit_weights_and_unique_ids():
    M, N, P, Be = 4, 2, 3, 3
    ms, lps, rs, ans = [], [], [], []
    for g in range(P):
        m, lp, r, a, _ = make_group(seed=10 + g)
        ms.append(m), lps.append(lp), rs.append(torch.tensor(r)), ans.append(a)
    adv, tid, w, tau, stats = batch_credit(torch.cat(ms), torch.cat(lps), torch.cat(rs), M, N, ans,
                                           rho=0.25, prompts_per_block=Be)
    assert abs(float(w.sum()) - P / Be) < 1e-6
    K = M * (N + 1)
    ids_per_group = [set(tid[g * K:(g + 1) * K].unique().tolist()) - {-1} for g in range(P)]
    assert not (ids_per_group[0] & ids_per_group[1]) and not (ids_per_group[1] & ids_per_group[2])
    assert torch.allclose(tau[0, 0], torch.tensor(core.tau_from_K(12, 0.25)))


def test_budget_stats_counts_only_new_continuation_tokens():
    M, N = 4, 2
    mask, lp, rewards, anchors, roots = make_group(seed=5)
    st = budget_stats(mask, M, N, [anchors])
    root_tok = sum(int(mask[m].sum()) for m in range(M))
    new = 0
    for m in range(M):
        for n in range(N):
            r = M + m * N + n
            new += sum(e - s for s, e in turn_segments(mask[r].tolist())[anchors[m][n]:])
    assert st["budget/gen_tokens_per_prompt"] == root_tok + new


def test_anchor_rng_is_deterministic_and_seed_dependent():
    a = anchor_rng(0, 5, "uid-x").integers(0, 10**9, 4).tolist()
    b = anchor_rng(0, 5, "uid-x").integers(0, 10**9, 4).tolist()
    c = anchor_rng(1, 5, "uid-x").integers(0, 10**9, 4).tolist()
    assert a == b and a != c


# ----------------------------------------------------------------------------- loss cores
def _rows_from_group(seed=0, drift=0.3):
    M, N = 4, 2
    mask, lp_old, rewards, anchors, _ = make_group(seed=seed)
    adv, tid, w, tau, _ = batch_credit(mask, lp_old, torch.tensor(rewards), M, N, [anchors],
                                       rho=0.25, prompts_per_block=1)
    g = torch.Generator().manual_seed(seed)
    lp = (lp_old + drift * torch.randn(lp_old.shape, generator=g) * mask).clone().requires_grad_(True)
    return lp, lp_old, adv, tid, w, tau


@pytest.mark.parametrize("seed", range(6))
def test_treeopsd_core_equals_reference_loss_and_grad(seed):
    lp, lp_old, adv, tid, w, tau = _rows_from_group(seed)
    loss, metrics = treeopsd_fkl_core(lp, lp_old, adv, tid, w, tau)
    (g,) = torch.autograd.grad(loss, lp)
    # reference: flatten owned tokens, turn index by node id, V per node
    owned = tid >= 0
    ids = tid[owned]
    uniq, inv = torch.unique(ids, return_inverse=True)
    V = torch.zeros(len(uniq)).index_put_((inv,), adv[owned].double())
    q = torch.tensor(core.teacher_q(V.numpy(), float(tau[0, 0])))
    lp2 = lp.detach()[owned].clone().requires_grad_(True)
    ref = core.treeopsd_loss(lp2, lp_old[owned], inv, q, float(tau[0, 0]))
    (g2,) = torch.autograd.grad(ref, lp2)
    # the training core runs in fp32 (as on the GPU); the reference in fp64
    assert torch.allclose(loss.double(), ref, rtol=1e-5, atol=1e-7)
    assert torch.allclose(g[owned].double(), g2, rtol=1e-4, atol=1e-8)
    assert (g[~owned] == 0).all()
    assert 0 <= metrics["treeopsd/drift_rel"]


def test_first_block_treeopsd_equals_treepo_gradient():
    lp, lp_old, adv, tid, w, tau = _rows_from_group(2, drift=0.0)
    on = lp_old.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(treeopsd_fkl_core(on, lp_old, adv, tid, w, tau)[0], on)
    on2 = lp_old.clone().requires_grad_(True)
    (g2,) = torch.autograd.grad(treepo_fixed_core(on2, adv, tid, w), on2)
    assert torch.allclose(g1, g2, atol=1e-7)  # PREREG G7 / Eq. 35 (fp32 core)


def test_treepo_clip_equals_fixed_on_policy():
    lp, lp_old, adv, tid, w, tau = _rows_from_group(4, drift=0.0)
    a = lp_old.clone().requires_grad_(True)
    (g1,) = torch.autograd.grad(treepo_clip_core(a, lp_old, adv, tid, w)[0], a)
    b = lp_old.clone().requires_grad_(True)
    (g2,) = torch.autograd.grad(treepo_fixed_core(b, adv, tid, w), b)
    assert torch.allclose(g1, g2, atol=1e-12)


# ----------------------------------------------------------------------------- verl glue
@needs_verl
def test_losses_registered_and_callable_through_verl_registry():
    import treeopsd.verl_ext.losses  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    lp, lp_old, adv, tid, w, tau = _rows_from_group(1)
    fn = get_policy_loss_fn("treeopsd_fkl")
    loss, m = fn(old_log_prob=lp_old, log_prob=lp, advantages=adv, response_mask=tid >= 0,
                 loss_agg_mode="token-sum", config=None, rollout_is_weights=None,
                 tree_turn_id=tid, tree_token_weight=w, tree_tau=tau)
    ref, _ = treeopsd_fkl_core(lp, lp_old, adv, tid, w, tau)
    assert torch.allclose(loss, ref)
    with pytest.raises(RuntimeError):
        fn(old_log_prob=lp_old, log_prob=lp, advantages=adv, response_mask=tid >= 0)


@needs_verl
def test_patched_ppo_loss_forwards_tree_inputs():
    """G4 (CPU part): verl's own ppo_loss (patched) must pass tree_* to our loss and
    return the same value as the reference on a padded-format micro-batch."""
    from tensordict import TensorDict
    from omegaconf import OmegaConf

    import treeopsd.verl_ext.losses  # noqa: F401
    from verl.workers.utils import losses as vl

    if "tree_fields" not in open(vl.__file__).read():
        pytest.skip("verl in this env is not patched (env/patches/verl-v0.9.1-treeopsd.patch)")
    lp, lp_old, adv, tid, w, tau = _rows_from_group(0)
    B, L = lp.shape
    data = TensorDict({"response_mask": (tid >= 0).long(), "old_log_probs": lp_old.float(),
                       "advantages": adv.float(), "tree_turn_id": tid, "tree_token_weight": w.float(),
                       "tree_tau": tau.float()}, batch_size=[B])
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)

    class Cfg:
        global_batch_info = {}
        loss_scale_factor = None
        loss_agg_mode = "token-sum"
        policy_loss = {"loss_mode": "treeopsd_fkl"}
        entropy_coeff = 0.0
        use_kl_loss = False

    orig = vl.no_padding_2_padding
    vl.no_padding_2_padding = lambda x, d: x  # model output already padded here
    try:
        loss, metrics = vl.ppo_loss(Cfg(), {"log_probs": lp.float()}, data)
    finally:
        vl.no_padding_2_padding = orig
    ref, _ = treeopsd_fkl_core(lp.float(), lp_old.float(), adv.float(), tid, w.float(), tau.float())
    assert torch.allclose(loss.float(), ref.float(), atol=1e-6)


@needs_verl
def test_tree_rollout_manager_order_prefix_and_meta():
    """G5 (CPU part): a fake AgentLoopManager stands in for vLLM. Checks the K-row
    layout, that every continuation request carries its root's exact token prefix,
    and the tree_meta / reassembly order."""
    from verl import DataProto
    from treeopsd.verl_ext.tree_rollout import TreeRolloutManager

    M, N, P, PL, RL = 4, 2, 2, 6, 60
    K = M * (N + 1)
    rng = np.random.default_rng(0)
    calls = []

    class FakeInner:
        def generate_sequences(self, batch):
            calls.append(batch)
            R = len(batch)
            prompts = torch.zeros(R, PL, dtype=torch.long)
            resp = torch.zeros(R, RL, dtype=torch.long)
            rmask = torch.zeros(R, RL, dtype=torch.long)
            am = torch.zeros(R, PL + RL, dtype=torch.long)
            snaps, msgs = [], []
            for i in range(R):
                pid = int(batch.non_tensor_batch["pid"][i])
                prompts[i, 2:] = torch.tensor([100 + pid] * (PL - 2))
                am[i, 2:PL] = 1
                pre = batch.non_tensor_batch["tree_prefix"][i]
                if pre:
                    n0 = len(pre["response_mask"])
                    resp[i, :n0] = torch.tensor(pre["prompt_ids"][PL - 2:])
                    rmask[i, :n0] = torch.tensor(pre["response_mask"])
                    pos, sn = n0, list(pre["snapshots"])
                else:
                    pos, sn = 0, []
                T = int(rng.integers(1, 4))
                for k in range(T):
                    resp[i, pos:pos + 5] = int(rng.integers(1000, 2000))
                    rmask[i, pos:pos + 5] = 1
                    pos += 5
                    if k < T - 1:
                        resp[i, pos:pos + 3] = 7
                        pos += 3
                        sn.append({"n_ids": PL - 2 + pos, "n_resp": pos, "n_messages": 3 + 2 * (len(sn) + 1),
                                   "assistant_turns": len(sn) + 1, "user_turns": len(sn) + 1})
                am[i, PL:PL + pos] = 1
                snaps.append(sn)
                msgs.append([{"role": "x", "content": str(j)} for j in range(3 + 2 * len(sn) + 1)])
            out = DataProto.from_dict(
                tensors={"prompts": prompts, "responses": resp, "response_mask": rmask, "attention_mask": am},
                non_tensors={"tree_snapshots": np.array(snaps + [None], dtype=object)[:-1],
                             "tree_messages": np.array(msgs + [None], dtype=object)[:-1],
                             "pid": batch.non_tensor_batch["pid"]})
            out.meta_info["timing"] = {"t": 1.0}
            return out

    pid = np.repeat(np.arange(P), K)
    uid = np.array([f"u{p}" for p in pid], dtype=object)
    prompts = DataProto.from_dict(tensors={"dummy": torch.zeros(P * K, 1)},
                                  non_tensors={"pid": pid, "uid": uid})
    prompts.meta_info["global_steps"] = 3
    mgr = TreeRolloutManager(FakeInner(), M, N, seed=0)
    out = mgr.generate_sequences(prompts)
    assert len(calls) == 2 and len(calls[0]) == P * M and len(calls[1]) == P * M * N
    assert len(out) == P * K
    assert (out.non_tensor_batch["pid"] == pid).all()  # reassembled in trainer order
    meta = out.non_tensor_batch["tree_meta"]
    for g in range(P):
        for j in range(K):
            r = g * K + j
            if j >= M:
                d = meta[r]
                root = g * K + d["root"]
                t = d["anchor"]
                if t > 0:
                    n_resp = out.non_tensor_batch["tree_snapshots"][root][t - 1]["n_resp"]
                    assert torch.equal(out.batch["responses"][r, :n_resp], out.batch["responses"][root, :n_resp])
                    assert torch.equal(out.batch["response_mask"][r, :n_resp], out.batch["response_mask"][root, :n_resp])


# ----------------------------------------------------------------------------- retriever
def test_exact_topk_matches_bruteforce_with_ties():
    from common import exact_topk

    g = torch.Generator().manual_seed(0)
    V = torch.randn(5000, 16, generator=g, dtype=torch.float32)
    V[123] = V[456]  # exact duplicate -> tie, lower index must win
    Q = torch.randn(20, 16, generator=g, dtype=torch.float32)
    Q[0] = V[456]
    s, i = exact_topk(Q, V, 3, chunk=777)
    S = (Q @ V.T).double().numpy()
    for r in range(20):
        order = np.lexsort((np.arange(5000), -S[r]))[:3]
        assert i[r].tolist() == order.tolist()
    assert i[0][0] == 123


def test_jsonl_corpus_random_access(tmp_path):
    from common import JsonlCorpus, split_title_text

    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"id": str(k), "contents": f'"T{k}"\ntext {k}'}) for k in range(50)) + "\n")
    c = JsonlCorpus(str(p))
    assert len(c) == 50 and split_title_text(c.get(37)) == ("T37", "text 37")
    c2 = JsonlCorpus(str(p))  # cached offsets path
    assert c2.get(0)["id"] == "0"


def test_server_batches_requests_with_fake_encoder(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import server

    n, d = 300, 8
    rng = np.random.default_rng(0)
    V = rng.standard_normal((n, d)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)  # unit vectors: a vector's own row is its top hit
    np.save(tmp_path / "v.npy", V)
    (tmp_path / "c.jsonl").write_text(
        "\n".join(json.dumps({"id": str(k), "contents": f'"T{k}"\nbody {k}'}) for k in range(n)) + "\n")

    class FakeEnc:
        def __init__(self, *a, **k):
            pass

        def encode(self, qs):
            return torch.tensor(np.stack([V[int(q.split()[-1])] for q in qs]))

    monkeypatch.setattr(server, "E5Encoder", FakeEnc)
    args = type("A", (), dict(backend="wiki18", vectors=str(tmp_path / "v.npy"), corpus=str(tmp_path / "c.jsonl"),
                              encoder="x", docids=None, device="cpu", batch_window_ms=5.0, max_batch=64,
                              max_queries=4, max_query_chars=512, doc_chars=100, log=str(tmp_path / "log.jsonl")))
    with TestClient(server.build_app(args)) as cl:
        r = cl.post("/retrieve", json={"queries": ["q 5", "q 17"], "topk": 3}).json()
        assert r["results"][0][0]["id"] == "5" and r["results"][1][0]["title"] == "T17"
        rv = cl.post("/retrieve_vec", json={"vectors": [V[9].tolist()], "topk": 2}).json()
        assert rv["ids"][0][0] == 9
        h = cl.get("/health").json()
        assert h["n_vectors"] == n and h["queries"] == 2


# ----------------------------------------------------------------------------- data/eval
def test_bcplus_decrypt_roundtrip():
    from prepare_bcplus import decrypt, encrypt

    s = "Which 19th-century treaty ... ? ünïcode"
    assert decrypt(encrypt(s)) == s


def test_search_tool_formatting_and_query_cleaning():
    pytest.importorskip("aiohttp")
    if not HAVE_VERL:
        pytest.skip("verl not importable")
    from treeopsd.verl_ext.search_tool import clean_queries, format_hits

    assert clean_queries({"query_list": ["a", " ", "b" * 600, "c", "d", "e"]}) == ["a", "b" * 512, "c", "d"]
    assert clean_queries({"query_list": "single"}) == ["single"]
    assert clean_queries({"query_list": 5}) == []
    t = format_hits(["q"], [[{"title": "T", "text": "x"}]], 1000)
    assert t == "Query: q\nDoc 1(Title: T) x"


def test_eval_scoring_end_to_end(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for bench, outs in {"hotpotqa": ["<answer>Paris</answer>", "<answer>no</answer>"],
                        "2wikimultihopqa": ["<answer>a</answer>"] * 2, "musique": ["x", "<answer>b</answer>"],
                        "bamboogle": ["<tool_response><answer>Paris</answer></tool_response>"] * 2,
                        "bcplus": ["<answer>42</answer>", "<answer>41</answer>"]}.items():
        d = tmp_path / bench
        d.mkdir()
        gts = {"hotpotqa": ["paris"], "2wikimultihopqa": ["a"], "musique": ["b"], "bamboogle": ["paris"],
               "bcplus": ["42"]}[bench]
        with open(d / "0.jsonl", "w") as f:
            for o in outs:
                f.write(json.dumps({"input": "Question: q?", "output": o, "gts": gts, "score": 0}) + "\n")
    tele = run / "telemetry"
    tele.mkdir()
    with open(tele / "steps.jsonl", "w") as f:
        for k in range(3):
            f.write(json.dumps({"budget/gen_tokens_per_prompt": 100 + k, "budget/tool_calls_per_prompt": 2,
                                "time/step_wall_s_prev": 3600.0}) + "\n")
    args = ["python3", os.path.join(ROOT, "scripts", "eval_qa.py"), "--run-dir", str(run), "--telemetry", str(tele)]
    for b in ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle", "bcplus"]:
        args += ["--dump", f"{b}={tmp_path / b}"]
    subprocess.run(args, check=True, capture_output=True)
    s = json.load(open(run / "summary.json"))
    assert s["em"] == {"hotpotqa": 50.0, "2wikimultihopqa": 100.0, "musique": 50.0, "bamboogle": 0.0, "bcplus": 50.0}
    assert abs(s["train_gpu_hours"] - 3.0) < 1e-9 and s["rollout_gen_tokens_per_prompt"] == 101
    out = subprocess.run(["bash", os.path.join(ROOT, "autoresearch.sh"), str(run)], capture_output=True,
                         text=True, check=True).stdout
    assert "METRIC em_headline_bcplus=50" in out and "METRIC em_qa_avg=50" in out
