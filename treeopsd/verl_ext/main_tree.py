"""Entry point: verl's PPO main (legacy RayPPOTrainer path) + TreeOPSD hooks.

    python3 -m treeopsd.verl_ext.main_tree <hydra overrides> \
        +treeopsd.tree=true +treeopsd.M=4 +treeopsd.N=2 +treeopsd.rho=0.25 \
        +treeopsd.weighting=likelihood +treeopsd.prompts_per_block=16 +treeopsd.seed=0

What it adds to verl v0.9.1 (commit 1876b06d), all inside the TaskRunner actor:
  * the tree-credit / telemetry wrapper around ray_trainer.compute_advantage (all arms);
  * if treeopsd.tree: wraps trainer.async_rollout_manager in TreeRolloutManager.
Worker-side pieces are loaded by name: the policy losses via VERL_EXTRA_MODULES (patched
verl/workers/utils/losses.py), the agent loop via rollout.agent.agent_loop_config_path.
We use the legacy trainer (trainer.use_v1=false): it calls generate_sequences and
compute_advantage synchronously, which is where the tree hooks live.
"""

from __future__ import annotations

import os

import hydra
import ray

import verl.trainer.main_ppo as vmain

import treeopsd.verl_ext.advantage  # noqa: F401  registers the 'tree_backup' name
from verl.trainer.main_ppo_v0 import BaseTaskRunner, TaskRunner
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

_ORIG_RUN = TaskRunner.__ray_metadata__.modified_class.run
_CONFIG_DIR = os.path.join(os.path.dirname(vmain.__file__), "config")


def _tree_settings(config) -> dict:
    t = config.get("treeopsd", {}) or {}
    return {
        "tree": bool(t.get("tree", False)),
        "M": int(t.get("M", 4)),
        "N": int(t.get("N", 2)),
        "rho": float(t.get("rho", 0.25)),
        "weighting": str(t.get("weighting", "likelihood")),
        "prompts_per_block": int(t.get("prompts_per_block", 16)),
        "seed": int(t.get("seed", 0)),
    }


def check_config(config) -> dict:
    """Fail fast on settings that would silently break the protocol (PREREG section 2)."""
    s = _tree_settings(config)
    a = config.actor_rollout_ref
    assert not config.trainer.use_v1, "use the legacy trainer: trainer.use_v1=false"
    assert not config.trainer.balance_batch, "trainer.balance_batch must be false (keeps prompt groups contiguous)"
    assert not a.actor.shuffle, "actor.shuffle must be false (blocks = disjoint consecutive prompt groups)"
    assert int(a.actor.ppo_epochs) == 1, "one pass: E optimizer blocks per collected batch"
    B = int(config.data.train_batch_size)
    Be = int(a.actor.ppo_mini_batch_size)
    assert B % Be == 0, "train_batch_size must be a multiple of ppo_mini_batch_size"
    if s["tree"]:
        K = s["M"] * (s["N"] + 1)
        assert int(a.rollout.n) == K, f"tree arms need rollout.n = K = M(N+1) = {K}"
        assert str(config.algorithm.adv_estimator) == "tree_backup"
        assert a.actor.policy_loss.loss_mode in ("treeopsd_fkl", "treepo_fixed", "treepo_clip")
        assert s["prompts_per_block"] == Be, "treeopsd.prompts_per_block must equal ppo_mini_batch_size"
    return s


@ray.remote
class TreeTaskRunner(BaseTaskRunner):
    def run(self, config):
        from treeopsd.verl_ext import advantage
        import verl.trainer.ppo.ray_trainer as rt

        s = check_config(config)
        advantage.install(s)
        if s["tree"]:
            from treeopsd.verl_ext.tree_rollout import TreeRolloutManager

            orig_init = rt.RayPPOTrainer.init_workers

            def init_workers(trainer, *a, **k):
                out = orig_init(trainer, *a, **k)
                trainer.async_rollout_manager = TreeRolloutManager(
                    trainer.async_rollout_manager, s["M"], s["N"], s["seed"]
                )
                return out

            rt.RayPPOTrainer.init_workers = init_workers
        return _ORIG_RUN(self, config)


@hydra.main(config_path=_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    validate_config(config=config, use_reference_policy=need_reference_policy(config),
                    use_critic=need_critic(config))
    vmain.run_ppo(config, task_runner_class=TreeTaskRunner)


if __name__ == "__main__":
    main()
