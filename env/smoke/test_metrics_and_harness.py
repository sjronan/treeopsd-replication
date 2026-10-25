"""Tests for EM / F1 reward and for the autoresearch.sh metric emitter."""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from treeopsd.metrics import (  # noqa: E402
    exact_match,
    extract_answer,
    mhqa_train_reward,
    normalize_answer,
    token_f1,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_normalize_and_em():
    assert normalize_answer("The  Eiffel-Tower!") == "eiffeltower"
    assert exact_match("the Beatles", ["Beatles"]) == 1.0
    assert exact_match("Beatles band", ["Beatles"]) == 0.0
    assert exact_match(None, ["x"]) == 0.0


def test_extract_last_answer_span():
    r = "<think>x</think><answer> Paris </answer> more <answer>Lyon</answer>"
    assert extract_answer(r) == "Lyon"
    assert extract_answer("no tags") is None


def test_thresholded_f1_reward():
    golds = ["Arthur's Magazine"]
    assert mhqa_train_reward("<answer>Arthur's Magazine</answer>", golds) == 1.0
    # prec 2/3, rec 1 -> F1 0.8 (kept); second case F1 0.25 < 0.3 (zeroed)
    f1 = token_f1("arthurs magazine story", "arthurs magazine")
    assert abs(f1 - 0.8) < 1e-9
    assert mhqa_train_reward("<answer>magazine of nothing at all here</answer>", golds) == 0.0
    assert mhqa_train_reward("no answer tag", golds) == 0.0


def test_autoresearch_sh_emits_metric_lines(tmp_path):
    summary = {
        "em": {"hotpotqa": 48.0, "2wikimultihopqa": 50.0, "musique": 22.0, "bamboogle": 54.0},
        "rollout_gen_tokens_per_prompt": 20000,
        "tool_calls_per_prompt": 14.5,
        "train_tokens_per_prompt": 84000,
        "train_gpu_hours": 12.5,
        "drift_rel_blocks234": 0.031,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    out = subprocess.run(
        ["bash", os.path.join(REPO, "autoresearch.sh"), str(tmp_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    assert "METRIC em_avg=43.5" in out
    assert "METRIC em_hotpotqa=48" in out
    assert "METRIC rollout_gen_tokens_per_prompt=20000" in out
    assert "METRIC train_gpu_hours=12.5" in out
    assert all(line.startswith("METRIC ") for line in out)


def test_autoresearch_sh_fails_without_summary(tmp_path):
    p = subprocess.run(["bash", os.path.join(REPO, "autoresearch.sh"), str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode != 0
