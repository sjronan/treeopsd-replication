"""verl custom reward function (reward.custom_reward_function.path/name=compute_score).

Training reward = thresholded token F1 (>= 0.3, else 0) against the best gold alias,
paper App. E.1.1. EM and raw F1 are returned alongside for logging. Only the text of the
final assistant turn counts: we take the LAST <answer>...</answer> span of the decoded
response (observations cannot contain an <answer> tag that the model did not write,
because tool output is wrapped in <tool_response>; we strip those blocks first).
"""

from __future__ import annotations

import re
from typing import Any

from treeopsd.metrics import exact_match, extract_answer, max_f1

_TOOL_RESP = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)


def _golds(ground_truth: Any):
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("target", ground_truth.get("ground_truth", []))
    if isinstance(ground_truth, str):
        return [ground_truth]
    return [str(g) for g in list(ground_truth)]


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    text = _TOOL_RESP.sub(" ", solution_str or "")
    pred = extract_answer(text)
    golds = _golds(ground_truth)
    f1 = max_f1(pred, golds)
    em = exact_match(pred, golds)
    score = f1 if f1 >= 0.3 else 0.0
    return {"score": score, "em": em, "f1": f1, "has_answer": float(pred is not None)}
