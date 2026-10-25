"""Answer extraction, EM and thresholded-F1 reward for the Multi-Hop QA track.

Normalisation follows the SQuAD / Search-R1 convention (lower-case, strip
punctuation, drop English articles a/an/the, collapse whitespace). The paper
(App. E.1.1) says "exact match" and "thresholded token F1 (>= 0.3)" without
giving a normaliser; PAPER.md A9 records this choice.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, Optional

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def extract_answer(response: str) -> Optional[str]:
    """Content of the LAST <answer>...</answer> span, or None if absent."""
    m = _ANSWER_RE.findall(response)
    if not m:
        return None
    return m[-1].strip()


def exact_match(pred: Optional[str], golds: Iterable[str]) -> float:
    if pred is None:
        return 0.0
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: Optional[str], gold: str) -> float:
    if pred is None:
        return 0.0
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


def max_f1(pred: Optional[str], golds: Iterable[str]) -> float:
    return max((token_f1(pred, g) for g in golds), default=0.0)


def mhqa_train_reward(response: str, golds: Iterable[str], threshold: float = 0.3) -> float:
    """App. E.1.1: token F1 if F1 >= 0.3 else 0 (max over gold aliases)."""
    f1 = max_f1(extract_answer(response), list(golds))
    return f1 if f1 >= threshold else 0.0
