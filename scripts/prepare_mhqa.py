#!/usr/bin/env python3
"""FlashRAG jsonl -> verl RL parquet with the paper's prompt (App. F.1), plus fixed
eval subsets and the a-priori horizon strata (PREREG F5).

    python3 scripts/prepare_mhqa.py --src $D/mhqa --out $D/prepared/mhqa

Writes
  train.parquet                90,447 HotpotQA distractor-train questions
  val_dev.parquet              in-training dev loop: 500 HotpotQA + 500 MuSiQue + 500 2Wiki
                               (fixed seed 20261021) + all 125 Bamboogle
  eval_<bench>.parquet         full sets for the fixed-final-checkpoint eval
  strata.json                  id -> {bench, hops / type}
  manifest.json                row counts + sha256 of every output (record in the worklog)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import pandas as pd

SYSTEM = "You are a helpful and harmless assistant."
# App. F.1, verbatim except PDF line breaks (PAPER.md A21: whitespace reconstructed)
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every "
    "time you get new information. After reasoning, if you find you lack some knowledge, you can call "
    "a search engine. Emit one Hermes tool call per assistant turn and no surrounding prose in this "
    "form:\n<tool_call>\n{\"name\":\"search\",\n \"arguments\":{\"query_list\":[\"fully formed search "
    "query\"]}}\n</tool_call>\nThe arguments value must be a JSON object and query_list must be a "
    "non-empty JSON array containing only non-empty strings. Never put a bare query, Python syntax, "
    "Markdown fences, or trailing commas inside <tool_call>. The search engine will return the top "
    "searched results between <tool_response> and </tool_response>. You can search as many times as "
    "you want. If you find no further external knowledge needed, directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: "
)
BENCH_FILES = {
    "hotpotqa": "hotpotqa/dev.jsonl",
    "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
    "musique": "musique/dev.jsonl",
    "bamboogle": "bamboogle/test.jsonl",
}
EXPECTED = {"train": 90447, "hotpotqa": 7405, "2wikimultihopqa": 12576, "musique": 2417, "bamboogle": 125}
SEED = 20261021


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def to_row(rec, bench, split, i, agent_name):
    q = rec["question"].strip()
    return {
        "data_source": f"mhqa/{bench}",
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_PREFIX + q}],
        "ability": "multihop_qa",
        "reward_model": {"style": "rule", "ground_truth": [str(a) for a in rec["golden_answers"]]},
        "extra_info": {"index": i, "split": split, "bench": bench, "id": rec["id"], "question": q},
        "agent_name": agent_name,
    }


def stratum(rec, bench):
    md = rec.get("metadata") or {}
    if bench == "musique":
        return {"hops": len(md.get("question_decomposition") or [])}
    if bench in ("hotpotqa", "2wikimultihopqa"):
        return {"type": md.get("type")}
    return {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-name", default="tree_tool_agent")
    ap.add_argument("--dev-per-bench", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = {}

    train = read_jsonl(os.path.join(a.src, "hotpotqa/train.jsonl"))
    assert len(train) == EXPECTED["train"], len(train)
    df = pd.DataFrame([to_row(r, "hotpotqa", "train", i, a.agent_name) for i, r in enumerate(train)])
    p = os.path.join(a.out, "train.parquet")
    df.to_parquet(p, index=False)
    man["train.parquet"] = {"rows": len(df)}

    strata, dev_rows = {}, []
    rng = random.Random(SEED)
    for bench, rel in BENCH_FILES.items():
        recs = read_jsonl(os.path.join(a.src, rel))
        assert len(recs) == EXPECTED[bench], (bench, len(recs))
        rows = [to_row(r, bench, "eval", i, a.agent_name) for i, r in enumerate(recs)]
        p = os.path.join(a.out, f"eval_{bench}.parquet")
        pd.DataFrame(rows).to_parquet(p, index=False)
        man[f"eval_{bench}.parquet"] = {"rows": len(rows)}
        for r in recs:
            strata[f"{bench}/{r['id']}"] = {"bench": bench, **stratum(r, bench)}
        if bench == "bamboogle" or len(rows) <= a.dev_per_bench:
            sel = list(range(len(rows)))
        else:
            sel = sorted(rng.sample(range(len(rows)), a.dev_per_bench))
        dev_rows += [rows[i] for i in sel]
    pd.DataFrame(dev_rows).to_parquet(os.path.join(a.out, "val_dev.parquet"), index=False)
    man["val_dev.parquet"] = {"rows": len(dev_rows)}
    json.dump(strata, open(os.path.join(a.out, "strata.json"), "w"))
    for k in list(man):
        man[k]["sha256"] = sha(os.path.join(a.out, k))
    man["strata.json"] = {"sha256": sha(os.path.join(a.out, "strata.json"))}
    hop_counts = {}
    for v in strata.values():
        if v["bench"] == "musique":
            hop_counts[v["hops"]] = hop_counts.get(v["hops"], 0) + 1
    man["musique_hops"] = hop_counts
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
