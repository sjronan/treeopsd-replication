"""Gate G3b: sanity of the tier-2 (BrowseComp-Plus) retriever. Retrieval-only recall of
the human-labelled evidence / gold documents for the 830 queries, through the same
/retrieve endpoint the agent uses. Prints recall@5 and recall@100 (no plaintext queries).

    python3 retriever/check_bcplus.py --server http://127.0.0.1:8001/retrieve \
        --parquet $D/prepared/bcplus/eval_bcplus.parquet

Compare with the Qwen3-Embedding-8B retriever-only numbers in the BrowseComp-Plus paper
(arXiv:2508.06600); a large shortfall means the query prompt/pooling differs from the one
the released vectors were built with -- stop and fix before any tier-2 eval.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    df = pd.read_parquet(a.parquet, columns=["extra_info"])
    infos = list(df["extra_info"])
    r5e, r100e, r5g, r100g = [], [], [], []
    for i in range(0, len(infos), 4):
        batch = infos[i:i + 4]
        res = requests.post(a.server, json={"queries": [b["question"] for b in batch], "topk": a.k},
                            timeout=600).json()["results"]
        for info, hits in zip(batch, res):
            ids = [h["id"] for h in hits]
            ev, gd = set(info["evidence_docids"]), set(info["gold_docids"])
            if ev:
                r5e.append(len(ev & set(ids[:5])) / len(ev))
                r100e.append(len(ev & set(ids[:100])) / len(ev))
            if gd:
                r5g.append(len(gd & set(ids[:5])) / len(gd))
                r100g.append(len(gd & set(ids[:100])) / len(gd))
    print(json.dumps({"n": len(infos), "evidence_recall@5": float(np.mean(r5e)),
                      "evidence_recall@100": float(np.mean(r100e)), "gold_recall@5": float(np.mean(r5g)),
                      "gold_recall@100": float(np.mean(r100g))}, indent=1))


if __name__ == "__main__":
    main()
