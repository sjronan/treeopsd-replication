"""Gate G3: the server's exact GPU search must return the same top-k ids as faiss-cpu
IndexFlatIP for 1,000 fixed queries (the first 1,000 HotpotQA dev questions).

    python3 retriever/check_vs_faiss.py --server http://127.0.0.1:8000/retrieve \
        --index $D/wiki18-index/e5_Flat.index --encoder $D/model-e5-base-v2 \
        --questions $D/mhqa/hotpotqa/dev.jsonl --corpus $D/wiki18/wiki-18.jsonl --n 1000

PASS: every mismatching top-3 id list is a score tie (|ds| < 1e-6 at every rank). Founder 2026-10-21 19:43 PT:
the earlier ">= 999/1000 identical" bar assumed rare ties; wiki-18's duplicate passages make ~5% of lists tie.
Also reports p50/p95 latency of 4-query calls. Exit code 0 on PASS, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import requests  # noqa: E402

from common import E5Encoder, JsonlCorpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()

    import faiss

    faiss.omp_set_num_threads(a.threads)
    qs = []
    with open(a.questions) as f:
        for line in f:
            qs.append(json.loads(line)["question"])
            if len(qs) >= a.n:
                break
    enc = E5Encoder(a.encoder, "cpu")
    Q = np.concatenate([enc.encode(qs[i:i + 64]).numpy() for i in range(0, len(qs), 64)]).astype(np.float32)
    idx = faiss.read_index(a.index, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    t = time.time()
    D, I = idx.search(Q, a.k + 2)
    print(f"faiss-cpu search {len(qs)} queries: {time.time()-t:.1f}s")
    corpus = JsonlCorpus(a.corpus)

    # (1) identity of the exact search: same query vectors into both engines
    base = a.server.rsplit("/", 1)[0]
    same, tie_only, bad = 0, 0, []
    for i in range(0, len(qs), 50):
        r = requests.post(base + "/retrieve_vec", json={"vectors": Q[i:i + 50].tolist(), "topk": a.k},
                          timeout=600).json()
        for j, (ids, sc) in enumerate(zip(r["ids"], r["scores"])):
            qi = i + j
            if list(ids) == I[qi, :a.k].tolist():
                same += 1
            else:
                # Tie-only: every rank where the ids differ holds the same score in both engines (|ds| < 1e-6), so the
                # lists differ only in the order of, or boundary choice among, equal-score passages
                # (wiki-18 has duplicate passages with identical vectors). The earlier test looked
                # only at a rank k / k+1 boundary tie and miscounted in-list swaps (G3 run 1).
                # Only ranks whose ids differ are compared: at a rank where both engines return the
                # same passage, a score gap is fp32 summation-order noise (GPU vs CPU), not a tie
                # question (G3 run 2: 1.1e-6 at an identical rank 1).
                fi = I[qi, :a.k].tolist()
                if all(abs(sc[r] - D[qi, r]) < 1e-6 for r in range(a.k) if ids[r] != fi[r]):
                    tie_only += 1
                bad.append({"q": qs[qi], "server": ids, "faiss": I[qi, :a.k].tolist(),
                            "server_scores": sc, "faiss_scores": D[qi, :a.k + 1].tolist()})
    # (2) end-to-end text endpoint: latency of 4-query calls + id agreement (GPU vs CPU
    #     query encoding may flip near-ties; reported, not gated)
    lat, e2e_same = [], 0
    for i in range(0, min(len(qs), 400), 4):
        batch = qs[i:i + 4]
        t = time.time()
        res = requests.post(a.server, json={"queries": batch, "topk": a.k}, timeout=300).json()["results"]
        lat.append(time.time() - t)
        for j, hits in enumerate(res):
            ref = [str(corpus.get(int(x)).get("id", int(x))) for x in I[i + j, :a.k]]
            e2e_same += int([h["id"] for h in hits] == ref)
    n = len(qs)
    p50, p95 = statistics.median(lat), sorted(lat)[int(0.95 * (len(lat) - 1))]
    print(json.dumps({"n": n, "identical": same, "mismatch": n - same, "mismatch_tie_only": tie_only, "e2e_identical_of_400": e2e_same,
                      "lat_p50_s": round(p50, 4), "lat_p95_s": round(p95, 4)}, indent=1))
    for b in bad:
        print("MISMATCH", json.dumps(b))
    ok = (n - same) == tie_only and p95 < 0.1
    print("G3", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
