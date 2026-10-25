"""Frozen exact-search retriever server (FastAPI).

    # MHQA: wiki-18 + E5-base-v2 (paper setting), on the RTX PRO 6000
    python3 retriever/server.py --backend wiki18 --device cuda:1 --port 8000 \
        --vectors $D/wiki18-index/e5_vectors.f32.npy --corpus $D/wiki18/wiki-18.jsonl \
        --encoder $D/model-e5-base-v2

    # Tier-2 headline: BrowseComp-Plus corpus + Qwen3-Embedding-8B released vectors
    python3 retriever/server.py --backend bcplus --device cuda:1 --port 8001 \
        --vectors $D/bcplus-index-q3e8b/vectors.f32.npy --docids $D/bcplus-index-q3e8b/docids.json \
        --corpus $D/bcplus-corpus/corpus.jsonl --encoder $D/model-qwen3-embedding-8b

POST /retrieve        {"queries": [str, ...], "topk": 3}
     -> {"results": [[{"id", "title", "text", "score"}, ...] per query]}
GET  /health          -> sizes, device, request counters

Requests arriving within `--batch-window-ms` are coalesced into ONE matmul pass over
the matrix (the pass is memory-bound: ~64.5 GB read per pass for wiki-18), so latency
under concurrent agent rollouts stays near one pass rather than one pass per call.
Every response is also appended (query, ids) to --log for the F4 replay check.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from common import E5Encoder, JsonlCorpus, Qwen3EmbEncoder, exact_topk, load_matrix, split_title_text  # noqa: E402


class Req(BaseModel):
    queries: List[str]
    topk: int = 3


class VecReq(BaseModel):
    vectors: List[List[float]]
    topk: int = 3


def build_app(args) -> FastAPI:
    app = FastAPI()
    t0 = time.time()
    V = load_matrix(args.vectors, args.device)
    corpus = JsonlCorpus(args.corpus)
    if args.backend == "wiki18":
        enc = E5Encoder(args.encoder, args.device)
        docids = None
        if len(corpus) != V.shape[0]:
            raise SystemExit(f"corpus rows {len(corpus)} != vectors {V.shape[0]}")
    else:
        enc = Qwen3EmbEncoder(args.encoder, args.device)
        docids = json.load(open(args.docids))
        # map vector row -> corpus row via docid
        row_of = {}
        for i in range(len(corpus)):
            row_of[str(corpus.get(i)["docid"])] = i
        vec2row = np.array([row_of[str(d)] for d in docids], dtype=np.int64)
    print(f"loaded {V.shape} on {args.device} in {time.time()-t0:.0f}s", flush=True)

    queue: asyncio.Queue = asyncio.Queue()
    stats = {"requests": 0, "queries": 0, "passes": 0, "pass_ms_total": 0.0}
    log_f = open(args.log, "a") if args.log else None

    def lookup(vrow: int, score: float) -> Dict[str, Any]:
        if docids is None:
            d = corpus.get(int(vrow))
            title, text = split_title_text(d)
            return {"id": str(d.get("id", vrow)), "title": title, "text": text, "score": score}
        d = corpus.get(int(vec2row[vrow]))
        text = d.get("text", "")[: args.doc_chars]
        title = d.get("title") or d.get("url", "")
        return {"id": str(d["docid"]), "title": title, "text": text, "score": score}

    async def worker():
        while True:
            items = [await queue.get()]
            deadline = time.time() + args.batch_window_ms / 1000
            while time.time() < deadline and sum(len(i[0].queries) for i in items) < args.max_batch:
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
            qs = [q for it in items for q in it[0].queries]
            k = max(it[0].topk for it in items)
            t = time.time()
            try:
                Q = await asyncio.to_thread(enc.encode, qs)
                scores, ids = await asyncio.to_thread(exact_topk, Q, V, k)
                err = None
            except Exception as e:  # pragma: no cover
                err = e
            stats["passes"] += 1
            stats["pass_ms_total"] += (time.time() - t) * 1000
            pos = 0
            for req, fut in items:
                if err is not None:
                    fut.set_exception(err)
                    continue
                res = []
                for j in range(len(req.queries)):
                    res.append([lookup(ids[pos + j, c], float(scores[pos + j, c])) for c in range(req.topk)])
                pos += len(req.queries)
                fut.set_result(res)

    @app.on_event("startup")
    async def _start():
        asyncio.get_running_loop().create_task(worker())

    @app.post("/retrieve")
    async def retrieve(req: Req):
        req.queries = [q[: args.max_query_chars] for q in req.queries][: args.max_queries]
        if not req.queries:
            return {"results": []}
        fut = asyncio.get_running_loop().create_future()
        await queue.put((req, fut))
        res = await fut
        stats["requests"] += 1
        stats["queries"] += len(req.queries)
        if log_f:
            for q, hits in zip(req.queries, res):
                ids = [h["id"] for h in hits]
                log_f.write(json.dumps({"q": q, "ids": ids,
                                        "h": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]}) + "\n")
            log_f.flush()
        return {"results": res}

    @app.post("/retrieve_vec")
    async def retrieve_vec(req: VecReq):
        """Search with caller-supplied query vectors (isolates the search for gate G3)."""
        Q = torch.tensor(req.vectors, dtype=torch.float32, device=args.device)
        scores, ids = await asyncio.to_thread(exact_topk, Q, V, req.topk)
        return {"ids": ids.tolist(), "scores": scores.tolist()}

    @app.get("/health")
    async def health():
        return {"backend": args.backend, "n_vectors": int(V.shape[0]), "dim": int(V.shape[1]),
                "device": args.device, **stats}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["wiki18", "bcplus"], required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--docids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--batch-window-ms", type=float, default=10.0)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--max-queries", type=int, default=4)
    ap.add_argument("--max-query-chars", type=int, default=512)
    ap.add_argument("--doc-chars", type=int, default=2500)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
