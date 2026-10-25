"""One-time conversions (CPU) into plain arrays the server memory-maps.

  wiki18:  e5_Flat.index (FAISS IndexFlatIP)       -> e5_vectors.f32.npy   [21M, 768]
  bcplus:  Tevatron qwen3-embedding-8b *.pkl shards -> vectors.f32.npy + docids.json
           browsecomp-plus-corpus parquet           -> corpus.jsonl ({"docid","title","url","text"})

The BC+ indexes are Python pickles: loading a pickle executes code, so run the bcplus
conversion only inside the replication container with networking disabled, after the
fetch script verified each shard's sha256 against scripts/manifest.tsv:
  docker run --rm --network none -v $D:$D treeopsd-repl python3 retriever/build_vectors.py bcplus ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np


def wiki18(index_path: str, out_path: str, chunk: int = 1_000_000):
    import faiss

    idx = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    n, d = idx.ntotal, idx.d
    print(f"index: {type(idx).__name__} ntotal={n} d={d} metric={idx.metric_type}", flush=True)
    assert idx.metric_type == faiss.METRIC_INNER_PRODUCT
    out = np.lib.format.open_memmap(out_path + ".tmp", mode="w+", dtype=np.float32, shape=(n, d))
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        out[s:e] = idx.reconstruct_n(s, e - s)
        print(f"  {e}/{n}", flush=True)
    out.flush()
    del out
    os.replace(out_path + ".tmp", out_path)
    print("wrote", out_path)


def bcplus_index(shard_dir: str, out_dir: str):
    shards = sorted(glob.glob(os.path.join(shard_dir, "corpus.shard*_of_*.pkl")))
    assert shards, f"no shards in {shard_dir}"
    embs, ids = [], []
    for p in shards:
        with open(p, "rb") as f:
            e, look = pickle.load(f)  # trusted only after sha256 verification (see docstring)
        embs.append(np.asarray(e, dtype=np.float32))
        ids.extend([str(x) for x in look])
        print(p, embs[-1].shape, flush=True)
    E = np.concatenate(embs)
    norms = np.linalg.norm(E, axis=1)
    print(f"vectors {E.shape}, norm min/mean/max {norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "vectors.f32.npy"), E)
    json.dump(ids, open(os.path.join(out_dir, "docids.json"), "w"))


def bcplus_corpus(parquet_dir: str, out_path: str):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(parquet_dir, "train-*.parquet")))
    n = 0
    with open(out_path + ".tmp", "w") as f:
        for p in files:
            t = pq.read_table(p).to_pylist()
            for r in t:
                text = r.get("text", "")
                title = text.split("\n", 1)[0][:200] if text else ""
                f.write(json.dumps({"docid": str(r["docid"]), "url": r.get("url", ""), "title": title,
                                    "text": text}) + "\n")
                n += 1
    os.replace(out_path + ".tmp", out_path)
    print(f"wrote {n} docs to {out_path} (expect 100195)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["wiki18", "bcplus-index", "bcplus-corpus"])
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    {"wiki18": wiki18, "bcplus-index": bcplus_index, "bcplus-corpus": bcplus_corpus}[a.what](a.src, a.dst)
