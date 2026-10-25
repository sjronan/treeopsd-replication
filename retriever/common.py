"""Shared pieces of the frozen retriever: corpus access, encoders, exact top-k search.

Exact search = one fp32 matmul over the full matrix of passage vectors held on one GPU,
then top-k with a deterministic tie-break (higher score first, then lower doc index).
That is the same computation as FAISS IndexFlatIP; gate G3 checks ids against
faiss-cpu on 1,000 fixed queries.
"""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import torch


# ----------------------------------------------------------------------------- corpus
class JsonlCorpus:
    """Random access to a JSONL corpus by row number through a byte-offset table
    (built once, cached next to the corpus as <file>.offsets.npy)."""

    def __init__(self, path: str):
        self.path = path
        off_path = path + ".offsets.npy"
        if os.path.exists(off_path):
            self.offsets = np.load(off_path)
        else:
            offs = []
            pos = 0
            with open(path, "rb") as f:
                for line in f:
                    offs.append(pos)
                    pos += len(line)
            self.offsets = np.asarray(offs, dtype=np.int64)
            np.save(off_path, self.offsets)
        self._f = open(path, "rb")

    def __len__(self):
        return int(self.offsets.shape[0])

    def get(self, i: int) -> dict:
        self._f.seek(int(self.offsets[i]))
        return json.loads(self._f.readline())


def split_title_text(doc: dict) -> Tuple[str, str]:
    """Search-R1 wiki-18 rows: {"id", "contents": '"Title"\\ntext'}."""
    c = doc.get("contents")
    if c is None:
        return doc.get("title", ""), doc.get("text", "")
    title, _, text = c.partition("\n")
    return title.strip().strip('"'), text.strip()


# ----------------------------------------------------------------------------- encoders
class E5Encoder:
    """intfloat/e5-base-v2 query encoder: 'query: ' prefix, mean pooling, L2-normalised
    (matches how Search-R1/FlashRAG built e5_Flat.index)."""

    def __init__(self, model_dir: str, device: str, max_len: int = 256):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, queries: Sequence[str]) -> torch.Tensor:
        b = self.tok([f"query: {q}" for q in queries], padding=True, truncation=True,
                     max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
        return torch.nn.functional.normalize(e, dim=-1).float()


class Qwen3EmbEncoder:
    """Qwen3-Embedding query encoder (instruction prefix, last-token pooling,
    L2-normalised), the encoder of the BrowseComp-Plus released indexes."""

    TASK = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, model_dir: str, device: str, max_len: int = 512):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, queries: Sequence[str]) -> torch.Tensor:
        texts = [f"Instruct: {self.TASK}\nQuery:{q}" for q in queries]
        b = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                     return_tensors="pt").to(self.device)
        h = self.model(**b).last_hidden_state
        e = h[:, -1]  # left padding -> last position is the last real token
        return torch.nn.functional.normalize(e.float(), dim=-1)


# ----------------------------------------------------------------------------- search
def exact_topk(Q: torch.Tensor, V: torch.Tensor, k: int, chunk: int = 4_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k of queries Q [q, d] against V [n, d] (both on one device).
    Deterministic: ties broken by lower row index. Returns (scores [q,k], ids [q,k])."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    best_s, best_i = None, None
    kk = k + 8  # slack so exact ties at the boundary survive the per-chunk cut
    for start in range(0, V.shape[0], chunk):
        S = Q @ V[start:start + chunk].T
        s, i = torch.topk(S, min(kk, S.shape[1]), dim=1)
        i = i + start
        if best_s is None:
            best_s, best_i = s, i
        else:
            cs, ci = torch.cat([best_s, s], 1), torch.cat([best_i, i], 1)
            best_s, idx = torch.topk(cs, min(kk, cs.shape[1]), dim=1)
            best_i = torch.gather(ci, 1, idx)
    s, i = best_s.double().cpu().numpy(), best_i.cpu().numpy()
    out_s = np.empty((Q.shape[0], k))
    out_i = np.empty((Q.shape[0], k), dtype=np.int64)
    for r in range(Q.shape[0]):
        order = np.lexsort((i[r], -s[r]))[:k]
        out_s[r], out_i[r] = s[r][order], i[r][order]
    return out_s, out_i


def load_matrix(path: str, device: str, dtype=torch.float32) -> torch.Tensor:
    """Load a [n, d] float32 .npy (memory-mapped) onto `device` in slabs."""
    arr = np.load(path, mmap_mode="r")
    n, d = arr.shape
    V = torch.empty((n, d), dtype=dtype, device=device)
    step = 1_000_000
    for s in range(0, n, step):
        V[s:s + step] = torch.from_numpy(np.ascontiguousarray(arr[s:s + step])).to(device=device, dtype=dtype)
    return V
