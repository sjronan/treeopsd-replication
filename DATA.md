# Data, frozen retrieval corpora, indexes and models

Nothing here has been downloaded yet. Every file is pinned in `scripts/manifest.tsv`:
- Each entry has a Hugging Face repo, a **revision hash**, a path, a byte size and the **LFS sha256**.
- The manifest was generated from the HF API on 2026-10-21.
- `scripts/fetch.py` downloads with resume and verifies size and sha256.
- The script is a dry run unless you pass `--yes`.

```bash
export TREEOPSD_DATA_ROOT=/path/to/fast/disk/treeopsd     # target root (variable)
python3 scripts/fetch.py --list                           # groups, sizes, pinned revisions
STAGE=exp1 ./scripts/fetch_data.sh                        # dry run: URLs + total
STAGE=exp1 YES=1 ./scripts/fetch_data.sh                  # real download + verification
```

Layout: `$TREEOPSD_DATA_ROOT/<group>/<path-in-repo>`. Each completed group writes a `.verified`
file with the sha256 of every file. Only small git files (configs, READMEs) have no published
sha256. For those we record our own hash on download.

## Stages

| STAGE | groups | size | used by |
|---|---|---|---|
| `shakedown` | mhqa, model-qwen3-1.7b, model-e5-base-v2, wiki18, wiki18-index | 74.9 GB | S1 |
| `exp1` | shakedown + model-qwen3-4b-thinking | 82.9 GB | EXP-1, EXP-2 |
| `bcplus` | bcplus queries, bcplus corpus, BC+ Qwen3-Embedding-8B and BM25 indexes, Qwen3-Embedding-8B, OpenResearcher questions (seed_42) | 24.0 GB | EXP-1b eval |
| `or-train` | OpenResearcher corpus (parquet) | 29.8 GB | EXP-1b training environment |
| `or-dense` | OpenResearcher Qwen3-Embedding-8B index (a single pickle) | 122.0 GB | optional, EXP-1b |
| `scale30b` | Qwen3-30B-A3B-Thinking-2507 | 61.1 GB | EXP-7 only |

Peak disk for `exp1` is about 148 GB. The two index parts (64.6 GB) and the concatenated
`e5_Flat.index` (64.6 GB) coexist until the parts are deleted. Add another 13 GB for the
gunzipped corpus.

## 1. Multi-Hop QA: retrieval corpus and index (frozen, the paper's exact setting)

The paper uses "a local E5 retrieval server built over a Wikipedia dump (Karpukhin et al., 2020)",
and App. F.1 calls it the "fixed Wikipedia-2018 index". This is Search-R1's wiki-18 setup:
- **Corpus:** the DPR 2018-12-20 Wikipedia dump, split into 100-word passages (21,015,324 passages
  according to DPR).
- **Encoder:** `intfloat/e5-base-v2` (768-dimensional). Queries get the `query: ` prefix and
  passages the `passage: ` prefix.
- **Index:** a FAISS `IndexFlatIP` (exact inner product).

| group | repo @ revision | file | bytes | sha256 |
|---|---|---|---|---|
| wiki18 | `PeterJinGo/wiki-18-corpus` @ `69c1c00ffe7c5554c68d8548355cb22e46aabc51` | `wiki-18.jsonl.gz` | 5,123,307,260 | `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db` |
| wiki18-index | `PeterJinGo/wiki-18-e5-index` @ `a4d31160a035f30764604f4827cd8f1d0315eb86` | `part_aa` | 42,949,672,960 | `a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023` |
| wiki18-index | same | `part_ab` | 21,609,402,413 | `b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11` |
| model-e5-base-v2 | `intfloat/e5-base-v2` @ `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `model.safetensors` + tokenizer/config | 0.44 GB | in manifest |

**Gotcha (found 2026-10-21):** `wiki-18.jsonl.gz` is a gzipped **tar** archive with one member, `data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl` (14,393,573,105 bytes, 21,015,324 lines). `gzip -d` (as the Search-R1 README suggests) yields a tar stream with a 512-byte header glued to the first record. Extract it with `tar -xzOf`.

`cat part_aa part_ab > e5_Flat.index` produces a file of 64,559,075,373 bytes. This matches the
Search-R1 README. No hash is published for the concatenated file, so `fetch_data.sh` records ours.

FlashRAG mirrors the same corpus unzipped, in a different packaging:
`RUC-NLPIR/FlashRAG_datasets: retrieval-corpus/wiki18_100w.zip` (5,130,719,280 B, sha256
`e5b987a0...56db`). It is not used, but it is a usable fallback.

**BM25 fallback (not the paper's retriever).** A Pyserini/Lucene BM25 index over the same
`wiki-18.jsonl` is about 10 to 12 GB and takes a few CPU-hours to build on the Grace CPU. It
needs no GPU and no 64 GB index. We use it only for the shakedown if the dense server is not
ready. **Any EXP-1 result must use E5-flat**, because switching retrievers changes absolute EM
by several points (Search-R1 reports both).

**Serving.** We do not use faiss-gpu. FAISS GPU wheels for sm_120 and aarch64 are not reliably
available. Instead:
- Reconstruct the 21.0 M × 768 float32 vectors once with `faiss-cpu` (`index.reconstruct_n`).
  That is 64.6 GB.
- Hold them on the RTX PRO 6000 (96 GB) as one fp32 tensor and score with a single
  `torch.mm` + `topk`.
- This is exact inner product, the same computation as `IndexFlatIP`. A fixed batch of 1,000
  queries is checked against `faiss-cpu` for identical top-3 ids before EXP-1. That check is
  Phase-0 gate G3.
- The same server runs on a DGX Spark (128 GB unified) as an eval-time replica.

## 2. Multi-Hop QA: questions

`RUC-NLPIR/FlashRAG_datasets` @ `bcafb8dd07d453be3cbeeeb3f78be1841bddf92c` (group `mhqa`, 0.68 GB).
Each line is `{"id", "question", "golden_answers": [...]}`; the format was verified on the
Bamboogle file.

| file | expected rows (paper) | bytes | sha256 |
|---|---|---|---|
| `hotpotqa/train.jsonl` | 90,447 (distractor train) | 569,520,788 | `a81274abafa899ec0ee073102edbe6bb694a8a1174201b4e43e2bc6c98964d1a` |
| `hotpotqa/dev.jsonl` | 7,405 | 48,191,374 | `434ec155867019396312f3d466ce3406c71fe8a2917f49f63bb646ec5ad2ff52` |
| `2wikimultihopqa/dev.jsonl` | 12,576 | 54,682,944 | `b4d74d73d1cb3c0632c665a85e1e5060e3411968b807721618cfd2464fc905e3` |
| `musique/dev.jsonl` | 2,417 | 5,620,439 | `d1cc08bcc91b060ba67c1e12881789e1315fb9f85d46f03a3413dd5e0f2ed02c` |
| `bamboogle/test.jsonl` | 125 (verified: 125 lines) | 17,031 | `c9703dae6bb1ceb9e2df77be45da28cb12aa040d2f471507890a461296968f3f` (ours; git file) |

The row counts must match on download. If they do not, stop and log the mismatch. Search-R1's own
`PeterJinGo/nq_hotpotqa_train` mixes NQ with HotpotQA, so we do not use it for training. The paper
trains on HotpotQA only.

**Horizon strata, fixed a priori for F5:**
- MuSiQue ids carry hop counts (`2hop__`, `3hop1__`, `3hop2__`, `4hop1__`, ...).
- 2Wiki has a `type` field (comparison, inference, compositional, bridge_comparison).
- HotpotQA has bridge vs comparison.

The strata are extracted by `scripts/prepare_mhqa.py`, which is still to be written (Phase 0),
and frozen into `data/mhqa/strata.json` before any run.

**EM.**
- Take the content of the last `<answer>...</answer>` span.
- Normalise it: lowercase, strip punctuation, drop a/an/the, collapse whitespace.
- EM is 1 if it equals the normalised form of any gold alias. A missing answer tag scores 0.
- The **training reward** is the token F1 against the best alias if that F1 is ≥ 0.3, else 0.

The implementation is `treeopsd/metrics.py`, with tests. The paper does not state its normaliser
(PAPER.md A9).

## 3. Deep Research track (EXP-1b, gated)

| group | repo @ revision | size | notes |
|---|---|---|---|
| bcplus | `Tevatron/browsecomp-plus` @ `144cff8e35b5eaef7e526346aa60774a9deb941f` | 2.78 GB | 830 test queries with evidence/gold docs. Queries are distributed obfuscated against contamination; decrypt with the BrowseComp-Plus repo script (`texttron/BrowseComp-Plus` @ `046949032b03`). Verify the count is 830. |
| bcplus-corpus | `Tevatron/browsecomp-plus-corpus` @ `b27b02bc3e45511b8b82a13e6f90ce761df726f6` | 1.76 GB | about 100K web documents (fixed) |
| bcplus-index-q3e8b | `Tevatron/browsecomp-plus-indexes` @ `b3f37f70c33829eb09d04784a54277a31871fd63` | 1.64 GB | Qwen3-Embedding-8B doc embeddings (4 pickles) |
| bcplus-index-bm25 | same | 2.17 GB | Lucene BM25 index (Pyserini, needs Java 21) |
| model-qwen3-embedding-8b | `Qwen/Qwen3-Embedding-8B` @ `1d8ad4ca9b3d` | 15.15 GB | query encoder for the dense indexes |
| or-questions | `OpenResearcher/OpenResearcher-Dataset` @ `1edca18c5d7c9c105762da4f3c5c2b83855b3c6e`, `seed_42/` only | 0.46 GB | 16 seed dirs (seed_42..57) hold 16 trajectories per question; one seed gives the question list. We extract unique questions + answers and drop teacher messages. Expect 6,102. |
| or-corpus | `OpenResearcher/OpenResearcher-Corpus` @ `8367703babe6e080899a48782f365c284c32b668` | 29.75 GB | about 11B-token closed web corpus |
| or-index-q3e8b | `OpenResearcher/OpenResearcher-Indexes` @ `a4240938a009d3dc6a6f733439928cd7a112609c` | 122.03 GB | one `corpus.pkl` (embeddings + docids) |

**Pickle hazard.** The Tevatron and OpenResearcher indexes are Python pickles, and loading a pickle
executes code. We load them only:
- inside the replication container, with network disabled;
- after sha256 verification against the manifest;
- once, in a converter that writes plain `.npy` and `.json`.

The pickles are never loaded on a shared host.

**Retriever for EXP-1b (PAPER.md A8).**
- *Evaluation (BC+):* the Qwen3-Embedding-8B dense index, the setting OpenResearcher reports, with
  BM25 as a secondary. Both are fixed corpora.
- *Training (OpenResearcher corpus):* BM25, built locally, primary. The 122 GB dense index needs
  the 15 GB encoder online during rollouts plus about 122 GB of host RAM (the Grace LPDDR5X can hold
  it). It is optional, and EXP-1b reports which retriever was used.

The **live-web** benchmarks (BrowseComp, GAIA, xbench-DeepSearch) are **not** part of this
replication's primary metrics: there is no frozen snapshot of them (F4, PAPER.md A7).

## 4. Models

| group | HF id @ revision | size | role |
|---|---|---|---|
| model-qwen3-1.7b | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 4.08 GB | S1 shakedown (paper App. D.1), `enable_thinking=True` (A10) |
| model-qwen3-4b-thinking | `Qwen/Qwen3-4B-Thinking-2507` @ `768f209d9ea81521153ed38c47d515654e938aea` | 8.06 GB | **main** (paper Table 1/4) |
| model-qwen3-30b-a3b-thinking | `Qwen/Qwen3-30B-A3B-Thinking-2507` @ `144afc2f379b542fdd4e85a1fcd5e1f79112d95d` | 61.08 GB | EXP-7 only |
| model-e5-base-v2 | `intfloat/e5-base-v2` @ `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | 0.44 GB | retriever query encoder |

The variant the paper uses is **Thinking-2507**, not the hybrid Qwen3-4B and not
Qwen3-4B-Instruct-2507 (`cdbee75f...`). The study brief's "Qwen3-4B" means
`Qwen/Qwen3-4B-Thinking-2507`.

## 5. What gets recorded per staging run

All of the following go into `experiments/worklog.md` and the `autoresearch.jsonl` config header:
- the `.verified` hash lists;
- the container image digest;
- row counts;
- the retriever self-check: top-3 ids for 1,000 fixed queries, torch vs faiss-cpu;
- the vLLM prefix-cache setting.
