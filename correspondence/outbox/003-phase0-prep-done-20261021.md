# 003 - Phase 0 prep done (CPU only); GPU gates start when the Stations free (~12:00 PT)

Decisions from your 001 are folded in (PREREG amendment 1: headline = frozen BrowseComp-Plus slice, QA average secondary; Thinking-2507 for all arms from one checkpoint; tokens/tool calls/GPU-hours logged from run 1).

Data (left Station /models/treeopsd, sha256 verified vs scripts/manifest.tsv):
- Qwen3-4B-Thinking-2507, Qwen3-1.7B, e5-base-v2, Qwen3-Embedding-8B at pinned revisions (see manifest).
- wiki-18 corpus 21,015,324 passages, sha256 43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d (note: the published .jsonl.gz is a tar archive; plain gunzip corrupts record 1).
- e5 flat index sha256 69c98463fdb41fc08737d88513c597725f311c44f7ba6dca4b05d8c7c658d166.
- QA: train 90,447; eval 7,405 / 12,576 / 2,417 / 125; dev 1,625; MuSiQue hops 2/3/4 = 1,252 / 760 / 405.
- BrowseComp-Plus: 830 queries (eval_bcplus.parquet sha256 ecb2bd6a40fa090ad941ec4c15f03113f74314c1d1c60834e25730fa7a853f2c), corpus 100,195 docs.

Code: verl 0.9.1 extension (tree rollout, credit hook with telemetry, TreeOPSD / literal TreePO / clipped TreePO losses on one credit tensor), 2-hunk verl patch, exact-matmul retriever with faiss cross-check and query log for F4 replay, data prep, two-tier eval. 86 CPU tests pass. Runbook: RUNBOOK-PHASE0.md.

Deviation to note (D9): the BrowseComp-Plus slice uses the training search tool, not the paper's browser tools, and models are trained on multi-hop QA, so BC+ numbers measure transfer and are not level-comparable to the paper's 13.98. Arms remain comparable to each other.

Plan agreed with the Founder: screening pass first (all EXP-1 arms to step 30 of a 100-step schedule, continuation not restart), which also reads M1 (drift) and the F2 cost numbers; full runs and seeds 2-3 only if the screen is alive. The 1.7B plumbing shakedown runs first.

Open: your view on M1 (drift < 2%) as an early kill signal for the loss claim (my 002, question b).
