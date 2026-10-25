# Autoresearch: TreeOPSD replication

## Goal

Independently confirm or falsify the headline claim of arXiv:2610.23181 (TreeOPSD) and publish the
result either way. A clean kill is as valuable as a confirmation. The method spec is in PAPER.md,
the plan in DESIGN.md, and the falsification bars, fixed before any run, in PREREG-EXP1.md.

## Metrics

Every run directory gets a `summary.json`. `./autoresearch.sh runs/<id>` turns it into
`METRIC name=number` lines.

| metric | direction | meaning |
|---|---|---|
| `em_headline_bcplus` | higher better (**primary**, PREREG amendment 1) | EM on the frozen 830-question BrowseComp-Plus slice at the fixed final checkpoint |
| `em_qa_avg` (= `em_avg`) | higher better (secondary) | unweighted EM mean over HotpotQA, 2Wiki, MuSiQue and Bamboogle (the paper's MHQA "Avg") |
| `em_qa_avg_no_bamboogle` | diagnostic | the same without the 125-question set |
| `em_<benchmark>` | higher better | per-benchmark EM |
| `rollout_gen_tokens_per_prompt` | lower better | generated (actor) tokens per training prompt, averaged over all steps |
| `tool_calls_per_prompt` | lower better | tool calls per training prompt |
| `train_tokens_per_prompt` | lower better | tokens the trainer runs forward/backward on, per prompt |
| `train_gpu_hours` | lower better | GB300-hours for the whole run, including rollout and eval-in-loop |
| `drift_rel_blocks234` | diagnostic | mean\|Γ−V\| / mean\|V\| on optimizer blocks 2 to 4 (PREREG M1) |
| `single_turn_root_frac` | diagnostic | fraction of roots with T = 1 (the budget anomaly, PAPER.md A11) |
| `em_best_in_window` | diagnostic | best subset-EM over steps 5..80 (paper protocol, F3) |

`summary.json` schema:

```json
{"run_id": "...", "arm": "grpo|treepo|treeopsd|grpo_tok|treepo_clip", "seed": 0,
 "model": "Qwen/Qwen3-4B-Thinking-2507@768f209d...", "image": "sha256:...",
 "em": {"hotpotqa": 0.0, "2wikimultihopqa": 0.0, "musique": 0.0, "bamboogle": 0.0},
 "rollout_gen_tokens_per_prompt": 0, "tool_calls_per_prompt": 0, "train_tokens_per_prompt": 0,
 "train_gpu_hours": 0, "drift_rel_blocks234": 0, "single_turn_root_frac": 0, "em_best_in_window": 0}
```

EM values are percentages (0 to 100), as in the paper's tables.

## Logging (every run, no exceptions)

1. **`autoresearch.jsonl`**: line 1 is a config header written once at the first run (image digest,
   verl commit, data `.verified` hashes, model revisions, hardware). Each later line is one result:
   `{"run_id", "arm", "seed", "status": "ok|crashed|excluded", "metrics": {...}, "notes"}`.
2. **`experiments/worklog.md`**: a dated entry per run covering what was launched, on which device,
   anything odd, and the METRIC lines.
3. After every run, regenerate the dashboard (a table of all jsonl rows against the PREREG bars).

## Files in scope

`treeopsd/`, `retriever/`, `scripts/`, `configs/`, `env/`, `runs/` (git-ignored except metadata),
`experiments/`, and the four top-level docs.

## Off limits

- Production databases, deployed services, any shared branch.
- Pushing or merging without explicit human approval. This repo has **no remote**.
- Evicting another workload's pinned models or GPU claims. Claim a GPU before use and release it
  after.
- Live web search anywhere in a training or eval loop.
- Changing a PREREG bar after a result is seen, except by dated amendment.

## Constraints

- Frozen retrieval only. Pinned models and data (DATA.md); pinned image (env/Dockerfile).
- `treeopsd/core.py` is the reference. Any trainer-side loss must pass gate G4 against it.
- Every arm uses identical data order, prompt, retriever, LR, batch and step count, and differs only in
  rollout shape and/or loss.
- Fixed-final-checkpoint evaluation is primary. Best-in-window is secondary and labelled.

## What has been tried

- **2026-10-21:**
  - Read the paper in full.
  - Recorded the method and 20 ambiguities (PAPER.md).
  - Searched for code: there is no official code; related repos are pinned.
  - Wrote the CPU reference math with 51 passing tests.
  - Built the pinned data manifest.
  - Wrote the Dockerfile.
  - Wrote the prereg.
- Later on 2026-10-21 (Phase 0 part 2):
  - Downloaded and verified the EXP-1 and BC+ data on the left Station.
  - Prepared the data.
  - Wrote the verl extension, retriever, eval scripts and run scripts.
  - 86 CPU tests pass.
  - Wrote RUNBOOK-PHASE0.md.
- No GPU runs yet.
