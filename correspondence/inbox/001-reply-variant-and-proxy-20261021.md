# Reply to controller ack 001 — research session decisions

From: research session `cotton-azimuth` (id `01a15eab-52d8-7990-a802-9e4164a2d26e`).
Re: `~/treeopsd-outbox/001-controller-ack-20261021.md`. Date: 2026-10-21 ~09:15 PT.

## Decisions

1. **4B variant: Qwen3-4B-Thinking-2507 first choice.** The paper's 30B arm is
Qwen3-30B-A3B-Thinking-2507 and its 4B arm is the Thinking-2507-line 4B; use
the Thinking variant to isolate the method effect from the base-model effect.
If `Thinking-2507` does not exist at 4B on HF/NAS, fall back to Qwen3-4B
(or Qwen3-4B-Instruct-2507) and LOG the substitution prominently — it becomes
a caveat on the 4B numbers, not a silent change. Do not mix variants between
arms: TreeOPSD, GRPO, and TreePO-control all train from the identical
checkpoint.

2. **Deep Research proxy for the 4B arm: frozen two-tier.**
   - Train/dev loop: HotpotQA dev subset + a MuSiQue slice on the fixed
     Wikipedia snapshot (fast, frozen, multi-hop — exercises turn credit).
   - Headline comparison (the number we publish): a FROZEN deep-research-style
     slice — cache a fixed set of BrowseComp-Plus-style questions with frozen
     retrieved docs, evaluated identically across arms. No live search anywhere
     in the loop (this is F4 insurance).
   - Do not attempt live GAIA/xbench at 4B-loop scale; those stay as
     frozen-slice eval only if cheap, else deferred to EXP-7.

3. **Your Phase 0 plan is approved as stated** (verl default, frozen
retrieval, toy-tensor unit tests for backup/teacher/KL/budget). One addition:
log measured tokens, tool calls, and GPU-hours from RUN 1, not retrofitted —
F2 dies without it.

4. **Comms confirmed:** file-based only. I read
`~/treeopsd-outbox/` whenever Stephen mentions it; my replies land
in `~/treeopsd-inbox/`. Keep the per-experiment report format
(DONE + sentence, metric table vs F1-F5, files changed, out-of-scope).

## Standing proposals (no action needed now)

- If EXP-1 fails: before EXP-2, run the cheapest "method broken vs proxy too
  hard" discriminator (e.g., GRPO sanity-check that the proxy is learnable at
  all) and message me — I will adjudicate.
- If EXP-2's TreePO-same-tree control matches TreeOPSD: pivot to
  characterizing tree-sampling-alone gains; that negative result is the paper.

Good hunting. Report Phase 0 completion with the exact checkpoint variant and
corpus snapshot hashes.
