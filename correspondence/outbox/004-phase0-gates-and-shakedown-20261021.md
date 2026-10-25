# 004 - Phase 0 GPU gates passed; 1.7B shakedown done; M1 preview near the line

Date: 2026-10-21, 22:40 PT. From: Station controller. Re: your 003/004 rulings (all applied; the Founder's no-pause protocol reports at steps 10/25/50/75/100, so the step-25 report is the screen you asked for at step 30).

## Gates (right and left DGX Station, GB300 sm_103)

| Gate | Result |
|---|---|
| Image | verl 0.9.1 @1876b06 + vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0; 86 CPU tests pass; base pins verified unchanged |
| G1a | GB300 (10, 3); verl CPU tests 28 passed |
| G1b / G1c | 2-step GRPO and TreeOPSD on Qwen3-1.7B: pass; telemetry complete |
| G3 | exact search vs faiss-cpu, 1,000 HotpotQA queries: 951 identical, 49 differ only among equal-score passages (wiki-18 duplicates); p95 0.078 s |
| G7 (GPU) | first optimizer block: TreeOPSD signal exactly 0, drift_rel max 2.5e-6; blocks 2-4 > 0 |
| G6 | untrained Qwen3-4B-Thinking-2507, fixed 500 HotpotQA dev: EM 30.8 (paper ReAct 31.61), F1 0.389, answer rate 89.6%, 3.9 turns |

## Deviations (logged in experiments/worklog.md)

- **D12:** transformers 5.19.0 (vLLM 0.32 requirement) is outside verl 1876b06's `<5.11` pin. Founder approved; G1, G7 and S1a are the evidence it trains correctly.
- **D13 (opened and reverted):** briefly SDPA without remove-padding because the image lacked flash-attn. Now flash-attn 2.8.3 + remove-padding, the paper's verl configuration; actor passes about 7x faster.
- **Weight-sync bucket = 20 GB.** vLLM 0.32 requires tied lm_head/embed_tokens in one load call. This changes no math.

## S1a shakedown: Qwen3-1.7B TreeOPSD, B=64, 30 steps (flash-attn image)

- 30/30 steps, no errors or NaNs; checkpoints at 10/20/30; mean step 112 s.
- Training reward: steps 1-5 mean 0.41, steps 26-30 mean 0.45 (noisy, B=64).
- Validation: MuSiQue 0.114 -> 0.151, Bamboogle 0.335 -> 0.374 (start -> step 30).
- Tree health:
  - n_nodes always >= 17.6 (floor 12);
  - |V| max/bound <= 0.38;
  - empty leaves 0;
  - trajectory-equivalents mean 7.96 (design target ~8).
- Behaviour: tool calls per prompt rose from 6.3 to 10.4 over 30 steps.

## M1 preview (1.7B, so it informs only)

Relative drift of the TreeOPSD target vs the TreePO credit on blocks 2-4 (1,368 micro-batch calls):
- median 1.73%, IQR 1.19% - 2.78%, p90 4.39%;
- 60% of calls below the 2% line;
- no trend (first half 1.66%, second half 1.78%);
- sign reversals: median 0.

At 1.7B the posterior-distillation term moves the target only slightly, sitting right on your M1 line. The 4B arms decide it. Question for you: is "median over blocks 2-4 across the screen" the statistic you want M1 judged on, or a per-step fraction below 2%?

## F2 (cost) from the 2-step 1.7B checks

A tree prompt generated about 6.8K tokens but trained on about 23.5K tokens (shared prefixes are re-scored per branch in training). Budget-matched in sampled tokens is therefore not matched in training compute. The 4B smoke runs (running now) give the per-arm numbers and fix G_TOK for arm D.

## Next

- S1b 4B smoke, TreeOPSD and GRPO, 10 steps each (both Stations, now) -> per-step cost, G_TOK, the EXP-1 compute check.
- Then EXP-1 seed 0, all arms to step 100 with no-pause reports at 10/25/50/75.
