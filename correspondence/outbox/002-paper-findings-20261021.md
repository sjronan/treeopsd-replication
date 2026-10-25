# 002 - Full-paper read: findings that change EXP-1 (controller, 2026-10-21 ~09:45 PT)

Repo: . (branch autoresearch/treeopsd-20261021, local only). Read PAPER.md (20 ambiguities, section 11 pins related repos), DESIGN.md, PREREG-EXP1.md. Unit tests of the method math: 51/51 pass (env/smoke/run.sh).

No official code exists (arXiv, HF paper page, GitHub, author repos, HF hub checked). Framework: verl v0.9.1 (1876b06d) on the vLLM v0.32.0 arm64 image, Search-R1 lineage, TreeOPSD and TreePO as two registered losses over the same credit tensor.

Findings that change the brief:
1. At 4B on multi-hop QA the paper's own gain is small: +0.76 over GRPO, +0.73 over TreePO. Large gains (+2.9..+4.0) are on BrowseComp-Plus and the live-web sets. The brief's "+2 EM at <=80% tokens" is ~2.6x the paper's claim, and F1 "within ~1 point" would kill the QA result by construction. Prereg uses the paper's numbers as the bar; the brief's bar is kept as a "STRONG" outcome.
2. Two brief facts are off: the 30B table does have a per-benchmark Deep Research breakdown (Table 1); the QA training reward is thresholded F1 (>=0.3), not EM.
3. Backup weights from full-span likelihoods are likely near one-hot with long thinking spans; ablations built in.
4. TreePO control update rule is ambiguous ("fixed coefficient"): literal (no ratio/clip) primary, PPO-clipped secondary.
5. TreeOPSD and TreePO give identical gradients on the first optimizer block; their difference is bounded by policy drift on blocks 2-4 (Eq. 40). Cheap mechanism test M1: if relative drift < 2%, the loss cannot be doing the work; a drift-dose follow-up is pre-declared.
6. Budget match holds only in sampled tokens: the trainer processes 12 full rows per prompt vs GRPO's 8 (~1.27x training compute); single-turn answers turn a branch into a full rollout (up to 1.5x). Prediction: F2 trips on GPU-hours.
7. Unspecified: checkpoint-selection rule, behavior log-prob source, GRPO clip/aggregation, Deep Research retriever, answer normalisation.
8. The Deep Research average is half live-web; only BrowseComp-Plus is frozen and testable.
9. The brief's EXP-2 configs mix nominal budgets (6, 8, 12, 12, 4). Proposed iso-budget lines instead: budget 8 = (8,0), (4,2), (2,6); budget 12 = (12,0), (8,1), (4,4); plus the TreePO same-tree control.

Models: Qwen/Qwen3-4B-Thinking-2507 (main), Qwen/Qwen3-30B-A3B-Thinking-2507 (scale-up), Qwen/Qwen3-1.7B (shakedown). Retrieval: Search-R1 frozen wiki-18 + E5-base-v2 flat index (5.1 GB corpus + 64.6 GB index), FlashRAG question sets; exact matmul on an RTX PRO 6000, cross-checked against CPU faiss.

Compute (assumed until gate G6 measures): EXP-1 3 arms x 1 seed ~35 GB300-hours (~18 h wall on both Stations); 3 seeds ~106; EXP-1b on BrowseComp-Plus ~37; EXP-2 ~64.

Controller decisions (proceeding unless you object in the inbox): paper-number bars with the brief's as STRONG; iso-budget EXP-2 lines + same-tree control; literal TreePO primary; thresholded-F1 reward, EM and F1 both reported; 1 seed of all EXP-1 arms first, then 2 more seeds for the arms that matter; add EXP-1b (BrowseComp-Plus, frozen) since that is where the paper's large effect lives.

Questions for you: (a) agree with adding EXP-1b ahead of EXP-2? (b) any objection to treating M1 (drift < 2%) as an early kill signal for the loss claim?
