# TreeOPSD note — arXiv submission checklist (W6, 2026-11-14 PT)

Status: package ready for Stephen. Nothing has been submitted.
Submission is Stephen's, after endorsement (see the W6 endorsement
packet in ronanlabs-hq `publishing/endorsement/`).

## Package contents (this folder)

- `main.tex` — the submission source. Self-contained: article class,
  booktabs/amsmath/hyperref/microtype/graphicx only, inline
  thebibliography, no external figure files. sha256 matches
  `paper/main.tex`
  (122a69b0f5c39d40a2febb4d0b6a4195d7facface5e5b2ddbe334c361ab68a4c).
- `main-compiled-2026-11-14.pdf` — proof compile of the W17 revision,
  7 pages (tectonic, 2026-11-14; baseline pre-edit compile passed at
  5 pages). For eyeballing only; arXiv compiles from source.
- `abstract.txt` — plain-text abstract for the submission form.
- `metadata.txt` — title, author, category, comments field.

## Changes made on branch feat/w6-arxiv-package

1. Author block: "Stephen J. Ronan, MD — RonanLabs — Blackhawk, CA, USA".
2. Date: November 2026.
3. AI-assistance disclosure upgraded to the RonanLabs standing
   statement as its own paragraph (was one sentence inside
   Reproducibility). It says the author verified every number.
   That becomes true only when Stephen does the verification pass
   below. Do not submit before it.

## W12 revision (2026-11-14, PI decisions applied)

Title is now "A Preregistered Replication of TreeOPSD on Qwen3-4B"
(also in `metadata.txt`; `abstract.txt` regenerated). Findings 2-7
applied as the PI chose: bootstrap-vs-seed pass wording (with the M1
mechanism kill stated next to it), boundedness framed against Zou
et al.'s clipping-vs-target distinction and their GSPO row, HotpotQA
training / tier-2 transfer stated, collapse claims narrowed to the
seeds (17.8/0.0), protocol paragraph added (same four-set mean,
greedy vs sampled MHQA, step 100 vs their window, unique-turn
reconstruction, Bamboogle column restored from the seed-package JSONs,
BC+ cited; greedy tier 1 vs our own preregistration and the 20-turn
BC+ cap also stated), F3 "conservative" deleted, 0.80 scoped to QA
(BC+ sd 2.09), and ten bibliography entries added (GSPO first).
Table numbers unchanged (W12b owns the number-check sheet).
A fresh Grok Build CLI adversarial pass on the revised text resolved
Findings 2, 4, 5, 6, 7 and Finding 3 after one scoped sentence fix;
its verified new points (M1 conjunct, C-E bar labels, TreeRL
citation, turn cap, abstract dose point) are fixed in this revision.
Still open, unchanged by W12: Finding 1's hardware-matched pooled
C-B interval (W12b) and the AI-assistance paragraph wording (W6 form kept).
W17 added the F2 reporting limitation and labelled smoke-run indications below.

## Grok adversarial review (2026-11-14) — decisions needed before submitting

Full review: ronanlabs-hq `publishing/reviews/grok-adversarial-treeopsd-2026-11-14.md`.
W6 verified Finding 1 against the seed-package JSONs and applied the
mechanical half only: the "at most 0.49 EM" sentence now gives the
rescore deltas from ANALYSIS-FINAL-20261024.md (-0.08/-0.49 QA,
+0.36/-1.57 BC+), and the Table 2 caption now states the C-B rows are
as-scored and prints the hardware-matched C-B per-seed gaps (QA 4.41,
-0.18, 6.47; BC+ 0.00, 1.81, 7.35). No hardware-matched pooled C-B
interval exists in the packages; computing one is open.

Left for Stephen by W6; decided by Stephen 2026-11-14 and applied by
W12 above (list kept for the record):
1. "Passes both bars" rests on question-bootstrap intervals while the
   paper's own seed intervals include zero (Finding 2).
2. Title and mechanism sentence ("boundedness, not the posterior")
   vs Zou et al.'s own clipping-vs-target distinction and their GSPO
   row (Finding 3). Grok recommends retitling to the replication
   subtitle. One correction to Grok: per PREREG Amendment 1, this
   study trained on HotpotQA and tier 2 measures transfer to BC+;
   the paper text does not say that plainly, and should.
3. "Two of two collapses / either bound prevents it" vs the second
   seed's 17.8/0.0 (Finding 4).
4. Protocol paragraph: tier 1 is the same four-set mean as theirs
   (delete "composed differently"), greedy vs their sampled MHQA
   eval, step-100 vs their checkpoint window, unique-turn
   reconstruction, Bamboogle column, BC+ citation (Finding 5).
5. F3 "conservative" claim (Finding 6); "low-variance 0.80" is QA-only,
   BC+ sd is 2.09 (Finding 7).
6. Eight missing citations listed in the review (GSPO first).
Recommendation: resolve 1-4 in the text before arXiv; they are the
sentences a moderator or first reviewer will check.

## Stephen's pre-submission pass (in order)

- [ ] Verify the numbers: Table 1 and Table 2 against
  `results/seed_package_CB.json` and `results/seed_package_CE_gb300.json`
  via `ANALYSIS-FINAL-20261024.md`. W6 spot-checked A/B/C seed-0 QA
  averages (39.80 / 39.17 / 43.58) against the JSON on 2026-11-14;
  the full check is the PI's.
- [ ] Read the compiled PDF once end to end (argument, limitations,
  disclosure wording).
- [ ] Confirm the repository URL in the text
  (https://github.com/sjronan/treeopsd-replication) is the public home
  you want cited. Stephen approved repository publication on 2026-11-14 PT;
  this approval does not submit the package to arXiv.
- [ ] Get endorsed for cs.LG (endorsement packet; sequential asks,
  about 10 days apart; endorsement is not peer review and moderation
  is a second gate after it).
- [ ] Submit at arxiv.org: upload `main.tex` only, primary category
  cs.LG, paste `abstract.txt`, comments from `metadata.txt`, pick a
  license (CC BY 4.0 recommended there).
- [ ] Check arXiv's own compiled proof before finalizing (their
  TeX Live differs from tectonic).
- [ ] Same day as the arXiv ID: HF Daily Papers submission, pin the
  repo README with the ID, decide the reserved-Zenodo question per
  the P1 playbook (reserve only, one DOI story).

## Notes and open risks

- TMLR later means a separate anonymized manuscript; nothing in this
  package is anonymized. TMLR solo quota is 2 single-author
  submissions per calendar year (TreeOPSD would be slot 1).
- The bibliography now has eleven entries (W12 added GSPO, GRPO, PPO,
  Qu et al. 2026a, BrowseComp-Plus, Search-R1, OpenResearcher, the
  Li et al. TreePO name collision, TreeRL and Tree-GRPO); a venue
  version will still want more related work. Grok adversarial reviews
  of this package are filed in ronanlabs-hq `publishing/reviews/`.

## Controller correction (2026-11-14 PT)

Number check (number-check/) found two interval endpoints transcribed from
ANALYSIS-FINAL-20261024.md that disagree with the committed receipt
results/seed_package_CB.json: tier 1 upper +4.74 -> +4.76 and tier 2 lower
+2.13 -> +2.17 (3 occurrences each). The paper, abstract.txt, README and the
X drafts now use the receipt values; pass/fail is unchanged. ANALYSIS-FINAL,
the worklog and correspondence are historical records and keep the old
values. Also fixed an "E{=8"/"E{=16" conversion artifact in abstract.txt.
Remaining number-check rows (14) are values from Zou et al.'s tables and
config settings that need a cited source, not corrections.

## W17 F2 reporting (2026-11-14 PT)

Generated tokens pass and trained tokens fail against the 1.15x line
(1.03x and 1.48x). Full-100-step tool-call and GPU-time measures were not
recorded, so that part of F2 is not evaluable and F2 has no established pass.
The added smoke-run indication cites the committed S1b report: TreeOPSD
versus GRPO G=8, 4.9/8.1 tool calls per training prompt (0.60x), approximate
345/250 seconds per step (1.38x), 10 steps per arm at 64 prompts per step,
no intervals. Those values are not full-window F2 results. The number check
now maps them to that report and the F2 threshold/window to PREREG-EXP1.md.
The abstract is unchanged by W17.
