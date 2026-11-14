# TreeOPSD note — arXiv submission checklist (W6, 2026-11-14 PT)

Status: package ready for Stephen. Nothing has been submitted.
Submission is Stephen's, after endorsement (see the W6 endorsement
packet in ronanlabs-hq `publishing/endorsement/`).

## Package contents (this folder)

- `main.tex` — the submission source. Self-contained: article class,
  booktabs/amsmath/hyperref/microtype only, inline thebibliography,
  no external figure files. sha256 matches `paper/main.tex`
  (229a04b02f9a37834429704567a7bda420640fc7736a7b0843947f40809c8fb1).
- `main-compiled-2026-11-14.pdf` — proof compile, 5 pages (tectonic,
  2026-11-14; baseline pre-edit compile also passed at 5 pages).
  For eyeballing only; arXiv compiles from source.
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

## Grok adversarial review (2026-11-14) — decisions needed before submitting

Full review: ronanlabs-hq `publishing/reviews/grok-adversarial-treeopsd-2026-11-14.md`.
W6 verified Finding 1 against the seed-package JSONs and applied the
mechanical half only: the "at most 0.49 EM" sentence now gives the
rescore deltas from ANALYSIS-FINAL-20261024.md (-0.08/-0.49 QA,
+0.36/-1.57 BC+), and the Table 2 caption now states the C-B rows are
as-scored and prints the hardware-matched C-B per-seed gaps (QA 4.41,
-0.18, 6.47; BC+ 0.00, 1.81, 7.35). No hardware-matched pooled C-B
interval exists in the packages; computing one is open.

Left for Stephen, not edited by W6 (they change the paper's argument):
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
  you want cited. Do not push this branch to it until you approve;
  the cited repo currently holds the October version.
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
- The one bibliography entry (arXiv:2610.23181) is deliberate; a
  reviewer will want more related work for a venue version, less so
  for an arXiv note. Grok adversarial review of this package is filed
  in ronanlabs-hq `publishing/reviews/`.
