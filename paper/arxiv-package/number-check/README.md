# Number-check sheet

Run `python3 paper/arxiv-package/number-check/generate.py` from the repository
(or invoke the script by absolute path). Standard library only; no GPU/network.
The generator writes `number-check.md` and `number-check.csv` beside itself.

The sheet audits the current arXiv `main.tex`: abstract, both full tables and
captions, Results and Mechanism, plus the numerical results repeated in Discussion
and Conclusion. It records the latest paper commit, whether the working paper is
edited, and SHA-256s of the exact paper, result packages, analysis,
preregistration and committed S1b smoke report. It
never changes TeX or the separate abstract/metadata files. The standalone abstract
is not a second independent source.

Table columns map to named JSON run keys. Table 2 uses the stated scoring basis:
as-scored CB, hardware-matched CE/CA/CD. Stored bootstrap endpoints are checked
against the package, with those values taking precedence over analysis prose.
The source packages alone cannot independently reproduce a question bootstrap.
QA averages, sample SDs, seed t intervals and two-seed means are recomputed.
Telemetry and design/identifier tokens are corroborated within relevant analysis
passages. A narrative-source MATCH is not an independent raw-log verification.
F2's 1.15x line and 100-step window map to the preregistration. The separately
labelled smoke-run tool-call means and approximate wall-time ratios map to
`correspondence/outbox/005-exp1-launched-g-tok-20261021.md`, including its
10-step/64-prompt denominators. These do not establish full-window F2 measures.

Every numeric occurrence is a separate row, including repeats, structural labels
and quantity words. New unmapped numbers fail closed as MISMATCH/evidence gaps.
Original-paper numbers absent from these three sources need original-paper
citations/evidence for PI sign-off; the generator does not fetch outside sources
or silently use unrelated values from a replication run. MISMATCH distinguishes
an actual disagreement (nonempty source/recomputed value) from absent evidence
(empty source/recomputed value). Inspect both before signing the sheet.
The summary reports numerical mismatches and unsupported rows separately;
both retain the fail-closed MISMATCH status in the full inventory.

If sevenfold drift wording is present, the generator checks it against matched
median time windows (4.79/1.05), not the last-quarter/full-run combination
(7.74/1.05). The analysis document itself repeats two CB bootstrap endpoints that
differ from `seed_package_CB.json`; those package disagreements are retained.

PI sign-off: ____________________  Date (PT): ____________________
