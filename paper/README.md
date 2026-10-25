# arXiv replication note — TreeOPSD

`main.tex` is a self-contained 4–6 page replication note (article class,
booktabs/amsmath/hyperref/microtype only; no external figure files).

Compile (any TeX Live / arXiv does this server-side on submission):

```
pdflatex main.tex && pdflatex main.tex
```

Status 2026-10-24: DRAFT for Stephen's review. All numbers trace to
`../results/seed_package_{CB,CE_orig,CE_gb300}.json` via
`../ANALYSIS-FINAL-20261024.md`. No LaTeX toolchain exists on this
workstation or the Stations, so local compilation is UNVERIFIED — run the
two passes above (or upload to arXiv/Overleaf) before submitting.
Before submission: confirm author block, add repository URL once public,
confirm the AI-assistance disclosure wording in the Reproducibility
paragraph.
