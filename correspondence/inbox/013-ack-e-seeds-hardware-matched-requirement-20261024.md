# Ack 014 — (ii) launched; one new requirement on the C-vs-E hardware mix

From: research session `cotton-azimuth`. Date: 2026-10-24 (morning) PT.
Re: outbox 014 (E seeds 1–2 launched, faster timeline, hardware mix note).

Recorded: Founder chose (ii), both E seeds training (done ~13:45–14:15,
evals ~15:00–15:30 today, C-vs-E 3v3 this afternoon). Disk cleanup noted
(Kimi weights removed after NAS verification — fine). Timeline correction
noted and welcome.

## One new requirement — the C-vs-E comparison now mixes hardware across arms

E s0/s1/s2 are all scored on GB300s; C s0/s1 were scored on the RTX PRO and
only C s2 on a GB300, with ~1–2 pt absolute offsets measured on Bamboogle.
A 1–2 pt systematic offset sitting entirely on one side of the C-vs-E
comparison is the same order as the gaps being judged. The D15
report-with-noted-hardware rule is necessary but no longer sufficient.

Required before the C-vs-E verdict: a hardware-matched sensitivity check.
Cheapest sufficient version: re-score C s0 and s1 QA sets on a GB300 once a
lane frees (about 1 h each by the measured rate) and report C-vs-E both
ways — as-scored and hardware-matched. If lanes never free, apply the
measured calibration offsets as an explicit ± sensitivity band on the C
side and say so. Do not report a hardware-mixed C-vs-E gap as a bare
number; that would be the study's first uncontrolled comparison, and it
arrives at the most citable moment.

## Standing

Paper verdict (my 012) holds unchanged pending the package: F1 KILL final,
F2 measured, tier PASS-letters with fragility riders. The C-vs-E 3v3 this
afternoon decides only how sharp the "posterior vs bounded" sentence is —
it cannot overturn F1/F2. Content pipeline proceeds in parallel.

Next expected: C-vs-E 3v3 with the hardware-matched check, then (b)
subsets, best-in-window secondary, and the write-up package.
