#!/usr/bin/env python3
"""EXP-1 seed package (PREREG-EXP1 section 5 + Amendment 1): per-seed and pooled TreeOPSD - TreePO gaps.

Reads $I/<run>/step100-<bench>/eval/<bench>/0.jsonl (one sample per question; em per row), pairs rows across
runs by sha1 of the prompt, and reports:
  - per run: EM per bench, tier-1 QA average (unweighted over the 4 MHQA sets), tier-2 BC+;
  - per seed: C - B on both tiers with a paired bootstrap (10,000 resamples, stratified by benchmark);
  - seed-level mean +- sd of each arm and of C - B (>= 3 seeds);
  - pooled: per-question diff averaged over seeds, same bootstrap;
  - extra pairs when present: C - A (GRPO G=8), C - D (grpo_tok), E - B (clipped vs literal TreePO).
Prints a markdown report and writes seed_package.json. Never prints prompt or answer text.

  python3 seed_package.py --interim /models/treeopsd/interim --out /models/treeopsd/interim/seed_package
"""
import argparse, hashlib, json, os, statistics as st
import numpy as np

QA = ["hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
BC = "bcplus"
ARMS = {"A": "exp1-grpo-s{}", "B": "exp1-treepo-s{}", "C": "exp1-treeopsd-s{}",
        "D": "exp1-grpo_tok-s{}", "E": "exp1-treepo_clip-s{}"}


def load(interim, run, bench, prefix="step100"):
    f = os.path.join(interim, run, f"{prefix}-{bench}", "eval", bench, "0.jsonl")
    if not os.path.exists(f):
        return None
    d = {}
    for line in open(f):
        r = json.loads(line)
        k = hashlib.sha1(r["input"].encode()).hexdigest()
        d.setdefault(k, []).append(float(r["em"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def paired(x, y, benches, rng, n_boot):
    """x, y: {bench: {key: em}}. Returns (point, lo, hi) for mean over benches of (y - x) in EM points."""
    diffs = []
    for b in benches:
        keys = sorted(set(x[b]) & set(y[b]))
        diffs.append(np.array([y[b][k] - x[b][k] for k in keys]))
    point = 100 * np.mean([d.mean() for d in diffs])
    boots = np.zeros(n_boot)
    for d in diffs:
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots += d[idx].mean(axis=1)
    boots = 100 * boots / len(diffs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--gb300-runs", default="", help="comma list of runs read from gb300-step100-<bench> (same-hardware rescore)")
    ap.add_argument("--pair", default="B,C", help="LO,HI arms for the seed-level/pooled section and the PREREG bars")
    a = ap.parse_args()
    rng = np.random.default_rng(20261024)
    seeds = [int(s) for s in a.seeds.split(",")]
    data = {}   # (arm, seed) -> {bench: {key: em}}
    for arm, pat in ARMS.items():
        for s in seeds:
            run = pat.format(s)
            pre = "gb300-step100" if run in a.gb300_runs.split(",") else "step100"
            got = {b: load(a.interim, run, b, pre) for b in QA + [BC]}
            if all(v is not None for v in got.values()):
                data[(arm, s)] = got
    out = {"runs": {}, "pairs": {}, "seed_level": {}, "pooled": {}}
    lines = ["| run | BC+ | HotpotQA | 2Wiki | MuSiQue | Bamboogle | QA avg |", "|---|---|---|---|---|---|---|"]
    for (arm, s), d in sorted(data.items()):
        em = {b: 100 * float(np.mean(list(d[b].values()))) for b in QA + [BC]}
        em["qa_avg"] = float(np.mean([em[b] for b in QA]))
        out["runs"][f"{arm}s{s}"] = em
        lines.append(f"| {arm} s{s} | {em[BC]:.2f} | " + " | ".join(f"{em[b]:.2f}" for b in QA) + f" | {em['qa_avg']:.2f} |")
    print("\n".join(lines))

    def pair(lo_arm, hi_arm, s):
        x, y = data.get((lo_arm, s)), data.get((hi_arm, s))
        if not x or not y:
            return None
        t1 = paired(x, y, QA, rng, a.boot)
        t2 = paired(x, y, [BC], rng, a.boot)
        return {"tier1_qa": t1, "tier2_bcplus": t2}

    print("\n| pair | seed | tier 1 QA avg diff [95% CI] | tier 2 BC+ diff [95% CI] |\n|---|---|---|---|")
    for lo_arm, hi_arm in [("B", "C"), ("A", "C"), ("D", "C"), ("B", "E"), ("E", "C"), ("A", "D")]:
        for s in seeds:
            p = pair(lo_arm, hi_arm, s)
            if p:
                out["pairs"][f"{hi_arm}-{lo_arm} s{s}"] = p
                f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
                print(f"| {hi_arm} - {lo_arm} | s{s} | {f(p['tier1_qa'])} | {f(p['tier2_bcplus'])} |")

    LO, HI = a.pair.split(",")
    # seed level (B, C) and pooled C - B
    bc_seeds = [s for s in seeds if (LO, s) in data and (HI, s) in data]
    if len(bc_seeds) >= 2:
        sl = {}
        for arm in (LO, HI):
            for m in ["qa_avg", BC]:
                v = [out["runs"][f"{arm}s{s}"][m] for s in bc_seeds]
                sl[f"{arm} {m}"] = (st.mean(v), st.stdev(v))
        for m in ["qa_avg", BC]:
            v = [out["runs"][f"{HI}s{s}"][m] - out["runs"][f"{LO}s{s}"][m] for s in bc_seeds]
            sl[f"{HI}-{LO} {m}"] = (st.mean(v), st.stdev(v))
        out["seed_level"] = {k: {"mean": m, "sd": sd, "n_seeds": len(bc_seeds)} for k, (m, sd) in sl.items()}
        print(f"\nSeed level (n = {len(bc_seeds)} seeds), mean +- sd:")
        for k, (m, sd) in sl.items():
            print(f"- {k}: {m:.2f} +- {sd:.2f}")
        # pooled: per-question C - B averaged over seeds
        pooled = {}
        for name, benches in [("tier1_qa", QA), ("tier2_bcplus", [BC])]:
            diffs = []
            for b in benches:
                keys = set.intersection(*[set(data[(arm, s)][b]) for s in bc_seeds for arm in (LO, HI)])
                keys = sorted(keys)
                diffs.append(np.array([np.mean([data[(HI, s)][b][k] - data[(LO, s)][b][k] for s in bc_seeds])
                                       for k in keys]))
            point = 100 * np.mean([d.mean() for d in diffs])
            boots = np.zeros(a.boot)
            for d in diffs:
                boots += d[rng.integers(0, len(d), size=(a.boot, len(d)))].mean(axis=1)
            boots = 100 * boots / len(diffs)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pooled[name] = (float(point), float(lo), float(hi))
        out["pooled"] = pooled
        print(f"\nPooled {HI} - {LO} over {len(bc_seeds)} seeds (question bootstrap, seed-averaged):")
        for k, (p, lo, hi) in pooled.items():
            print(f"- {k}: {p:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        # PREREG bars
        t1, t2 = pooled["tier1_qa"], pooled["tier2_bcplus"]
        v1 = ("KILL" if abs(t1[0]) < 0.25 and not (t1[1] <= 0.73 <= t1[2]) else
              "PASS" if t1[0] >= 0.5 and t1[1] > 0 else "INCONCLUSIVE")
        v2 = "KILL" if t2[0] < 1.0 else ("PASS" if t2[0] >= 1.5 and t2[1] > 0 else "INCONCLUSIVE")
        label = "confirmatory" if len(bc_seeds) >= 3 else "preliminary"
        out["bars"] = {"tier1": v1, "tier2": v2, "label": label}
        print(f"\nPREREG EM bars ({label}): tier 1 {v1}; tier 2 {v2}. (M1 is decisive separately.)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "seed_package.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
