# F3 best-in-window secondary over the interim subset grid (10/25/50/75/100); run on the left host.
import json, os
os.chdir("/models/treeopsd/interim")
rows = {}
for arm, pat in [("A", "exp1-grpo-s{}"), ("B", "exp1-treepo-s{}"), ("C", "exp1-treeopsd-s{}"), ("D", "exp1-grpo_tok-s{}"), ("E", "exp1-treepo_clip-s{}")]:
    for s in (0, 1, 2):
        r = pat.format(s); g = {}
        for n in (10, 25, 50, 75, 100):
            f = f"{r}/step{n}/summary.json"
            if os.path.exists(f):
                d = json.load(open(f)); g[n] = (d["em"]["qa500i"], d["em"]["bc200i"])
        if g: rows[(arm, s)] = g
out = {}
for (arm, s), g in sorted(rows.items()):
    steps = sorted(g); q = {n: g[n][0] for n in steps}; b = {n: g[n][1] for n in steps}
    bq = max(q, key=q.get); bb = max(b, key=b.get); fin = g.get(100)
    out[(arm, s)] = (q[bq], b[bb], fin)
    inf = f"{q[bq]-fin[0]:+.1f} / {b[bb]-fin[1]:+.1f}" if fin else "-"
    print(f"| {arm} s{s} | " + " ".join(f"{q[n]:.1f}" for n in steps) + " | " + " ".join(f"{b[n]:.1f}" for n in steps)
          + f" | {q[bq]:.1f} @{bq} | {b[bb]:.1f} @{bb} | {inf} |")
for s in (0, 1, 2):
    if ("B", s) in out and ("C", s) in out:
        c, b_ = out[("C", s)], out[("B", s)]
        print(f"s{s} C-B: best-in-window qa {c[0]-b_[0]:+.1f} bc {c[1]-b_[1]:+.1f} | final(subset) qa {c[2][0]-b_[2][0]:+.1f} bc {c[2][1]-b_[2][1]:+.1f}")
