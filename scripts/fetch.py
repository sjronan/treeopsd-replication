#!/usr/bin/env python3
"""Pinned, resumable, checksum-verified downloader for the replication's data and models.

Reads scripts/manifest.tsv (generated 2026-10-21 from the Hugging Face API: every
file is pinned to a repo revision, with byte size and LFS sha256).

DRY RUN BY DEFAULT: prints what would be fetched and the total size. Pass --yes to
download. Python standard library only, plus `curl` for resumable transfers.

  python3 scripts/fetch.py --groups mhqa,model-e5-base-v2          # dry run
  python3 scripts/fetch.py --groups mhqa --root /data/treeopsd --yes
  python3 scripts/fetch.py --list                                   # groups + sizes

Layout: <root>/<group>/<path-in-repo>. Each finished group gets a
<root>/<group>/.verified file listing the sha256 of every file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.tsv")


def load_manifest():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.reader(f, delimiter="\t"):
            if not r or r[0].startswith("#"):
                continue
            group, kind, repo, rev, path, size, sha = r
            rows.append(dict(group=group, kind=kind, repo=repo, rev=rev, path=path,
                             size=int(size) if size not in ("", "None") else None,
                             sha=None if sha == "-" else sha))
    return rows


def url(r):
    pre = "datasets/" if r["kind"] == "dataset" else ""
    return f"https://huggingface.co/{pre}{r['repo']}/resolve/{r['rev']}/{r['path']}"


def sha256(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_one(r, root):
    dst = os.path.join(root, r["group"], r["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and r["size"] is not None and os.path.getsize(dst) == r["size"]:
        pass  # already complete; still verified below
    else:
        cmd = ["curl", "-fsSL", "--retry", "8", "--retry-delay", "5", "-C", "-", "-o", dst, url(r)]
        tok = os.environ.get("HF_TOKEN")
        if tok:  # only needed for gated repos (e.g. Llama); never logged
            cmd[1:1] = ["-H", f"Authorization: Bearer {tok}"]
        print(f"start {dst} ({(r['size'] or 0)/1e9:.2f} GB)", flush=True)
        subprocess.run(cmd, check=True)
    got = os.path.getsize(dst)
    if r["size"] is not None and got != r["size"]:
        raise SystemExit(f"size mismatch {dst}: {got} != {r['size']}")
    digest = sha256(dst)
    if r["sha"] and digest != r["sha"]:
        raise SystemExit(f"sha256 mismatch {dst}: {digest} != {r['sha']}")
    return dst, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="", help="comma-separated group names")
    ap.add_argument("--root", default=os.environ.get("TREEOPSD_DATA_ROOT", "./data"))
    ap.add_argument("--yes", action="store_true", help="actually download")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = load_manifest()
    sizes = OrderedDict()
    for r in rows:
        sizes.setdefault(r["group"], [0, 0, r["repo"], r["rev"]])
        sizes[r["group"]][0] += r["size"] or 0
        sizes[r["group"]][1] += 1
    if a.list or not a.groups:
        for g, (b, n, repo, rev) in sizes.items():
            print(f"{g:30s} {b/1e9:8.2f} GB  {n:4d} files  {repo}@{rev[:12]}")
        return
    want = [g.strip() for g in a.groups.split(",") if g.strip()]
    bad = [g for g in want if g not in sizes]
    if bad:
        raise SystemExit(f"unknown groups: {bad}")
    sel = [r for r in rows if r["group"] in want]
    total = sum(r["size"] or 0 for r in sel)
    print(f"{len(sel)} files, {total/1e9:.2f} GB -> {os.path.abspath(a.root)}")
    if not a.yes:
        for r in sel:
            print(f"  [dry-run] {url(r)}  ({(r['size'] or 0)/1e6:.1f} MB)")
        print("dry run only; re-run with --yes to download")
        return
    for g in want:
        out = []
        for r in (x for x in sel if x["group"] == g):
            dst, d = fetch_one(r, a.root)
            out.append(f"{d}  {os.path.relpath(dst, os.path.join(a.root, g))}")
            print(f"ok {dst} sha256={d}", flush=True)
        with open(os.path.join(a.root, g, ".verified"), "w") as f:
            f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    sys.exit(main())
