"""
Dedup-aware re-split (paper item 3).

The original split (prepare_dataset.py) took the first 15% of each class in
parquet row order. dedup_audit.py showed the result: 81% of test images have
an exact or near-duplicate on the train side. This script builds a split
where that cannot happen:

  1. pool Training + Testing
  2. cluster by exact bytes (MD5) and by 256-bit dHash (16x16) at
     Hamming <= 8, union-find so transitive near-duplicates share a cluster.
     NOTE: the original audit used a 64-bit (8x8) dHash at Hamming <= 4.
     On MRI that hash is too coarse - dark background + centred brain makes
     unrelated scans collide, which over-counted glioma/pituitary leakage
     ~30x. The 256-bit hash was validated against the exact-MD5 matches
     (it recovers them all) and separates true re-encodes (Hamming 0-8) from
     unrelated scans of the same class (Hamming 17+).
  3. keep ONE representative per cluster (drops in-split redundancy too)
  4. stratified 85/15 split over clusters, seeded

Images are copied (not linked) so the output is a plain ImageFolder that
train_ablation.py consumes with --train_dir/--test_dir. A manifest records
which cluster every original file belonged to, so the mapping is auditable.

    python src/make_clean_split.py --src dataset --out dataset_clean --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import shutil

import numpy as np
from PIL import Image

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def md5_bytes(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=16):
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())   # 16 x 16 = 256 bits


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out", default="dataset_clean")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--hamming", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. pool
    files = []  # (class, path, origin_split)
    for split in ("Training", "Testing"):
        for c in CLASSES:
            d = os.path.join(args.src, split, c)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                files.append((c, os.path.join(d, fn), split))
    n = len(files)
    print("pooled {} images".format(n), flush=True)

    # 2. cluster - within class only; cross-class duplicates would be label
    #    noise, which we report but do not merge
    uf = UnionFind(n)
    md5s, hashes = [], np.zeros((n, 32), dtype=np.uint8)
    for i, (c, p, _) in enumerate(files):
        md5s.append(md5_bytes(p))
        hashes[i] = dhash(p)
        if (i + 1) % 1000 == 0:
            print("  hashed {}/{}".format(i + 1, n), flush=True)

    by_md5 = {}
    for i, m in enumerate(md5s):
        if m in by_md5:
            uf.union(by_md5[m], i)
        else:
            by_md5[m] = i

    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    for c in CLASSES:
        idx = np.array([i for i, f in enumerate(files) if f[0] == c])
        h = hashes[idx]
        for a in range(len(idx) - 1):
            d = popcount[h[a] ^ h[a + 1:]].sum(1)
            for b in np.nonzero(d <= args.hamming)[0]:
                uf.union(int(idx[a]), int(idx[a + 1 + b]))
        print("  clustered class {}".format(c), flush=True)

    cluster_of = [uf.find(i) for i in range(n)]
    clusters = {}
    for i, r in enumerate(cluster_of):
        clusters.setdefault(r, []).append(i)
    print("{} images -> {} unique clusters ({} redundant)".format(
        n, len(clusters), n - len(clusters)), flush=True)

    # 3+4. one representative per cluster, stratified split over clusters
    rng = random.Random(args.seed)
    counts = {"Training": {c: 0 for c in CLASSES}, "Testing": {c: 0 for c in CLASSES}}
    manifest = []
    for c in CLASSES:
        reps = [members[0] for members in clusters.values() if files[members[0]][0] == c]
        rng.shuffle(reps)
        n_test = int(round(args.test_frac * len(reps)))
        assign = {i: ("Testing" if k < n_test else "Training") for k, i in enumerate(reps)}
        for i, split in assign.items():
            c_, p, origin = files[i]
            dst = os.path.join(args.out, split, c_)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
            counts[split][c_] += 1
            members = clusters[cluster_of[i]]
            manifest.append({
                "file": os.path.basename(p), "class": c_, "new_split": split,
                "original_split": origin, "cluster_size": len(members),
                "cluster_members": [os.path.basename(files[m][1]) for m in members],
            })

    report = {
        "source": args.src, "out": args.out, "seed": args.seed,
        "hamming_threshold": args.hamming, "n_pooled": n,
        "n_clusters": len(clusters), "n_redundant_dropped": n - len(clusters),
        "counts": counts,
        "n_train": sum(counts["Training"].values()),
        "n_test": sum(counts["Testing"].values()),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "split_manifest.json"), "w") as f:
        json.dump(manifest, f)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
