"""
Duplicate / leakage audit (brief section 6). REPORT ONLY - changes nothing.

Published audits of this benchmark family report severe contamination:
exact-duplicate files, near-identical crops and resampled copies straddling
train/test, and patients appearing on both sides. Our split is also
non-random (prepare_dataset.py takes the first 15% of each class in parquet
row order), so we cannot assume any of that was avoided.

This script measures our own copy:
  1. exact duplicates   - MD5 over file bytes
  2. pixel duplicates   - SHA1 over decoded, resized pixel data (catches
                          re-encodes that differ byte-wise but not visually)
  3. near duplicates    - 64-bit dHash, Hamming distance <= 4
  4. train/test leakage - any of the above crossing the split boundary

It writes dedup_report.json plus clean_test_manifest.txt, a list of test
filenames with no train-side match, so metrics can later be recomputed on a
contamination-free subset WITHOUT altering the existing split.
"""

import hashlib
import json
import os
from collections import defaultdict

from PIL import Image

ROOT = "dataset"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
HAMMING_THRESHOLD = 4


def md5_bytes(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha1(path, size=64):
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return hashlib.sha1(im.tobytes()).hexdigest()


def dhash(path, size=8):
    """64-bit difference hash: robust to mild rescaling and re-compression."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        px = list(im.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def collect(split):
    out = []
    for c in CLASSES:
        d = os.path.join(ROOT, split, c)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((c, fn, os.path.join(d, fn)))
    return out


def main():
    train, test = collect("Training"), collect("Testing")
    print(f"train={len(train)} test={len(test)}", flush=True)

    rec = {}
    for split, items in (("Training", train), ("Testing", test)):
        for i, (c, fn, p) in enumerate(items):
            rec[p] = {
                "split": split, "cls": c, "fn": fn,
                "md5": md5_bytes(p), "psha": pixel_sha1(p), "dh": dhash(p),
            }
            if (i + 1) % 500 == 0:
                print(f"  hashed {split} {i+1}/{len(items)}", flush=True)

    report = {"n_train": len(train), "n_test": len(test)}

    # ---- within-corpus duplicate groups --------------------------------
    for key, label in (("md5", "exact_file"), ("psha", "pixel")):
        groups = defaultdict(list)
        for p, r in rec.items():
            groups[r[key]].append(p)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        report[f"{label}_duplicate_groups"] = len(dup)
        report[f"{label}_redundant_images"] = sum(len(v) - 1 for v in dup.values())

    # ---- cross-split leakage -------------------------------------------
    tr = {p: r for p, r in rec.items() if r["split"] == "Training"}
    te = {p: r for p, r in rec.items() if r["split"] == "Testing"}

    tr_md5 = defaultdict(list)
    tr_psha = defaultdict(list)
    for p, r in tr.items():
        tr_md5[r["md5"]].append(p)
        tr_psha[r["psha"]].append(p)
    tr_dh = [(r["dh"], p) for p, r in tr.items()]

    leaked, contaminated = [], set()
    for p, r in te.items():
        hit = None
        if r["md5"] in tr_md5:
            hit = ("exact_file", tr_md5[r["md5"]][0], 0)
        elif r["psha"] in tr_psha:
            hit = ("pixel_identical", tr_psha[r["psha"]][0], 0)
        else:
            for dh_t, pt in tr_dh:
                d = bin(dh_t ^ r["dh"]).count("1")
                if d <= HAMMING_THRESHOLD:
                    hit = ("near_duplicate", pt, d)
                    break
        if hit:
            contaminated.add(p)
            leaked.append({
                "test_file": r["fn"], "test_class": r["cls"],
                "kind": hit[0], "train_match": os.path.basename(hit[1]),
                "hamming": hit[2],
            })

    report["test_images_with_train_match"] = len(leaked)
    report["test_contamination_pct"] = round(100 * len(leaked) / max(len(te), 1), 2)
    report["by_kind"] = {
        k: sum(1 for x in leaked if x["kind"] == k)
        for k in ("exact_file", "pixel_identical", "near_duplicate")
    }
    report["by_class"] = {
        c: sum(1 for x in leaked if x["test_class"] == c) for c in CLASSES
    }
    report["examples"] = leaked[:15]
    report["hamming_threshold"] = HAMMING_THRESHOLD

    clean = [r["fn"] for p, r in te.items() if p not in contaminated]
    report["n_clean_test"] = len(clean)

    with open("dedup_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open("clean_test_manifest.txt", "w") as f:
        f.write("\n".join(sorted(clean)))

    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    print(f"\nwrote dedup_report.json and clean_test_manifest.txt ({len(clean)} clean test images)")


if __name__ == "__main__":
    main()
