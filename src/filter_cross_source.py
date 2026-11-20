"""
Build the honest cross-source evaluation set (paper item 5).

dataset_ood was collected as an out-of-distribution set, but it is a Roboflow
re-export of the same Kaggle source the training data derives from. Measured
with a 256-bit dHash against every training and test image:

    glioma      ~95% of images have a near-duplicate in train  -> not OOD
    meningioma  mostly genuinely unseen
    notumor     partially overlapping
    pituitary   mostly genuinely unseen

Reporting the whole set as "OOD" overstates the claim; discarding it entirely
throws away a real cross-source signal. This script splits it properly:

    overlapping   >= one train/test image within --hamming  -> RESAMPLING subset
                  (same images, different resize/re-encode: measures robustness
                  to preprocessing shift)
    disjoint      no near match anywhere                     -> CROSS-SOURCE subset
                  (genuinely unseen images from another export pipeline)

Both are legitimate evaluations; they measure different things and the paper
must report them separately. Writes a manifest and re-scores existing
prediction CSVs on each subset - no retraining needed.

    python src/filter_cross_source.py --ood dataset_ood/Testing --src dataset \\
        --preds "runs_*/*/preds_dataset_ood.csv" --out results/cross_source.json
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def dhash(path, size=16):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", default="dataset_ood/Testing")
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--hamming", type=int, default=8,
                    help="256-bit dHash distance below which an image counts as a re-encode.")
    ap.add_argument("--preds", default=None, help="glob of prediction CSVs to re-score")
    ap.add_argument("--out", default="results/cross_source.json")
    args = ap.parse_args()

    manifest, stats = {}, defaultdict(lambda: {"overlapping": 0, "disjoint": 0})
    for c in CLASSES:
        ref = []
        for split in ("Training", "Testing"):
            d = os.path.join(args.src, split, c)
            if os.path.isdir(d):
                ref += [os.path.join(d, f) for f in sorted(os.listdir(d))]
        R = np.stack([dhash(p) for p in ref])

        od = os.path.join(args.ood, c)
        files = sorted(os.listdir(od))
        H = np.stack([dhash(os.path.join(od, f)) for f in files])
        dist = POPCOUNT[H[:, None, :] ^ R[None, :, :]].sum(-1).min(1)
        for f, d in zip(files, dist):
            kind = "overlapping" if d <= args.hamming else "disjoint"
            manifest[f] = {"class": c, "kind": kind, "min_hamming": int(d)}
            stats[c][kind] += 1
        print("  %-12s n=%3d  overlapping=%3d  disjoint=%3d  median-hamming=%d"
              % (c, len(files), stats[c]["overlapping"], stats[c]["disjoint"], int(np.median(dist))),
              flush=True)

    n_over = sum(v["overlapping"] for v in stats.values())
    n_dis = sum(v["disjoint"] for v in stats.values())
    report = {"hamming_threshold": args.hamming, "n_total": len(manifest),
              "n_overlapping": n_over, "n_disjoint": n_dis,
              "by_class": {k: dict(v) for k, v in stats.items()},
              "manifest": manifest, "rescored": {}}
    print("\n  TOTAL  %d images -> %d overlapping (resampling), %d disjoint (cross-source)"
          % (len(manifest), n_over, n_dis))

    if args.preds:
        import csv
        over = {f for f, v in manifest.items() if v["kind"] == "overlapping"}
        dis = {f for f, v in manifest.items() if v["kind"] == "disjoint"}
        print("\n  re-scored predictions (whole / resampling / cross-source):")
        for path in sorted(glob.glob(args.preds)):
            with open(path) as fh:
                rows = list(csv.DictReader(fh))

            def acc(keep):
                sub = [r for r in rows if r["filename"] in keep]
                return (sum(r["true_label"] == r["pred_label"] for r in sub) / len(sub),
                        len(sub)) if sub else (float("nan"), 0)

            whole = sum(r["true_label"] == r["pred_label"] for r in rows) / len(rows)
            a_o, n_o = acc(over)
            a_d, n_d = acc(dis)
            run = os.path.basename(os.path.dirname(path))
            split = os.path.basename(os.path.dirname(os.path.dirname(path)))
            report["rescored"]["%s/%s" % (split, run)] = {
                "whole": whole, "resampling_acc": a_o, "resampling_n": n_o,
                "cross_source_acc": a_d, "cross_source_n": n_d}
            print("    %-34s %6.2f%%   %6.2f%% (n=%d)   %6.2f%% (n=%d)"
                  % (split + "/" + run, 100 * whole, 100 * a_o, n_o, 100 * a_d, n_d))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.splitext(args.out)[0] + "_disjoint_manifest.txt", "w") as f:
        f.write("\n".join(sorted(k for k, v in manifest.items() if v["kind"] == "disjoint")) + "\n")
    print("\n  wrote %s" % args.out)


if __name__ == "__main__":
    main()
