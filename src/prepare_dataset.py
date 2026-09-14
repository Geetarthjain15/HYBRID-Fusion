import argparse
import io
import os
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

LABELS = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}


def main():
    parser = argparse.ArgumentParser(
        description="Download the public Brain-Tumor-MRI-Dataset (Hemg/Brain-Tumor-MRI-Dataset on "
                     "Hugging Face, matching glioma/meningioma/notumor/pituitary classes) and lay it "
                     "out as dataset/Training/<class> and dataset/Testing/<class> for train.py/eval.py."
    )
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Cap images per class (before the 85/15 split) for a faster demo run.")
    parser.add_argument("--out-dir", default="dataset")
    args = parser.parse_args()

    parquet_path = hf_hub_download(
        repo_id="Hemg/Brain-Tumor-MRI-Dataset", repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    df = pd.read_parquet(parquet_path)

    for cls in LABELS.values():
        os.makedirs(f"{args.out_dir}/Training/{cls}", exist_ok=True)
        os.makedirs(f"{args.out_dir}/Testing/{cls}", exist_ok=True)

    if args.max_per_class:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), args.max_per_class), random_state=42)
        )

    totals = df["label"].value_counts().to_dict()
    test_cutoff = {lbl: int(n * 0.15) for lbl, n in totals.items()}
    seen = {c: 0 for c in LABELS.values()}
    counts = {"Training": {c: 0 for c in LABELS.values()}, "Testing": {c: 0 for c in LABELS.values()}}

    for i, row in df.iterrows():
        cls = LABELS[row["label"]]
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        split = "Testing" if seen[cls] < test_cutoff[row["label"]] else "Training"
        seen[cls] += 1
        img.save(f"{args.out_dir}/{split}/{cls}/{cls}_{i:05d}.jpg", quality=95)
        counts[split][cls] += 1

    for split in ["Training", "Testing"]:
        print(split, counts[split], "total:", sum(counts[split].values()))


if __name__ == "__main__":
    main()
