"""
Dump RAW per-image predictions to CSV. No metrics are computed here.

Every metric in the analysis is derived from this CSV, so the CSV is the
audit trail: open it, pick any row, run that image through the Streamlit
app, and the probabilities should match.

Output columns:
    filename, true_label, pred_label, p_<class0>, p_<class1>, ...

Usage:
    python src/dump_predictions.py --model hybrid --data_dir dataset/Testing --out_csv preds_hybrid_test.csv
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

from hybrid_model import HybridBrainTumorModel

# Identical to src/eval.py and the Streamlit app's inference pipeline
# (HF AutoImageProcessor for google/vit-base-patch16-224: mean=0.5/std=0.5).
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def main():
    parser = argparse.ArgumentParser(
        description="Write raw per-image softmax probabilities to a CSV."
    )
    parser.add_argument("--model", choices=["hybrid", "vit"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # --- device -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch version      : {torch.__version__}")
    print(f"cuda available     : {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"device             : {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(os.cpu_count())
        print(f"device             : cpu ({os.cpu_count()} threads)")

    # --- data -------------------------------------------------------------
    dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    class_names = dataset.classes  # alphabetical, matches training
    print(f"data_dir           : {args.data_dir}")
    print(f"classes            : {class_names}")
    print(f"images             : {len(dataset)}")

    # shuffle=False so loader order matches dataset.samples order
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    paths = [p for p, _ in dataset.samples]

    # --- model ------------------------------------------------------------
    if args.model == "hybrid":
        model = HybridBrainTumorModel(
            vit_weights_path="src/vit_brain_tumor_5class.pth"
        ).to(device)
        model.load_state_dict(torch.load("src/hybrid_brain_tumor.pth", map_location=device))
        is_hybrid = True
    else:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=4, ignore_mismatched_sizes=True
        ).to(device)
        model.load_state_dict(torch.load("src/vit_brain_tumor_5class.pth", map_location=device))
        is_hybrid = False
    model.eval()
    print(f"model              : {args.model}")

    # --- inference --------------------------------------------------------
    rows = []
    idx = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images) if is_hybrid else model(images).logits
            probs = F.softmax(outputs, dim=1).cpu()
            preds = probs.argmax(1)

            for j in range(images.size(0)):
                rows.append(
                    [
                        os.path.basename(paths[idx]),
                        class_names[labels[j].item()],
                        class_names[preds[j].item()],
                    ]
                    + [f"{probs[j][k].item():.6f}" for k in range(len(class_names))]
                )
                idx += 1
            print(f"  processed {idx}/{len(dataset)}", flush=True)

    # --- write ------------------------------------------------------------
    header = ["filename", "true_label", "pred_label"] + [f"p_{c}" for c in class_names]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # sanity line only - the real numbers get computed from the CSV later
    correct = sum(1 for r in rows if r[1] == r[2])
    print(f"WROTE {args.out_csv} ({len(rows)} rows)")
    print(f"sanity check accuracy: {correct}/{len(rows)} = {correct / len(rows):.4f}")


if __name__ == "__main__":
    main()
