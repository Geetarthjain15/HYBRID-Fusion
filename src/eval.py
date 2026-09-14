import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from hybrid_model import HybridBrainTumorModel

# Must match streamlit_app.py's actual inference pipeline (HF
# AutoImageProcessor for google/vit-base-patch16-224: mean=0.5/std=0.5).
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def evaluate(model, loader, device, class_names, is_hybrid):
    model.eval()
    correct, total = 0, 0
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) if is_hybrid else model(images).logits
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                class_total[class_names[l]] += 1
                class_correct[class_names[l]] += int(p == l)
    per_class = {c: (class_correct[c] / class_total[c] if class_total[c] else 0.0) for c in class_names}
    return correct / total, per_class


def main():
    parser = argparse.ArgumentParser(description="Re-check a saved checkpoint's accuracy on a dataset split.")
    parser.add_argument("--model", choices=["hybrid", "vit"], default="hybrid")
    parser.add_argument("--data_dir", default="dataset/Testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=16)
    class_names = dataset.classes

    if args.model == "hybrid":
        model = HybridBrainTumorModel(vit_weights_path="src/vit_brain_tumor_5class.pth").to(device)
        model.load_state_dict(torch.load("src/hybrid_brain_tumor.pth", map_location=device))
        is_hybrid = True
    else:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=4, ignore_mismatched_sizes=True
        ).to(device)
        model.load_state_dict(torch.load("src/vit_brain_tumor_5class.pth", map_location=device))
        is_hybrid = False

    acc, per_class = evaluate(model, loader, device, class_names, is_hybrid)
    print(f"Overall accuracy on {args.data_dir}: {acc:.4f}")
    for c, a in per_class.items():
        print(f"  {c}: {a:.4f}")


if __name__ == "__main__":
    main()
