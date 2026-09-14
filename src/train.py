import copy
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from hybrid_model import HybridBrainTumorModel

# MPS benchmarked slower than plain CPU for this hybrid ViT+CNN model on
# Apple Silicon (frequent CPU<->GPU syncs from unsupported ops), so CPU
# wins unless a real CUDA GPU is present.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    # torch defaults to using only half the physical cores on some systems;
    # pin it to all of them (~2.7x speedup measured on this machine).
    torch.set_num_threads(os.cpu_count())
train_dir = "dataset/Training"
test_dir = "dataset/Testing"

# Normalization must match streamlit_app.py's actual inference pipeline
# (HF AutoImageProcessor for google/vit-base-patch16-224: mean=0.5/std=0.5,
# NOT ImageNet stats) or the model is trained on a different input
# distribution than it sees at inference time.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Carve a validation split out of Training, kept separate from Testing so
# Testing stays an honest, untouched final score evaluated only once.
full_train = datasets.ImageFolder(train_dir)
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(full_train), generator=generator).tolist()
val_size = int(0.15 * len(indices))
val_idx, train_idx = indices[:val_size], indices[val_size:]

train_set = Subset(datasets.ImageFolder(train_dir, transform=train_transform), train_idx)
val_set = Subset(datasets.ImageFolder(train_dir, transform=eval_transform), val_idx)
test_set = datasets.ImageFolder(test_dir, transform=eval_transform)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)
test_loader = DataLoader(test_set, batch_size=16)
class_names = full_train.classes

model = HybridBrainTumorModel(vit_weights_path="src/vit_brain_tumor_5class.pth").to(device)

# Freeze most of the pretrained backbones. Fine-tuning all of EfficientNet
# + ViT on a small MRI dataset is what drives the train/val gap; only the
# last CNN block, the last 2 ViT layers, and the fusion head stay trainable.
for p in model.cnn_features.parameters():
    p.requires_grad = False
for p in model.cnn_features[-1].parameters():
    p.requires_grad = True

for p in model.vit_branch.parameters():
    p.requires_grad = False
for layer in model.vit_branch.vit.encoder.layer[-2:]:
    for p in layer.parameters():
        p.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.05)

EPOCHS = 15
PATIENCE = 4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                class_total[class_names[l]] += 1
                class_correct[class_names[l]] += int(p == l)
    per_class = {c: (class_correct[c] / class_total[c] if class_total[c] else 0.0) for c in class_names}
    return correct / total, per_class


best_val_acc = 0.0
best_state = None
epochs_without_improvement = 0
train_acc = 0.0

num_train_batches = len(train_loader)
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_train, total_train = 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_train += (outputs.argmax(1) == labels).sum().item()
        total_train += labels.size(0)
        if (batch_idx + 1) % 20 == 0:
            print(f"  epoch {epoch+1} batch {batch_idx+1}/{num_train_batches}", flush=True)
    scheduler.step()

    train_acc = correct_train / total_train
    val_acc, _ = evaluate(val_loader)
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {PATIENCE} epochs)")
            break

model.load_state_dict(best_state)
test_acc, test_per_class = evaluate(test_loader)
print(f"\nBest Val Acc: {best_val_acc:.4f}")
print(f"Final Test Acc (held-out, evaluated once): {test_acc:.4f}")
for c, a in test_per_class.items():
    print(f"  {c}: {a:.4f}")

torch.save(model.state_dict(), "src/hybrid_brain_tumor.pth")

metrics = {
    "train_acc": train_acc,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
    "per_class_test_acc": test_per_class,
}
with open("src/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
