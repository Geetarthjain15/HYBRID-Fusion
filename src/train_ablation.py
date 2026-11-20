"""
Ablation trainer: one architecture arm, one seed, one run.

The paper's central claim is that CNN+ViT fusion beats either branch alone.
That claim is only supported if all three arms are trained under IDENTICAL
conditions - same split, same schedule, same augmentation, same seed control.
This script is that harness. Hyperparameters are copied verbatim from the
original train.py so the fusion arm reproduces the existing checkpoint's
regimen; the only thing that varies between arms is which branches exist.

    python src/train_ablation.py --arch cnn    --seed 42
    python src/train_ablation.py --arch vit    --seed 42
    python src/train_ablation.py --arch fusion --seed 42

Each run writes runs/<arch>_seed<N>/{checkpoint.pth,history.json,metrics.json}.
"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_model import HybridBrainTumorModel
from fusion_v2 import FusionV2, ModelEMA, tta_logits
from seeding import seed_everything, seed_worker, make_generator, environment_fingerprint

# ---- hyperparameters: verbatim from train.py ---------------------------
IMAGE_SIZE = 224
VAL_FRACTION = 0.15
VAL_SPLIT_SEED = 42          # split held fixed across arms AND seeds, so the
                             # seed sweep measures training noise, not split noise
EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])


class CNNOnly(nn.Module):
    """Fusion model with the ViT branch removed. Head width matches the
    fusion head so the arms differ in inputs, not head capacity."""

    def __init__(self, num_classes=4):
        super().__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn_features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.avgpool(self.cnn_features(x)).flatten(1))


class ViTOnly(nn.Module):
    """Fusion model with the CNN branch removed. Same head shape as CNNOnly."""

    def __init__(self, num_classes=4, warm_start=None, vit_source="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            vit_source, num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        if warm_start and os.path.exists(warm_start):
            self.vit.load_state_dict(torch.load(warm_start, map_location="cpu"))
        self.head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        cls = self.vit.vit(pixel_values=x).last_hidden_state[:, 0, :]
        return self.head(cls)


def build(arch, warm_start, unfreeze_all_cnn=False, vit_source="google/vit-base-patch16-224",
          unfreeze_all_vit=False, multiscale=True, gated=True):
    """Return the model with the freezing policy applied. Policy is identical
    across arms: last CNN block, last 2 ViT layers, and the head stay trainable.

    unfreeze_all_cnn is the capacity control: EfficientNet-B0 is 4.7M params
    total, so the strongest possible CNN-only arm is the fully fine-tuned one.
    If fusion still wins against that, the gain is not a parameter-count artefact."""
    if arch == "cnn":
        m = CNNOnly()
        for p in m.cnn_features.parameters():
            p.requires_grad = bool(unfreeze_all_cnn)
        for p in m.cnn_features[-1].parameters():
            p.requires_grad = True
    elif arch == "vit":
        m = ViTOnly(warm_start=warm_start, vit_source=vit_source)
        for p in m.vit.parameters():
            p.requires_grad = False
        for layer in m.vit.vit.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
    elif arch == "fusion":
        m = HybridBrainTumorModel(vit_weights_path=warm_start)
        for p in m.cnn_features.parameters():
            p.requires_grad = bool(unfreeze_all_cnn)
        for p in m.cnn_features[-1].parameters():
            p.requires_grad = True
        for p in m.vit_branch.parameters():
            p.requires_grad = False
        for layer in m.vit_branch.vit.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in m.fusion_head.parameters():
            p.requires_grad = True
    elif arch == "fusion_v2":
        m = FusionV2(vit_source=vit_source, multiscale=multiscale, gated=gated)
        for p in m.cnn_stages.parameters():
            p.requires_grad = bool(unfreeze_all_cnn)
        for p in m.cnn_stages[-1].parameters():
            p.requires_grad = True
        for p in m.vit.parameters():
            p.requires_grad = bool(unfreeze_all_vit)
        for layer in m.vit.vit.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for mod in (m.cnn_proj, m.vit_proj, m.gate, m.head, m.aux_cnn, m.aux_vit):
            for p in mod.parameters():
                p.requires_grad = True
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader, device, class_names, tta=False):
    model.eval()
    correct = total = 0
    per_c = {c: [0, 0] for c in class_names}
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = (tta_logits(model, images) if tta else model(images)).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for p, l in zip(preds.tolist(), labels.tolist()):
            per_c[class_names[l]][1] += 1
            per_c[class_names[l]][0] += int(p == l)
    return correct / total, {c: (k / n if n else 0.0) for c, (k, n) in per_c.items()}


@torch.no_grad()
def dump_predictions(model, root, device, out_csv, workers=0, tta=False):
    """Raw per-image softmax to CSV - the audit trail every metric derives from.
    Same columns as src/dump_predictions.py so the existing metrics pipeline
    consumes these files unchanged."""
    import csv
    ds = datasets.ImageFolder(root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, num_workers=workers)
    model.eval()
    rows, i = [], 0
    for x, y in loader:
        xb = x.to(device)
        p = (tta_logits(model, xb) if tta else torch.softmax(model(xb), 1)).cpu()
        for j in range(len(y)):
            fn = os.path.basename(ds.samples[i][0]); i += 1
            rows.append([fn, ds.classes[y[j]], ds.classes[int(p[j].argmax())]]
                        + ["{:.6f}".format(v) for v in p[j].tolist()])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in ds.classes])
        w.writerows(rows)
    acc = sum(r[1] == r[2] for r in rows) / len(rows)
    print("  wrote {} ({} rows, acc={:.4f})".format(out_csv, len(rows), acc), flush=True)
    return acc


@torch.no_grad()
def dump_gates(model, root, device, out_csv, workers=0):
    """Per-image gate mass for fusion_v2. Columns: filename, true_label,
    gate_cnn, gate_vit. diagnostics.py reads these for TEST 2."""
    import csv
    ds = datasets.ImageFolder(root, transform=eval_transform)
    loader = DataLoader(ds, batch_size=32, num_workers=workers)
    model.eval()
    rows, i = [], 0
    for x, y in loader:
        _, _, _, g = model(x.to(device), return_aux=True)
        g = g.cpu()
        for j in range(len(y)):
            fn = os.path.basename(ds.samples[i][0]); i += 1
            rows.append([fn, ds.classes[y[j]], "%.6f" % g[j, 0], "%.6f" % g[j, 1]])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "gate_cnn", "gate_vit"])
        w.writerows(rows)
    m = sum(float(r[2]) for r in rows) / len(rows)
    print("  wrote {} (gate_cnn mean {:.3f} / gate_vit {:.3f})".format(out_csv, m, 1 - m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["cnn", "vit", "fusion", "fusion_v2"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_dir", default="dataset/Training")
    ap.add_argument("--test_dir", default="dataset/Testing")
    ap.add_argument("--warm_start", default=None,
                    help="ViT checkpoint to warm-start from. Omitted = ImageNet init. "
                         "The existing src/vit_brain_tumor_5class.pth measures at chance, "
                         "so the default is deliberately NOT to use it.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (4 on Kaggle)")
    ap.add_argument("--unfreeze_all_cnn", action="store_true",
                    help="Capacity control: fine-tune every EfficientNet block, not just the last.")
    ap.add_argument("--vit_source", default="google/vit-base-patch16-224",
                    help="HF model id or a local directory holding the ViT-B/16 weights.")
    ap.add_argument("--unfreeze_all_vit", action="store_true",
                    help="fusion_v2: fine-tune every ViT layer, not just the last 2.")
    ap.add_argument("--backbone_lr_mult", type=float, default=0.1,
                    help="fusion_v2: LR multiplier for pretrained backbones vs fresh fusion parts.")
    ap.add_argument("--aux_weight", type=float, default=0.3,
                    help="fusion_v2: weight on the per-branch auxiliary losses.")
    ap.add_argument("--ema", action="store_true", help="Keep an EMA of weights and evaluate it.")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--tta", action="store_true", help="Horizontal-flip TTA at eval time.")
    ap.add_argument("--no_multiscale", action="store_true")
    ap.add_argument("--no_gate", action="store_true")
    ap.add_argument("--extra_eval", nargs="*", default=[],
                    help="Additional ImageFolder roots to dump predictions for (e.g. the "
                         "resampled 'dataset_ood/Testing' set).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())

    seed_info = seed_everything(args.seed)
    run = "{}_seed{}".format(args.arch, args.seed) + ("_" + args.tag if args.tag else "")
    out_dir = os.path.join(args.out_root, run)
    os.makedirs(out_dir, exist_ok=True)

    # split: fixed permutation, identical for every arm and seed
    full_train = datasets.ImageFolder(args.train_dir)
    g = torch.Generator().manual_seed(VAL_SPLIT_SEED)
    idx = torch.randperm(len(full_train), generator=g).tolist()
    n_val = int(VAL_FRACTION * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_set = Subset(datasets.ImageFolder(args.train_dir, transform=train_transform), train_idx)
    val_set = Subset(datasets.ImageFolder(args.train_dir, transform=eval_transform), val_idx)
    test_set = datasets.ImageFolder(args.test_dir, transform=eval_transform)
    class_names = full_train.classes

    pin = device.type == "cuda"
    # BatchNorm1d in the head cannot take a batch of 1; only drop the tail
    # batch when it would be exactly that, so epoch content stays intact.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              generator=make_generator(args.seed), worker_init_fn=seed_worker,
                              num_workers=args.workers, pin_memory=pin,
                              persistent_workers=args.workers > 0,
                              drop_last=(len(train_set) % BATCH_SIZE == 1))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=args.workers, pin_memory=pin)

    model = build(args.arch, args.warm_start, args.unfreeze_all_cnn, args.vit_source,
                  args.unfreeze_all_vit, not args.no_multiscale, not args.no_gate).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in trainable)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    if args.arch == "fusion_v2":
        optimizer = torch.optim.AdamW(model.param_groups(LR, args.backbone_lr_mult),
                                      weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    header = {
        "run": run, "arch": args.arch, "seed": args.seed,
        "warm_start": args.warm_start, "device": str(device),
        "unfreeze_all_cnn": args.unfreeze_all_cnn, "vit_source": args.vit_source,
        "unfreeze_all_vit": args.unfreeze_all_vit, "aux_weight": args.aux_weight,
        "backbone_lr_mult": args.backbone_lr_mult, "ema": args.ema, "tta": args.tta,
        "train_dir": args.train_dir, "test_dir": args.test_dir,
        "n_train": len(train_set), "n_val": len(val_set), "n_test": len(test_set),
        "params_total": n_total, "params_trainable": n_train_p,
        "epochs": args.epochs, "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
        "seeding": seed_info, "env": environment_fingerprint(),
    }
    print(json.dumps(header, indent=2), flush=True)

    history, best_val, best_state, stale = [], 0.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot_loss = correct = seen = 0
        te = time.time()
        for bi, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.arch == "fusion_v2" and args.aux_weight > 0:
                out, a_cnn, a_vit, _ = model(images, return_aux=True)
                loss = (criterion(out, labels)
                        + args.aux_weight * (criterion(a_cnn, labels) + criterion(a_vit, labels)))
            else:
                out = model(images)
                loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            tot_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            seen += labels.size(0)
            if (bi + 1) % 50 == 0:
                print("  ep{} batch {}/{} loss={:.4f}".format(
                    epoch + 1, bi + 1, len(train_loader), tot_loss / (bi + 1)), flush=True)
        scheduler.step()

        tr_acc = correct / seen
        eval_model = ema.ema if ema is not None else model
        val_acc, _ = evaluate(eval_model, val_loader, device, class_names, args.tta)
        rec = {"epoch": epoch + 1, "train_loss": tot_loss / len(train_loader),
               "train_acc": tr_acc, "val_acc": val_acc,
               "lr": scheduler.get_last_lr()[0], "secs": round(time.time() - te, 1)}
        history.append(rec)
        print("Epoch {}/{} loss={:.4f} train={:.4f} val={:.4f} ({:.0f}s)".format(
            epoch + 1, args.epochs, rec["train_loss"], tr_acc, val_acc, rec["secs"]), flush=True)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_val:
            best_val, best_state, stale = val_acc, copy.deepcopy(eval_model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stop at epoch {} (no val gain in {})".format(
                    epoch + 1, PATIENCE), flush=True)
                break

    model.load_state_dict(best_state)
    test_acc, test_per_class = evaluate(model, test_loader, device, class_names, args.tta)
    if args.arch == "fusion_v2":
        # TEST 2 of the diagnostic protocol: dump the per-image mixing weight so
        # branch utilisation is an observable, not an assumption. Written for
        # every eval set, because the interesting question is whether the gate
        # SHIFTS between in-distribution and shifted data.
        for root, name in [(args.test_dir, "test")] + [
                (r, os.path.basename(os.path.dirname(r.rstrip("/\\"))) or "extra")
                for r in args.extra_eval]:
            dump_gates(model, root, device, os.path.join(out_dir, "gates_%s.csv" % name),
                       args.workers)
    torch.save(best_state, os.path.join(out_dir, "checkpoint.pth"))

    metrics = dict(header)
    metrics.update({
        "best_val_acc": best_val, "test_acc": test_acc,
        "per_class_test_acc": test_per_class,
        "final_train_acc": history[-1]["train_acc"],
        "epochs_run": len(history), "total_minutes": round((time.time() - t0) / 60, 1),
    })
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n[{}] best_val={:.4f} test={:.4f} ({:.0f} min, {} epochs)".format(
        run, best_val, test_acc, metrics["total_minutes"], len(history)), flush=True)
    for c, a in test_per_class.items():
        print("  {}: {:.4f}".format(c, a), flush=True)

    dump_predictions(model, args.test_dir, device,
                     os.path.join(out_dir, "preds_test.csv"), args.workers, args.tta)
    extra = {}
    for root in args.extra_eval:
        name = os.path.basename(os.path.dirname(root.rstrip("/\\"))) or "extra"
        extra[name] = dump_predictions(model, root, device,
                                       os.path.join(out_dir, "preds_{}.csv".format(name)),
                                       args.workers, args.tta)
    if extra:
        metrics["extra_eval_acc"] = extra
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
