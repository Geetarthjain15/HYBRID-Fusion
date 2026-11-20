"""
Pre-training sanity checks (brief section 8).

Catches shape, dtype, device, gradient and leakage faults in ~1 minute so we
never burn a training run on a bug. Exits non-zero if any check fails, so a
.bat can gate training on it.

Checks:
  1. data          - batches load, shapes/dtypes/ranges correct
  2. forward       - model output dimensions correct
  3. loss          - loss computes and is finite
  4. backward      - gradients flow to exactly the parameters we unfroze
  5. overfit       - model can drive a tiny batch to ~0 loss (learning works)
  6. split hygiene - train/val indices disjoint; no filename overlap train/test
  7. determinism   - same seed gives identical batches and identical outputs
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config as cfg_mod
from hybrid_model import HybridBrainTumorModel
from seeding import seed_everything, make_generator, seed_worker, environment_fingerprint

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((PASS, name, detail))
        print(f"[{PASS}] {name}: {detail}", flush=True)
        return True
    except Exception as e:
        results.append((FAIL, name, repr(e)))
        print(f"[{FAIL}] {name}: {e!r}", flush=True)
        return False


def build(cfg):
    tfm = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(list(cfg.norm_mean), list(cfg.norm_std)),
    ])
    ds = datasets.ImageFolder(cfg.train_dir, transform=tfm)
    return ds


def main():
    cfg = cfg_mod.BASELINE
    env = environment_fingerprint()
    seed_info = seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"env": env, "seed": seed_info, "device": str(device)}, indent=2), flush=True)

    ds = build(cfg)
    k = len(ds.classes)

    # ---- 1. data -------------------------------------------------------
    def _data():
        g = make_generator(cfg.seed)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=g, worker_init_fn=seed_worker)
        x, y = next(iter(dl))
        assert x.ndim == 4, f"expected 4D input, got {tuple(x.shape)}"
        assert x.shape[1:] == (3, cfg.image_size, cfg.image_size), tuple(x.shape)
        assert x.dtype == torch.float32, x.dtype
        assert y.dtype == torch.int64, y.dtype
        assert int(y.min()) >= 0 and int(y.max()) < k, (int(y.min()), int(y.max()))
        lo, hi = float(x.min()), float(x.max())
        assert -1.05 <= lo and hi <= 1.05, f"normalised range looks wrong: [{lo:.3f},{hi:.3f}]"
        return f"x={tuple(x.shape)} {x.dtype}, y={tuple(y.shape)} in [0,{k-1}], range [{lo:.2f},{hi:.2f}]"
    ok_data = check("data batches", _data)

    # ---- model ---------------------------------------------------------
    vw = cfg.vit_weights_path if cfg.use_vit_warm_start else None
    model = HybridBrainTumorModel(vit_weights_path=vw, num_classes=cfg.num_classes).to(device)

    # replicate the baseline freeze policy
    for p in model.cnn_features.parameters():
        p.requires_grad = False
    if cfg.unfreeze_cnn_last_block:
        for p in model.cnn_features[-1].parameters():
            p.requires_grad = True
    for p in model.vit_branch.parameters():
        p.requires_grad = False
    if cfg.unfreeze_vit_last_n_layers:
        for layer in model.vit_branch.vit.encoder.layer[-cfg.unfreeze_vit_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    g = make_generator(cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    generator=g, worker_init_fn=seed_worker)
    x, y = next(iter(dl))
    x, y = x.to(device), y.to(device)

    # ---- 2. forward ----------------------------------------------------
    def _fwd():
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (x.shape[0], cfg.num_classes), tuple(out.shape)
        assert torch.isfinite(out).all(), "non-finite logits"
        return f"logits={tuple(out.shape)}; params total={total:,} trainable={trainable:,} ({100*trainable/total:.1f}%)"
    check("forward pass", _fwd)

    # ---- 3. loss -------------------------------------------------------
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    def _loss():
        model.train()
        loss = crit(model(x), y)
        assert torch.isfinite(loss), "non-finite loss"
        # 4 balanced classes -> chance cross-entropy is about ln(4)=1.386
        assert 0.1 < float(loss) < 5.0, f"implausible initial loss {float(loss):.3f}"
        return f"loss={float(loss):.4f} (chance ~= {torch.log(torch.tensor(float(k))):.3f})"
    check("loss computes", _loss)

    # ---- 4. backward ---------------------------------------------------
    def _bwd():
        model.train()
        model.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        with_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
                     and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0]
        frozen_with_grad = [n for n, p in model.named_parameters()
                            if not p.requires_grad and p.grad is not None]
        assert not frozen_with_grad, f"frozen params received grads: {frozen_with_grad[:3]}"
        assert any("fusion_head" in n for n in with_grad), "fusion head got no gradient"
        assert any("cnn_features" in n for n in with_grad), "CNN branch got no gradient"
        assert any("vit_branch" in n for n in with_grad), "ViT branch got no gradient"
        return f"{len(with_grad)} tensors received finite non-zero grads; 0 frozen tensors did"
    check("gradient flow", _bwd)

    # ---- 5. overfit a tiny batch ---------------------------------------
    def _overfit():
        seed_everything(cfg.seed, cfg.deterministic)
        m = HybridBrainTumorModel(vit_weights_path=vw, num_classes=cfg.num_classes).to(device)
        xb, yb = x[:8], y[:8]
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        c = nn.CrossEntropyLoss()  # no smoothing: we want to reach ~0
        m.train()
        first = last = None
        for i in range(60):
            opt.zero_grad(set_to_none=True)
            l = c(m(xb), yb)
            l.backward()
            opt.step()
            if i == 0:
                first = float(l)
            last = float(l)
        assert last < first * 0.5, f"loss did not halve: {first:.3f} -> {last:.3f}"
        return f"tiny-batch loss {first:.3f} -> {last:.3f} in 60 steps (model can learn)"
    check("overfit tiny batch", _overfit)

    # ---- 6. split hygiene ----------------------------------------------
    def _splits():
        gen = torch.Generator().manual_seed(cfg.val_split_seed)
        idx = torch.randperm(len(ds), generator=gen).tolist()
        v = int(cfg.val_fraction * len(idx))
        val_idx, train_idx = set(idx[:v]), set(idx[v:])
        assert not (val_idx & train_idx), "train/val indices overlap"
        assert len(val_idx) + len(train_idx) == len(ds)

        test_ds = datasets.ImageFolder(cfg.test_dir)
        tr_names = {os.path.basename(p) for p, _ in ds.samples}
        te_names = {os.path.basename(p) for p, _ in test_ds.samples}
        overlap = tr_names & te_names
        assert not overlap, f"{len(overlap)} filenames appear in BOTH train and test"
        return (f"train {len(train_idx)} / val {len(val_idx)} disjoint; "
                f"test {len(te_names)} filenames, 0 shared with train")
    check("split hygiene", _splits)

    # ---- 7. determinism -------------------------------------------------
    def _determinism():
        def one():
            seed_everything(cfg.seed, cfg.deterministic)
            gg = make_generator(cfg.seed)
            d = DataLoader(ds, batch_size=8, shuffle=True, generator=gg, worker_init_fn=seed_worker)
            xa, ya = next(iter(d))
            m = HybridBrainTumorModel(vit_weights_path=vw, num_classes=cfg.num_classes).to(device)
            m.eval()
            with torch.no_grad():
                return xa, ya, m(xa.to(device))
        x1, y1, o1 = one()
        x2, y2, o2 = one()
        assert torch.equal(y1, y2), "same seed produced a different batch order"
        assert torch.allclose(x1, x2), "same seed produced different image tensors"
        assert torch.allclose(o1, o2, atol=1e-5), "same seed produced different outputs"
        return "identical batches and identical outputs across two seeded runs"
    check("determinism", _determinism)

    # ---- summary --------------------------------------------------------
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("\n" + "=" * 58)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    print("=" * 58)
    if n_fail:
        print("DO NOT START TRAINING - fix the failures above.")
    else:
        print("All checks passed. Safe to train.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
