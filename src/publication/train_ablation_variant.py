"""
Phase 3 ablation trainer - a wrapper, not a fork.

Every constant, transform, evaluation routine and dump helper is IMPORTED from
src/train_ablation.py. That file is frozen: this module does not edit it, and
importing rather than copying means the ablations cannot drift from the trainer
that produced the reference result.

What this wrapper adds:
  - the eight named Phase 3 variants, each resolving to one controlled change
    against the reference configuration;
  - the two branch-only architectures, which the frozen trainer has no notion of;
  - output under publication_runs/ablations/<split>/<variant>/seed_<seed>/;
  - refusal to overwrite a completed run, with an explicit --resume;
  - a saved config.json carrying the resolved variant, seed, split, environment
    and the exact argument vector.

Label isolation: the external set is scored ONCE, after training and after
validation-based selection have both finished. Its labels reach no optimiser,
no early-stopping decision and no configuration choice.

    python src/publication/train_ablation_variant.py --variant no_gate --seed 42 \
        --split clean --project-root . --dry-run
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_ablation as TA            # frozen: imported, never modified
from fusion_v2 import FusionV2, ModelEMA, tta_logits
from ablation_models import build_branch_only
from run_ablations import VARIANTS, SEEDS, PRIMARY_SPLIT
import session_scope


class VariantError(Exception):
    """A Phase 3 stop condition was hit."""


def resolve(variant: str) -> dict:
    """Turn a variant name into concrete switches. One change per variant."""
    if variant not in VARIANTS:
        raise VariantError("unknown variant %r; known: %s" % (variant, sorted(VARIANTS)))
    f = VARIANTS[variant]["flags"]
    return {
        "variant": variant,
        "arch": VARIANTS[variant]["arch"],
        "component_changed": VARIANTS[variant]["component_changed"],
        "unfreeze_all_cnn": "--unfreeze_all_cnn" in f,
        "unfreeze_all_vit": "--unfreeze_all_vit" in f,
        "backbone_lr_mult": float(f[f.index("--backbone_lr_mult") + 1]) if "--backbone_lr_mult" in f else 0.1,
        "aux_weight": float(f[f.index("--aux_weight") + 1]) if "--aux_weight" in f else 0.0,
        "ema": "--ema" in f,
        "tta": "--tta" in f,
        "gated": "--no_gate" not in f,
        "multiscale": "--no_multiscale" not in f,
    }


def build_model(cfg: dict, vit_source: str):
    """Reference architecture, with exactly one component altered."""
    arch = cfg["arch"]
    if arch == "fusion_v2":
        m = FusionV2(vit_source=vit_source, multiscale=cfg["multiscale"], gated=cfg["gated"])
        for p in m.cnn_stages.parameters():
            p.requires_grad = cfg["unfreeze_all_cnn"]
        for p in m.cnn_stages[-1].parameters():
            p.requires_grad = True
        for p in m.vit.parameters():
            p.requires_grad = cfg["unfreeze_all_vit"]
        for layer in m.vit.vit.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        for mod in (m.cnn_proj, m.vit_proj, m.gate, m.head, m.aux_cnn, m.aux_vit):
            for p in mod.parameters():
                p.requires_grad = True
        return m
    if arch in ("cnn_branch_only", "vit_branch_only"):
        return build_branch_only(arch, cfg["unfreeze_all_cnn"], cfg["unfreeze_all_vit"],
                                 vit_source, cfg["multiscale"])
    raise VariantError("unhandled arch %r" % arch)


def run_eval_only(args, cfg, record, out_dir, root, device, test_dir,
                  class_names_hint=None) -> int:
    """Score existing weights. Trains nothing, updates nothing, writes no
    checkpoint. The digest of the weights actually loaded is recorded, so the
    result can always be traced back to the exact model that produced it."""
    import hashlib

    os.makedirs(out_dir, exist_ok=True)
    log_fh = open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8")

    def log(msg):
        line = "[%s] %s" % (_dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    h = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    digest = h.hexdigest()
    log("EVALUATION ONLY - %s seed %d" % (args.variant, args.seed))
    log("checkpoint: %s" % args.checkpoint)
    log("sha256: %s" % digest)
    if args.expect_checkpoint_sha256 and digest != args.expect_checkpoint_sha256:
        log_fh.close()
        raise VariantError(
            "checkpoint digest mismatch.\n  expected %s\n  observed %s\n"
            "A checkpoint differing by one byte is a different model; refusing to "
            "score unverified weights."
            % (args.expect_checkpoint_sha256, digest))
    log("digest verified against the expected value")

    # The weights are the verified REFERENCE model, so they load into the
    # reference architecture - not into the variant's resolved architecture.
    ref = FusionV2(vit_source=args.vit_source, multiscale=True, gated=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ref_state = ref.state_dict()
    missing = sorted(set(ref_state) - set(state))
    unexpected = sorted(set(state) - set(ref_state))
    if missing or unexpected:
        log_fh.close()
        raise VariantError("strict load failed: %d missing, %d unexpected keys"
                           % (len(missing), len(unexpected)))
    ref.load_state_dict(state, strict=True)
    ref.to(device).eval()
    log("loaded strictly into the reference architecture (%d keys)" % len(ref_state))

    use_tta = cfg["tta"]
    log("test-time augmentation: %s" % ("ENABLED" if use_tta else "DISABLED (this is the ablation)"))

    ds = datasets.ImageFolder(test_dir, transform=TA.eval_transform)
    loader = DataLoader(ds, batch_size=TA.BATCH_SIZE, num_workers=args.workers)
    class_names = ds.classes
    test_acc, per_class = TA.evaluate(ref, loader, device, class_names, use_tta)
    log("clean test accuracy %.4f" % test_acc)

    TA.dump_predictions(ref, test_dir, device,
                        os.path.join(out_dir, "preds_test.csv"), args.workers, use_tta)
    if args.extra_eval:
        TA.dump_predictions(ref, args.extra_eval, device,
                            os.path.join(out_dir, "preds_dataset_ood.csv"),
                            args.workers, use_tta)
    # The reference is gated, so gate telemetry is available and worth keeping.
    TA.dump_gates(ref, test_dir, device, os.path.join(out_dir, "gates_test.csv"), args.workers)
    if args.extra_eval:
        TA.dump_gates(ref, args.extra_eval, device,
                      os.path.join(out_dir, "gates_dataset_ood.csv"), args.workers)

    with open(os.path.join(out_dir, "source_checkpoint_sha256.txt"), "w",
              encoding="utf-8") as f:
        f.write("%s  %s\n" % (digest, os.path.abspath(args.checkpoint)))

    record.update({
        "mode": "evaluation_only",
        "trained": False,
        "weights_updated": False,
        "checkpoint_written": False,
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "source_checkpoint_sha256": digest,
        "tta_used": use_tta,
        "test_acc": test_acc,
        "per_class_test_acc": per_class,
        "architecture_loaded": "FusionV2 reference (gated, multi-scale)",
        "note": ("Evaluation-only. The verified reference weights were scored with "
                 "TTA disabled; nothing was trained and no checkpoint was written. "
                 "Comparing this against the reference isolates the inference-time "
                 "effect of TTA from any training-time difference."),
    })
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(TA.environment_fingerprint(), f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    ck = os.path.join(out_dir, "checkpoint.pth")
    if os.path.exists(ck):
        log_fh.close()
        raise VariantError("evaluation-only run wrote a checkpoint; that must not happen")
    log("DONE %s seed=%d test=%.4f (no training, no checkpoint written)"
        % (args.variant, args.seed, test_acc))
    log_fh.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--seed", type=int, default=42, choices=SEEDS)
    ap.add_argument("--split", default=PRIMARY_SPLIT, choices=["clean", "original"])
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_runs/ablations")
    ap.add_argument("--train-dir", default=None)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--extra-eval", default=None,
                    help="External set, scored once AFTER training and selection.")
    ap.add_argument("--vit-source", default="google/vit-base-patch16-224")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=TA.EPOCHS)
    ap.add_argument("--resume", action="store_true",
                    help="Continue a run whose metrics.json is absent but whose "
                         "directory exists. Never silently replaces a completed run.")
    ap.add_argument("--eval-only", action="store_true",
                    help="Re-score an existing checkpoint without training "
                         "(used by no_tta).")
    ap.add_argument("--authorize-training", default=None,
                    help="Unlock training. Must name a registered session, and that "
                         "session must have approved the variant.")
    ap.add_argument("--checkpoint", default=None,
                    help="Evaluation-only: the weights to score. Required with "
                         "--eval-only; never written to.")
    ap.add_argument("--expect-checkpoint-sha256", default=None,
                    help="Evaluation-only: refuse to load weights whose digest differs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve the configuration, build the model, report shapes "
                         "and parameter counts, then exit without training.")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    cfg = resolve(args.variant)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(root, args.out_root)
    rel_out = os.path.relpath(out_root, root).replace("\\", "/")
    if not rel_out.startswith("publication_"):
        raise VariantError("refusing to write outside publication_*: %s" % rel_out)
    out_dir = os.path.join(out_root, args.split, args.variant, "seed_%d" % args.seed)

    done_marker = os.path.join(out_dir, "metrics.json")
    if os.path.exists(done_marker) and not args.resume:
        raise VariantError("run already complete: %s. Refusing to overwrite. Pass "
                           "--resume only if you intend to continue an INCOMPLETE run."
                           % os.path.relpath(done_marker, root))
    if os.path.exists(done_marker) and args.resume:
        raise VariantError("--resume given but the run is already complete (%s exists). "
                           "Resume is for incomplete runs; delete nothing and take no "
                           "action." % os.path.relpath(done_marker, root))

    train_dir = args.train_dir or os.path.join(
        root, "dataset_clean" if args.split == "clean" else "dataset", "Training")
    test_dir = args.test_dir or os.path.join(
        root, "dataset_clean" if args.split == "clean" else "dataset", "Testing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_info = TA.seed_everything(args.seed)
    model = build_model(cfg, args.vit_source)
    n_total = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    record = {
        "phase": "3 - FusionV2 component ablation",
        "variant": args.variant, "seed": args.seed, "split": args.split,
        "resolved_config": cfg,
        "component_changed": cfg["component_changed"],
        "question": VARIANTS[args.variant]["question"],
        "train_dir": train_dir, "test_dir": test_dir,
        "external_eval_dir": args.extra_eval,
        "params_total": n_total, "params_trainable": n_train_p,
        "device": str(device), "seeding": seed_info,
        "env": TA.environment_fingerprint(),
        "argv": sys.argv[1:],
        "batch_size": TA.BATCH_SIZE, "lr": TA.LR, "epochs": args.epochs,
        "weight_decay": TA.WEIGHT_DECAY, "label_smoothing": TA.LABEL_SMOOTHING,
        "patience": TA.PATIENCE, "val_fraction": TA.VAL_FRACTION,
        "val_split_seed": TA.VAL_SPLIT_SEED,
        "label_isolation": ("The external set is scored once, after training and after "
                            "validation-based model selection. Its labels never reach "
                            "the optimiser, early stopping or configuration choice."),
        "output_dir": os.path.relpath(out_dir, root).replace("\\", "/"),
    }

    if args.dry_run:
        x = torch.randn(2, 3, TA.IMAGE_SIZE, TA.IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            out = model(x)
        # FusionV2 always CONSTRUCTS self.gate; forward() consults it only when
        # gated=True. Reporting mere presence would misstate the no_gate ablation,
        # so presence and use are reported separately, along with the parameters
        # that become inert when the gate is bypassed.
        gate_mod = getattr(model, "gate", None)
        gate_used = bool(getattr(model, "gated", False)) and gate_mod is not None
        gate_params = sum(p.numel() for p in gate_mod.parameters()) if gate_mod else 0
        with torch.no_grad():
            probe = model(x, return_aux=True) if gate_mod is not None else None
        record["dry_run"] = {
            "forward_output_shape": list(out.shape),
            "has_cnn_branch": hasattr(model, "cnn_stages"),
            "has_vit_branch": hasattr(model, "vit"),
            "gate_module_constructed": gate_mod is not None,
            "gate_used_in_forward": gate_used,
            "gate_emits_mass": probe is not None and probe[3] is not None,
            "gate_parameters": gate_params,
            "inert_gate_parameters": 0 if gate_used else gate_params,
            "params_excluding_inert_gate": n_train_p - (0 if gate_used else gate_params),
            "trained": False,
        }
        print(json.dumps({k: record[k] for k in
                          ("variant", "seed", "split", "params_total", "params_trainable",
                           "component_changed", "dry_run", "output_dir")}, indent=2))
        print("\nDRY RUN - no training performed, nothing written.")
        return 0

    # ---- authorisation gate ------------------------------------------------
    # Training stays refused by default. It unlocks only for a named session and
    # only for the variants that session approved, so a mistyped --variant cannot
    # quietly start an unapproved run.
    token = args.authorize_training
    if token not in session_scope.SESSIONS:
        raise VariantError(
            "Training is gated. Pass --authorize-training <session> naming a "
            "registered session (%s), or --dry-run to plan. Got %r."
            % (", ".join(sorted(session_scope.SESSIONS)), token))

    # ---- evaluation-only path (no_tta) -----------------------------------
    # Scores existing weights. It must not train, must not update weights, and
    # must not write a checkpoint - otherwise it stops measuring "the reference
    # model without TTA" and starts measuring a different model.
    if args.eval_only:
        ok, reason = session_scope.load(token).is_eval_in_scope(args.variant)
        if not ok:
            raise VariantError("refusing evaluation-only %r: %s" % (args.variant, reason))
        if not args.checkpoint:
            raise VariantError("--eval-only requires --checkpoint")
        if not os.path.exists(args.checkpoint):
            raise VariantError("checkpoint not found: %s" % args.checkpoint)
        return run_eval_only(args, cfg, record, out_dir, root, device,
                             test_dir, class_names_hint=None)

    allowed, reason = session_scope.is_in_scope(token, args.variant)
    if not allowed:
        owner = session_scope.owning_session(args.variant)
        extra = (" It belongs to %s." % owner) if owner and owner != token else ""
        raise VariantError("refusing to train %r: %s%s" % (args.variant, reason, extra))

    os.makedirs(out_dir, exist_ok=True)
    log_fh = open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8")

    def log(msg):
        line = "[%s] %s" % (_dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log("variant=%s seed=%d split=%s device=%s" % (args.variant, args.seed, args.split, device))
    log("component changed: %s" % cfg["component_changed"])
    log("params total=%s trainable=%s" % (format(n_total, ","), format(n_train_p, ",")))

    # ---- data: the split logic is the frozen trainer's, not a copy ---------
    full_train = datasets.ImageFolder(train_dir)
    gen = torch.Generator().manual_seed(TA.VAL_SPLIT_SEED)
    idx = torch.randperm(len(full_train), generator=gen).tolist()
    n_val = int(TA.VAL_FRACTION * len(idx))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_set = Subset(datasets.ImageFolder(train_dir, transform=TA.train_transform), train_idx)
    val_set = Subset(datasets.ImageFolder(train_dir, transform=TA.eval_transform), val_idx)
    test_set = datasets.ImageFolder(test_dir, transform=TA.eval_transform)
    class_names = full_train.classes
    log("data: train=%d val=%d test=%d classes=%s"
        % (len(train_set), len(val_set), len(test_set), class_names))

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=TA.BATCH_SIZE, shuffle=True,
                              generator=TA.make_generator(args.seed),
                              worker_init_fn=TA.seed_worker, num_workers=args.workers,
                              pin_memory=pin, persistent_workers=args.workers > 0,
                              drop_last=(len(train_set) % TA.BATCH_SIZE == 1))
    val_loader = DataLoader(val_set, batch_size=TA.BATCH_SIZE,
                            num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=TA.BATCH_SIZE,
                             num_workers=args.workers, pin_memory=pin)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=TA.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.param_groups(TA.LR, cfg["backbone_lr_mult"]),
                                  weight_decay=TA.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = ModelEMA(model) if cfg["ema"] else None

    record["batches_per_epoch"] = len(train_loader)
    record["n_train"] = len(train_set)
    record["n_val"] = len(val_set)
    record["n_test"] = len(test_set)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(TA.environment_fingerprint(), f, indent=2)

    history, best_val, best_state, stale = [], 0.0, None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot_loss = correct = seen = 0
        te = time.time()
        for bi, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if cfg["aux_weight"] > 0:
                out, a1, a2, _ = model(images, return_aux=True)
                aux = criterion(a1, labels)
                if a2 is not None:
                    aux = aux + criterion(a2, labels)
                loss = criterion(out, labels) + cfg["aux_weight"] * aux
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
                log("  ep%d batch %d/%d loss=%.4f"
                    % (epoch + 1, bi + 1, len(train_loader), tot_loss / (bi + 1)))
        scheduler.step()

        tr_acc = correct / seen
        eval_model = ema.ema if ema is not None else model
        val_acc, _ = TA.evaluate(eval_model, val_loader, device, class_names, cfg["tta"])
        rec = {"epoch": epoch + 1, "train_loss": tot_loss / len(train_loader),
               "train_acc": tr_acc, "val_acc": val_acc,
               "train_val_gap": tr_acc - val_acc,
               "lr": scheduler.get_last_lr()[0], "secs": round(time.time() - te, 1)}
        history.append(rec)
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        log("Epoch %d/%d loss=%.4f train=%.4f val=%.4f gap=%+.4f (%.0fs)"
            % (epoch + 1, args.epochs, rec["train_loss"], tr_acc, val_acc,
               rec["train_val_gap"], rec["secs"]))

        if val_acc > best_val:
            best_val, best_state, stale = val_acc, copy.deepcopy(eval_model.state_dict()), 0
            log("  new best validation accuracy %.4f at epoch %d" % (best_val, epoch + 1))
        else:
            stale += 1
            if stale >= TA.PATIENCE:
                log("Early stopping at epoch %d (no validation gain in %d epochs)"
                    % (epoch + 1, TA.PATIENCE))
                break

    model.load_state_dict(best_state)
    test_acc, test_per_class = TA.evaluate(model, test_loader, device, class_names, cfg["tta"])
    torch.save(best_state, os.path.join(out_dir, "checkpoint.pth"))
    log("checkpoint saved from the BEST-validation epoch, not the last")
    log("clean test accuracy %.4f" % test_acc)

    TA.dump_predictions(model, test_dir, device,
                        os.path.join(out_dir, "preds_test.csv"), args.workers, cfg["tta"])
    if args.extra_eval:
        TA.dump_predictions(model, args.extra_eval, device,
                            os.path.join(out_dir, "preds_dataset_ood.csv"),
                            args.workers, cfg["tta"])
    if cfg["gated"] and cfg["arch"] == "fusion_v2":
        TA.dump_gates(model, test_dir, device,
                      os.path.join(out_dir, "gates_test.csv"), args.workers)
        if args.extra_eval:
            TA.dump_gates(model, args.extra_eval, device,
                          os.path.join(out_dir, "gates_dataset_ood.csv"), args.workers)

    gaps = [h["train_val_gap"] for h in history]
    best_i = max(range(len(history)), key=lambda i: history[i]["val_acc"])
    record.update({
        "best_val_acc": best_val, "test_acc": test_acc,
        "per_class_test_acc": test_per_class,
        "epochs_run": len(history), "total_minutes": round((time.time() - t0) / 60, 1),
        "overfitting": {
            "best_epoch": history[best_i]["epoch"],
            "max_train_val_gap_pp": 100 * max(gaps),
            "gap_at_best_epoch_pp": 100 * gaps[best_i],
            "final_train_val_gap_pp": 100 * gaps[-1],
            "early_stopping_fired": best_i < len(history) - 1,
            "checkpoint_is_best_epoch": True,
            "note": ("The recipe is unchanged from the reviewed reference so the "
                     "ablation stays controlled. Overfitting is measured here, not "
                     "newly defended against: altering regularisation would confound "
                     "the component under test with a change in fitting capacity."),
        },
    })
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    log("DONE %s seed=%d test=%.4f best_val=%.4f epochs=%d max_gap=%.2fpp %.1f min"
        % (args.variant, args.seed, test_acc, best_val, len(history),
           100 * max(gaps), record["total_minutes"]))
    log_fh.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VariantError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
