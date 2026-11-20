"""
Phase 3 addendum - recovered-checkpoint verification and import.

A file named checkpoint.pth in a directory named fusion_v2_seed42 is a claim.
This script turns it into evidence, or rejects it. Nothing is imported on the
strength of its path.

Gates, in order, cheapest first. A failure at any gate stops that seed and
leaves the file in staging:

  1. CONTAINER   size, SHA-256, torch archive structure, top-level keys,
                 state-dict key count.
  2. LOAD        CPU, weights_only=True, strict=True against the exact
                 full_fusion_v2 reference architecture. Zero missing keys, zero
                 unexpected keys, every tensor shape matching.
  3. ARCHITECTURE  the loaded module really is the reference: gated, multi-scale,
                 cnn_proj.in_features == 1432, proj_dim == 512.
  4. GATE        the gate is still the learned nn.Sequential and still produces
                 input-dependent output. A checkpoint whose gate had been frozen
                 to a constant would pass every other gate and silently destroy
                 the ablation, so it is checked directly: gate mass must vary
                 across a batch of different images.
  5. SAMPLE      a fixed 32-image subset is scored first. The full set is only
                 attempted if the sample matches.
  6. FULL        all 848 clean-test images, with the recorded preprocessing and
                 TTA, compared filename-by-filename against preds_test.csv.

The clean-split images were built on Kaggle and never downloaded, so the test
set is reconstructed by looking up dataset_clean/split_manifest.json against the
source images in dataset/. dataset_clean/ itself is never written to.

    python src/publication/verify_recovered_checkpoints.py --project-root . \
        --staging "D:/Research Model/kaggle_checkpoint_staging"
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import train_ablation as TA
from fusion_v2 import FusionV2, tta_logits, TAPS, FINAL_CH
from run_ablations import SEEDS

TOL_PP = 0.01           # accuracy reproduction tolerance, percentage points
PROB_TOL = 1e-4         # per-probability tolerance before an explanation is required


class RecoveryError(Exception):
    """A stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_staged(staging: str, seed: int):
    """Accept either extension. The task names .pth; the staged files are .zip.
    Both are torch archives - the extension is not what makes it valid."""
    d = os.path.join(staging, "runs_v2_clean", "fusion_v2_seed%d" % seed)
    for name in ("checkpoint.pth", "checkpoint.zip"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------- gate 1: container ---

def inspect_container(path: str) -> dict:
    info = {"path": path, "bytes": os.path.getsize(path), "sha256": sha256(path),
            "filename": os.path.basename(path)}
    with open(path, "rb") as f:
        magic = f.read(4)
    info["magic_hex"] = magic.hex()
    info["is_zip_container"] = magic == b"PK\x03\x04"
    if info["is_zip_container"]:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
        info["archive_entries"] = len(names)
        info["is_torch_archive"] = any(n.endswith("data.pkl") for n in names)
        info["archive_roots"] = sorted({n.split("/")[0] for n in names})
    else:
        info["is_torch_archive"] = False
    return info


# ----------------------------------------------------------------- gate 2: load ---

def build_reference_model(vit_source: str) -> FusionV2:
    """The exact Phase 3 full_fusion_v2 architecture. Defaults only - this must
    not drift from src/fusion_v2.py, which is why nothing is passed but the
    weight source."""
    return FusionV2(vit_source=vit_source, multiscale=True, gated=True)


def load_and_check(path: str, vit_source: str) -> dict:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    out = {"container_type": type(obj).__name__}
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        state = obj
        out["top_level_keys"] = "raw state_dict (tensor values at top level)"
        out["n_top_level_keys"] = len(obj)
    elif isinstance(obj, dict):
        out["top_level_keys"] = sorted(obj.keys())
        out["n_top_level_keys"] = len(obj)
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj:
                state = obj[k]
                break
        else:
            raise RecoveryError("no state dict found; top-level keys: %s" % sorted(obj))
    else:
        raise RecoveryError("unsupported checkpoint container: %s" % type(obj).__name__)

    out["state_dict_key_count"] = len(state)
    model = build_reference_model(vit_source)
    ref_state = model.state_dict()
    out["reference_key_count"] = len(ref_state)

    missing = sorted(set(ref_state) - set(state))
    unexpected = sorted(set(state) - set(ref_state))
    shape_mismatch = {k: [list(ref_state[k].shape), list(state[k].shape)]
                      for k in set(ref_state) & set(state)
                      if tuple(ref_state[k].shape) != tuple(state[k].shape)}
    out.update({"missing_keys": missing, "unexpected_keys": unexpected,
                "shape_mismatches": shape_mismatch,
                "n_missing": len(missing), "n_unexpected": len(unexpected),
                "n_shape_mismatch": len(shape_mismatch)})
    if missing or unexpected or shape_mismatch:
        out["loaded"] = False
        return out, None
    model.load_state_dict(state, strict=True)
    model.eval()
    out["loaded"] = True
    out["params_total"] = sum(p.numel() for p in model.parameters())
    return out, model


# --------------------------------------------------- gate 3/4: architecture, gate ---

def check_architecture_and_gate(model: FusionV2) -> dict:
    expected_cnn_in = FINAL_CH + sum(TAPS.values())
    arch = {
        "gated_flag": bool(model.gated),
        "multiscale_flag": bool(model.multiscale),
        "cnn_proj_in_features": model.cnn_proj[0].in_features,
        "cnn_proj_in_features_expected": expected_cnn_in,
        "proj_dim": model.cnn_proj[0].out_features,
        "vit_proj_in_features": model.vit_proj[0].in_features,
        "head_out_features": model.head[-1].out_features,
        "gate_module_type": type(model.gate).__name__,
        "gate_layer_types": [type(m).__name__ for m in model.gate],
        "gate_output_dim": model.gate[-1].out_features,
    }
    arch["architecture_matches_reference"] = (
        arch["gated_flag"] and arch["multiscale_flag"]
        and arch["cnn_proj_in_features"] == expected_cnn_in
        and arch["proj_dim"] == 512 and arch["gate_output_dim"] == 2)

    # The gate must remain LEARNED and input-dependent. A gate collapsed to a
    # constant would still load cleanly and would silently invalidate every
    # gate-related conclusion, so it is probed rather than assumed.
    torch.manual_seed(0)
    x = torch.randn(8, 3, TA.IMAGE_SIZE, TA.IMAGE_SIZE)
    with torch.no_grad():
        _, _, _, g = model(x, return_aux=True)
    g = g.numpy()
    gate_params = sum(p.numel() for p in model.gate.parameters())
    arch.update({
        "gate_is_learned_module": isinstance(model.gate, torch.nn.Sequential)
                                  and gate_params > 0,
        "gate_parameter_count": gate_params,
        "gate_cnn_mean_on_probe": float(g[:, 0].mean()),
        "gate_cnn_std_on_probe": float(g[:, 0].std()),
        "gate_varies_across_inputs": bool(g[:, 0].std() > 1e-6),
        "gate_rows_sum_to_one": bool(np.abs(g.sum(1) - 1.0).max() < 1e-5),
    })
    arch["gate_intact"] = (arch["gate_is_learned_module"]
                           and arch["gate_varies_across_inputs"]
                           and arch["gate_rows_sum_to_one"])
    return arch


# ------------------------------------------------ clean test set reconstruction ---

def reconstruct_clean_test(root: str, workdir: str) -> dict:
    """Rebuild the 848-image clean test set by manifest lookup. dataset_clean/
    is read, never written."""
    man_p = os.path.join(root, "dataset_clean", "split_manifest.json")
    if not os.path.exists(man_p):
        raise RecoveryError("dataset_clean/split_manifest.json absent; the clean test "
                            "split cannot be reconstructed and reuse cannot be verified")
    manifest = json.load(open(man_p, encoding="utf-8"))
    wanted = [m for m in manifest if m["new_split"] == "Testing"]
    index = {}
    for split in ("Training", "Testing"):
        for cls in sorted(os.listdir(os.path.join(root, "dataset", split))):
            d = os.path.join(root, "dataset", split, cls)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    index[fn] = os.path.join(d, fn)
    copied, missing = 0, []
    for m in wanted:
        src = index.get(m["file"])
        if src is None:
            missing.append(m["file"])
            continue
        dst_dir = os.path.join(workdir, "Testing", m["class"])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, m["file"])
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        copied += 1
    if missing:
        raise RecoveryError("%d clean-test images not found in dataset/: %s"
                            % (len(missing), missing[:5]))
    return {"manifest": "dataset_clean/split_manifest.json",
            "manifest_sha256": sha256(man_p),
            "n_testing_in_manifest": len(wanted), "n_reconstructed": copied,
            "reconstructed_at": workdir,
            "note": "dataset_clean/ was read only; images sourced from dataset/"}


# ----------------------------------------------------------- gates 5/6: inference ---

def score(model, root_dir: str, use_tta: bool, limit_files=None, workers=0):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ds = datasets.ImageFolder(root_dir, transform=TA.eval_transform)
    keep = None
    if limit_files is not None:
        keep = set(limit_files)
        ds.samples = [s for s in ds.samples if os.path.basename(s[0]) in keep]
        ds.targets = [t for _, t in ds.samples]
    loader = DataLoader(ds, batch_size=32, num_workers=workers)
    rows, i = [], 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            p = (tta_logits(model, x) if use_tta else torch.softmax(model(x), 1)).cpu().numpy()
            for j in range(len(y)):
                fn = os.path.basename(ds.samples[i][0]); i += 1
                rows.append({"filename": fn, "true_label": ds.classes[int(y[j])],
                             "pred_label": ds.classes[int(p[j].argmax())],
                             "probs": p[j].tolist()})
    return rows, ds.classes


def compare(rows, recorded_csv: str, classes) -> dict:
    with open(recorded_csv, encoding="utf-8") as f:
        rec = {r["filename"]: r for r in csv.DictReader(f)}
    ours = {r["filename"]: r for r in rows}
    only_ours = sorted(set(ours) - set(rec))
    only_rec = sorted(set(rec) - set(ours))
    common = sorted(set(ours) & set(rec))
    label_mismatch = [f for f in common if ours[f]["true_label"] != rec[f]["true_label"]]
    pred_mismatch = [f for f in common if ours[f]["pred_label"] != rec[f]["pred_label"]]
    diffs = []
    for f in common:
        a = np.array(ours[f]["probs"])
        b = np.array([float(rec[f]["p_" + c]) for c in classes])
        diffs.append(np.abs(a - b))
    D = np.concatenate(diffs) if diffs else np.zeros(1)
    acc_ours = sum(ours[f]["pred_label"] == ours[f]["true_label"] for f in common) / max(len(common), 1)
    acc_rec = sum(rec[f]["pred_label"] == rec[f]["true_label"] for f in common) / max(len(common), 1)
    return {
        "n_compared": len(common),
        "filenames_only_in_recomputed": only_ours[:10],
        "filenames_only_in_recorded": only_rec[:10],
        "n_filename_mismatch": len(only_ours) + len(only_rec),
        "n_true_label_mismatch": len(label_mismatch),
        "n_pred_label_mismatch": len(pred_mismatch),
        "pred_label_mismatch_examples": pred_mismatch[:5],
        "max_abs_prob_diff": float(D.max()), "mean_abs_prob_diff": float(D.mean()),
        "accuracy_recomputed": acc_ours, "accuracy_recorded": acc_rec,
        "accuracy_diff_pp": abs(100 * acc_ours - 100 * acc_rec),
    }


# ------------------------------------------------------------------- driver ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--staging", required=True)
    ap.add_argument("--vit-source", default="google/vit-base-patch16-224")
    ap.add_argument("--sample", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-root", default="publication_results/ablations/checkpoint_recovery")
    ap.add_argument("--import-root", default="publication_runs/recovered_checkpoints")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    for r in (args.out_root, args.import_root):
        rr = r if os.path.isabs(r) else os.path.join(root, r)
        if not os.path.relpath(rr, root).replace("\\", "/").startswith("publication_"):
            raise RecoveryError("refusing to write outside publication_*: %s" % r)
    out_root = os.path.join(root, args.out_root)
    imp_root = os.path.join(root, args.import_root)
    os.makedirs(out_root, exist_ok=True)

    scratch = os.path.join(os.environ.get("TEMP", "/tmp"), "clean_test_reconstructed")
    print("0. reconstructing the clean test split from its manifest", flush=True)
    recon = reconstruct_clean_test(root, scratch)
    print("   %d/%d images reconstructed into the scratchpad (dataset_clean/ untouched)"
          % (recon["n_reconstructed"], recon["n_testing_in_manifest"]), flush=True)

    report = {
        "phase": "3 addendum - recovered checkpoint verification",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "staging_root": args.staging,
        "tolerances": {"accuracy_pp": TOL_PP, "per_probability": PROB_TOL},
        "clean_test_reconstruction": recon,
        "seeds": {}, "imports": [], "training_launched": False,
    }

    for seed in SEEDS:
        print("\n--- seed %d ---" % seed, flush=True)
        entry = {"seed": seed, "gates": {}, "decision": None, "rejection_reason": None}
        src = find_staged(args.staging, seed)
        if src is None:
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = "no checkpoint.pth or checkpoint.zip in staging"
            report["seeds"][seed] = entry
            print("   REJECTED - not found", flush=True)
            continue

        # gate 1
        cont = inspect_container(src)
        entry["gates"]["container"] = cont
        print("   container: %s | %d bytes | torch archive: %s | %d entries"
              % (cont["filename"], cont["bytes"], cont["is_torch_archive"],
                 cont.get("archive_entries", 0)), flush=True)
        if not cont["is_torch_archive"]:
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = "not a torch archive"
            report["seeds"][seed] = entry
            continue

        # gate 2
        load_info, model = load_and_check(src, args.vit_source)
        entry["gates"]["load"] = load_info
        print("   load: %d state keys vs %d reference | missing %d | unexpected %d | "
              "shape mismatch %d" % (load_info["state_dict_key_count"],
                                     load_info["reference_key_count"], load_info["n_missing"],
                                     load_info["n_unexpected"], load_info["n_shape_mismatch"]),
              flush=True)
        if not load_info["loaded"]:
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = (
                "strict load failed: %d missing, %d unexpected, %d shape mismatches"
                % (load_info["n_missing"], load_info["n_unexpected"],
                   load_info["n_shape_mismatch"]))
            report["seeds"][seed] = entry
            print("   REJECTED - %s" % entry["rejection_reason"], flush=True)
            continue

        # gates 3 and 4
        arch = check_architecture_and_gate(model)
        entry["gates"]["architecture_and_gate"] = arch
        print("   arch matches reference: %s | gate intact: %s (mean %.3f, sd %.3f, %s params)"
              % (arch["architecture_matches_reference"], arch["gate_intact"],
                 arch["gate_cnn_mean_on_probe"], arch["gate_cnn_std_on_probe"],
                 format(arch["gate_parameter_count"], ",")), flush=True)
        if not (arch["architecture_matches_reference"] and arch["gate_intact"]):
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = "architecture or gate check failed"
            report["seeds"][seed] = entry
            continue

        # metadata cross-check against the recorded run
        rec_dir = os.path.join(root, "runs_v2_clean", "fusion_v2_seed%d" % seed)
        rec_metrics = json.load(open(os.path.join(rec_dir, "metrics.json"), encoding="utf-8"))
        entry["recorded_metrics"] = {k: rec_metrics.get(k) for k in
                                     ("seed", "arch", "test_acc", "best_val_acc",
                                      "params_total", "params_trainable", "ema", "tta",
                                      "aux_weight", "backbone_lr_mult", "epochs_run")}
        entry["gates"]["metadata"] = {
            "recorded_seed": rec_metrics.get("seed"),
            "seed_from_path": seed,
            "seed_matches": rec_metrics.get("seed") == seed,
            "params_total_matches": rec_metrics.get("params_total") == load_info["params_total"],
        }
        print("   metadata: recorded seed %s matches path: %s | params %s matches: %s"
              % (rec_metrics.get("seed"), entry["gates"]["metadata"]["seed_matches"],
                 format(load_info["params_total"], ","),
                 entry["gates"]["metadata"]["params_total_matches"]), flush=True)

        recorded_csv = os.path.join(rec_dir, "preds_test.csv")
        with open(recorded_csv, encoding="utf-8") as f:
            rec_rows = list(csv.DictReader(f))
        classes = [c[2:] for c in rec_rows[0] if c.startswith("p_")]

        # gate 5 - fixed sample first
        sample_files = sorted(r["filename"] for r in rec_rows)[:args.sample]
        print("   sample: scoring %d fixed images with TTA..." % len(sample_files), flush=True)
        rows, _ = score(model, os.path.join(scratch, "Testing"), True, sample_files, args.workers)
        smp = compare(rows, recorded_csv, classes)
        entry["gates"]["sample"] = smp
        print("      max|dp| %.2e | mean|dp| %.2e | pred mismatches %d"
              % (smp["max_abs_prob_diff"], smp["mean_abs_prob_diff"],
                 smp["n_pred_label_mismatch"]), flush=True)
        if smp["n_pred_label_mismatch"] or smp["max_abs_prob_diff"] > PROB_TOL:
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = (
                "sample verification failed: %d predicted-label mismatches, max "
                "probability difference %.3e exceeds %.0e"
                % (smp["n_pred_label_mismatch"], smp["max_abs_prob_diff"], PROB_TOL))
            report["seeds"][seed] = entry
            print("   REJECTED - %s (full set not attempted)" % entry["rejection_reason"],
                  flush=True)
            continue

        # gate 6 - full clean test set
        print("   full set: scoring 848 images with TTA...", flush=True)
        rows, _ = score(model, os.path.join(scratch, "Testing"), True, None, args.workers)
        full = compare(rows, recorded_csv, classes)
        entry["gates"]["full_test"] = full
        print("      n=%d | acc recomputed %.4f vs recorded %.4f (diff %.4f pp)"
              % (full["n_compared"], full["accuracy_recomputed"],
                 full["accuracy_recorded"], full["accuracy_diff_pp"]), flush=True)
        print("      max|dp| %.2e | mean|dp| %.2e | filename mismatches %d | "
              "true-label mismatches %d | pred mismatches %d"
              % (full["max_abs_prob_diff"], full["mean_abs_prob_diff"],
                 full["n_filename_mismatch"], full["n_true_label_mismatch"],
                 full["n_pred_label_mismatch"]), flush=True)

        reasons = []
        if full["n_filename_mismatch"]:
            reasons.append("filename set differs")
        if full["n_true_label_mismatch"]:
            reasons.append("true labels differ")
        if full["accuracy_diff_pp"] > TOL_PP:
            reasons.append("accuracy differs by %.4f pp (> %.2f)"
                           % (full["accuracy_diff_pp"], TOL_PP))
        if full["n_pred_label_mismatch"]:
            reasons.append("%d predicted labels differ" % full["n_pred_label_mismatch"])
        if full["max_abs_prob_diff"] > PROB_TOL:
            reasons.append("max probability difference %.3e exceeds %.0e"
                           % (full["max_abs_prob_diff"], PROB_TOL))

        if reasons:
            entry["decision"] = "REJECTED"
            entry["rejection_reason"] = "; ".join(reasons)
            print("   REJECTED - %s" % entry["rejection_reason"], flush=True)
        else:
            dst_dir = os.path.join(imp_root, "clean", "full_fusion_v2", "seed_%d" % seed)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, "checkpoint.pth")
            shutil.copy2(src, dst)
            dst_hash = sha256(dst)
            entry["decision"] = "ACCEPTED"
            entry["import"] = {
                "source_path": src, "source_sha256": cont["sha256"],
                "destination_path": os.path.relpath(dst, root).replace("\\", "/"),
                "destination_sha256": dst_hash,
                "bytes_match": os.path.getsize(dst) == cont["bytes"],
                "hash_match": dst_hash == cont["sha256"],
                "renamed_from": cont["filename"],
            }
            report["imports"].append(entry["import"])
            print("   ACCEPTED - imported to %s (hash preserved: %s)"
                  % (entry["import"]["destination_path"], entry["import"]["hash_match"]),
                  flush=True)
        report["seeds"][seed] = entry

    accepted = [s for s, e in report["seeds"].items() if e["decision"] == "ACCEPTED"]
    report["n_accepted"] = len(accepted)
    report["n_rejected"] = len(SEEDS) - len(accepted)
    report["all_accepted"] = len(accepted) == len(SEEDS)
    report["acceptance_status"] = "PASS" if report["all_accepted"] else "FAIL"

    with open(os.path.join(out_root, "import_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_root, "CHECKPOINT_RECOVERY.md"), "w", encoding="utf-8") as f:
        f.write(render_md(report))

    print("\n%d/%d accepted -> %s" % (len(accepted), len(SEEDS), report["acceptance_status"]))
    print("wrote %s" % os.path.relpath(out_root, root))
    return 0 if report["all_accepted"] else 1


def render_md(r: dict) -> str:
    L = ["# Phase 3 Addendum - Recovered Checkpoint Verification", "",
         "Generated by `src/publication/verify_recovered_checkpoints.py` at %s."
         % r["generated_utc"], "",
         "**%d of 3 checkpoints accepted. Status: %s.**"
         % (r["n_accepted"], r["acceptance_status"]), "",
         "A checkpoint is accepted only if it loads strictly against the reference "
         "architecture, keeps a learned input-dependent gate, and reproduces the stored "
         "predictions. A path is a claim; only reproduction is evidence.", "",
         "## Clean test set reconstruction", "",
         "The clean-split images were built on Kaggle and never downloaded, so the test "
         "set was rebuilt by manifest lookup against `dataset/`. `dataset_clean/` was "
         "read, never written.", "",
         "| Field | Value |", "|---|---|",
         "| manifest | `%s` |" % r["clean_test_reconstruction"]["manifest"],
         "| manifest SHA-256 | `%s` |" % r["clean_test_reconstruction"]["manifest_sha256"][:16],
         "| images in manifest (Testing) | %d |" % r["clean_test_reconstruction"]["n_testing_in_manifest"],
         "| images reconstructed | %d |" % r["clean_test_reconstruction"]["n_reconstructed"], "",
         "## Verdicts", "", "| Seed | Decision | Accuracy (recomputed vs recorded) | max abs prob diff |",
         "|---|---|---|---|"]
    for seed, e in r["seeds"].items():
        f = e["gates"].get("full_test", {})
        acc = ("%.4f vs %.4f (%.4f pp)" % (f["accuracy_recomputed"], f["accuracy_recorded"],
                                           f["accuracy_diff_pp"])) if f else "-"
        L.append("| %s | **%s** | %s | %s |"
                 % (seed, e["decision"], acc,
                    ("%.2e" % f["max_abs_prob_diff"]) if f else "-"))
    L.append("")
    for seed, e in r["seeds"].items():
        L += ["## Seed %s - %s" % (seed, e["decision"]), ""]
        c = e["gates"].get("container", {})
        if c:
            L += ["**Container**", "",
                  "- file: `%s` (the task names `.pth`; the staged file is `%s` - the "
                  "extension differs, the content is a torch archive)"
                  % (c["filename"], c["filename"]),
                  "- bytes: %d" % c["bytes"], "- SHA-256: `%s`" % c["sha256"],
                  "- magic: `%s` | torch archive: %s | entries: %s"
                  % (c["magic_hex"], c.get("is_torch_archive"), c.get("archive_entries")), ""]
        ld = e["gates"].get("load", {})
        if ld:
            L += ["**Strict load**", "",
                  "- container type: `%s`" % ld["container_type"],
                  "- top-level keys: %s" % ld["top_level_keys"],
                  "- state-dict keys: %d (reference: %d)"
                  % (ld["state_dict_key_count"], ld["reference_key_count"]),
                  "- missing: %d | unexpected: %d | shape mismatches: %d"
                  % (ld["n_missing"], ld["n_unexpected"], ld["n_shape_mismatch"]),
                  "- loaded strictly: **%s**" % ld["loaded"], ""]
        a = e["gates"].get("architecture_and_gate", {})
        if a:
            L += ["**Architecture and gate**", "",
                  "- gated: %s | multi-scale: %s | cnn_proj in_features: %d (expected %d)"
                  % (a["gated_flag"], a["multiscale_flag"], a["cnn_proj_in_features"],
                     a["cnn_proj_in_features_expected"]),
                  "- gate module: `%s` %s, %s parameters"
                  % (a["gate_module_type"], a["gate_layer_types"],
                     format(a["gate_parameter_count"], ",")),
                  "- gate probe: cnn mass mean %.4f, s.d. %.4f across 8 different inputs"
                  % (a["gate_cnn_mean_on_probe"], a["gate_cnn_std_on_probe"]),
                  "- gate varies across inputs: **%s** (a constant gate would load "
                  "cleanly yet invalidate every gate conclusion, so it is probed)"
                  % a["gate_varies_across_inputs"],
                  "- rows sum to 1: %s | gate intact: **%s**"
                  % (a["gate_rows_sum_to_one"], a["gate_intact"]), ""]
        m = e["gates"].get("metadata", {})
        if m:
            L += ["**Metadata cross-check**", "",
                  "- seed recorded in metrics.json: %s; seed from path: %s; match: **%s**"
                  % (m["recorded_seed"], m["seed_from_path"], m["seed_matches"]),
                  "- parameter count matches the recorded run: **%s**"
                  % m["params_total_matches"], ""]
        for label, key in (("Fixed sample", "sample"), ("Full clean test set", "full_test")):
            g = e["gates"].get(key)
            if not g:
                continue
            L += ["**%s**" % label, "",
                  "- images compared: %d" % g["n_compared"],
                  "- filename mismatches: %d | true-label mismatches: %d | "
                  "predicted-label mismatches: %d"
                  % (g["n_filename_mismatch"], g["n_true_label_mismatch"],
                     g["n_pred_label_mismatch"]),
                  "- max absolute probability difference: %.3e" % g["max_abs_prob_diff"],
                  "- mean absolute probability difference: %.3e" % g["mean_abs_prob_diff"],
                  "- accuracy recomputed %.6f vs recorded %.6f (%.4f pp)"
                  % (g["accuracy_recomputed"], g["accuracy_recorded"], g["accuracy_diff_pp"]), ""]
        if e["decision"] == "REJECTED":
            L += ["**Rejected:** %s" % e["rejection_reason"],
                  "", "The file remains in staging. It was not imported.", ""]
        elif e.get("import"):
            i = e["import"]
            L += ["**Imported**", "",
                  "- source: `%s`" % i["source_path"],
                  "- destination: `%s`" % i["destination_path"],
                  "- source SHA-256: `%s`" % i["source_sha256"],
                  "- destination SHA-256: `%s`" % i["destination_sha256"],
                  "- byte-identical copy: **%s**" % i["hash_match"], ""]
    L += ["## Scope", "",
          "- `runs_v2_clean/` was read only. No checkpoint was copied into it.",
          "- `dataset_clean/` was read only. The reconstructed test set lives in the "
          "session scratchpad.",
          "- `src/fusion_v2.py` and `src/train_ablation.py` were imported, never modified.",
          "- No gate value was hardcoded anywhere; the gate was probed as a learned module.",
          "- No training was launched and nothing was uploaded.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RecoveryError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
