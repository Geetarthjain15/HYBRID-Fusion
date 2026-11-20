"""
Phase 3A - archive verification and controlled import.

Takes the two Kaggle result archives from a claim to evidence, or refuses them.
Nothing is imported on the strength of a filename or a log line.

Order, cheapest first, halting at the first failure:

  1. CHECKSUM    the .sha256 sidecar is read as data; the digest is recomputed
                 independently and compared. No extraction before this passes.
  2. SAFETY      every member name is checked for absolute paths, traversal,
                 symlinks, device files and duplicates. A zip that can write
                 outside its extraction root is rejected before a byte is read.
  3. STAGE       extraction into an isolated staging directory, never into an
                 existing experiment directory.
  4. CONTRACT    per-run artifact requirements from session_artifact_contract,
                 which derives them from the experimental DESIGN - so a gateless
                 model is required NOT to have gate files, rather than merely
                 excused for lacking them.
  5. RECOMPUTE   accuracy and macro-F1 recomputed from the prediction CSVs and
                 compared against the stored metrics. A summary number is never
                 trusted on its own.
  6. IMPORT      only fully accepted runs, into new directories, refusing to
                 overwrite. An existing destination is verified by hash and
                 reported, never replaced.

    python src/publication/verify_session_archives.py --project-root . \\
        --downloads kaggle/downloads --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from session_artifact_contract import (VARIANT_DESIGN, design, required_artifacts,
                                       forbidden_artifacts, expected_cells,
                                       EXPECTED_ROWS, CLASSES, PROB_SUM_TOL,
                                       METRIC_TOL_PP, CLEAN_TEST_N)

ARCHIVES = {
    "session_1": {"zip": "fusionv2_session1_results.zip",
                  "expect": ["no_gate", "no_multiscale", "no_auxiliary_loss"]},
    "session_2": {"zip": "fusionv2_session2_results.zip",
                  "expect": ["no_ema", "cnn_branch_only", "vit_branch_only", "no_tta"]},
}
RECOVERED_CKPT_MANIFEST = "publication_results/ablations/session2/checkpoint_manifest.json"
RUN_RE = re.compile(r"^clean/(?P<variant>[A-Za-z0-9_]+)/seed_(?P<seed>\d+)/(?P<file>[^/]+)$")


class VerifyError(Exception):
    """A Phase 3A stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def banner(m):
    print("\n" + "=" * 72 + "\n  %s\n" % m + "=" * 72, flush=True)


def stamp(m):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), m), flush=True)


# ------------------------------------------------------------- 1. checksum ---

def read_sidecar(downloads: str, zip_name: str) -> dict:
    """The sidecar is DATA. Accept either spelling; parse, never trust."""
    cands = [zip_name + ".sha256", zip_name + ".sha256.txt"]
    for c in cands:
        p = os.path.join(downloads, c)
        if os.path.exists(p):
            raw = open(p, encoding="utf-8").read().strip()
            m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", raw)
            if not m:
                raise VerifyError("%s does not parse as '<64-hex>  <filename>': %r"
                                  % (c, raw[:120]))
            digest, named = m.group(1).lower(), os.path.basename(m.group(2))
            if named != zip_name:
                raise VerifyError("%s names %r but should name %r" % (c, named, zip_name))
            return {"sidecar": c, "recorded_sha256": digest, "named_file": named,
                    "raw": raw}
    raise VerifyError("no checksum sidecar found for %s (looked for %s)"
                      % (zip_name, ", ".join(cands)))


def verify_checksum(downloads: str, zip_name: str) -> dict:
    side = read_sidecar(downloads, zip_name)
    p = os.path.join(downloads, zip_name)
    if not os.path.exists(p):
        raise VerifyError("archive absent: %s" % p)
    stamp("  hashing %s (%.2f GB)" % (zip_name, os.path.getsize(p) / 2 ** 30))
    computed = sha256(p)
    ok = computed == side["recorded_sha256"]
    stamp("  recorded  %s" % side["recorded_sha256"])
    stamp("  computed  %s  -> %s" % (computed, "MATCH" if ok else "MISMATCH"))
    if not ok:
        raise VerifyError("checksum mismatch for %s: recorded %s, computed %s"
                          % (zip_name, side["recorded_sha256"], computed))
    return {**side, "archive": zip_name, "bytes": os.path.getsize(p),
            "computed_sha256": computed, "match": True}


# --------------------------------------------------------------- 2. safety ---

def inspect_members(zip_path: str) -> dict:
    """Reject anything that could write outside the extraction root."""
    problems, names = [], []
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
    for i in infos:
        n = i.filename
        names.append(n)
        if n.startswith("/") or n.startswith("\\") or re.match(r"^[A-Za-z]:", n):
            problems.append("absolute path: %s" % n)
        if ".." in n.replace("\\", "/").split("/"):
            problems.append("path traversal: %s" % n)
        # high 16 bits of external_attr hold the unix mode; 0xA000 == symlink
        if (i.external_attr >> 16) & 0xF000 == 0xA000:
            problems.append("symlink member: %s" % n)
        if i.create_system == 3 and ((i.external_attr >> 16) & 0xF000) not in (
                0x8000, 0x4000, 0):
            problems.append("non-regular member: %s" % n)
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append("duplicate members: %s" % dupes[:5])
    return {"n_members": len(names), "problems": problems,
            "safe": not problems, "names": names}


# ------------------------------------------------------- 4/5. per-run checks ---

def load_predictions(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])
    if not rows:
        raise VerifyError("%s: empty" % path)
    prob_cols = [c for c in rows[0] if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols]
    if classes != sorted(classes):
        raise VerifyError("%s: probability columns not in sorted class order: %s"
                          % (path, classes))
    if classes != CLASSES:
        raise VerifyError("%s: class set %s differs from the documented %s"
                          % (path, classes, CLASSES))
    files = [r["filename"] for r in rows]
    if len(set(files)) != len(files):
        raise VerifyError("%s: duplicate filenames" % path)
    labels = {r["true_label"] for r in rows} | {r["pred_label"] for r in rows}
    unknown = sorted(labels - set(classes))
    if unknown:
        raise VerifyError("%s: labels outside the class set: %s" % (path, unknown))
    P = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
    if not np.isfinite(P).all():
        raise VerifyError("%s: non-finite probabilities" % path)
    if P.min() < 0.0 or P.max() > 1.0:
        raise VerifyError("%s: probabilities outside [0,1] (%.6f..%.6f)"
                          % (path, P.min(), P.max()))
    dev = float(np.abs(P.sum(1) - 1.0).max())
    if dev > PROB_SUM_TOL:
        raise VerifyError("%s: probability rows deviate from 1 by %.2e (tol %.0e)"
                          % (path, dev, PROB_SUM_TOL))
    y = np.array([classes.index(r["true_label"]) for r in rows])
    pred = np.array([classes.index(r["pred_label"]) for r in rows])
    argmax = P.argmax(1)
    strict = 0
    for i in np.nonzero(pred != argmax)[0]:
        top2 = np.sort(P[i])[-2:]
        if top2[1] - top2[0] > 1e-9:
            strict += 1
    if strict:
        raise VerifyError("%s: pred_label disagrees with argmax on %d strictly "
                          "ranked rows" % (path, strict))
    return {"files": files, "y": y, "pred": pred, "P": P, "classes": classes,
            "n": len(rows), "max_prob_sum_deviation": dev}


def accuracy_macro_f1(y, pred, k: int):
    acc = float((pred == y).mean())
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, float(np.mean(f1s))


def validate_run(run_dir: str, variant: str, seed: int, recovered: dict) -> dict:
    d = design(variant)
    res = {"variant": variant, "seed": seed, "mode": d["mode"],
           "has_gate": d["has_gate"], "run_dir": run_dir,
           "required": required_artifacts(variant),
           "forbidden": forbidden_artifacts(variant),
           "missing": [], "unexpected_present": [], "problems": [],
           "files": {}, "checks": {}}

    for name in res["required"]:
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            res["missing"].append(name)
        else:
            res["files"][name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    for name in res["forbidden"]:
        if os.path.exists(os.path.join(run_dir, name)):
            res["unexpected_present"].append(name)

    if res["missing"] or res["unexpected_present"]:
        res["accepted"] = False
        bits = []
        if res["missing"]:
            bits.append("missing: %s" % ", ".join(res["missing"]))
        if res["unexpected_present"]:
            bits.append("must not exist: %s" % ", ".join(res["unexpected_present"]))
        res["reason"] = "; ".join(bits)
        return res

    # --- config / seed / split ---------------------------------------------
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    met = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
    res["checks"]["seed_matches"] = cfg.get("seed") == seed and met.get("seed") == seed
    res["checks"]["variant_matches"] = cfg.get("variant") == variant
    res["checks"]["split_is_clean"] = cfg.get("split") == "clean"
    if not all(res["checks"][k] for k in
               ("seed_matches", "variant_matches", "split_is_clean")):
        res["accepted"] = False
        res["reason"] = ("config disagrees with path: variant=%r seed=%r split=%r"
                         % (cfg.get("variant"), cfg.get("seed"), cfg.get("split")))
        return res

    # --- the single intended difference from the reference -----------------
    rc = cfg.get("resolved_config", {})
    if variant == "no_ema":
        res["checks"]["ema_disabled"] = rc.get("ema") is False
        res["checks"]["tta_unchanged"] = rc.get("tta") is True
        res["checks"]["gated_unchanged"] = rc.get("gated") is True
        res["checks"]["multiscale_unchanged"] = rc.get("multiscale") is True
        res["checks"]["aux_unchanged"] = rc.get("aux_weight") == 0.3
        # Evaluation must have used the live weights, not an EMA shadow.
        of = met.get("overfitting", {})
        res["checks"]["checkpoint_is_best_epoch"] = of.get("checkpoint_is_best_epoch") is True
        res["checks"]["no_ema_shadow_used"] = rc.get("ema") is False
    if variant == "no_tta":
        res["checks"]["tta_disabled"] = met.get("tta_used") is False
        res["checks"]["did_not_train"] = met.get("trained") is False
        res["checks"]["weights_not_updated"] = met.get("weights_updated") is False
        res["checks"]["no_checkpoint_written"] = met.get("checkpoint_written") is False
        recorded = (open(os.path.join(run_dir, "source_checkpoint_sha256.txt"),
                         encoding="utf-8").read().split() or [""])[0].lower()
        expected = recovered.get(seed, {}).get("sha256")
        res["source_checkpoint_sha256"] = recorded
        res["expected_source_checkpoint_sha256"] = expected
        res["checks"]["source_checkpoint_matches_verified"] = (
            expected is not None and recorded == expected
            and met.get("source_checkpoint_sha256", "").lower() == expected)

    failed = [k for k, v in res["checks"].items() if v is False]
    if failed:
        res["accepted"] = False
        res["reason"] = "design checks failed: %s" % ", ".join(failed)
        return res

    # --- predictions and recomputation -------------------------------------
    try:
        pt = load_predictions(os.path.join(run_dir, "preds_test.csv"))
        po = load_predictions(os.path.join(run_dir, "preds_dataset_ood.csv"))
    except VerifyError as e:
        res["accepted"] = False
        res["reason"] = str(e)
        return res

    if pt["n"] != EXPECTED_ROWS["preds_test.csv"]:
        res["accepted"] = False
        res["reason"] = ("clean test set has %d rows, contract requires %d"
                         % (pt["n"], EXPECTED_ROWS["preds_test.csv"]))
        return res
    if po["n"] != EXPECTED_ROWS["preds_dataset_ood.csv"]:
        res["accepted"] = False
        res["reason"] = ("external set has %d rows, contract requires %d"
                         % (po["n"], EXPECTED_ROWS["preds_dataset_ood.csv"]))
        return res

    acc, f1 = accuracy_macro_f1(pt["y"], pt["pred"], len(pt["classes"]))
    ood_acc, _ = accuracy_macro_f1(po["y"], po["pred"], len(po["classes"]))
    stored = met.get("test_acc")
    res["recomputed"] = {"clean_test_accuracy": acc, "clean_test_macro_f1": f1,
                         "external_accuracy": ood_acc,
                         "stored_test_acc": stored,
                         "max_prob_sum_deviation_test": pt["max_prob_sum_deviation"]}
    if stored is None:
        res["accepted"] = False
        res["reason"] = "metrics.json has no test_acc to compare against"
        return res
    diff_pp = abs(100 * acc - 100 * stored)
    res["recomputed"]["abs_diff_pp"] = diff_pp
    res["checks"]["metrics_reproduce"] = diff_pp <= METRIC_TOL_PP
    if diff_pp > METRIC_TOL_PP:
        res["accepted"] = False
        res["reason"] = ("recomputed accuracy %.6f differs from stored %.6f by "
                         "%.4f pp (tolerance %.2f pp)" % (acc, stored, diff_pp, METRIC_TOL_PP))
        return res

    # gate files, where the design says they must exist
    if d["has_gate"]:
        for g in ("gates_test.csv", "gates_dataset_ood.csv"):
            with open(os.path.join(run_dir, g), encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) != EXPECTED_ROWS[g]:
                res["accepted"] = False
                res["reason"] = "%s has %d rows, expected %d" % (g, len(rows), EXPECTED_ROWS[g])
                return res
            vals = np.array([float(r["gate_cnn"]) for r in rows])
            if not np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
                res["accepted"] = False
                res["reason"] = "%s: gate mass outside [0,1]" % g
                return res
            res.setdefault("gate", {})[g] = {"n": len(rows),
                                             "gate_cnn_mean": float(vals.mean())}

    res["accepted"] = True
    res["reason"] = "all contract, design, schema and recomputation checks passed"
    return res


# -------------------------------------------------------------- 6. import ---

def import_run(run_dir: str, dest: str, entry: dict) -> dict:
    if os.path.exists(dest) and os.listdir(dest):
        # Never replace. Verify identity and report.
        same, differing = True, []
        for name in os.listdir(run_dir):
            a, b = os.path.join(run_dir, name), os.path.join(dest, name)
            if not os.path.exists(b) or sha256(a) != sha256(b):
                same = False
                differing.append(name)
        return {"action": "skipped_existing", "identical": same,
                "differing_files": differing[:5], "destination": dest}
    os.makedirs(dest, exist_ok=True)
    copied = {}
    for name in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest, name)
        shutil.copy2(src, dst)
        h = sha256(dst)
        if h != entry["files"].get(name, {}).get("sha256", h):
            raise VerifyError("copy changed %s" % name)
        copied[name] = {"bytes": os.path.getsize(dst), "sha256": h}
    return {"action": "imported", "destination": dest, "files": copied,
            "n_files": len(copied)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--downloads", default="kaggle/downloads")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--out-root", default="publication_results/ablations/phase3a")
    ap.add_argument("--import-root", default="publication_runs/ablations/clean")
    ap.add_argument("--do-import", action="store_true")
    ap.add_argument("--keep-stage", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    downloads = os.path.join(root, args.downloads)
    out_root = os.path.join(root, args.out_root)
    for r in (args.out_root, args.import_root):
        if not r.replace("\\", "/").startswith("publication_"):
            raise VerifyError("refusing to write outside publication_*: %s" % r)
    os.makedirs(out_root, exist_ok=True)
    stage = args.stage or os.path.join(os.environ.get("TEMP", "/tmp"), "phase3a_stage")

    report = {"phase": "3A - archive verification and controlled import",
              "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
              "training_performed": False, "archives": {}, "safety": {},
              "runs": [], "imports": [], "stage_dir": stage}

    recovered = {}
    rp = os.path.join(root, RECOVERED_CKPT_MANIFEST)
    if os.path.exists(rp):
        for c in json.load(open(rp, encoding="utf-8"))["checkpoints"]:
            recovered[c["seed"]] = c
    report["recovered_checkpoints_known"] = sorted(recovered)

    banner("STEP 1 - CHECKSUM VERIFICATION")
    for tag, spec in ARCHIVES.items():
        stamp("%s: %s" % (tag, spec["zip"]))
        report["archives"][tag] = verify_checksum(downloads, spec["zip"])

    banner("STEP 2 - SAFE ARCHIVE INSPECTION")
    for tag, spec in ARCHIVES.items():
        info = inspect_members(os.path.join(downloads, spec["zip"]))
        report["safety"][tag] = {k: v for k, v in info.items() if k != "names"}
        stamp("%s: %d members, safe=%s" % (tag, info["n_members"], info["safe"]))
        if not info["safe"]:
            for p in info["problems"][:10]:
                stamp("    %s" % p)
            raise VerifyError("%s contains unsafe members" % tag)

    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    stamp("staging root: %s (isolated; no existing experiment directory touched)" % stage)
    for tag, spec in ARCHIVES.items():
        d = os.path.join(stage, tag)
        os.makedirs(d, exist_ok=True)
        stamp("extracting %s ..." % spec["zip"])
        with zipfile.ZipFile(os.path.join(downloads, spec["zip"])) as z:
            for m in z.namelist():
                target = os.path.realpath(os.path.join(d, m))
                if not target.startswith(os.path.realpath(d)):
                    raise VerifyError("member escapes staging root: %s" % m)
            z.extractall(d)
        stamp("  extracted into %s" % d)

    banner("STEP 4/5 - CONTRACT AND RECOMPUTATION")
    expected = expected_cells()
    for tag, spec in ARCHIVES.items():
        base = os.path.join(stage, tag, "clean")
        found = []
        for variant in sorted(os.listdir(base)):
            for seed_dir in sorted(os.listdir(os.path.join(base, variant))):
                seed = int(seed_dir.split("_")[1])
                found.append((variant, seed))
        stamp("%s: %d run directories" % (tag, len(found)))
        key = "session_1_training" if tag == "session_1" else None
        want = set(expected["session_1_training"]) if tag == "session_1" else \
            set(expected["session_2_training"]) | set(expected["session_2_eval_only"])
        extra = sorted(set(found) - want)
        absent = sorted(want - set(found))
        report["safety"][tag]["unexpected_runs"] = extra
        report["safety"][tag]["absent_runs"] = absent
        if extra:
            raise VerifyError("%s contains runs outside its documented matrix: %s"
                              % (tag, extra))
        if absent:
            raise VerifyError("%s is missing documented runs: %s" % (tag, absent))

        for variant, seed in found:
            rd = os.path.join(base, variant, "seed_%d" % seed)
            r = validate_run(rd, variant, seed, recovered)
            r["source_archive"] = spec["zip"]
            r["source_member_prefix"] = "clean/%s/seed_%d/" % (variant, seed)
            r["session"] = tag
            report["runs"].append(r)
            stamp("  %-9s %-18s seed %d  %s  %s"
                  % (tag[-1], variant, seed, "ACCEPT" if r["accepted"] else "REJECT",
                     (("acc %.4f (stored %.4f, %.4f pp)"
                       % (r["recomputed"]["clean_test_accuracy"],
                          r["recomputed"]["stored_test_acc"],
                          r["recomputed"]["abs_diff_pp"]))
                      if r.get("recomputed") else r["reason"][:70])))

    acc_runs = [r for r in report["runs"] if r["accepted"]]
    rej = [r for r in report["runs"] if not r["accepted"]]
    report["n_accepted"] = len(acc_runs)
    report["n_rejected"] = len(rej)

    if args.do_import and not rej:
        banner("STEP 6 - CONTROLLED IMPORT")
        for r in acc_runs:
            dest = os.path.join(root, args.import_root, r["variant"],
                                "seed_%d" % r["seed"])
            imp = import_run(r["run_dir"], dest, r)
            imp.update({"variant": r["variant"], "seed": r["seed"],
                        "mode": r["mode"], "session": r["session"],
                        "source_archive": r["source_archive"],
                        "source_member_prefix": r["source_member_prefix"],
                        "source_archive_sha256":
                            report["archives"][r["session"]]["computed_sha256"],
                        "validation_status": "accepted"})
            report["imports"].append(imp)
            stamp("  %-18s seed %d -> %s (%s)"
                  % (r["variant"], r["seed"], os.path.relpath(dest, root), imp["action"]))
    elif rej:
        stamp("import skipped: %d run(s) rejected" % len(rej))

    report["acceptance_status"] = "PASS" if (not rej and report["n_accepted"] == 21) else "FAIL"
    with open(os.path.join(out_root, "archive_verification.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_root, "import_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_utc": report["generated_utc"],
                   "archives": report["archives"], "imports": report["imports"],
                   "n_imported": len(report["imports"])}, f, indent=2)
    with open(os.path.join(out_root, "ARCHIVE_VERIFICATION.md"), "w",
              encoding="utf-8") as f:
        f.write(render_md(report))

    if not args.keep_stage and os.path.isdir(stage):
        shutil.rmtree(stage)
        stamp("staging removed")

    banner("PHASE 3A: %d accepted, %d rejected -> %s"
           % (report["n_accepted"], report["n_rejected"], report["acceptance_status"]))
    for r in rej:
        stamp("  REJECT %s seed %d: %s" % (r["variant"], r["seed"], r["reason"]))
    return 0 if report["acceptance_status"] == "PASS" else 1


def render_md(r: dict) -> str:
    L = ["# Phase 3A - Archive Verification", "",
         "Generated by `src/publication/verify_session_archives.py` at %s."
         % r["generated_utc"], "",
         "**%d accepted, %d rejected. Status: %s.**"
         % (r["n_accepted"], r["n_rejected"], r["acceptance_status"]), "",
         "No training was performed. Archives were checksum-verified before any "
         "extraction, and extraction went to an isolated staging directory.", "",
         "## Checksums", "", "| Archive | Bytes | Recorded | Computed | Match |",
         "|---|---:|---|---|---|"]
    for tag, a in r["archives"].items():
        L.append("| `%s` | %d | `%s` | `%s` | %s |"
                 % (a["archive"], a["bytes"], a["recorded_sha256"][:24],
                    a["computed_sha256"][:24], "YES" if a["match"] else "**NO**"))
    L += ["", "## Member safety", "", "| Archive | Members | Safe | Problems |",
          "|---|---:|---|---|"]
    for tag, s in r["safety"].items():
        L.append("| %s | %d | %s | %s |" % (tag, s["n_members"], s["safe"],
                                            "; ".join(s["problems"]) or "none"))
    L += ["", "## Runs", "",
          "| Session | Variant | Seed | Mode | Gate | Verdict | Recomputed acc | Stored | diff (pp) |",
          "|---|---|---:|---|---|---|---:|---:|---:|"]
    for run in r["runs"]:
        rec = run.get("recomputed", {})
        L.append("| %s | `%s` | %d | %s | %s | %s | %s | %s | %s |"
                 % (run["session"], run["variant"], run["seed"], run["mode"],
                    run["has_gate"], "ACCEPT" if run["accepted"] else "**REJECT**",
                    ("%.4f" % rec["clean_test_accuracy"]) if rec else "-",
                    ("%.4f" % rec["stored_test_acc"]) if rec else "-",
                    ("%.4f" % rec["abs_diff_pp"]) if rec else "-"))
    L.append("")
    rej = [x for x in r["runs"] if not x["accepted"]]
    if rej:
        L += ["### Rejections", ""]
        L += ["- `%s` seed %d: %s" % (x["variant"], x["seed"], x["reason"]) for x in rej]
        L.append("")
    L += ["## Contract applied", "",
          "Requirements come from the experimental design, not from what is on "
          "disk. A gateless model is required NOT to have gate files; their "
          "presence would mean a gate ran.", "",
          "| Variant | Mode | Gate | Required | Forbidden |", "|---|---|---|---:|---|"]
    for v, d in VARIANT_DESIGN.items():
        L.append("| `%s` | %s | %s | %d | %s |"
                 % (v, d["mode"], d["has_gate"], len(required_artifacts(v)),
                    ", ".join(forbidden_artifacts(v)) or "none"))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except VerifyError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
