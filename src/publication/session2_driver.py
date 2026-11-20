"""
Session 2 driver - preflight, smoke test, nine training cells, three evaluations.

Order is enforced in code rather than left to a human following a plan:

  1. environment      CUDA, GPU model, dependency versions
  2. code integrity   SHA-256 of the uploaded package
  3. split            rebuilt and fingerprinted BEFORE any gradient step
  4. checkpoints      every recovered checkpoint hash verified BEFORE no_tta
  5. tests            all 17 Phase 3 tests
  6. smoke            one short run in an isolated directory
  7. training grid    nine cells, halting on the first failure
  8. evaluation       three no_tta cells against the verified checkpoints
  9. packaging        zip plus a SHA-256 manifest of every artifact

no_tta is deliberately last: it depends on checkpoint verification, and running
it early would mean a hash failure surfaced after seven hours of training rather
than before any of it.

    python src/publication/session2_driver.py --project-root . --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _SRC)

from session2_config import (SESSION, SPLIT, SEEDS, APPROVED_VARIANTS,
                             EVAL_ONLY_VARIANTS, EXPECTED, RECOVERED_CHECKPOINTS,
                             COST_MIN, training_cells, eval_cells, all_cells)
from validate_run_outputs import sha256
import validate_run_outputs as V


class SessionError(Exception):
    """A Session 2 stop condition was hit."""


def banner(msg):
    print("\n" + "=" * 72 + "\n  %s\n" % msg + "=" * 72, flush=True)


def stamp(msg):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ preflight ---

def environment_report() -> dict:
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    stamp("python %s | torch %s | transformers %s" %
          (info["python"], info["torch"], info["transformers"]))
    stamp("cuda %s | gpu %s (x%d)" % (info["cuda_available"], info["gpu"], info["gpu_count"]))
    if not info["cuda_available"]:
        raise SessionError("no CUDA device; Session 2 needs a GPU")
    return info


def verify_code_package(root: str) -> dict:
    """Hash the code actually in use, so results trace to a specific package."""
    files = {}
    for rel in sorted(os.listdir(os.path.join(root, "src", "publication"))):
        if rel.endswith(".py"):
            p = os.path.join(root, "src", "publication", rel)
            files["src/publication/" + rel] = sha256(p)
    for rel in ("fusion_v2.py", "train_ablation.py", "make_clean_split.py",
                "seeding.py", "hybrid_model.py"):
        p = os.path.join(root, "src", rel)
        if os.path.exists(p):
            files["src/" + rel] = sha256(p)
    combined = hashlib.sha256(
        "".join("%s:%s" % (k, v) for k, v in sorted(files.items())).encode()).hexdigest()
    stamp("code package: %d files, combined digest %s" % (len(files), combined[:16]))
    return {"files": files, "combined_sha256": combined, "n_files": len(files)}


def rebuild_and_verify_split(root: str, data_root: str, python: str) -> dict:
    out_dir = os.path.join(root, "dataset_clean")
    report_p = os.path.join(out_dir, "split_report.json")
    manifest_p = os.path.join(out_dir, "split_manifest.json")

    if not os.path.exists(report_p):
        src = os.path.join(data_root, "dataset")
        if not os.path.isdir(os.path.join(src, "Training")):
            raise SessionError("source dataset not found at %s" % src)
        stamp("rebuilding the duplicate-clean split (seed %d, hamming %d)"
              % (EXPECTED["split_seed"], EXPECTED["hamming_threshold"]))
        r = subprocess.run([python, os.path.join(root, "src", "make_clean_split.py"),
                            "--src", src, "--out", out_dir,
                            "--seed", str(EXPECTED["split_seed"]),
                            "--hamming", str(EXPECTED["hamming_threshold"])], cwd=root)
        if r.returncode != 0:
            raise SessionError("make_clean_split.py failed with rc=%d" % r.returncode)
    else:
        stamp("dataset_clean present; verifying rather than rebuilding")

    rep = json.load(open(report_p, encoding="utf-8"))
    checks, failures = {}, []

    def expect(name, got, want):
        ok = got == want
        checks[name] = {"expected": want, "observed": got, "pass": ok}
        if not ok:
            failures.append("%s: expected %r, observed %r" % (name, want, got))

    expect("n_pooled", rep.get("n_pooled"), EXPECTED["n_pooled"])
    expect("n_clusters", rep.get("n_clusters"), EXPECTED["n_clusters"])
    expect("n_train", rep.get("n_train"), EXPECTED["n_train"])
    expect("n_test", rep.get("n_test"), EXPECTED["n_test"])
    expect("hamming_threshold", rep.get("hamming_threshold"), EXPECTED["hamming_threshold"])
    expect("class_counts", rep.get("counts"), EXPECTED["class_counts"])
    expect("split_manifest_sha256",
           sha256(manifest_p) if os.path.exists(manifest_p) else None,
           EXPECTED["split_manifest_sha256"])
    for split_name, key in (("Training", "n_train"), ("Testing", "n_test")):
        d = os.path.join(out_dir, split_name)
        n = sum(len(fs) for _, _, fs in os.walk(d)) if os.path.isdir(d) else 0
        expect("images_on_disk_%s" % split_name.lower(), n, EXPECTED[key])

    # The model trains on 4080 after the seeded 15% carve-out, not on 4800.
    n_total = rep.get("n_train", 0)
    n_val = int(0.15 * n_total)
    expect("n_val_after_carve_out", n_val, EXPECTED["n_val"])
    expect("n_model_train_after_carve_out", n_total - n_val, EXPECTED["n_model_train"])

    for name, c in checks.items():
        stamp("  %-32s %s" % (name, "PASS" if c["pass"] else
                              "FAIL expected=%r observed=%r" % (c["expected"], c["observed"])))
    if failures:
        raise SessionError("split verification failed: %s" % "; ".join(failures))
    stamp("split verified: %d train / %d val / %d test"
          % (EXPECTED["n_model_train"], EXPECTED["n_val"], EXPECTED["n_test"]))
    return {"checks": checks, "passed": True, "split_report": rep}


def verify_checkpoints(ckpt_root: str) -> dict:
    """Every recovered checkpoint must match its verified digest BEFORE no_tta.
    A one-byte difference is a different model."""
    results, failures = {}, []
    for seed, meta in sorted(RECOVERED_CHECKPOINTS.items()):
        p = os.path.join(ckpt_root, meta["relative_path"])
        entry = {"expected_sha256": meta["sha256"], "expected_bytes": meta["bytes"],
                 "path": p, "present": os.path.exists(p)}
        if not entry["present"]:
            entry.update({"pass": False, "error": "not found"})
            failures.append("seed %d: checkpoint not found at %s" % (seed, p))
        else:
            entry["observed_bytes"] = os.path.getsize(p)
            entry["observed_sha256"] = sha256(p)
            entry["pass"] = (entry["observed_sha256"] == meta["sha256"]
                             and entry["observed_bytes"] == meta["bytes"])
            if not entry["pass"]:
                failures.append("seed %d: digest or size mismatch (expected %s / %d, "
                                "observed %s / %d)"
                                % (seed, meta["sha256"][:16], meta["bytes"],
                                   entry["observed_sha256"][:16], entry["observed_bytes"]))
        results[seed] = entry
        stamp("  seed %d %s %s" % (seed, "PASS" if entry.get("pass") else "FAIL",
                                   entry.get("observed_sha256", "")[:16]))
    if failures:
        raise SessionError("checkpoint verification failed: %s" % "; ".join(failures))
    stamp("all %d recovered checkpoints verified" % len(results))
    return {"results": results, "passed": True}


def run_tests(root: str, python: str) -> dict:
    stamp("running the 17-test Phase 3 suite")
    r = subprocess.run([python, os.path.join(root, "src", "publication", "test_ablations.py"),
                        "--project-root", root], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines()
            if "passed ->" in l or l.startswith("  FAIL") or l.startswith("  ERROR")]
    for l in tail:
        stamp("  " + l.strip())
    if r.returncode != 0:
        raise SessionError("Phase 3 test suite failed; refusing to train")
    return {"returncode": r.returncode, "summary": tail}


# ------------------------------------------------------------------ execution ---

def validate_cell(d: str, variant: str, mode: str) -> dict:
    """Completeness for this session's artifact contract. Evaluation-only cells
    have a different contract: they must NOT contain a checkpoint."""
    import session2_config as C
    if mode == "eval_only":
        required = list(C.EVAL_ONLY_ARTIFACTS)
        if variant in C.VARIANTS_WITH_GATES:
            required += C.GATE_ARTIFACTS
        res = {"run_dir": d, "variant": variant, "mode": mode,
               "missing": [], "problems": [], "artifacts": {}}
        for name in required:
            p = os.path.join(d, name)
            if not os.path.exists(p):
                res["missing"].append(name)
                continue
            e = {"bytes": os.path.getsize(p)}
            if name.endswith(".csv"):
                e.update(V.check_csv(p, C.EXPECTED_ROWS.get(name)))
                if not e["ok"]:
                    res["problems"].append("%s: %s" % (name, e.get("error")))
            res["artifacts"][name] = e
        for name in C.EVAL_ONLY_FORBIDDEN:
            if os.path.exists(os.path.join(d, name)):
                res["problems"].append("%s must NOT exist for an evaluation-only cell"
                                       % name)
        res["complete"] = not res["missing"] and not res["problems"]
        res["verdict"] = "COMPLETE" if res["complete"] else "INCOMPLETE"
        res["verdict_detail"] = ("all %d artifacts present, no checkpoint written"
                                 % len(required)) if res["complete"] else \
            ("missing: %s" % ", ".join(res["missing"]) if res["missing"]
             else "; ".join(res["problems"]))
        return res
    # Gate CSVs exist only where a gate runs in the forward pass. The
    # single-branch controls have no gate at all, so demanding the CSVs would
    # fail a perfectly correct run - which is exactly what happened on the first
    # attempt at cnn_branch_only.
    return V.validate(d, variant, with_hashes=False,
                      gates_expected=variant in C.VARIANTS_WITH_GATES)


def run_cell(root, cell, out_root, train_dir, test_dir, ood_dir, python, workers,
             ckpt_root, epochs=None):
    variant, seed, mode = cell["variant"], cell["seed"], cell["mode"]
    cmd = [python, os.path.join(root, "src", "publication", "train_ablation_variant.py"),
           "--variant", variant, "--seed", str(seed), "--split", SPLIT,
           "--project-root", root, "--out-root", out_root,
           "--train-dir", train_dir, "--test-dir", test_dir,
           "--workers", str(workers), "--authorize-training", SESSION]
    if ood_dir:
        cmd += ["--extra-eval", ood_dir]
    if mode == "eval_only":
        meta = RECOVERED_CHECKPOINTS[seed]
        cmd += ["--eval-only",
                "--checkpoint", os.path.join(ckpt_root, meta["relative_path"]),
                "--expect-checkpoint-sha256", meta["sha256"]]
    elif epochs:
        cmd += ["--epochs", str(epochs)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=root)
    return {"returncode": r.returncode, "minutes": round((time.time() - t0) / 60, 1),
            "command": " ".join(cmd)}


def smoke_test(root, out_root, train_dir, test_dir, ood_dir, python, workers, ckpt_root):
    smoke_root = os.path.join(out_root, "_smoke_test")
    banner("SMOKE TEST (2 epochs, isolated directory)")
    full_tree = os.path.abspath(os.path.join(out_root, SPLIT))
    if os.path.abspath(smoke_root).startswith(full_tree):
        raise SessionError("smoke output would land inside the publication tree")
    stamp("output root: %s (separate from %s)" % (smoke_root, full_tree))
    cell = {"variant": APPROVED_VARIANTS[0], "seed": SEEDS[0], "mode": "train"}
    res = run_cell(root, cell, smoke_root, train_dir, test_dir, ood_dir, python,
                   workers, ckpt_root, epochs=2)
    if res["returncode"] != 0:
        raise SessionError("smoke test failed (rc=%d); refusing to start the grid"
                           % res["returncode"])
    d = os.path.join(smoke_root, SPLIT, cell["variant"], "seed_%d" % cell["seed"])
    v = validate_cell(d, cell["variant"], "train")
    stamp("smoke run %s: %s" % (v["verdict"], v["verdict_detail"]))
    if not v["complete"]:
        raise SessionError("smoke run incomplete: %s" % v["verdict_detail"])
    stamp("smoke passed in %.1f min; it lives in _smoke_test/ and is never mixed "
          "with publication runs" % res["minutes"])
    return {"ok": True, "minutes": res["minutes"], "validation": v}


def package(root, out_root, results, preflight, zip_path) -> dict:
    manifest = {"session": SESSION, "split": SPLIT,
                "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "preflight": preflight, "runs": results, "files": []}
    base = os.path.join(out_root, SPLIT)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, filenames in os.walk(base):
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                arc = os.path.relpath(p, out_root).replace("\\", "/")
                z.write(p, arc)
                manifest["files"].append({"path": arc, "bytes": os.path.getsize(p),
                                          "sha256": sha256(p)})
        mp = os.path.join(out_root, "run_manifest.json")
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        z.write(mp, "run_manifest.json")
    manifest["zip"] = {"path": zip_path, "bytes": os.path.getsize(zip_path),
                       "sha256": sha256(zip_path)}
    with open(zip_path + ".sha256", "w", encoding="utf-8") as f:
        f.write("%s  %s\n" % (manifest["zip"]["sha256"], os.path.basename(zip_path)))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--checkpoint-root", default=None,
                    help="Directory containing clean/full_fusion_v2/seed_<n>/checkpoint.pth")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--authorize-training", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_root = os.path.abspath(args.data_root) if args.data_root else root
    ckpt_root = os.path.abspath(args.checkpoint_root) if args.checkpoint_root else \
        os.path.join(root, "publication_runs", "recovered_checkpoints")
    out_root = args.out_root or os.path.join(root, "publication_runs", SESSION)
    python = sys.executable
    train_plan, eval_plan = training_cells(), eval_cells()

    banner("%s - %d training + %d evaluation-only cells on the %s split"
           % (SESSION, len(train_plan), len(eval_plan), SPLIT))
    stamp("training   : %s x seeds %s" % (", ".join(APPROVED_VARIANTS),
                                          "/".join(map(str, SEEDS))))
    stamp("evaluation : %s x seeds %s (verified recovered checkpoints)"
          % (", ".join(EVAL_ONLY_VARIANTS), "/".join(map(str, SEEDS))))

    if args.dry_run:
        stamp("DRY RUN: listing the plan, then exiting without training")
        for c in all_cells():
            d = os.path.join(out_root, SPLIT, c["variant"], "seed_%d" % c["seed"])
            stamp("  %-11s %-18s seed %d -> %s"
                  % (c["mode"], c["variant"], c["seed"], os.path.relpath(d, root)))
        total = sum(COST_MIN[c["variant"]] for c in all_cells())
        stamp("estimated %.1f GPU-hours" % (total / 60.0))
        stamp("nothing trained, nothing written")
        return 0

    if args.authorize_training != SESSION:
        raise SessionError("training requires --authorize-training %s" % SESSION)

    banner("PREFLIGHT")
    preflight = {"environment": environment_report(),
                 "code_package": verify_code_package(root)}
    stamp("verifying the duplicate-clean split")
    preflight["split"] = rebuild_and_verify_split(root, data_root, python)
    stamp("verifying recovered checkpoints (required before no_tta)")
    preflight["checkpoints"] = verify_checkpoints(ckpt_root)
    preflight["tests"] = run_tests(root, python)

    train_dir = os.path.join(root, "dataset_clean", "Training")
    test_dir = os.path.join(root, "dataset_clean", "Testing")
    ood_dir = os.path.join(data_root, "dataset_ood", "Testing")
    if not os.path.isdir(ood_dir):
        raise SessionError("external evaluation set not found at %s" % ood_dir)

    if not args.skip_smoke:
        preflight["smoke"] = smoke_test(root, out_root, train_dir, test_dir, ood_dir,
                                        python, args.workers, ckpt_root)

    results = []
    for phase, plan in (("TRAINING GRID", train_plan), ("EVALUATION ONLY", eval_plan)):
        banner("%s - %d cells" % (phase, len(plan)))
        for i, c in enumerate(plan, 1):
            variant, seed, mode = c["variant"], c["seed"], c["mode"]
            d = os.path.join(out_root, SPLIT, variant, "seed_%d" % seed)
            if os.path.isdir(d):
                v = validate_cell(d, variant, mode)
                if v["complete"]:
                    stamp("[%d/%d] SKIP %s seed %d - validated complete"
                          % (i, len(plan), variant, seed))
                    results.append({**c, "status": "skipped", "validation": v})
                    continue
                stamp("[%d/%d] %s seed %d exists but INCOMPLETE (%s) - re-running"
                      % (i, len(plan), variant, seed, v["verdict_detail"]))
                for fn in os.listdir(d):
                    os.remove(os.path.join(d, fn))

            stamp("[%d/%d] %s %s seed %d" % (i, len(plan), mode.upper(), variant, seed))
            res = run_cell(root, c, out_root, train_dir, test_dir, ood_dir, python,
                           args.workers, ckpt_root)
            if res["returncode"] != 0:
                raise SessionError("cell %s seed %d failed with rc=%d - halting rather "
                                   "than continuing silently" % (variant, seed, res["returncode"]))
            v = validate_cell(d, variant, mode)
            if not v["complete"]:
                raise SessionError("cell %s seed %d finished but is incomplete: %s"
                                   % (variant, seed, v["verdict_detail"]))
            o = v.get("overfitting", {})
            if o.get("available"):
                stamp("[%d/%d] OK %s seed %d in %.1f min | best epoch %s | max gap "
                      "%.2f pp | %s" % (i, len(plan), variant, seed, res["minutes"],
                                        o.get("best_epoch"), o.get("max_train_val_gap_pp", 0),
                                        o.get("verdict")))
                for f in o.get("flags", []):
                    stamp("        OVERFIT FLAG: %s" % f)
            else:
                stamp("[%d/%d] OK %s seed %d in %.1f min (evaluation-only)"
                      % (i, len(plan), variant, seed, res["minutes"]))
            results.append({**c, "status": "completed", "minutes": res["minutes"],
                            "validation": v})

    banner("PACKAGING")
    zip_path = os.path.abspath(os.path.join(root, "fusionv2_session2_results.zip"))
    man = package(root, out_root, results, preflight, zip_path)
    stamp("zip: %s (%.1f MB, sha256 %s)"
          % (zip_path, man["zip"]["bytes"] / 2 ** 20, man["zip"]["sha256"][:16]))
    stamp("manifest covers %d files" % len(man["files"]))

    n_done = sum(1 for r in results if r["status"] in ("completed", "skipped"))
    flagged = [r for r in results if r["validation"].get("overfitting", {}).get("flags")]
    banner("SESSION 2 COMPLETE: %d/%d cells" % (n_done, len(all_cells())))
    if flagged:
        stamp("%d run(s) carry overfitting flags - review before using the numbers"
              % len(flagged))
    return 0 if n_done == len(all_cells()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SessionError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
