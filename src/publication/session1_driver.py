"""
Session 1 driver - preflight, smoke test, then the nine approved cells.

Order matters and is enforced. The split is rebuilt and fingerprinted BEFORE any
gradient step, because a silently different split produces numbers that look
fine and mean nothing. The smoke test runs in its own directory, so a two-epoch
probe can never be mistaken for a finished run. The grid then halts on the first
failure rather than continuing and leaving a partial matrix that reads as
complete.

Resume is by VALIDATION, not by the presence of metrics.json. A process killed
between writing metrics and writing predictions leaves a directory that looks
finished; trusting metrics.json would skip that cell forever.

    python src/publication/session1_driver.py --project-root . --dry-run
    python src/publication/session1_driver.py --project-root /kaggle/working \\
        --data-root /kaggle/input/<slug> --authorize-training session_1
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

from session1_config import (SESSION, SPLIT, SEEDS, APPROVED_VARIANTS, EXPECTED,
                             cells, is_in_scope)
from validate_run_outputs import validate, sha256


class SessionError(Exception):
    """A Session 1 stop condition was hit."""


def banner(msg):
    print("\n" + "=" * 72 + "\n  %s\n" % msg + "=" * 72, flush=True)


def stamp(msg):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ preflight ---

def rebuild_and_verify_split(root: str, data_root: str, python: str) -> dict:
    """Rebuild dataset_clean with the same script and seed, then check every
    fingerprint. Any mismatch stops the session before training."""
    out_dir = os.path.join(root, "dataset_clean")
    report_p = os.path.join(out_dir, "split_report.json")
    manifest_p = os.path.join(out_dir, "split_manifest.json")

    if not os.path.exists(report_p):
        src = os.path.join(data_root, "dataset")
        if not os.path.isdir(os.path.join(src, "Training")):
            raise SessionError("source dataset not found at %s" % src)
        stamp("rebuilding the duplicate-clean split (seed %d, hamming %d)"
              % (EXPECTED["split_seed"], EXPECTED["hamming_threshold"]))
        cmd = [python, os.path.join(root, "src", "make_clean_split.py"),
               "--src", src, "--out", out_dir,
               "--seed", str(EXPECTED["split_seed"]),
               "--hamming", str(EXPECTED["hamming_threshold"])]
        r = subprocess.run(cmd, cwd=root)
        if r.returncode != 0:
            raise SessionError("make_clean_split.py failed with rc=%d" % r.returncode)
    else:
        stamp("dataset_clean already present; verifying rather than rebuilding")

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

    man_hash = sha256(manifest_p) if os.path.exists(manifest_p) else None
    expect("split_manifest_sha256", man_hash, EXPECTED["split_manifest_sha256"])

    # Count the images actually on disk: the report describes intent, the
    # filesystem describes reality, and only the latter is what trains.
    for split_name, key in (("Training", "n_train"), ("Testing", "n_test")):
        d = os.path.join(out_dir, split_name)
        n = sum(len(fs) for _, _, fs in os.walk(d)) if os.path.isdir(d) else 0
        expect("images_on_disk_%s" % split_name.lower(), n, EXPECTED[key])

    result = {"checks": checks, "failures": failures, "passed": not failures,
              "split_report": rep, "split_manifest_sha256": man_hash,
              "dataset_clean_dir": out_dir}
    for name, c in checks.items():
        stamp("  %-28s %s" % (name, "PASS" if c["pass"] else
                              "FAIL expected=%r observed=%r" % (c["expected"], c["observed"])))
    if failures:
        raise SessionError("split verification failed: %s" % "; ".join(failures))
    stamp("split verified: %d train / %d test, manifest hash matches"
          % (rep["n_train"], rep["n_test"]))
    return result


def run_tests(root: str, python: str) -> dict:
    stamp("running the 17-test Phase 3 suite")
    r = subprocess.run([python, os.path.join(root, "src", "publication", "test_ablations.py"),
                        "--project-root", root], capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if "passed ->" in l or l.startswith("  FAIL")
            or l.startswith("  ERROR")]
    for l in tail:
        stamp("  " + l.strip())
    if r.returncode != 0:
        raise SessionError("Phase 3 test suite failed; refusing to train")
    return {"returncode": r.returncode, "summary": tail}


# ------------------------------------------------------------------ execution ---

def train_cell(root, variant, seed, out_root, train_dir, test_dir, ood_dir,
               python, workers, epochs=None, authorize=SESSION):
    cmd = [python, os.path.join(root, "src", "publication", "train_ablation_variant.py"),
           "--variant", variant, "--seed", str(seed), "--split", SPLIT,
           "--project-root", root, "--out-root", out_root,
           "--train-dir", train_dir, "--test-dir", test_dir,
           "--workers", str(workers), "--authorize-training", authorize]
    if ood_dir:
        cmd += ["--extra-eval", ood_dir]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=root)
    return {"returncode": r.returncode, "minutes": round((time.time() - t0) / 60, 1),
            "command": " ".join(cmd)}


def smoke_test(root, out_root, train_dir, test_dir, ood_dir, python, workers) -> dict:
    """Two epochs of the first approved variant, into a SEPARATE directory tree.
    Requirement: this must not deposit metrics.json anywhere a full run would
    later be written, or resume logic would skip a cell that never really ran."""
    smoke_root = os.path.join(out_root, "_smoke_test")
    banner("SMOKE TEST (2 epochs, isolated directory)")
    stamp("output root: %s" % smoke_root)
    if os.path.commonpath([os.path.abspath(smoke_root),
                           os.path.abspath(os.path.join(out_root, SPLIT))]) == \
            os.path.abspath(os.path.join(out_root, SPLIT)):
        raise SessionError("smoke output would land inside the full-run tree")
    res = train_cell(root, APPROVED_VARIANTS[0], SEEDS[0], smoke_root,
                     train_dir, test_dir, ood_dir, python, workers, epochs=2)
    if res["returncode"] != 0:
        raise SessionError("smoke test failed (rc=%d); refusing to start the grid"
                           % res["returncode"])
    d = os.path.join(smoke_root, SPLIT, APPROVED_VARIANTS[0], "seed_%d" % SEEDS[0])
    v = validate(d, APPROVED_VARIANTS[0], with_hashes=False)
    stamp("smoke run %s: %s" % (v["verdict"], v["verdict_detail"]))
    if not v["complete"]:
        raise SessionError("smoke run produced an incomplete artifact set: %s"
                           % v["verdict_detail"])
    stamp("smoke test passed in %.1f min; it lives in _smoke_test/ and is never "
          "mistaken for a graded run" % res["minutes"])
    return {"ok": True, "minutes": res["minutes"], "dir": d, "validation": v}


def package(root, out_root, results, preflight, zip_path) -> dict:
    """Zip the completed runs and write a SHA-256 manifest of every file."""
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
    ap.add_argument("--data-root", default=None,
                    help="Directory holding dataset/ and dataset_ood/. Defaults to the "
                         "project root.")
    ap.add_argument("--out-root", default=None,
                    help="Defaults to <project-root>/publication_runs/session_1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--authorize-training", default=None)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    data_root = os.path.abspath(args.data_root) if args.data_root else root
    out_root = args.out_root or os.path.join(root, "publication_runs", SESSION)
    python = sys.executable
    plan = cells()

    banner("%s - %d cells (%s x seeds %s) on the %s split"
           % (SESSION, len(plan), ", ".join(APPROVED_VARIANTS),
              "/".join(map(str, SEEDS)), SPLIT))
    for v in APPROVED_VARIANTS:
        allowed, reason = is_in_scope(v)
        stamp("  %-20s %s" % (v, reason))

    if args.dry_run:
        stamp("DRY RUN: listing the plan, then exiting without training")
        for c in plan:
            d = os.path.join(out_root, SPLIT, c["variant"], "seed_%d" % c["seed"])
            stamp("  %-20s seed %d -> %s" % (c["variant"], c["seed"],
                                             os.path.relpath(d, root)))
        stamp("nothing trained, nothing written")
        return 0

    if args.authorize_training != SESSION:
        raise SessionError("training requires --authorize-training %s" % SESSION)

    banner("PREFLIGHT")
    preflight = {"split": rebuild_and_verify_split(root, data_root, python),
                 "tests": run_tests(root, python)}

    train_dir = os.path.join(root, "dataset_clean", "Training")
    test_dir = os.path.join(root, "dataset_clean", "Testing")
    ood_dir = os.path.join(data_root, "dataset_ood", "Testing")
    if not os.path.isdir(ood_dir):
        raise SessionError("external evaluation set not found at %s" % ood_dir)

    if not args.skip_smoke:
        preflight["smoke"] = smoke_test(root, out_root, train_dir, test_dir, ood_dir,
                                        python, args.workers)

    banner("GRID - %d cells" % len(plan))
    results = []
    for i, c in enumerate(plan, 1):
        variant, seed = c["variant"], c["seed"]
        d = os.path.join(out_root, SPLIT, variant, "seed_%d" % seed)
        # Resume by validation, never by the presence of metrics.json.
        if os.path.isdir(d):
            v = validate(d, variant, with_hashes=False)
            if v["complete"]:
                stamp("[%d/%d] SKIP %s seed %d - validated complete"
                      % (i, len(plan), variant, seed))
                results.append({"variant": variant, "seed": seed, "status": "skipped",
                                "validation": v})
                continue
            stamp("[%d/%d] %s seed %d exists but is INCOMPLETE (%s) - re-running"
                  % (i, len(plan), variant, seed, v["verdict_detail"]))
            for fn in os.listdir(d):
                os.remove(os.path.join(d, fn))

        stamp("[%d/%d] RUN %s seed %d" % (i, len(plan), variant, seed))
        res = train_cell(root, variant, seed, out_root, train_dir, test_dir, ood_dir,
                         python, args.workers)
        if res["returncode"] != 0:
            raise SessionError("cell %s seed %d failed with rc=%d - halting the grid "
                               "rather than continuing silently"
                               % (variant, seed, res["returncode"]))
        v = validate(d, variant, with_hashes=False)
        if not v["complete"]:
            raise SessionError("cell %s seed %d finished but is incomplete: %s"
                               % (variant, seed, v["verdict_detail"]))
        o = v.get("overfitting", {})
        stamp("[%d/%d] OK %s seed %d in %.1f min | best epoch %s | max gap %.2f pp | %s"
              % (i, len(plan), variant, seed, res["minutes"], o.get("best_epoch"),
                 o.get("max_train_val_gap_pp", float("nan")), o.get("verdict")))
        for f in o.get("flags", []):
            stamp("        OVERFIT FLAG: %s" % f)
        results.append({"variant": variant, "seed": seed, "status": "completed",
                        "minutes": res["minutes"], "validation": v})

    banner("PACKAGING")
    zip_path = os.path.join(os.path.dirname(out_root.rstrip("/\\")), "..",
                            "fusionv2_session1_results.zip")
    zip_path = os.path.abspath(os.path.join(root, "fusionv2_session1_results.zip"))
    man = package(root, out_root, results, preflight, zip_path)
    stamp("zip: %s (%.1f MB, sha256 %s)"
          % (zip_path, man["zip"]["bytes"] / 2 ** 20, man["zip"]["sha256"][:16]))
    stamp("manifest covers %d files" % len(man["files"]))

    n_done = sum(1 for r in results if r["status"] in ("completed", "skipped"))
    flagged = [r for r in results
               if r["validation"].get("overfitting", {}).get("flags")]
    banner("SESSION 1 COMPLETE: %d/%d cells" % (n_done, len(plan)))
    if flagged:
        stamp("%d run(s) carry overfitting flags - review before using the numbers"
              % len(flagged))
    return 0 if n_done == len(plan) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SessionError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
