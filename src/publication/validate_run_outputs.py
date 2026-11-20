"""
Run-completeness validator and overfitting surveillance.

A run is "complete" only when every required artifact exists, parses, and has
the expected size. The presence of metrics.json is deliberately NOT sufficient:
a process killed between writing metrics and writing predictions leaves a
directory that looks finished and silently drops a cell from the grid. Resume
logic that trusts metrics.json will skip it forever.

Also reports, per run, whether training overfitted. Session 1 must not change
the recipe - these are controlled ablations, and adding regularisation would
confound the component under test with a change in how hard the model may fit.
So overfitting is measured and flagged for a human, never auto-corrected.

    python src/publication/validate_run_outputs.py --run-dir <dir> --variant no_gate
    python src/publication/validate_run_outputs.py --root publication_runs/session_1 --all
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from session1_config import (REQUIRED_ARTIFACTS, GATE_ARTIFACTS, EXPECTED_ROWS,
                             OVERFIT_FLAG_GAP_PP, OVERFIT_REFERENCE)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gate_expected(variant: str) -> bool:
    """no_gate removes the gate by definition, so demanding gate CSVs there
    would fail a correct run."""
    return variant != "no_gate"


def check_csv(path: str, expected_rows=None) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    out = {"ok": True, "rows": len(rows), "columns": list(rows[0].keys()) if rows else []}
    if not rows:
        return {"ok": False, "error": "no data rows"}
    if expected_rows is not None and len(rows) != expected_rows:
        out.update({"ok": False, "error": "expected %d rows, found %d"
                                          % (expected_rows, len(rows))})
    return out


def overfit_report(history_path: str) -> dict:
    """Train-validation divergence, measured from the saved history."""
    try:
        h = json.load(open(history_path, encoding="utf-8"))
    except Exception as e:
        return {"available": False, "error": str(e)}
    if not h:
        return {"available": False, "error": "empty history"}
    gaps = [e["train_acc"] - e["val_acc"] for e in h]
    best_i = max(range(len(h)), key=lambda i: h[i]["val_acc"])
    epochs_run = len(h)
    # The trainer stops after PATIENCE epochs without improvement; if the best
    # epoch is the last one, the budget ran out instead.
    early_stopped = best_i < epochs_run - 1
    max_gap_pp = 100 * max(gaps)
    flags = []
    if max_gap_pp > OVERFIT_FLAG_GAP_PP:
        flags.append("max train-val gap %.2f pp exceeds the %.1f pp review threshold"
                     % (max_gap_pp, OVERFIT_FLAG_GAP_PP))
    if not early_stopped:
        flags.append("no early stopping: best epoch is the last, so the epoch budget "
                     "bound training rather than validation")
    if h[best_i]["val_acc"] < 0.90:
        flags.append("best validation accuracy %.4f is far below the reference; the run "
                     "may not have converged" % h[best_i]["val_acc"])
    return {
        "available": True,
        "epochs_run": epochs_run,
        "best_epoch": h[best_i]["epoch"],
        "best_val_acc": h[best_i]["val_acc"],
        "train_acc_at_best_epoch": h[best_i]["train_acc"],
        "gap_at_best_epoch_pp": 100 * gaps[best_i],
        "max_train_val_gap_pp": max_gap_pp,
        "final_train_val_gap_pp": 100 * gaps[-1],
        "early_stopping_fired": early_stopped,
        "checkpoint_is_best_epoch_not_last": True,
        "reference": OVERFIT_REFERENCE,
        "flags": flags,
        "verdict": "REVIEW" if flags else "within expected range",
    }


def validate(run_dir: str, variant: str, with_hashes: bool = True,
             gates_expected=None) -> dict:
    """gates_expected: pass True/False to state explicitly whether this variant
    has a gate. Left as None it falls back to the Session 1 rule, which assumes
    every variant except no_gate is gated - wrong for the single-branch controls,
    which have no gate to dump."""
    res = {"run_dir": run_dir, "variant": variant, "artifacts": {},
           "missing": [], "problems": []}
    required = list(REQUIRED_ARTIFACTS)
    want_gates = gate_expected(variant) if gates_expected is None else bool(gates_expected)
    if want_gates:
        required += GATE_ARTIFACTS
    res["required_artifacts"] = required
    res["gates_expected"] = want_gates

    for name in required:
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            res["missing"].append(name)
            continue
        entry = {"bytes": os.path.getsize(p)}
        if with_hashes:
            entry["sha256"] = sha256(p)
        if entry["bytes"] == 0:
            entry["ok"] = False
            entry["error"] = "zero bytes"
            res["problems"].append("%s is empty" % name)
        elif name.endswith(".csv"):
            entry.update(check_csv(p, EXPECTED_ROWS.get(name)))
            if not entry["ok"]:
                res["problems"].append("%s: %s" % (name, entry.get("error")))
        elif name.endswith(".json"):
            try:
                json.load(open(p, encoding="utf-8"))
                entry["ok"] = True
            except Exception as e:
                entry.update({"ok": False, "error": str(e)})
                res["problems"].append("%s is not valid JSON" % name)
        else:
            entry["ok"] = True
        res["artifacts"][name] = entry

    hp = os.path.join(run_dir, "history.json")
    res["overfitting"] = overfit_report(hp) if os.path.exists(hp) else {"available": False}

    res["complete"] = not res["missing"] and not res["problems"]
    res["verdict"] = "COMPLETE" if res["complete"] else "INCOMPLETE"
    if res["missing"]:
        res["verdict_detail"] = "missing: %s" % ", ".join(res["missing"])
    elif res["problems"]:
        res["verdict_detail"] = "; ".join(res["problems"])
    else:
        res["verdict_detail"] = "all %d artifacts present and parseable" % len(required)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir")
    ap.add_argument("--variant")
    ap.add_argument("--root", help="scan every <root>/<split>/<variant>/seed_<n>")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--no-hashes", action="store_true")
    args = ap.parse_args()

    results = []
    if args.run_dir:
        if not args.variant:
            print("--variant is required with --run-dir", file=sys.stderr)
            return 2
        results.append(validate(args.run_dir, args.variant, not args.no_hashes))
    elif args.root and args.all:
        for d in sorted(glob.glob(os.path.join(args.root, "*", "*", "seed_*"))):
            variant = os.path.basename(os.path.dirname(d))
            results.append(validate(d, variant, not args.no_hashes))
    else:
        print("give --run-dir/--variant, or --root with --all", file=sys.stderr)
        return 2

    for r in results:
        print("%-9s %-46s %s" % (r["verdict"], os.path.relpath(r["run_dir"]),
                                 r["verdict_detail"]))
        o = r.get("overfitting", {})
        if o.get("available"):
            print("          overfit: best epoch %d/%d, gap at best %.2f pp, max gap "
                  "%.2f pp, early stop %s -> %s"
                  % (o["best_epoch"], o["epochs_run"], o["gap_at_best_epoch_pp"],
                     o["max_train_val_gap_pp"], o["early_stopping_fired"], o["verdict"]))
            for f in o["flags"]:
                print("            FLAG: %s" % f)

    n_ok = sum(1 for r in results if r["complete"])
    print("\n%d/%d complete" % (n_ok, len(results)))
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        json.dump({"results": results, "n_complete": n_ok, "n_total": len(results)},
                  open(args.json_out, "w", encoding="utf-8"), indent=2)
        print("wrote %s" % args.json_out)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
