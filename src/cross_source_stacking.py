"""Leakage-controlled cross-source calibration by logistic stacking.

This is a source-adaptation experiment, not a zero-shot result.  It uses labels
from the duplicate-disjoint target-source set, but every reported prediction is
out-of-fold: the image being scored is never used to fit its meta-learner.

The input features are seed-averaged class probabilities from four already
trained model families (CNN, ViT, v1 fusion and v2 fusion).  Keeping the
regularisation fixed at C=1 avoids selecting a hyperparameter on the reported
folds.  Repeated stratified five-fold evaluation measures split sensitivity.

Run from the project root:

    python src/cross_source_stacking.py
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold


DEFAULT_GROUPS = {
    "cnn": "runs_clean/cnn_seed*_unfrozen/preds_dataset_ood.csv",
    "vit": "runs_clean/vit_seed*/preds_dataset_ood.csv",
    "fusion": "runs_clean/fusion_seed*/preds_dataset_ood.csv",
    "fusion_v2": "runs_v2_clean/fusion_v2_seed*/preds_dataset_ood.csv",
}


def load_group(project_root, pattern):
    paths = sorted(glob.glob(os.path.join(project_root, pattern)))
    if not paths:
        raise FileNotFoundError("No prediction files matched: " + pattern)

    frames = [pd.read_csv(path).sort_values("filename").reset_index(drop=True)
              for path in paths]
    reference = frames[0]
    prob_cols = [column for column in reference.columns if column.startswith("p_")]
    required = ["filename", "true_label"] + prob_cols
    for path, frame in zip(paths, frames):
        if list(frame[required].columns) != required:
            raise ValueError("Prediction schema differs: " + path)
        if not frame["filename"].equals(reference["filename"]):
            raise ValueError("Image ordering/content differs: " + path)
        if not frame["true_label"].equals(reference["true_label"]):
            raise ValueError("Labels differ: " + path)

    probabilities = np.mean([frame[prob_cols].to_numpy(float) for frame in frames], axis=0)
    return paths, reference[["filename", "true_label"]], prob_cols, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=os.path.dirname(os.path.dirname(__file__)))
    parser.add_argument("--manifest", default="results/cross_source.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/cross_source_stacking.json")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    with open(os.path.join(project_root, args.manifest), encoding="utf-8") as handle:
        manifest = json.load(handle)["manifest"]

    family_probabilities = {}
    family_files = {}
    reference = prob_cols = None
    for name, pattern in DEFAULT_GROUPS.items():
        paths, frame, columns, probabilities = load_group(project_root, pattern)
        if reference is None:
            reference, prob_cols = frame, columns
        elif not frame.equals(reference) or columns != prob_cols:
            raise ValueError("Model families do not describe identical examples/classes")
        family_probabilities[name] = probabilities
        family_files[name] = [os.path.relpath(path, project_root) for path in paths]

    disjoint = np.array([manifest[name]["kind"] == "disjoint"
                         for name in reference["filename"]])
    class_names = [column[2:] for column in prob_cols]
    class_to_index = {name: index for index, name in enumerate(class_names)}
    y = np.array([class_to_index[label] for label in reference["true_label"]])[disjoint]
    X = np.concatenate([family_probabilities[name][disjoint]
                        for name in DEFAULT_GROUPS], axis=1)

    baselines = {}
    for name, probabilities in family_probabilities.items():
        prediction = probabilities[disjoint].argmax(axis=1)
        baselines[name] = {
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, average="macro")),
        }

    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)
    fold_results = []
    probability_sum = np.zeros((len(y), len(class_names)), dtype=float)
    prediction_count = np.zeros(len(y), dtype=int)

    for fold, (train_index, test_index) in enumerate(splitter.split(X, y), start=1):
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(X[train_index], y[train_index])
        probabilities = model.predict_proba(X[test_index])
        prediction = probabilities.argmax(axis=1)
        probability_sum[test_index] += probabilities
        prediction_count[test_index] += 1
        fold_results.append({
            "fold": fold,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "accuracy": float(accuracy_score(y[test_index], prediction)),
            "macro_f1": float(f1_score(y[test_index], prediction, average="macro")),
        })

    if not np.all(prediction_count == args.repeats):
        raise RuntimeError("Each image must receive one out-of-fold prediction per repeat")
    consensus = (probability_sum / prediction_count[:, None]).argmax(axis=1)
    accuracy_values = np.array([row["accuracy"] for row in fold_results])
    f1_values = np.array([row["macro_f1"] for row in fold_results])

    result = {
        "protocol": "supervised target-source calibration; all reported meta-predictions out-of-fold",
        "n_disjoint": int(len(y)),
        "classes": class_names,
        "families": family_files,
        "feature_count": int(X.shape[1]),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "meta_learner": "multinomial logistic regression, fixed C=1.0",
        "baselines": baselines,
        "repeated_fold_summary": {
            "accuracy_mean": float(accuracy_values.mean()),
            "accuracy_sd": float(accuracy_values.std(ddof=1)),
            "macro_f1_mean": float(f1_values.mean()),
            "macro_f1_sd": float(f1_values.std(ddof=1)),
            "accuracy_min": float(accuracy_values.min()),
            "accuracy_max": float(accuracy_values.max()),
        },
        "consensus_oof": {
            "accuracy": float(accuracy_score(y, consensus)),
            "macro_f1": float(f1_score(y, consensus, average="macro")),
        },
        "fold_results": fold_results,
    }

    output_path = os.path.join(project_root, args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    summary = result["repeated_fold_summary"]
    print("Duplicate-disjoint target images:", len(y))
    for name, metrics in baselines.items():
        print("  {:<10} accuracy={:.2%} macro-F1={:.2%}".format(
            name, metrics["accuracy"], metrics["macro_f1"]))
    print("  stack      accuracy={:.2%} +/- {:.2%}; macro-F1={:.2%} +/- {:.2%}".format(
        summary["accuracy_mean"], summary["accuracy_sd"],
        summary["macro_f1_mean"], summary["macro_f1_sd"]))
    print("  consensus  accuracy={:.2%}; macro-F1={:.2%}".format(
        result["consensus_oof"]["accuracy"], result["consensus_oof"]["macro_f1"]))
    print("Wrote", output_path)


if __name__ == "__main__":
    main()
