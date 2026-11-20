## T1 Main results (in-distribution test)

| Model | original: Acc | clean: Acc | original: Macro-F1 | clean: Macro-F1 |
|---|---|---|---|---|
| cnn | 92.36 $\pm$ 1.04 | 91.94 $\pm$ 0.48 | 92.13 $\pm$ 1.07 | 92.47 $\pm$ 0.43 |
| cnn/unfrozen | 99.46 $\pm$ 0.05 | 98.70 $\pm$ 0.20 | 99.46 $\pm$ 0.06 | 98.62 $\pm$ 0.23 |
| fusion | 98.89 $\pm$ 0.44 | 97.05 $\pm$ 0.43 | 98.86 $\pm$ 0.45 | 97.01 $\pm$ 0.41 |
| fusion_v2 | 99.14 $\pm$ 0.10 | 98.27 $\pm$ 0.07 | 99.12 $\pm$ 0.09 | 98.28 $\pm$ 0.04 |
| vit | 98.70 $\pm$ 0.20 | 97.05 $\pm$ 0.31 | 98.66 $\pm$ 0.21 | 97.03 $\pm$ 0.29 |

## T3 Evaluation regimes (original split)

| Model | In-distribution | Resampled (n=126) | Cross-source (n=373) |
|---|---|---|---|
| cnn | 92.36 $\pm$ 1.04 | 88.89 $\pm$ 3.46 | 43.25 $\pm$ 2.28 |
| cnn/unfrozen | 99.46 $\pm$ 0.05 | 96.83 | 56.12 $\pm$ 5.89 |
| fusion | 98.89 $\pm$ 0.44 | 96.83 $\pm$ 0.79 | 70.96 $\pm$ 2.87 |
| fusion_v2 | 99.14 $\pm$ 0.10 | 97.09 $\pm$ 0.92 | 68.28 $\pm$ 3.39 |
| vit | 98.70 $\pm$ 0.20 | 97.62 $\pm$ 1.37 | 70.96 $\pm$ 5.27 |

## T3 Evaluation regimes (clean split)

| Model | In-distribution | Resampled (n=126) | Cross-source (n=373) |
|---|---|---|---|
| cnn | 91.94 $\pm$ 0.48 | 89.95 $\pm$ 1.65 | 48.44 $\pm$ 2.03 |
| cnn/unfrozen | 98.70 $\pm$ 0.20 | 96.03 $\pm$ 0.79 | 55.23 $\pm$ 4.77 |
| fusion | 97.05 $\pm$ 0.43 | 97.35 $\pm$ 0.92 | 74.17 $\pm$ 5.05 |
| fusion_v2 | 98.27 $\pm$ 0.07 | 97.09 $\pm$ 0.46 | 69.53 $\pm$ 1.98 |
| vit | 97.05 $\pm$ 0.31 | 97.62 $\pm$ 0.79 | 73.82 $\pm$ 1.62 |

## T4 Per-class recall (original split, in-distribution)

| Model | Glioma | Meningioma | Notumor | Pituitary |
|---|---|---|---|---|
| cnn | 91.63 $\pm$ 1.45 | 81.17 $\pm$ 3.55 | 98.67 $\pm$ 0.88 | 96.32 $\pm$ 1.44 |
| cnn/unfrozen | 99.59 | 98.92 $\pm$ 0.23 | 99.67 | 99.62 |
| fusion | 99.31 $\pm$ 0.24 | 98.24 $\pm$ 0.62 | 99.67 | 98.23 $\pm$ 1.16 |
| fusion_v2 | 99.59 | 98.51 $\pm$ 0.23 | 99.33 $\pm$ 0.58 | 99.11 $\pm$ 0.44 |
| vit | 98.49 $\pm$ 0.86 | 98.51 $\pm$ 0.47 | 99.67 | 97.97 $\pm$ 0.44 |

## T4 Per-class recall (clean split, in-distribution)

| Model | Glioma | Meningioma | Notumor | Pituitary |
|---|---|---|---|---|
| cnn | 85.73 $\pm$ 0.48 | 92.84 $\pm$ 0.67 | 95.98 $\pm$ 1.00 | 95.15 $\pm$ 1.11 |
| cnn/unfrozen | 98.08 $\pm$ 0.24 | 98.54 $\pm$ 0.25 | 97.99 $\pm$ 1.32 | 99.74 $\pm$ 0.22 |
| fusion | 97.39 $\pm$ 0.24 | 96.64 $\pm$ 1.66 | 95.69 $\pm$ 0.86 | 97.70 $\pm$ 0.00 |
| fusion_v2 | 98.22 $\pm$ 1.04 | 97.37 $\pm$ 0.44 | 97.99 $\pm$ 0.50 | 99.23 $\pm$ 0.38 |
| vit | 97.12 $\pm$ 1.09 | 96.78 $\pm$ 0.51 | 95.40 $\pm$ 0.50 | 97.96 $\pm$ 0.59 |

## T5 Calibration and confidence gate

| Model | Split | ECE | Brier | Conf (correct) | Conf (wrong) | Gate-accepted acc |
|---|---|---|---|---|---|---|
| cnn | original | 0.0838 $\pm$ 0.0035 | 0.1238 $\pm$ 0.0098 | 0.860 $\pm$ 0.004 | 0.607 $\pm$ 0.011 | 96.10 $\pm$ 0.57 |
| cnn/unfrozen | original | 0.0701 $\pm$ 0.0031 | 0.0171 $\pm$ 0.0001 | 0.926 $\pm$ 0.004 | 0.720 $\pm$ 0.027 | 99.62 $\pm$ 0.09 |
| fusion | original | 0.0718 $\pm$ 0.0039 | 0.0265 $\pm$ 0.0047 | 0.920 $\pm$ 0.003 | 0.699 $\pm$ 0.055 | 99.26 $\pm$ 0.29 |
| fusion_v2 | original | 0.0756 $\pm$ 0.0017 | 0.0200 $\pm$ 0.0014 | 0.922 $\pm$ 0.002 | 0.676 $\pm$ 0.025 | 99.39 $\pm$ 0.05 |
| vit | original | 0.0726 $\pm$ 0.0058 | 0.0296 $\pm$ 0.0052 | 0.920 $\pm$ 0.006 | 0.689 $\pm$ 0.016 | 99.17 $\pm$ 0.06 |
| cnn | clean | 0.0857 $\pm$ 0.0110 | 0.1394 $\pm$ 0.0024 | 0.853 $\pm$ 0.009 | 0.636 $\pm$ 0.011 | 95.13 $\pm$ 0.15 |
| cnn/unfrozen | clean | 0.0791 $\pm$ 0.0023 | 0.0294 $\pm$ 0.0027 | 0.916 $\pm$ 0.003 | 0.616 $\pm$ 0.035 | 99.32 $\pm$ 0.25 |
| fusion | clean | 0.0697 $\pm$ 0.0034 | 0.0483 $\pm$ 0.0043 | 0.917 $\pm$ 0.003 | 0.675 $\pm$ 0.018 | 98.23 $\pm$ 0.36 |
| fusion_v2 | clean | 0.0752 $\pm$ 0.0032 | 0.0335 $\pm$ 0.0026 | 0.919 $\pm$ 0.001 | 0.707 $\pm$ 0.061 | 98.69 $\pm$ 0.31 |
| vit | clean | 0.0685 $\pm$ 0.0047 | 0.0497 $\pm$ 0.0036 | 0.915 $\pm$ 0.004 | 0.677 $\pm$ 0.033 | 98.15 $\pm$ 0.13 |

## T2 Diagnostic protocol

| Test | Setting | Value 1 | Value 2 | Value 3 | Verdict |
|---|---|---|---|---|---|
| T1 oracle | clean / test | 98.70 | 99.29 | +0.59 | HEADROOM EXISTS |
| T3 equivalence | clean / test vs vit | +0.00 | 0.481 | 0.31 | EQUIVALENT |
| T1 oracle | clean / dataset_ood | 79.83 | 81.43 | +1.60 | HEADROOM EXISTS |
| T3 equivalence | clean / dataset_ood vs vit | +0.20 | 0.000 | 1.30 | ERROR-SWAPPING |
| T1 oracle | crosssource_clean / dataset_ood | 73.82 | 75.78 | +1.97 | HEADROOM EXISTS |
| T3 equivalence | crosssource_clean / dataset_ood vs vit | +0.37 | 0.000 | 1.62 | ERROR-SWAPPING |
| T1 oracle | crosssource_original / dataset_ood | 70.96 | 75.51 | +4.56 | HEADROOM EXISTS |
| T3 equivalence | crosssource_original / dataset_ood vs vit | +0.01 | 0.000 | 5.27 | ERROR-SWAPPING |
| T2 gate mass | crosssource_v2_clean | 0.162 | 0.838 | -- | BOTH BRANCHES USED |
| T2 gate mass | crosssource_v2_original | 0.296 | 0.704 | -- | BOTH BRANCHES USED |
| T1 oracle | original / test | 99.46 | 99.59 | +0.13 | NO HEADROOM |
| T3 equivalence | original / test vs vit | +0.19 | 0.344 | 0.20 | EQUIVALENT |
| T1 oracle | original / dataset_ood | 77.69 | 81.30 | +3.61 | HEADROOM EXISTS |
| T3 equivalence | original / dataset_ood vs vit | -0.21 | 0.000 | 4.29 | ERROR-SWAPPING |
| T2 gate mass | v2_clean | 0.115 | 0.885 | -- | BOTH BRANCHES USED |
| T2 gate mass | v2_original | 0.243 | 0.757 | -- | BOTH BRANCHES USED |
