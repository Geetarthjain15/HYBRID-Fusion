# Phase X7 limitation register

These eight structural limitations are carried forward from the X0 audit.

1. **No anatomical ground truth.** No lesion masks, boxes or expert localisation annotations are available.
2. **Coarse CNN attribution grid.** The CNN attribution begins on the final low-resolution spatial feature grid and is upsampled.
3. **CNN–ViT grid mismatch.** The two attribution paths have different native spatial grids and require an explicit resampling choice.
4. **Model-space coordinates.** Preprocessing resizes images without preserving original acquisition geometry; maps remain in model coordinates.
5. **Prediction–map protocol mismatch.** Reported predictions use flip augmentation, whereas each attribution map uses an unflipped pass.
6. **Attention is not explanation.** Token mixing weights alone do not establish class evidence; the accepted ViT method is gradient-by-input, not attention.
7. **Gate mass is not causal attribution.** The learned gate measures feature-branch utilisation and was not trained to weight spatial maps.
8. **Plausibility is not faithfulness.** Perturbation tests measure model sensitivity, not anatomy or clinical validity.

Additional study limitations

- Evaluation is image-level; patient-level independence cannot be verified from available metadata.
- Scanner, institution, acquisition and demographic metadata are unavailable.
- External evidence is limited to one duplicate-disjoint source.
- Three training seeds quantify limited training variability, not population-level uncertainty.
- Cross-source associations do not establish a mechanism for the accuracy decrease.

Source: `publication_results/explainability/audit/explainability_architecture_audit.json`, question 22.
