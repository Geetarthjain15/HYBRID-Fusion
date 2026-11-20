# Phase X7 — publication reporting

## Classification evidence

- Original split accuracy: 99.14 ± 0.10%.
- Duplicate-clean accuracy: 98.27 ± 0.07%.
- Duplicate-disjoint cross-source accuracy: 69.53 ± 1.98% over 373 unique images.

The external decrease is class-specific. Pituitary recall is 13.81 ± 4.62%, and 260 of 333 pituitary image–seed predictions are no-tumour.

## Explanation evidence

X3 produced synchronized CNN Grad-CAM++ and ViT Gradient × Input maps without changing fused probabilities. X4 established determinism, class sensitivity, native shapes, orientation and cleanup behaviour. X5 measured faithfulness and stability under a locked perturbation protocol. X6 applied that protocol unchanged to the duplicate-disjoint external set.

The X6 changes are mixed across methods and transformations. They support a source-associated change in confidence, branch behaviour, attribution agreement and selected faithfulness/stability measurements; they do not support a universal explanation-degradation claim.

## Claim boundary

These results concern image-level classification and model sensitivity. They do not establish patient-level independence, anatomical localisation, a causal mechanism, diagnostic utility or clinical readiness.

## Streamlit decision

No Streamlit wrapper was created. Original X2 is a recorded FAIL and X2b is its accepted replacement; the X7 app gate literally requires X2–X5 to pass. Reporting proceeds, but the optional app is withheld rather than silently redefining that gate.
