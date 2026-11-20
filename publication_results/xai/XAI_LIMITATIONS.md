# XAI limitations

No verified tumor masks, bounding boxes, or clinician annotations are available. Therefore, tumor localization accuracy cannot be computed and is reported as **not available**. Grad-CAM++ is a class-conditioned sensitivity map from the final CNN feature stage. It neither spatially decomposes the ViT branch nor proves clinically correct reasoning. A red maximum on a no-tumor image represents the strongest within-image support for the no-tumor score after per-image normalization; it does not represent tumor tissue.
