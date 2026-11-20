"""
Phase X0 supplementary probe - ViT attention availability.

The main probe established that `output_attentions=True` returns nothing under
the attention implementation the verified model loads with. That is a blocking
fact for any dual-path design that wants ViT attention, so it is investigated
here rather than assumed away.

The decisive question is not "can attention be switched on" but "does switching
it on change the model's predictions". If it does, an instrumented model is no
longer the verified model and no explanation computed from it may be attached to
the verified results.

Read-only: nothing is written except the JSON under the audit directory, and the
checkpoint is loaded with weights_only=True and never written back.

    env_win/Scripts/python.exe src/publication/explainability/audit_x0_attention.py --project-root .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for p in (str(PUB), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

TOL = 1e-6


def probe_input(batch=2):
    g = torch.Generator().manual_seed(20260926)
    return torch.rand((batch, 3, 224, 224), generator=g) * 2.0 - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_results/explainability/audit")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    os.chdir(root)
    if not args.out_root.replace("\\", "/").startswith("publication_results/"):
        raise SystemExit("refusing to write outside publication_results/")
    out = root / args.out_root
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    from fusion_v2_gradcam import load_verified_model

    R = {"probe": "ViT attention availability and prediction neutrality",
         "tolerance": TOL}

    model, _dev, sha = load_verified_model(root, seed=42, device="cpu")
    model.eval()
    R["checkpoint_sha256"] = sha
    x = probe_input(2)

    with torch.no_grad():
        probs_ref = torch.softmax(model(x), -1)
        logits_ref = model(x)
    R["reference_probs_row0"] = probs_ref[0].tolist()

    inner = model.vit.vit
    R["attn_implementation_as_loaded"] = getattr(model.vit.config,
                                                 "_attn_implementation", "unknown")

    # --- 1. flag alone, under the loaded implementation ---------------------
    try:
        with torch.no_grad():
            o = inner(pixel_values=x, output_attentions=True)
        R["flag_only"] = {
            "attentions_is_none": getattr(o, "attentions", None) is None,
            "note": "transformers warns that output_attentions is unsupported unless "
                    "attn_implementation is eager/eager_paged/flex_attention",
        }
    except Exception as e:
        R["flag_only"] = {"error": "%s: %s" % (type(e).__name__, e)}

    # --- 2. switch to eager, then ask again ---------------------------------
    eager = {}
    try:
        model.vit.set_attn_implementation("eager")
        eager["attn_implementation_after_switch"] = getattr(
            model.vit.config, "_attn_implementation", "unknown")
        with torch.no_grad():
            o = model.vit.vit(pixel_values=x, output_attentions=True)
        att = getattr(o, "attentions", None)
        eager["attentions_returned"] = att is not None
        if att is not None:
            eager["n_layers"] = len(att)
            eager["shape_layer0"] = list(att[0].shape)
            s = att[0].sum(-1)
            eager["rows_sum_to_one"] = bool(
                torch.allclose(s, torch.ones_like(s), atol=1e-4))
            eager["cls_row_shape"] = list(att[-1][:, :, 0, 1:].shape)
            eager["n_patch_tokens"] = int(att[0].shape[-1]) - 1
            import math
            g = int(math.isqrt(eager["n_patch_tokens"]))
            eager["patch_grid"] = [g, g] if g * g == eager["n_patch_tokens"] else None

        # THE decisive check: did switching change the prediction?
        with torch.no_grad():
            probs_eager = torch.softmax(model(x), -1)
            logits_eager = model(x)
        dprob = (probs_eager - probs_ref).abs().max()
        dlogit = (logits_eager - logits_ref).abs().max()
        eager["max_abs_prob_delta_vs_verified"] = float(dprob)
        eager["max_abs_logit_delta_vs_verified"] = float(dlogit)
        eager["changes_fused_probs_beyond_tolerance"] = bool(dprob > TOL)
        eager["argmax_unchanged"] = bool(torch.equal(probs_eager.argmax(-1),
                                                     probs_ref.argmax(-1)))
    except Exception as e:
        eager["error"] = "%s: %s" % (type(e).__name__, e)
        eager["traceback"] = traceback.format_exc()
    R["eager_switch"] = eager

    # --- 3. can we switch back, and is it exact? ----------------------------
    back = {}
    try:
        model.vit.set_attn_implementation(R["attn_implementation_as_loaded"])
        back["attn_implementation_restored"] = getattr(
            model.vit.config, "_attn_implementation", "unknown")
        with torch.no_grad():
            probs_back = torch.softmax(model(x), -1)
        back["exact_restore"] = bool(torch.equal(probs_back, probs_ref))
        back["max_abs_prob_delta"] = float((probs_back - probs_ref).abs().max())
    except Exception as e:
        back["error"] = "%s: %s" % (type(e).__name__, e)
    R["restore"] = back

    with open(out / "_probe_attention.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)

    print("attn impl as loaded      :", R["attn_implementation_as_loaded"])
    print("flag alone -> attentions :", "None" if R.get("flag_only", {}).get(
        "attentions_is_none") else R.get("flag_only"))
    print("after eager switch       :", eager.get("attentions_returned"),
          eager.get("shape_layer0"))
    print("  attention rows sum to 1:", eager.get("rows_sum_to_one"))
    print("  patch grid             :", eager.get("patch_grid"))
    print("  max prob delta         : %.3e" % eager.get(
        "max_abs_prob_delta_vs_verified", float("nan")))
    print("  CHANGES PREDICTIONS?   :", eager.get("changes_fused_probs_beyond_tolerance"))
    print("  argmax unchanged       :", eager.get("argmax_unchanged"))
    print("restore exact            :", back.get("exact_restore"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
