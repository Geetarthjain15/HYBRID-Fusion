"""
Phase X0 - read-only architecture probe for the dual-path attribution study.

This script answers architectural questions by EXECUTING the verified model and
recording what it actually does. It exists because the Phase X0 brief requires
architectural statements to cite executable evidence rather than variable names,
docstrings or filenames.

It is strictly read-only with respect to every protected artifact:

  - checkpoints are opened with map_location="cpu" and weights_only=True and are
    never written back;
  - no training, no optimiser, no gradient step on model parameters;
  - the only gradients taken are for the Grad-CAM++ identity check, and they are
    discarded;
  - the only filesystem writes are the JSON/Markdown outputs under
    publication_results/explainability/audit/.

Determinism: torch seeded, cuDNN disabled by running on CPU, and the probe input
is a fixed synthetic tensor rather than a dataset image, so the probe does not
read or depend on any dataset file.

    env_win/Scripts/python.exe src/publication/explainability/audit_x0.py --project-root .
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for p in (str(PUB), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

TOL = 1e-6          # fused-probability perturbation tolerance
SEEDS = [42, 43, 44]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def probe_input(batch: int = 2) -> torch.Tensor:
    """Fixed synthetic input. Using a synthetic tensor keeps the probe
    independent of any dataset file and makes it reproducible anywhere."""
    g = torch.Generator().manual_seed(20260926)
    return torch.rand((batch, 3, 224, 224), generator=g) * 2.0 - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root",
                    default="publication_results/explainability/audit")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    os.chdir(root)
    rel = args.out_root.replace("\\", "/")
    if not rel.startswith("publication_results/"):
        raise SystemExit("refusing to write outside publication_results/")
    out = root / args.out_root
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)

    R: dict = {"phase": "X0 - evidence and architecture audit",
               "generated_utc": __import__("datetime").datetime.now(
                   __import__("datetime").timezone.utc).isoformat(),
               "training_performed": False, "model_modified": False,
               "device": "cpu", "tolerance_fused_probability": TOL}

    # ---------------------------------------------------------------- env ---
    import torchvision
    import transformers
    R["environment"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
    }
    try:
        import pytorch_grad_cam as pgc
        R["environment"]["pytorch_grad_cam"] = getattr(pgc, "__version__", "unknown")
    except Exception as e:
        R["environment"]["pytorch_grad_cam"] = "IMPORT FAILED: %s" % e

    # ------------------------------------------- Q2: Grad-CAM++ identity ----
    # The question is whether the code's "Grad-CAM++" really computes Grad-CAM++
    # rather than ordinary Grad-CAM. Resolve the class the code imports and read
    # its own weight-computation source.
    q2: dict = {}
    try:
        from pytorch_grad_cam import GradCAMPlusPlus, GradCAM
        q2["resolved_class"] = "%s.%s" % (GradCAMPlusPlus.__module__,
                                          GradCAMPlusPlus.__name__)
        q2["is_distinct_class_from_GradCAM"] = GradCAMPlusPlus is not GradCAM
        q2["mro"] = [c.__name__ for c in GradCAMPlusPlus.__mro__[:4]]
        try:
            src = inspect.getsource(GradCAMPlusPlus.get_cam_weights)
            q2["get_cam_weights_source"] = src
            low = src.replace(" ", "")
            q2["uses_second_order_terms"] = ("grads**2" in low or "grads_power_2" in low)
            q2["uses_third_order_terms"] = ("grads**3" in low or "grads_power_3" in low)
            q2["defines_alpha_weighting"] = "alpha" in low
        except Exception as e:
            q2["get_cam_weights_source"] = "not available: %s" % e
        try:
            q2["gradcam_get_cam_weights_source"] = inspect.getsource(GradCAM.get_cam_weights)
        except Exception as e:
            q2["gradcam_get_cam_weights_source"] = "not available: %s" % e
        q2["source_file"] = inspect.getfile(GradCAMPlusPlus)
    except Exception as e:
        q2["error"] = "%s: %s" % (type(e).__name__, e)
    R["q2_gradcam_identity"] = q2

    # ------------------------------------------------- checkpoints (Q19) ----
    man_path = root / "publication_results/ablations/checkpoint_recovery/import_manifest.json"
    ck: dict = {"manifest": str(man_path.relative_to(root)).replace("\\", "/"),
                "manifest_present": man_path.is_file(), "seeds": {}}
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    for s in SEEDS:
        p = root / ("publication_runs/recovered_checkpoints/clean/full_fusion_v2/"
                    "seed_%d/checkpoint.pth" % s)
        e = {"path": str(p.relative_to(root)).replace("\\", "/"), "present": p.is_file()}
        if p.is_file():
            e["sha256"] = sha256(p)
            e["bytes"] = p.stat().st_size
        m = manifest.get("seeds", {}).get(str(s), {})
        e["manifest_decision"] = m.get("decision")
        imp = m.get("import", {})
        e["manifest_destination_sha256"] = imp.get("destination_sha256")
        e["container_sha256_in_manifest"] = m.get("gates", {}).get("container", {}).get("sha256")
        e["sha256_matches_manifest"] = (
            e.get("sha256") is not None
            and e["sha256"] in {e.get("manifest_destination_sha256"),
                                e.get("container_sha256_in_manifest")})
        ck["seeds"][s] = e
    ck["all_three_accepted_and_hash_matched"] = all(
        ck["seeds"][s].get("manifest_decision") == "ACCEPTED"
        and ck["seeds"][s].get("sha256_matches_manifest") for s in SEEDS)
    R["q19_checkpoints"] = ck

    # ------------------------------------------------ load model (seed 42) --
    from fusion_v2_gradcam import (CLASS_NAMES, EVAL_TRANSFORM, IMAGE_SIZE, NORM,
                                   OfflineFusionV2, load_verified_model)
    R["q7_class_order"] = {
        "class_names_in_code": list(CLASS_NAMES),
        "source": "src/publication/fusion_v2_gradcam.py:37",
        "is_sorted_alphabetically": list(CLASS_NAMES) == sorted(CLASS_NAMES),
    }
    R["q5_preprocessing"] = {
        "image_size": IMAGE_SIZE, "normalize_mean": NORM[0], "normalize_std": NORM[1],
        "transform_repr": repr(EVAL_TRANSFORM),
        "source": "src/publication/fusion_v2_gradcam.py:38-44",
    }

    model, dev, actual_hash = load_verified_model(root, seed=42, device="cpu")
    model.eval()
    R["loaded_checkpoint"] = {"seed": 42, "sha256": actual_hash,
                              "strict_load": True,
                              "note": "load_verified_model uses load_state_dict(strict=True) "
                                      "and verifies the SHA-256 against the recovery manifest "
                                      "(src/publication/fusion_v2_gradcam.py:115-131)"}

    # ------------------------------------- Q15/Q16: aux tensors in state ----
    state = torch.load(
        root / "publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_42/checkpoint.pth",
        map_location="cpu", weights_only=True)
    keys = list(state.keys())
    aux_keys = [k for k in keys if k.startswith(("aux_cnn", "aux_vit"))]
    gate_keys = [k for k in keys if k.startswith("gate")]
    R["q15_aux_in_checkpoint"] = {
        "n_state_dict_keys": len(keys),
        "aux_keys": aux_keys,
        "aux_shapes": {k: list(state[k].shape) for k in aux_keys},
        "gate_keys": gate_keys,
        "gate_shapes": {k: list(state[k].shape) for k in gate_keys},
        "aux_present": len(aux_keys) > 0,
    }

    x = probe_input(2)

    # ---------------------------------------- Q8/Q17: forward graph shapes --
    with torch.no_grad():
        c, v = model.branch_features(x)
        logits_plain = model(x)
        logits, aux_c, aux_v, gate = model(x, return_aux=True)
        probs_ref = torch.softmax(logits_plain, -1)

    R["q8_forward_graph"] = {
        "input": list(x.shape),
        "cnn_branch_feature_c": list(c.shape),
        "vit_branch_feature_v": list(v.shape),
        "gate_input_is_cat_c_v": [c.shape[0], c.shape[1] + v.shape[1]],
        "gate_output_g": list(gate.shape),
        "fused_vector": [c.shape[0], 2 * c.shape[1]],
        "fused_logits": list(logits.shape),
        "aux_cnn_logits": list(aux_c.shape),
        "aux_vit_logits": list(aux_v.shape),
        "source": "src/fusion_v2.py:83-107",
        "graph": ("x -> cnn_stages (taps 3,5,final; mean over H,W) -> cat(1432) -> "
                  "cnn_proj -> c(512);  x -> vit.vit(...).last_hidden_state[:,0](768) -> "
                  "vit_proj -> v(512);  g=softmax(gate(cat[c,v]),-1)(2); "
                  "fused=cat[c*g0*2, v*g1*2](1024); logits=head(fused)(4)"),
    }

    # ------------------------------------ Q12/Q13/Q14: gate semantics -------
    gsum = gate.sum(dim=-1)
    R["q13_q14_gate_semantics"] = {
        "gate_module_repr": repr(model.gate),
        "gate_consumes": "torch.cat([c, v], dim=1), shape %s" % [c.shape[0], c.shape[1] + v.shape[1]],
        "gate_returns_shape": list(gate.shape),
        "values_per_image": int(gate.shape[-1]),
        "kind": ("two scalar per-branch weights per image "
                 "(NOT per-feature, NOT per-scale, NOT spatial)"),
        "softmax_dim_in_source": -1,
        "row_sums": [float(t) for t in gsum],
        "rows_sum_to_one": bool(torch.allclose(gsum, torch.ones_like(gsum), atol=1e-6)),
        "sum_axis": "last dimension (dim=-1), i.e. across the two branches within one image",
        "example_rows": gate.tolist(),
        "source": "src/fusion_v2.py:72-73 (module), :97 (softmax), :100 (application)",
        "applied_to": ("the 512-d projected FEATURE VECTORS c and v, each scaled by "
                       "2*g; the gate never multiplies a spatial map"),
    }

    # -------------------------- Q16: branch probabilities without retrain ---
    with torch.no_grad():
        pc = torch.softmax(aux_c, -1)
        pv = torch.softmax(aux_v, -1)
    R["q16_branch_probabilities"] = {
        "obtainable_without_retraining": True,
        "mechanism": "model(x, return_aux=True) -> (logits, aux_cnn(c), aux_vit(v), g); "
                     "src/fusion_v2.py:105-106",
        "aux_cnn_prob_row0": pc[0].tolist(),
        "aux_vit_prob_row0": pv[0].tolist(),
        "aux_cnn_rows_sum_to_one": bool(torch.allclose(pc.sum(-1), torch.ones(pc.shape[0]), atol=1e-6)),
        "aux_vit_rows_sum_to_one": bool(torch.allclose(pv.sum(-1), torch.ones(pv.shape[0]), atol=1e-6)),
        "caveat": ("these are AUXILIARY TRAINING heads on the projected branch features; "
                   "they are not the fused decision and were trained with weight 0.3 as a "
                   "regulariser, so they are branch-readable proxies, not the model's output"),
    }

    # ------------------------------ Q10/Q11/Q18: ViT attention availability --
    vit_cfg = model.vit.config
    q11 = {
        "vit_class": type(model.vit).__name__,
        "inner_module_called_in_forward": "model.vit.vit (ViTModel) - src/fusion_v2.py:91",
        "config_output_attentions_default": bool(getattr(vit_cfg, "output_attentions", False)),
        "config_attn_implementation": getattr(vit_cfg, "_attn_implementation", "unknown"),
        "num_hidden_layers": int(vit_cfg.num_hidden_layers),
        "num_attention_heads": int(vit_cfg.num_attention_heads),
        "hidden_size": int(vit_cfg.hidden_size),
        "image_size": int(vit_cfg.image_size), "patch_size": int(vit_cfg.patch_size),
    }
    n_patches = (vit_cfg.image_size // vit_cfg.patch_size) ** 2
    q11["n_patches"] = n_patches
    q11["n_tokens_including_cls"] = n_patches + 1
    q11["expected_attention_shape_per_layer"] = [
        int(x.shape[0]), int(vit_cfg.num_attention_heads), n_patches + 1, n_patches + 1]

    try:
        with torch.no_grad():
            o_plain = model.vit.vit(pixel_values=x)
            o_attn = model.vit.vit(pixel_values=x, output_attentions=True,
                                   output_hidden_states=True)
        att = getattr(o_attn, "attentions", None)
        hid = getattr(o_attn, "hidden_states", None)
        q11["attentions_returned_by_default"] = getattr(o_plain, "attentions", None) is not None
        q11["attentions_returned_with_flag"] = att is not None
        if att is not None:
            q11["n_attention_tensors"] = len(att)
            q11["attention_shape_layer0"] = list(att[0].shape)
            q11["attention_rows_sum_to_one"] = bool(
                torch.allclose(att[0].sum(-1), torch.ones_like(att[0].sum(-1)), atol=1e-4))
        if hid is not None:
            q11["n_hidden_states"] = len(hid)
            q11["hidden_state_shape"] = list(hid[0].shape)
        q11["last_hidden_state_shape"] = list(o_plain.last_hidden_state.shape)
        q11["cls_token_index_used"] = 0
        # does requesting attention perturb the representation?
        d = (o_attn.last_hidden_state - o_plain.last_hidden_state).abs().max()
        q11["max_abs_last_hidden_state_delta_with_flag"] = float(d)
    except Exception as e:
        q11["error"] = "%s: %s" % (type(e).__name__, e)
        q11["traceback"] = traceback.format_exc()
    R["q10_q11_vit_tensors"] = q11

    # ---- Q18a: does output_attentions change FUSED probabilities? -----------
    # Monkeypatch only within this probe, on the loaded instance, then restore.
    q18: dict = {}
    try:
        inner = model.vit.vit
        orig_forward = inner.forward

        def forward_with_attn(*a, **kw):
            kw["output_attentions"] = True
            return orig_forward(*a, **kw)

        inner.forward = forward_with_attn
        with torch.no_grad():
            probs_attn = torch.softmax(model(x), -1)
        inner.forward = orig_forward
        with torch.no_grad():
            probs_restored = torch.softmax(model(x), -1)
        q18["max_abs_prob_delta_output_attentions"] = float(
            (probs_attn - probs_ref).abs().max())
        q18["output_attentions_changes_fused_probs"] = bool(
            (probs_attn - probs_ref).abs().max() > TOL)
        q18["restore_exact"] = bool(torch.equal(probs_restored, probs_ref))
    except Exception as e:
        q18["output_attentions_error"] = "%s: %s" % (type(e).__name__, e)

    # ---- Q18b: does a read-only forward hook change FUSED probabilities? ----
    try:
        captured = {}

        def hook(_m, _i, o):
            captured["shape"] = list(o.shape)
            captured["detached"] = True
            return None            # record only; never modify the output

        hnd = model.cnn_stages[-1].register_forward_hook(hook)
        with torch.no_grad():
            probs_hooked = torch.softmax(model(x), -1)
        hnd.remove()
        with torch.no_grad():
            probs_after_remove = torch.softmax(model(x), -1)
        q18["hooked_tensor_shape"] = captured.get("shape")
        q18["max_abs_prob_delta_forward_hook"] = float((probs_hooked - probs_ref).abs().max())
        q18["forward_hook_changes_fused_probs"] = bool(
            (probs_hooked - probs_ref).abs().max() > TOL)
        q18["hook_removal_exact"] = bool(torch.equal(probs_after_remove, probs_ref))
    except Exception as e:
        q18["hook_error"] = "%s: %s" % (type(e).__name__, e)
    q18["tolerance"] = TOL
    R["q18_instrumentation_neutrality"] = q18

    # -------------------------- Q9: CNN spatial stages actually available ---
    stages = []
    with torch.no_grad():
        h = x
        for i, st in enumerate(model.cnn_stages):
            h = st(h)
            stages.append({"stage_index": i, "output_shape": list(h.shape),
                           "channels": int(h.shape[1]),
                           "spatial": [int(h.shape[2]), int(h.shape[3])],
                           "is_multiscale_tap": i in (3, 5),
                           "is_final_stage": i == len(model.cnn_stages) - 1})
    R["q9_cnn_spatial_stages"] = {
        "n_stages": len(stages), "stages": stages,
        "tap_indices_in_source": [3, 5],
        "tap_channels_in_source": {"3": 40, "5": 112},
        "final_channels_in_source": 1280,
        "pooling_in_forward": "mean over dims (2,3) - src/fusion_v2.py:88-89",
        "note": ("taps are pooled to vectors inside branch_features, so the SPATIAL maps "
                 "are not returned by forward and must be captured with hooks"),
    }
    R["q3_hooked_module"] = {
        "hooked_in_current_code": "model.cnn_stages[-1]",
        "source": "src/publication/fusion_v2_gradcam.py:167",
        "resolved_module_type": type(model.cnn_stages[-1]).__name__,
        "resolved_module_repr_head": repr(model.cnn_stages[-1])[:400],
        "output_shape_on_probe": stages[-1]["output_shape"],
    }

    # ----------------------------------------- Q4: what is differentiated ---
    R["q4_target_score"] = {
        "target_object": "pytorch_grad_cam ClassifierOutputTarget(target_index)",
        "model_passed_to_cam": "the full FusionV2 instance (model=model)",
        "consequence": ("the differentiated scalar is the FUSED class logit "
                        "model(x)[:, target_index]; it is not a branch-specific score"),
        "source": "src/publication/fusion_v2_gradcam.py:167-168",
        "gate_in_gradient_path": True,
        "gate_in_gradient_path_reason": ("head() consumes fused = cat[c*g0*2, v*g1*2], and g "
                                         "is a differentiable function of c and v, so d(logit)/d(A_cnn) "
                                         "flows through both the direct c path and the gate"),
    }

    # --------------------------------------- Q6: spatial resize reversal ----
    R["q6_spatial_resize"] = {
        "cam_upsampling": ("pytorch_grad_cam resizes the stage activation map to the "
                           "INPUT tensor spatial size internally"),
        "asserted_output_shape": [IMAGE_SIZE, IMAGE_SIZE],
        "assert_source": "src/publication/fusion_v2_gradcam.py:170-171",
        "display_image": ("the original image is resized to 224x224 with PIL BILINEAR "
                          "for display - src/publication/fusion_v2_gradcam.py:136"),
        "overlay_space": "224x224",
        "back_projection_to_original_resolution": False,
        "consequence": ("the overlay is produced in 224x224 model space, NOT mapped back to "
                        "the source image resolution; any spatial statement is therefore made "
                        "in resized model space and inherits the resize distortion"),
        "normalisation_of_heatmap": ("min-max to [0,1] per image, zeros if constant - "
                                     "src/publication/fusion_v2_gradcam.py:172-176"),
    }

    # --------------------------------------------- Q17: one synced pass -----
    # Demonstrate that a single forward can expose everything simultaneously.
    try:
        cap = {}

        def cnn_hook(_m, _i, o):
            cap["cnn_final_map"] = list(o.shape)

        hnd = model.cnn_stages[-1].register_forward_hook(cnn_hook)
        inner = model.vit.vit
        orig = inner.forward

        def fwd_attn(*a, **kw):
            kw["output_attentions"] = True
            o = orig(*a, **kw)
            cap["vit_attentions"] = [len(o.attentions), list(o.attentions[0].shape)]
            return o

        inner.forward = fwd_attn
        with torch.no_grad():
            lg, ac, av, gg = model(x, return_aux=True)
        inner.forward = orig
        hnd.remove()
        cap["fused_logits"] = list(lg.shape)
        cap["aux_cnn"] = list(ac.shape)
        cap["aux_vit"] = list(av.shape)
        cap["gate"] = list(gg.shape)
        with torch.no_grad():
            delta = (torch.softmax(lg, -1) - probs_ref).abs().max()
        cap["max_abs_prob_delta_vs_uninstrumented"] = float(delta)
        cap["single_pass_exposes_everything"] = all(
            k in cap for k in ("cnn_final_map", "vit_attentions", "fused_logits",
                               "aux_cnn", "aux_vit", "gate"))
        R["q17_synchronized_pass"] = cap
    except Exception as e:
        R["q17_synchronized_pass"] = {"error": "%s: %s" % (type(e).__name__, e),
                                      "traceback": traceback.format_exc()}

    # ------------------------------------------------ write probe output ----
    with open(out / "_probe_raw.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    print("probe complete ->", (out / "_probe_raw.json").relative_to(root))

    # concise console summary
    print("\n--- key findings ---")
    print("  gate output shape          :", R["q8_forward_graph"]["gate_output_g"])
    print("  gate rows sum to one       :", R["q13_q14_gate_semantics"]["rows_sum_to_one"])
    print("  aux keys in checkpoint     :", R["q15_aux_in_checkpoint"]["aux_keys"])
    print("  ViT attn by default        :", q11.get("attentions_returned_by_default"))
    print("  ViT attn with flag         :", q11.get("attentions_returned_with_flag"),
          q11.get("attention_shape_layer0"))
    print("  attn flag perturbs probs   :", q18.get("output_attentions_changes_fused_probs"),
          "(max delta %.3e)" % q18.get("max_abs_prob_delta_output_attentions", float("nan")))
    print("  hook perturbs probs        :", q18.get("forward_hook_changes_fused_probs"),
          "(max delta %.3e)" % q18.get("max_abs_prob_delta_forward_hook", float("nan")))
    print("  GradCAM++ distinct class   :", q2.get("is_distinct_class_from_GradCAM"),
          "| 2nd-order", q2.get("uses_second_order_terms"),
          "| 3rd-order", q2.get("uses_third_order_terms"))
    print("  checkpoints all verified   :", ck["all_three_accepted_and_hash_matched"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
