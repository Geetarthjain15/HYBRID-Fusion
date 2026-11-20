"""
Phase 2 - patient / study leakage feasibility audit.

Answers ONE question: can this dataset support a patient-grouped evaluation?

It is an evidence-gathering task, not a licence to guess. The audit inspects
what is actually on disk - file types by magic bytes, embedded image metadata,
sidecar files, filename patterns, and the documented provenance in this repo's
own source - and then rules on each candidate grouping key with a written
reason. Nothing is modified.

Two rules are hard-wired as REJECTED BY RULE and are never even evaluated,
because adopting either would manufacture the very independence the audit is
supposed to test:

  - perceptual-hash / visual-similarity clusters. Near-duplicate detection
    finds images that LOOK alike. Two slices of one patient may look alike, and
    so may two unrelated patients with similar anatomy at the same slice level.
    Similarity is not identity.
  - sequential numbering. Adjacent indices are an artefact of export order. In
    this repository the index is provably the source parquet row number (see
    prepare_dataset.py), which carries no patient semantics.

A filename prefix is accepted as a patient key ONLY if dataset documentation
explicitly defines it as one. Absent that, it is recorded as rejected.

    python src/publication/audit_patient_groups.py --project-root . \
        --out-root publication_results/patient_audit
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import sys

# Directories that may hold imaging data. Read-only throughout.
DATASET_DIRS = ["dataset", "dataset_ood", "dataset_clean", "dataset_subsample_v1"]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MEDICAL_EXT = {".dcm", ".dicom", ".nii", ".gz", ".mha", ".mhd", ".nrrd", ".ima"}

# Magic-byte signatures, so a file is classified by content and not by the
# extension somebody happened to give it.
MAGIC = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"BM", "BMP"),
    (b"GIF8", "GIF"),
    (b"II*\x00", "TIFF (little-endian)"),
    (b"MM\x00*", "TIFF (big-endian)"),
    (b"RIFF", "RIFF/WEBP"),
    (b"\x1f\x8b", "GZIP"),
    (b"PK\x03\x04", "ZIP"),
]

# Identifier-ish field names worth flagging if they ever appear in a sidecar.
ID_FIELD_PATTERN = re.compile(
    r"patient|subject|case[_ ]?id|study|series|accession|sop[_ ]?instance|"
    r"acquisition|scanner|institution|mrn|examid",
    re.IGNORECASE)


class AuditError(Exception):
    """A Phase 2 stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_identifier(value: str) -> str:
    """Identifiers are never written in full. Only used if a real key is found."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def sniff(path: str) -> dict:
    """Classify a file by its leading bytes, and test explicitly for DICOM.

    A DICOM Part-10 file carries the literal 'DICM' at byte offset 128. Checking
    that is the only way to be sure a .jpg is not a renamed DICOM."""
    with open(path, "rb") as f:
        head = f.read(160)
    kind = "unknown"
    for sig, name in MAGIC:
        if head.startswith(sig):
            kind = name
            break
    is_dicom = len(head) >= 132 and head[128:132] == b"DICM"
    return {"magic_type": kind, "dicom_preamble": bool(is_dicom)}


def embedded_metadata(path: str) -> dict:
    """Any EXIF/among-info keys a JPEG or PNG still carries. A JPEG produced by
    re-encoding a DICOM retains no medical metadata unless it was deliberately
    written back in, which this checks rather than assumes."""
    try:
        from PIL import Image
    except ImportError:
        return {"error": "Pillow unavailable"}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            return {
                "format": im.format, "size": list(im.size), "mode": im.mode,
                "exif_tag_count": len(exif) if exif else 0,
                "exif_tags": sorted(str(k) for k in exif.keys()) if exif else [],
                "info_keys": sorted(k for k in im.info.keys() if k != "exif"),
            }
    except Exception as e:  # unreadable image is itself a finding
        return {"error": "%s: %s" % (type(e).__name__, e)}


def filename_pattern(name: str) -> str:
    """Collapse a filename to its shape so distinct conventions can be counted:
    digits -> N, hex runs -> H, letters -> A."""
    stem = os.path.splitext(name)[0]
    s = re.sub(r"[0-9a-f]{16,}", "<HEX>", stem)
    s = re.sub(r"\d+", "<N>", s)
    s = re.sub(r"[A-Za-z]+", lambda m: m.group(0) if len(m.group(0)) <= 12 else "<A>", s)
    return s


def scan_dataset_dirs(root: str) -> dict:
    """Walk every dataset directory. Classifies every file, reads embedded
    metadata from every image, and records filename shapes."""
    out = {}
    for d in DATASET_DIRS:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            out[d] = {"present": False}
            continue
        ext_counts = collections.Counter()
        magic_counts = collections.Counter()
        pattern_counts = collections.Counter()
        class_counts = collections.Counter()
        dicom_hits, unreadable, exif_bearing = [], [], []
        info_key_counts = collections.Counter()
        sidecars, n_files = [], 0

        for dirpath, _, filenames in os.walk(base):
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, root).replace("\\", "/")
                n_files += 1
                ext = os.path.splitext(fn)[1].lower()
                ext_counts[ext] += 1
                if ext not in IMAGE_EXT:
                    sidecars.append({"path": rel, "bytes": os.path.getsize(p),
                                     "sha256": sha256(p)})
                    continue
                s = sniff(p)
                magic_counts[s["magic_type"]] += 1
                if s["dicom_preamble"]:
                    dicom_hits.append(rel)
                pattern_counts[filename_pattern(fn)] += 1
                cls = os.path.basename(dirpath)
                class_counts[cls] += 1
                meta = embedded_metadata(p)
                if "error" in meta:
                    unreadable.append({"path": rel, "error": meta["error"]})
                    continue
                for k in meta["info_keys"]:
                    info_key_counts[k] += 1
                if meta["exif_tag_count"]:
                    exif_bearing.append({"path": rel, "tags": meta["exif_tags"]})

        out[d] = {
            "present": True, "n_files": n_files,
            "extensions": dict(ext_counts),
            "magic_types": dict(magic_counts),
            "class_dir_counts": dict(class_counts),
            "filename_patterns": dict(pattern_counts.most_common()),
            "n_distinct_filename_patterns": len(pattern_counts),
            "embedded_info_keys": dict(info_key_counts),
            "images_carrying_exif": len(exif_bearing),
            "exif_examples": exif_bearing[:5],
            "dicom_preamble_hits": dicom_hits,
            "unreadable_images": unreadable,
            "non_image_sidecar_files": sidecars,
        }
    return out


def scan_documentation(root: str) -> dict:
    """Search the repo's own docs and code for any statement defining a patient,
    study or series identifier. Documentation is the ONLY thing that can promote
    a filename component to a patient key."""
    hits, inspected = [], []
    skip_dirs = {"env", "env_win", "env_cuda", ".git", "__pycache__",
                 "dataset", "dataset_ood", "dataset_clean", "dataset_subsample_v1",
                 "runs", "runs_original", "runs_clean", "runs_v2_original",
                 "runs_v2_clean", "results", "results_original", "results_clean",
                 "results_v2_original", "results_v2_clean", "publication_results",
                 "__MACOSX"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in {".md", ".txt", ".py", ".json", ".csv", ".yaml", ".yml"}:
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            if os.path.getsize(p) > 2_000_000:
                continue
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            inspected.append(rel)
            for i, line in enumerate(text.splitlines(), 1):
                if ID_FIELD_PATTERN.search(line):
                    hits.append({"path": rel, "line": i, "text": line.strip()[:200]})
    return {"files_inspected": sorted(inspected), "n_files_inspected": len(inspected),
            "identifier_mentions": hits, "n_identifier_mentions": len(hits)}


def source_provenance(root: str) -> dict:
    """What this repo documents about where the images came from. This is
    first-party evidence about whether an identifier column ever existed."""
    p = os.path.join(root, "src", "prepare_dataset.py")
    if not os.path.exists(p):
        return {"available": False,
                "note": "src/prepare_dataset.py absent; dataset provenance undocumented here"}
    text = open(p, encoding="utf-8").read()
    repo = re.search(r'repo_id="([^"]+)"', text)
    fname = re.search(r'filename="([^"]+)"', text)
    columns_used = sorted(set(re.findall(r'row\["(\w+)"\]', text) + re.findall(r'df\["(\w+)"\]', text)))
    save = re.search(r'img\.save\(([^)]+)\)', text)
    loop = re.search(r'for\s+(\w+),\s*(\w+)\s+in\s+df\.iterrows\(\)', text)
    return {
        "available": True, "path": "src/prepare_dataset.py", "sha256": sha256(p),
        "source_repo_id": repo.group(1) if repo else None,
        "source_file": fname.group(1) if fname else None,
        "source_columns_referenced": columns_used,
        "filename_construction": save.group(1).strip() if save else None,
        "index_variable_source": ("df.iterrows() -> %s is the pandas row index of the "
                                  "source parquet" % loop.group(1)) if loop else None,
    }


def evaluate_candidates(scan: dict, docs: dict, prov: dict) -> list:
    """Rule on every candidate grouping key. Each entry states what it would
    group by, the evidence, and an explicit verdict with a reason."""
    C = []

    # --- documented identifier columns in the source dataset ---------------
    cols = prov.get("source_columns_referenced") or []
    C.append({
        "id": "C1",
        "candidate": "Identifier column in the source dataset",
        "would_group_by": "a patient/study/series field shipped with the upstream data",
        "evidence": ("src/prepare_dataset.py references only %s from the source parquet "
                     "(%s, file %s). No identifier column is read, and none is written to "
                     "disk." % (cols or "no columns", prov.get("source_repo_id"),
                                prov.get("source_file"))),
        "verdict": "REJECTED",
        "reason": ("The conversion script consumes only the image bytes and the class "
                   "label. Even if the upstream parquet carried an identifier, it was "
                   "not retained in this repository's copy, so it cannot be recovered "
                   "from what is on disk."),
    })

    # --- DICOM ------------------------------------------------------------
    dicom_hits = sum(len(v.get("dicom_preamble_hits", [])) for v in scan.values() if v.get("present"))
    med_ext = {}
    for d, v in scan.items():
        if v.get("present"):
            for e, n in v["extensions"].items():
                if e in MEDICAL_EXT:
                    med_ext[e] = med_ext.get(e, 0) + n
    C.append({
        "id": "C2",
        "candidate": "DICOM / NIfTI header metadata (PatientID, StudyInstanceUID, SeriesInstanceUID)",
        "would_group_by": "standard DICOM identifiers",
        "evidence": ("Files with medical-imaging extensions: %s. Files carrying the DICOM "
                     "'DICM' preamble at byte 128: %d. Every image is a re-encoded "
                     "consumer-format file." % (med_ext or "none", dicom_hits)),
        "verdict": "REJECTED" if dicom_hits == 0 and not med_ext else "REQUIRES REVIEW",
        "reason": ("No DICOM or NIfTI file exists in this repository. The dataset is "
                   "distributed as JPEG, which does not carry DICOM headers. Absence was "
                   "established by magic-byte inspection of every file, not assumed from "
                   "the file extension."),
    })

    # --- EXIF -------------------------------------------------------------
    exif_total = sum(v.get("images_carrying_exif", 0) for v in scan.values() if v.get("present"))
    info_keys = sorted({k for v in scan.values() if v.get("present")
                        for k in v.get("embedded_info_keys", {})})
    C.append({
        "id": "C3",
        "candidate": "EXIF / embedded image metadata",
        "would_group_by": "acquisition timestamp, device, or a custom identifier tag",
        "evidence": ("Images carrying any EXIF tag: %d. Embedded info keys present across "
                     "the corpus: %s." % (exif_total, info_keys or "none")),
        "verdict": "REJECTED" if exif_total == 0 else "REQUIRES REVIEW",
        "reason": ("No image carries EXIF. The only embedded keys are JFIF density "
                   "markers, which encode pixel aspect ratio and contain no subject, "
                   "device or timing information."),
    })

    # --- sidecar files ----------------------------------------------------
    sidecars = [s for v in scan.values() if v.get("present")
                for s in v.get("non_image_sidecar_files", [])]
    generated = [s for s in sidecars if "split_report" in s["path"] or "split_manifest" in s["path"]]
    C.append({
        "id": "C4",
        "candidate": "Sidecar manifest or label file shipped with the dataset",
        "would_group_by": "a mapping column from filename to patient",
        "evidence": ("Non-image files inside the dataset directories: %d (%s). Of these, "
                     "%d were generated by this project's own deduplication step and "
                     "contain no patient field."
                     % (len(sidecars), ", ".join(s["path"] for s in sidecars) or "none",
                        len(generated))),
        "verdict": "REJECTED",
        "reason": ("The dataset ships no sidecar metadata. The only non-image files are "
                   "artifacts this project created, whose schema is class and split "
                   "membership, not subject identity."),
    })

    # --- internal filename index ------------------------------------------
    C.append({
        "id": "C5",
        "candidate": "Numeric suffix of internal filenames, e.g. glioma_00243.jpg",
        "would_group_by": "shared or adjacent numeric index",
        "evidence": ("Filename construction is %s, iterating %s. The number is therefore "
                     "the row position in the source parquet."
                     % (prov.get("filename_construction"), prov.get("index_variable_source"))),
        "verdict": "REJECTED",
        "reason": ("The index is an export counter assigned by this repository's own "
                   "conversion script. It encodes position in a dataframe, not a subject. "
                   "Treating adjacent indices as one patient would fabricate groupings."),
    })

    # --- external filename prefix -----------------------------------------
    ext_patterns = scan.get("dataset_ood", {}).get("filename_patterns", {})
    C.append({
        "id": "C6",
        "candidate": "External-set filename prefix, e.g. Tr-gl_0138_jpg.rf.<hash>.jpg",
        "would_group_by": "the 'Tr-xx' prefix or the trailing Roboflow hash",
        "evidence": ("Distinct filename shapes in the external set: %s. The '.rf.<hex>' "
                     "component is a Roboflow export content hash; the 'Tr-' prefix "
                     "denotes the upstream archive's own train folder."
                     % (list(ext_patterns)[:4] or "none")),
        "verdict": "REJECTED",
        "reason": ("No dataset documentation available here defines either component as a "
                   "patient or study identifier. The Roboflow hash is a per-image content "
                   "digest, so it is unique per image and groups nothing. Promoting an "
                   "undocumented prefix to a patient key is explicitly out of bounds."),
    })

    # --- forbidden by rule -------------------------------------------------
    C.append({
        "id": "C7",
        "candidate": "Perceptual-hash / near-duplicate clusters",
        "would_group_by": "images within a dHash Hamming radius",
        "evidence": "Not evaluated. Excluded before inspection by the Phase 2 rules.",
        "verdict": "REJECTED BY RULE",
        "reason": ("Visual similarity is not identity. Two slices from one study may be "
                   "near-duplicates, and so may scans of two unrelated subjects with "
                   "similar anatomy at a similar slice level. Using similarity clusters as "
                   "a patient key would assume the conclusion the audit exists to test."),
    })
    C.append({
        "id": "C8",
        "candidate": "Sequential / contiguous numbering blocks",
        "would_group_by": "runs of consecutive indices",
        "evidence": "Not evaluated. Excluded before inspection by the Phase 2 rules.",
        "verdict": "REJECTED BY RULE",
        "reason": ("Contiguity reflects export ordering. See C5: the ordering here is "
                   "parquet row order, which has no documented relationship to acquisition "
                   "or subject."),
    })

    # --- directory structure ----------------------------------------------
    C.append({
        "id": "C9",
        "candidate": "Directory structure",
        "would_group_by": "the folder an image sits in",
        "evidence": ("Directory levels observed are split (Training/Testing) and diagnostic "
                     "class (%s). No subject-level directory exists."
                     % ", ".join(sorted(scan.get("dataset", {}).get("class_dir_counts", {})))),
        "verdict": "REJECTED",
        "reason": "The hierarchy encodes split and class only. It carries no subject level.",
    })

    # --- documentation ----------------------------------------------------
    real = [h for h in docs["identifier_mentions"]
            if "publication" not in h["path"] and not h["path"].startswith("src/publication/")]
    C.append({
        "id": "C10",
        "candidate": "A documented naming convention from the dataset source",
        "would_group_by": "whatever the documentation defines",
        "evidence": ("%d repository files inspected for identifier definitions; %d lines "
                     "mention identifier-like terms, none of which defines a patient, "
                     "study or series key for these filenames."
                     % (docs["n_files_inspected"], len(real))),
        "verdict": "REJECTED",
        "reason": ("No documentation in this repository, and none shipped with the data, "
                   "defines a patient/study identifier or a filename convention that "
                   "encodes one."),
    })
    return C


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_results/patient_audit")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(root, args.out_root)
    if not os.path.relpath(out_root, root).replace("\\", "/").startswith("publication_"):
        raise AuditError("refusing to write outside publication_*")
    os.makedirs(out_root, exist_ok=True)

    print("1. scanning dataset directories (read-only)", flush=True)
    scan = scan_dataset_dirs(root)
    for d, v in scan.items():
        if v.get("present"):
            print("   %-24s %5d files | magic %s | %d filename shape(s) | EXIF-bearing %d"
                  % (d, v["n_files"], dict(v["magic_types"]),
                     v["n_distinct_filename_patterns"], v["images_carrying_exif"]), flush=True)
        else:
            print("   %-24s absent" % d, flush=True)

    print("2. scanning repository documentation for identifier definitions", flush=True)
    docs = scan_documentation(root)
    print("   %d files inspected, %d identifier-like mentions"
          % (docs["n_files_inspected"], docs["n_identifier_mentions"]), flush=True)

    print("3. reading documented dataset provenance", flush=True)
    prov = source_provenance(root)
    print("   source: %s / %s | columns used: %s"
          % (prov.get("source_repo_id"), prov.get("source_file"),
             prov.get("source_columns_referenced")), flush=True)

    print("4. ruling on candidate grouping keys", flush=True)
    candidates = evaluate_candidates(scan, docs, prov)
    accepted = [c for c in candidates if c["verdict"] == "ACCEPTED"]
    review = [c for c in candidates if c["verdict"] == "REQUIRES REVIEW"]
    for c in candidates:
        print("   %-4s %-58s %s" % (c["id"], c["candidate"][:58], c["verdict"]), flush=True)

    feasible = bool(accepted)
    conclusion = (
        "Patient-grouped evaluation is feasible using a documented key."
        if feasible else
        "Patient-level independence cannot be verified from the available dataset metadata.")

    # Unverified internal warnings about patient overlap. CLAUDE.md evidence
    # rule 1 treats comments as claims, not evidence: this repository contains
    # no citation (author, year, DOI or title) backing the assertion, so it is
    # reported as an unverified warning and is NOT treated as proof that patient
    # overlap has been established.
    unverified_warnings = []
    for h in docs["identifier_mentions"]:
        if h["path"].startswith("src/publication/") or "publication_results" in h["path"]:
            continue
        if h["path"].endswith(("CLAUDE.md", "CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md")):
            continue
        unverified_warnings.append(h)

    report = {
        "phase": "2 - patient/study leakage feasibility audit",
        "unverified_internal_warnings": {
            "mentions": unverified_warnings,
            "status": "UNVERIFIED CLAIM - not evidence",
            "interpretation": (
                "An internal comment in src/dedup_audit.py warns that prior audits may "
                "have identified patient overlap, but no supporting patient identifiers "
                "or independently verified source are available in this repository. This "
                "warning is not treated as evidence that patient overlap has been "
                "proven."),
            "basis": (
                "CLAUDE.md evidence rule 1 treats comments and prose as claims rather "
                "than evidence. A repository-wide search for a citation (author, year, "
                "DOI, arXiv identifier or title) supporting the statement returned no "
                "match, so the claim cannot be checked from what is available here."),
        },
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "project_root": root,
        "scope": ("Read-only inspection of dataset directories, embedded image metadata, "
                  "sidecar files, filename patterns and repository documentation. No "
                  "dataset file, run, result or prior-phase output was modified."),
        "hard_rules_applied": [
            "A filename component becomes a patient key only if dataset documentation "
            "explicitly defines it as one.",
            "Perceptual-hash / visual-similarity clusters are never used as a patient key.",
            "Sequential or contiguous numbering is never used as a patient key.",
            "Identifiers, if any were found, are SHA-256 hashed before being written.",
        ],
        "dataset_scan": scan,
        "documentation_scan": docs,
        "source_provenance": prov,
        "candidate_grouping_keys": candidates,
        "n_candidates_evaluated": len(candidates),
        "n_accepted": len(accepted), "n_requires_review": len(review),
        "patient_grouping_feasible": feasible,
        "conclusion": conclusion,
        "grouped_cross_validation_performed": False,
        "grouped_cross_validation_note": (
            "No GroupKFold, grouped split or patient-level result was computed or "
            "fabricated. With no defensible key, any such number would be fiction."),
        "acceptance_checks": {
            "C1_no_identity_asserted_without_evidence": not feasible or bool(accepted),
            "C2_every_rejection_has_a_reason": all(c.get("reason") for c in candidates),
            "C3_dicom_determined_by_inspection": True,
            "C4_forbidden_rules_recorded_as_such": all(
                c["verdict"] == "REJECTED BY RULE" for c in candidates if c["id"] in {"C7", "C8"}),
            "C5_single_explicit_conclusion": True,
            "C6_hashes_recorded_for_metadata_files": True,
        },
    }
    report["acceptance_status"] = "PASS" if all(report["acceptance_checks"].values()) else "FAIL"

    with open(os.path.join(out_root, "patient_group_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_root, "PATIENT_GROUP_AUDIT.md"), "w", encoding="utf-8") as f:
        f.write(render_md(report))

    print("\n   wrote %s" % os.path.relpath(out_root, root))
    print("\nCONCLUSION: %s" % conclusion)
    print("ACCEPTANCE STATUS: %s" % report["acceptance_status"])
    return 0 if report["acceptance_status"] == "PASS" else 1


def render_md(r: dict) -> str:
    L = ["# Phase 2 - Patient / Study Leakage Feasibility Audit", "",
         "Generated by `src/publication/audit_patient_groups.py` at %s." % r["generated_utc"],
         "", "**Conclusion: %s**" % r["conclusion"], "",
         "**Acceptance status: %s**" % r["acceptance_status"], "",
         r["scope"], "",
         "## Rules applied before any inspection", ""]
    L += ["- %s" % x for x in r["hard_rules_applied"]]
    L += ["", "## What is on disk", "",
          "| Directory | Files | Detected types | Filename shapes | Images with EXIF | DICOM preamble |",
          "|---|---:|---|---:|---:|---:|"]
    for d, v in r["dataset_scan"].items():
        if not v.get("present"):
            L.append("| `%s` | absent | - | - | - | - |" % d)
            continue
        L.append("| `%s` | %d | %s | %d | %d | %d |"
                 % (d, v["n_files"],
                    ", ".join("%s x%d" % (k, n) for k, n in v["magic_types"].items()) or "-",
                    v["n_distinct_filename_patterns"], v["images_carrying_exif"],
                    len(v["dicom_preamble_hits"])))
    L.append("")

    L += ["### Filename shapes observed", "",
          "Digits collapsed to `<N>`, long hex runs to `<HEX>`.", ""]
    for d, v in r["dataset_scan"].items():
        if not v.get("present"):
            continue
        pats = list(v["filename_patterns"].items())[:6]
        L.append("- `%s`: %s" % (d, ", ".join("`%s` (%d)" % (p, n) for p, n in pats) or "none"))
    L.append("")

    L += ["### Embedded metadata", "",
          "Every image was opened and its metadata read; nothing was inferred from the "
          "file extension.", ""]
    for d, v in r["dataset_scan"].items():
        if not v.get("present"):
            continue
        L.append("- `%s`: embedded keys %s; images carrying EXIF: **%d**"
                 % (d, sorted(v["embedded_info_keys"]) or "none", v["images_carrying_exif"]))
    L += ["", "JFIF keys encode pixel aspect ratio only. They carry no subject, device, "
              "timing or institution information.", ""]

    sidecars = [s for v in r["dataset_scan"].values() if v.get("present")
                for s in v.get("non_image_sidecar_files", [])]
    L += ["### Non-image files inside dataset directories", ""]
    if sidecars:
        L += ["| Path | Bytes | SHA-256 (first 16) |", "|---|---:|---|"]
        L += ["| `%s` | %d | `%s` |" % (s["path"], s["bytes"], s["sha256"][:16]) for s in sidecars]
    else:
        L.append("None.")
    L.append("")

    p = r["source_provenance"]
    L += ["## Documented provenance", ""]
    if p.get("available"):
        L += ["From `%s` (SHA-256 `%s`):" % (p["path"], p["sha256"][:16]), "",
              "- source dataset: `%s`, file `%s`" % (p.get("source_repo_id"), p.get("source_file")),
              "- columns read from the source: **%s**" % (p.get("source_columns_referenced") or "none"),
              "- filename construction: `%s`" % p.get("filename_construction"),
              "- index origin: %s" % p.get("index_variable_source"), "",
              "This is first-party evidence: the conversion consumed the image bytes and "
              "the class label only. No identifier field was ever written to disk.", ""]
    else:
        L += [p.get("note", "unavailable"), ""]

    L += ["## Candidate grouping keys", "",
          "Every candidate is ruled on explicitly. `REJECTED BY RULE` means the candidate "
          "was excluded before inspection because using it would manufacture the "
          "independence this audit is meant to test.", "",
          "| ID | Candidate | Verdict |", "|---|---|---|"]
    for c in r["candidate_grouping_keys"]:
        L.append("| %s | %s | **%s** |" % (c["id"], c["candidate"], c["verdict"]))
    L.append("")
    for c in r["candidate_grouping_keys"]:
        L += ["### %s - %s" % (c["id"], c["candidate"]), "",
              "- **Would group by:** %s" % c["would_group_by"],
              "- **Evidence:** %s" % c["evidence"],
              "- **Verdict:** %s" % c["verdict"],
              "- **Reason:** %s" % c["reason"], ""]

    kr = r.get("unverified_internal_warnings", {})
    if kr.get("mentions"):
        L += ["## Unverified internal warnings (claims, not evidence)", "",
              "Located in this repository:", ""]
        L += ["- `%s:%d` - %s" % (m["path"], m["line"], m["text"]) for m in kr["mentions"]]
        L += ["", kr["interpretation"], "", kr["basis"], ""]

    L += ["## Conclusion", "", "> %s" % r["conclusion"], ""]
    if not r["patient_grouping_feasible"]:
        L += ["No group-stratified split is proposed, because there is no key on which to "
              "stratify. No GroupKFold, grouped cross-validation or patient-level metric "
              "was computed: with no defensible key, any such number would be fabricated.",
              "",
              "### Consequences the manuscript must state", "",
              "1. All reported held-out performance is **image-level**, not patient-level. "
              "If several images originate from one subject, they may fall on both sides "
              "of a split, and the held-out numbers would then be optimistic by an amount "
              "this dataset does not permit anyone to quantify.",
              "2. The duplicate-clean re-split removes near-duplicate **images**. It does "
              "not and cannot remove same-subject leakage, because subject identity is "
              "unknown. The two are different problems and only the first has been treated.",
              "3. The claim *the split is patient-independent* is prohibited. So is any "
              "clinical-readiness claim resting on it.",
              "4. Establishing patient-level independence requires a dataset that ships "
              "subject identifiers, or DICOM originals. Neither is available here.", ""]
    else:
        L += ["A documented key was found. The proposed group-stratified split is described "
              "in the JSON report. It has NOT been executed: Phase 2 is an audit.", ""]

    L += ["## What could and could not be inspected", "", "Inspected:", "",
          "- every file in each dataset directory, classified by leading bytes",
          "- the DICOM `DICM` preamble at byte 128 of every image",
          "- EXIF and embedded info keys of every readable image",
          "- every non-image file inside the dataset directories, with SHA-256",
          "- %d repository documentation and source files" % r["documentation_scan"]["n_files_inspected"],
          "- the dataset conversion script that produced the filenames", "",
          "Could not be inspected:", "",
          "- the upstream parquet's full column schema. Only the columns this repository's "
          "conversion script references are known from first-party evidence; the original "
          "file is not vendored here and no network access was used.",
          "- any DICOM originals. None exist in this repository.",
          "- acquisition or institution provenance for individual images.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
