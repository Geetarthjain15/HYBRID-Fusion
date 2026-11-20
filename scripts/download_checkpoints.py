"""Download and SHA-256 verify the FusionV2 demo checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


REPOSITORY = "Geetarthjain15/HYBRID-Fusion"
RELEASE_TAG = "v1.0.0-model-checkpoints"
CHECKPOINTS = {
    42: "9b328ba83df5d44b542e19faae68fd1493fee61e0c5b80e493c2f8a387ed592e",
    43: "9f97b6aa0c9ec3f1124e593c8c29e11d31f4aef8bc08e20a40e7ed1f29e80f25",
    44: "4155a2b71df99a6159e1e4a5dbe635f4e9ea7852a6704d804dd1fb5d43aca434",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(seed: int, project_root: Path, force: bool) -> None:
    expected = CHECKPOINTS[seed]
    destination = (
        project_root
        / "publication_runs"
        / "recovered_checkpoints"
        / "clean"
        / "full_fusion_v2"
        / f"seed_{seed}"
        / "checkpoint.pth"
    )
    if destination.is_file():
        actual = sha256(destination)
        if actual == expected:
            print(f"seed {seed}: already present and verified")
            return
        if not force:
            raise RuntimeError(
                f"seed {seed}: existing checkpoint has SHA-256 {actual}; "
                "refusing to overwrite it without --force"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = f"fusionv2_seed{seed}_checkpoint.pth"
    url = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{asset}"
    temporary = destination.with_suffix(".pth.download")
    print(f"seed {seed}: downloading {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "HYBRID-Fusion-setup"})
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != expected:
            raise RuntimeError(
                f"seed {seed}: SHA-256 mismatch; expected {expected}, got {actual}"
            )
        temporary.replace(destination)
        print(f"seed {seed}: verified and installed")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, choices=tuple(CHECKPOINTS), action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    try:
        for seed in args.seed or sorted(CHECKPOINTS):
            download(seed, root, args.force)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
