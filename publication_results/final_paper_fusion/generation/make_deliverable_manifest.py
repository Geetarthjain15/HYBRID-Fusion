from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
MANIFEST = OUT / "DELIVERABLE_MANIFEST.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


files = []
for path in sorted(OUT.rglob("*")):
    if not path.is_file() or path == MANIFEST:
        continue
    files.append({
        "path": path.relative_to(OUT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    })

payload = {
    "package": "final multi-scale EfficientNet-B0–ViT-B/16 fusion manuscript",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "file_count_excluding_manifest": len(files),
    "files": files,
}
MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"{MANIFEST}: {len(files)} files")
