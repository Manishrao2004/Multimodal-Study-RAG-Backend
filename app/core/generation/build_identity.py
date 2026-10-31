"""Fingerprint the loaded pipeline sources for reproducible local evaluations."""
from hashlib import sha256
from pathlib import Path


def pipeline_source_sha256():
    root = Path(__file__).resolve().parents[3]
    paths = sorted({
        *root.joinpath("app/core/generation").glob("*.py"),
        *root.joinpath("app/core/retrieval").glob("*.py"),
        *root.joinpath("app/core/attribution").glob("*.py"),
        *root.joinpath("app/core/vision").glob("*.py"),
        root / "app/api/routes_ask.py", root / "app/config.py",
        root / "app/models/schemas.py", root / "app/main.py",
    })
    digest = sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
