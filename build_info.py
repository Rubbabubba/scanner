from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PATCH_VERSION = "patch-046-controlled-profitability-filter-recalibration"
BUILD_TIMESTAMP_UTC = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
SYSTEM_NAME = "crypto-system"
DEFAULT_SERVICE_ROLE = "scanner"


def _safe_read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception:
        return b""


def _fingerprint_from_paths(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    h.update(PATCH_VERSION.encode())
    for path in paths:
        h.update(str(path).encode())
        h.update(_safe_read_bytes(path))
    return h.hexdigest()[:16]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def get_service_role() -> str:
    return str(os.getenv("SERVICE_ROLE", DEFAULT_SERVICE_ROLE) or DEFAULT_SERVICE_ROLE)


def get_env_name() -> str:
    return str(os.getenv("ENV_NAME", os.getenv("RENDER_SERVICE_NAME", "unknown")) or "unknown")


def get_release_stage() -> str:
    return str(os.getenv("RELEASE_STAGE", os.getenv("RELEASE_STAGE_CONFIGURED", "paper")) or "paper")


def build_payload(expected_files: list[str] | None = None) -> dict:
    root = _repo_root()
    expected = expected_files or [
        "app.py",
        "scoring.py",
        "kraken_spot.py",
        "kraken_futures.py",
        "requirements.txt",
        "PATCH_046_MARKER.txt",
    ]
    manifest = []
    digest_paths: list[Path] = []
    for rel in expected:
        path = root / rel
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        sha = hashlib.sha256(_safe_read_bytes(path)).hexdigest()[:16] if exists else None
        manifest.append({
            "name": rel,
            "exists": exists,
            "size_bytes": size,
            "sha256": sha,
        })
        if exists:
            digest_paths.append(path)
    return {
        "patch_version": PATCH_VERSION,
        "build_timestamp_utc": BUILD_TIMESTAMP_UTC,
        "build_fingerprint": _fingerprint_from_paths(digest_paths),
        "system_name": SYSTEM_NAME,
        "service_role": get_service_role(),
        "env_name": get_env_name(),
        "release_stage_configured": get_release_stage(),
        "artifact_integrity": {
            "base_dir": str(root),
            "expected_files": manifest,
        },
    }
