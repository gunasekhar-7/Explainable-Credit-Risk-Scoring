"""Artifact integrity helpers for model persistence.

joblib/pickle deserialization can execute arbitrary code. Model artifacts
must only be loaded from trusted sources. To enforce this boundary we write
a SHA-256 checksum file next to each serialized model at save time and
verify it at load time. A mismatch is fatal; a missing checksum file (legacy
artifact) degrades to a warning rather than a hard failure so existing
checkpoints keep working.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_checksum(path: Path) -> None:
    """Write a `<name>.sha256` file beside `path` containing its SHA-256 digest."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(digest, encoding="utf-8")


def checksum_status(path: Path) -> str:
    """Return ``"ok"``, ``"missing"`` (no sidecar) or ``"mismatch"`` (sidecar disagrees)."""
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    if not checksum_path.exists():
        return "missing"
    expected = checksum_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return "ok" if expected == actual else "mismatch"


def verify_checksum(path: Path) -> bool:
    """Return True if `path` matches its `.sha256` sidecar file.

    Missing sidecar -> False (caller decides whether to warn or fail).
    """
    return checksum_status(path) == "ok"


def assert_artifact_trusted(path: Path) -> None:
    """Gate a pickle load on its sidecar checksum.

    A *mismatch* means the bytes changed after the checksum was written, which is
    exactly the tampered-artifact case this control exists for, so it is fatal.
    A *missing* sidecar is treated as a legacy checkpoint and only warns, so
    pre-existing artifacts keep loading.

    Raises:
        FileNotFoundError: the artifact itself is absent.
        RuntimeError: the artifact does not match its recorded checksum.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")  # noqa: TRY003
    status = checksum_status(path)
    if status == "mismatch":
        raise RuntimeError(  # noqa: TRY003
            f"Checksum mismatch for {path}: artifact bytes do not match their .sha256 "
            "sidecar. Refusing to deserialize a possibly tampered pickle."
        )
    if status == "missing":
        logger.warning(
            "No checksum sidecar for %s; loading an unverified pickle can execute "
            "arbitrary code. Only load artifacts from trusted sources.",
            path,
        )
