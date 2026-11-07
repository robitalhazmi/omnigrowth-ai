"""Resilient HTTP downloader for public benchmark CSV files.

Streams each file to a temporary path, validates its CSV header against the
expected schema, then atomically moves it into ``data/raw/``. Every URL is
retried with exponential backoff before falling through to the next mirror.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.pipeline.sources import RemoteSource

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 16


class DownloadError(RuntimeError):
    """Raised when every URL of a source fails."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a successful download.

    Attributes:
        key: Logical dataset key.
        path: Final file location.
        url: URL the file was fetched from.
        sha256: Hex digest of the file contents.
        size_bytes: File size in bytes.
        rows: Number of data rows (header excluded).
    """

    key: str
    path: Path
    url: str
    sha256: str
    size_bytes: int
    rows: int


def file_sha256(path: Path) -> str:
    """Computes the SHA-256 digest of a file in streaming fashion."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    """Counts CSV data rows (handles quoted multi-line fields)."""
    with path.open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def validate_header(path: Path, required: tuple[str, ...]) -> None:
    """Checks that a CSV header contains all required columns.

    Args:
        path: CSV file to inspect.
        required: Column names that must be present.

    Raises:
        ValueError: If any required column is missing.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; header={header}")


def _stream_to_file(client: httpx.Client, url: str, dest: Path) -> None:
    """Streams a URL to ``dest`` raising on non-2xx responses."""
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                handle.write(chunk)


def download_source(
    source: RemoteSource,
    raw_dir: Path,
    *,
    timeout_s: float = 60.0,
    max_retries: int = 3,
    backoff_base_s: float = 1.0,
    client: httpx.Client | None = None,
) -> DownloadResult:
    """Downloads a source, trying each mirror with retry + exponential backoff.

    Args:
        source: Dataset descriptor.
        raw_dir: Destination directory.
        timeout_s: Per-request timeout in seconds.
        max_retries: Attempts per URL.
        backoff_base_s: Base delay; attempt ``i`` sleeps ``base * 2**i`` seconds.
        client: Optional pre-configured ``httpx.Client`` (useful for tests).

    Returns:
        A :class:`DownloadResult` describing the stored file.

    Raises:
        DownloadError: If every URL and retry fails.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / source.filename
    tmp = dest.with_suffix(dest.suffix + ".part")
    own_client = client is None
    http = client or httpx.Client(timeout=timeout_s, follow_redirects=True)
    errors: list[str] = []
    try:
        for url in source.urls:
            for attempt in range(max_retries):
                try:
                    _stream_to_file(http, url, tmp)
                    validate_header(tmp, source.required_columns)
                    tmp.replace(dest)
                    return DownloadResult(
                        key=source.key,
                        path=dest,
                        url=url,
                        sha256=file_sha256(dest),
                        size_bytes=dest.stat().st_size,
                        rows=count_csv_rows(dest),
                    )
                except (httpx.HTTPError, ValueError, OSError) as exc:
                    errors.append(f"{url} (attempt {attempt + 1}): {exc}")
                    logger.warning("Download failed: %s", errors[-1])
                    if attempt + 1 < max_retries:
                        time.sleep(backoff_base_s * (2**attempt))
    finally:
        tmp.unlink(missing_ok=True)
        if own_client:
            http.close()
    raise DownloadError(f"All mirrors failed for {source.key}: {errors}")
