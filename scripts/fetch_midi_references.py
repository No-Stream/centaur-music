"""Fetch external MIDI reference assets used by smoke tests and pieces."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

SankeyAsset = tuple[str, str, str, str, str]

ROOT = Path(__file__).resolve().parents[1]

ASSETS: tuple[SankeyAsset, ...] = (
    (
        "http://www.jsbach.net/midi/sankey/846-869.zip",
        "269257461b2f1f1dfc591bd1e9d8872ff66c86ce6f32aa440f03f2d2c36f0d24",
        "bwv846.mid",
        "midi_references/bach/well-tempered-clavier-i_bwv-846_(c)sankey.mid",
        "John Sankey WTC Book I BWV 846 MIDI",
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def _install_asset(asset: SankeyAsset) -> None:
    url, expected_archive_sha256, archive_member, output_relative_path, label = asset
    logger.info("Fetching %s from %s", label, url)
    archive_data = _download(url)
    archive_sha256 = _sha256_bytes(archive_data)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            f"{label} archive hash mismatch: got {archive_sha256}, "
            f"expected {expected_archive_sha256}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "asset.zip"
        archive_path.write_bytes(archive_data)
        with zipfile.ZipFile(archive_path) as archive:
            member_data = archive.read(archive_member)

    output_path = ROOT / output_relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(member_data)
    logger.info(
        "Installed %s to %s (sha256=%s)",
        label,
        output_path.relative_to(ROOT),
        _sha256_bytes(member_data),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    for asset in ASSETS:
        _install_asset(asset)


if __name__ == "__main__":
    main()
