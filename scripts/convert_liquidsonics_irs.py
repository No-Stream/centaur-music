"""Convert LiquidSonics Bricasti M7 Fusion-IR Sources to stereo WAV pairs.

Reads 4-channel 96kHz FLAC tail IRs and produces 44.1kHz stereo L/R WAV pairs
compatible with apply_bricasti's naming convention: "{ir_name}, 44K L.wav".

Channel layout (true stereo): ch0=LL, ch1=LR, ch2=RL, ch3=RR.
We extract ch0 (LL) as L and ch3 (RR) as R for the stereo pair.

Usage:
    PYTHONPATH=. uv run python scripts/convert_liquidsonics_irs.py
"""

import logging
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger: logging.Logger = logging.getLogger(__name__)

SOURCE_DIR = Path("../LiquidSonics Bricasti M7 Fusion-IR Sources")
OUTPUT_DIR = Path("irs/bricasti")

# Map LiquidSonics directory names to Samplicity-style category numbers/names.
# The codebase uses ir_names like "1 Halls 07 Large & Dark" matching Samplicity convention.
CATEGORY_MAP: dict[str, tuple[int, str]] = {
    "Halls1": (1, "Halls"),
    "Plates1": (2, "Plates"),
    "Rooms1": (3, "Rooms"),
    "Chambers1": (4, "Chambers"),
    "Spaces1": (5, "Spaces"),
    "Ambience1": (6, "Ambience"),
    "Halls2": (7, "Halls2"),
    "Plates2": (8, "Plates2"),
    "Rooms2": (9, "Rooms2"),
    "Spaces2": (10, "Spaces2"),
    "Nonlin": (11, "Nonlin"),
}

# Presets to convert: (category_dir, preset_dir) pairs.
PRESETS_TO_CONVERT: list[tuple[str, str]] = [
    # Halls -- the most-used category in the codebase
    ("Halls1", "01 Large Hall"),
    ("Halls1", "07 Large and Dark"),
    ("Halls1", "08 Large and Deep"),
    ("Halls1", "10 Concert Hall"),
    ("Halls1", "11 Gold Hall"),
    ("Halls1", "14 Clear Hall"),
    ("Halls1", "16 Amsterdam Hall"),
    ("Halls1", "25 Saint Sylvian"),
    ("Halls1", "32 Piano Hall"),
    # Plates
    ("Plates1", "02 Dark Plate"),
    ("Plates1", "06 Vocal Plate"),
    ("Plates1", "08 Rich Plate"),
    ("Plates1", "09 Gold Plate"),
    # Rooms
    ("Rooms1", "01 Studio A"),
    ("Rooms1", "08 Music Room"),
]

TARGET_SR = 44_100
SOURCE_SR = 96_000


def _build_ir_name(category_dir: str, preset_dir: str) -> str:
    """Build a Samplicity-style ir_name from LiquidSonics directory names.

    Example: ("Halls1", "07 Large and Dark") -> "1 Halls 07 Large & Dark"
    """
    cat_num, cat_name = CATEGORY_MAP[category_dir]
    # Preset dir is like "07 Large and Dark" -- extract number and name
    parts = preset_dir.split(" ", 1)
    preset_num = parts[0]
    preset_name = parts[1] if len(parts) > 1 else ""
    # Samplicity convention: "and" -> "&" in preset names
    preset_name = preset_name.replace(" and ", " & ")
    return f"{cat_num} {cat_name} {preset_num} {preset_name}"


def _resample(data: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Resample audio using polyphase filtering."""
    if from_sr == to_sr:
        return data
    ratio_gcd = gcd(from_sr, to_sr)
    up = to_sr // ratio_gcd
    down = from_sr // ratio_gcd
    return resample_poly(data, up, down).astype(np.float32)


def convert_preset(category_dir: str, preset_dir: str) -> None:
    """Convert a single preset's tail-1.flac to L/R WAV pair."""
    source_path = SOURCE_DIR / category_dir / preset_dir / "tail-1.flac"
    if not source_path.exists():
        logger.warning(f"Source not found, skipping: {source_path}")
        return

    ir_name = _build_ir_name(category_dir, preset_dir)
    out_l = OUTPUT_DIR / f"{ir_name}, 44K L.wav"
    out_r = OUTPUT_DIR / f"{ir_name}, 44K R.wav"

    if out_l.exists() and out_r.exists():
        logger.info(f"Already exists, skipping: {ir_name}")
        return

    data, sr = sf.read(source_path, dtype="float32")
    if sr != SOURCE_SR:
        logger.warning(
            f"Unexpected sample rate {sr} for {source_path}, expected {SOURCE_SR}"
        )
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"Expected 4-channel audio, got shape {data.shape} for {source_path}"
        )

    # True stereo layout: ch0=LL, ch1=LR, ch2=RL, ch3=RR
    left_ch = data[:, 0]  # LL
    right_ch = data[:, 3]  # RR

    # Resample 96kHz -> 44.1kHz
    left_44k = _resample(left_ch, sr, TARGET_SR)
    right_44k = _resample(right_ch, sr, TARGET_SR)

    # Write as 24-bit WAV (matching project convention)
    sf.write(str(out_l), left_44k, TARGET_SR, subtype="PCM_24")
    sf.write(str(out_r), right_44k, TARGET_SR, subtype="PCM_24")

    dur = len(left_44k) / TARGET_SR
    logger.info(f"Converted: {ir_name} ({dur:.2f}s)")


def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SOURCE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Source: {SOURCE_DIR.resolve()}")
    logger.info(f"Output: {OUTPUT_DIR.resolve()}")
    logger.info(f"Converting {len(PRESETS_TO_CONVERT)} presets...")

    for category_dir, preset_dir in PRESETS_TO_CONVERT:
        convert_preset(category_dir, preset_dir)

    # List results
    wav_files = sorted(OUTPUT_DIR.glob("*.wav"))
    logger.info(f"\nDone. {len(wav_files)} WAV files in {OUTPUT_DIR.resolve()}")
    for f in wav_files:
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
