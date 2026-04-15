"""Audio stem export tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from code_musics.render import export_piece_stems
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.stem_export import export_stem_bundle
from code_musics.stem_export_types import StemExportSpec


def _build_test_score_with_sends() -> Score:
    """Build a minimal score with 2 voices and 1 send bus."""
    score = Score(
        f0_hz=220.0,
        auto_master_gain_stage=False,
        master_effects=[],
    )
    score.send_buses = [
        SendBusSpec(
            name="room",
            effects=[
                EffectSpec(
                    "delay", {"delay_seconds": 0.05, "feedback": 0.2, "mix": 0.5}
                )
            ],
        )
    ]
    score.add_voice(
        "lead",
        normalize_lufs=None,
        velocity_humanize=None,
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )
    score.add_voice("bass", normalize_lufs=None, velocity_humanize=None)
    score.add_note("lead", start=0.0, duration=0.5, partial=4.0, amp=0.15)
    score.add_note("bass", start=0.1, duration=0.4, partial=2.0, amp=0.12)
    return score


def _build_test_score_different_durations() -> Score:
    """Build a score where voices have distinctly different durations."""
    score = Score(
        f0_hz=220.0,
        auto_master_gain_stage=False,
        master_effects=[],
    )
    score.add_voice("short", normalize_lufs=None, velocity_humanize=None)
    score.add_voice("long", normalize_lufs=None, velocity_humanize=None)
    score.add_note("short", start=0.0, duration=0.2, partial=4.0, amp=0.1)
    score.add_note("long", start=0.0, duration=0.5, partial=2.0, amp=0.1)
    return score


class TestWetBundle:
    """Wet (post-effects) stem bundle export."""

    def test_wet_bundle_writes_expected_files(self, tmp_path: Path) -> None:
        score = _build_test_score_with_sends()
        spec = StemExportSpec(piece_name="test", output_name="test_bundle")
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        assert (result.bundle_dir / "voices" / "lead.wav").exists()
        assert (result.bundle_dir / "voices" / "bass.wav").exists()
        assert (result.bundle_dir / "sends" / "room.wav").exists()
        assert (result.bundle_dir / "mix.wav").exists()
        assert (result.bundle_dir / "manifest.json").exists()

    def test_wet_stems_sum_approximates_mix(self) -> None:
        score = _build_test_score_with_sends()
        voice_stems, send_returns, mix_audio = score.render_for_stem_export(dry=False)

        all_signals = [*voice_stems.values(), *send_returns.values()]
        summed = Score._stack_signals(all_signals)

        # The mix goes through _apply_master_bus_processing which applies a
        # ceiling limiter even with no master_effects and auto_master_gain_stage
        # off, so we can't expect exact equality. But the signals should be
        # close in shape and energy.
        assert summed.shape[-1] == mix_audio.shape[-1]
        # RMS of the difference should be small relative to the signal
        diff_rms = np.sqrt(np.mean((summed - mix_audio) ** 2))
        signal_rms = np.sqrt(np.mean(mix_audio**2))
        assert diff_rms < 0.1 * signal_rms, (
            f"Stem sum differs too much from mix: diff_rms={diff_rms:.6f}, "
            f"signal_rms={signal_rms:.6f}"
        )


class TestDryBundle:
    """Dry (post-normalization only) stem bundle export."""

    def test_dry_bundle_has_no_sends_dir(self, tmp_path: Path) -> None:
        score = _build_test_score_with_sends()
        spec = StemExportSpec(piece_name="test", output_name="test_dry", dry=True)
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        assert (result.bundle_dir / "voices" / "lead.wav").exists()
        assert (result.bundle_dir / "voices" / "bass.wav").exists()
        sends_dir = result.bundle_dir / "sends"
        assert not sends_dir.exists() or not list(sends_dir.iterdir())

    def test_dry_stems_are_mono(self, tmp_path: Path) -> None:
        score = _build_test_score_with_sends()
        spec = StemExportSpec(piece_name="test", output_name="test_dry", dry=True)
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        for stem_info in result.manifest.voice_stems:
            assert stem_info.channels == 1, f"{stem_info.name} should be mono (dry)"


class TestZeroPadding:
    """Stems are zero-padded to a uniform length."""

    def test_stems_are_zero_padded_to_same_length(self, tmp_path: Path) -> None:
        score = _build_test_score_different_durations()
        spec = StemExportSpec(piece_name="test", output_name="test_pad")
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        frame_counts: set[int] = set()
        for stem_info in result.manifest.voice_stems:
            wav_path = result.bundle_dir / stem_info.path
            data = sf.read(wav_path)[0]
            frame_counts.add(data.shape[0])

        assert len(frame_counts) == 1, (
            f"All stems should have the same frame count, got {frame_counts}"
        )


class TestMixFlag:
    """include_mix=False omits the reference mix."""

    def test_no_mix_flag_omits_reference(self, tmp_path: Path) -> None:
        score = _build_test_score_with_sends()
        spec = StemExportSpec(
            piece_name="test", output_name="test_no_mix", include_mix=False
        )
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        assert not (result.bundle_dir / "mix.wav").exists()
        assert result.manifest.mix_path is None


class TestManifest:
    """Manifest metadata correctness."""

    def test_manifest_metadata(self, tmp_path: Path) -> None:
        score = _build_test_score_with_sends()
        spec = StemExportSpec(piece_name="test", output_name="test_manifest")
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        manifest_data = json.loads(result.manifest_path.read_text())

        assert manifest_data["schema_version"] == 1
        assert manifest_data["sample_rate"] == 44100
        assert manifest_data["bit_depth"] == 24
        assert manifest_data["dry"] is False
        assert len(manifest_data["voice_stems"]) == 2
        assert len(manifest_data["send_stems"]) == 1
        assert manifest_data["mix_path"] == "mix.wav"
        assert manifest_data["piece_name"] == "test"
        assert manifest_data["output_name"] == "test_manifest"


class TestExportPieceStems:
    """Integration via render.py orchestration layer."""

    def test_export_piece_stems_unknown_piece_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown piece"):
            export_piece_stems("definitely_not_a_real_piece", output_dir=tmp_path)


class TestClippingProtection:
    """Uniform gain scaling prevents clipping on export."""

    def test_hot_stems_are_attenuated_uniformly(self, tmp_path: Path) -> None:
        """A voice with effects that push peaks above 0 dBFS should be scaled down."""
        score = Score(
            f0_hz=220.0,
            auto_master_gain_stage=False,
            master_effects=[],
        )
        # High amp + saturation will push this well above 0 dBFS
        score.add_voice(
            "hot",
            normalize_lufs=None,
            velocity_humanize=None,
            effects=[EffectSpec("saturation", {"drive": 0.9})],
        )
        score.add_voice("quiet", normalize_lufs=None, velocity_humanize=None)
        score.add_note("hot", start=0.0, duration=0.3, partial=2.0, amp=0.9)
        score.add_note("quiet", start=0.0, duration=0.3, partial=4.0, amp=0.01)

        spec = StemExportSpec(
            piece_name="test", output_name="test_clip", include_mix=False
        )
        result = export_stem_bundle(score, tmp_path / "bundle", spec=spec)

        # No stem should clip (peak must be <= 0 dBFS)
        for stem_info in result.manifest.voice_stems:
            assert stem_info.peak_dbfs <= 0.0, (
                f"{stem_info.name} clipped at {stem_info.peak_dbfs} dBFS"
            )

        # Gain was applied (manifest records it)
        assert result.manifest.stem_gain_db <= 0.0

        # Both stems have the same sample count (zero-padded)
        counts = {s.sample_count for s in result.manifest.voice_stems}
        assert len(counts) == 1


class TestStemExportSpec:
    """Spec validation."""

    def test_bit_depth_validation_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="bit_depth"):
            StemExportSpec(piece_name="x", output_name="x", bit_depth=48)

    @pytest.mark.parametrize("valid_depth", [16, 24, 32])
    def test_bit_depth_accepts_valid(self, valid_depth: int) -> None:
        spec = StemExportSpec(piece_name="x", output_name="x", bit_depth=valid_depth)
        assert spec.bit_depth == valid_depth
