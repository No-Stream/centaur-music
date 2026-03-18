"""Short A/B demos for the effect chain."""

from __future__ import annotations

import logging

import numpy as np

from code_musics.composition import line
from code_musics.pieces.septimal import PieceDefinition
from code_musics.score import EffectSpec, Score
from code_musics.synth import SAMPLE_RATE, normalize

logger = logging.getLogger(__name__)

_SECTION_GAP_SECONDS = 0.7


def _build_demo_score(
    *,
    lead_effects: list[EffectSpec] | None = None,
    pad_effects: list[EffectSpec] | None = None,
    master_effects: list[EffectSpec] | None = None,
) -> Score:
    """Build a short motif that makes effect changes easy to hear."""
    score = Score(
        f0=55.0,
        master_effects=list(master_effects or []),
    )

    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "round_bass",
            "attack": 0.04,
            "decay": 0.16,
            "sustain_level": 0.74,
            "release": 0.34,
        },
        pan=-0.06,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
            "attack": 0.35,
            "decay": 0.24,
            "sustain_level": 0.62,
            "release": 1.8,
        },
        effects=list(pad_effects or []),
        pan=0.08,
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "attack": 0.03,
            "decay": 0.16,
            "sustain_level": 0.50,
            "release": 0.28,
        },
        effects=list(lead_effects or []),
        pan=0.16,
    )

    bass_line = line(
        tones=[2.0, 2.0, 3.0, 4.0, 3.0, 2.0],
        rhythm=(0.75, 0.75, 0.75, 0.75, 0.55, 1.10),
        amp_db=-18.0,
    )
    lead_line = line(
        tones=[8.0, 9.0, 10.0, 12.0, 10.0, 9.0, 8.0],
        rhythm=(0.35, 0.35, 0.45, 0.65, 0.45, 0.40, 1.15),
        amp_db=-17.0,
    )

    score.add_phrase("bass", bass_line, start=0.0)
    score.add_phrase("lead", lead_line, start=0.25)

    for start, duration, partial, amp_db in [
        (0.0, 2.2, 5.0, -27.0),
        (0.0, 2.2, 6.0, -28.5),
        (1.15, 1.9, 7.0, -30.0),
        (1.15, 1.9, 9.0, -31.0),
    ]:
        score.add_note(
            "pad",
            start=start,
            duration=duration,
            partial=partial,
            amp_db=amp_db,
        )

    return score


def _to_stereo(signal: np.ndarray) -> np.ndarray:
    """Standardize mono and stereo renders for concatenation."""
    if signal.ndim == 2:
        return signal.astype(np.float64)
    return np.stack([signal, signal]).astype(np.float64)


def _silence(seconds: float) -> np.ndarray:
    """Return stereo silence for a short separator gap."""
    n_samples = int(round(seconds * SAMPLE_RATE))
    return np.zeros((2, n_samples), dtype=np.float64)


def render_effects_showcase_demo() -> np.ndarray:
    """Render dry/effected sections for quick A/B listening."""
    sections: list[tuple[str, dict[str, list[EffectSpec]]]] = [
        ("dry_reference", {}),
        (
            "chorus_on",
            {
                "pad_effects": [
                    EffectSpec("chorus", {"preset": "juno_wide", "mix": 0.36})
                ],
                "lead_effects": [
                    EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.24})
                ],
            },
        ),
        ("dry_reference", {}),
        (
            "native_saturation_on",
            {
                "master_effects": [
                    EffectSpec(
                        "saturation",
                        {
                            "preset": "tube_warm",
                            "mix": 0.62,
                            "drive": 1.62,
                            "bias": 0.16,
                            "even_harmonics": 0.24,
                            "tone_tilt": 0.18,
                        },
                    )
                ]
            },
        ),
        ("dry_reference", {}),
        (
            "chow_tape_on",
            {
                "master_effects": [
                    EffectSpec(
                        "chow_tape",
                        {
                            "drive": 0.82,
                            "saturation": 0.78,
                            "bias": 0.58,
                            "mix": 88.0,
                        },
                    )
                ]
            },
        ),
    ]

    rendered_sections: list[np.ndarray] = []
    for label, config in sections:
        logger.info("Rendering effect showcase section: %s", label)
        try:
            section_audio = _to_stereo(_build_demo_score(**config).render())
        except Exception as exc:
            if label != "chow_tape_on":
                raise
            logger.warning("Skipping optional Chow Tape section: %s", exc)
            continue
        rendered_sections.append(section_audio)

    if not rendered_sections:
        raise ValueError("No showcase sections were rendered")

    joined_sections: list[np.ndarray] = []
    for index, section_audio in enumerate(rendered_sections):
        if index > 0:
            joined_sections.append(_silence(_SECTION_GAP_SECONDS))
        joined_sections.append(section_audio)

    return normalize(np.concatenate(joined_sections, axis=1), peak=0.9)


PIECES = {
    "effects_showcase": PieceDefinition(
        name="effects_showcase",
        output_name="23_effects_showcase.wav",
        render_audio=render_effects_showcase_demo,
    )
}
