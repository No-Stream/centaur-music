"""Vital instrument engine -- renders voices through pedalboard's VSTi hosting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from mido import Message

from code_musics.engines._mpe_utils import (
    DEFAULT_GLOBAL_GLIDE_TIME_SECONDS,
    build_cc_messages,
    build_global_bend_messages,
    build_mpe_config_messages,
    build_mpe_note_messages,
)
from code_musics.synth import SAMPLE_RATE, has_external_plugin, load_external_plugin

logger: logging.Logger = logging.getLogger(__name__)


def render_voice(
    *,
    notes: list[dict[str, Any]],
    total_duration: float,
    sample_rate: int = SAMPLE_RATE,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a full voice through Vital using MPE-style per-note pitch bend.

    Parameters
    ----------
    notes
        Each dict has: ``freq`` (Hz), ``start`` (seconds), ``duration`` (seconds),
        ``velocity`` (0.0--1.0), ``amp`` (linear amplitude, used for velocity scaling).
        Optional glide fields: ``glide_from`` (Hz, starting frequency for a pitch
        sweep toward ``freq``) and ``glide_time`` (seconds, defaults to full
        ``duration``).
    total_duration
        Total duration of the voice in seconds (the latest note-off time).
    sample_rate
        Sample rate for rendering.
    params
        Optional engine params.  Recognised keys:

        - ``preset_path`` -- path to a ``.vital`` preset file
        - ``raw_state`` -- serialised plugin state (bytes)
        - ``vital_params`` -- dict of ``{param_name: raw_value}`` for direct
          parameter control via Vital's VST3 parameter interface
        - ``tail_seconds`` -- extra render time after last note-off (default 2.0)
        - ``release_padding`` -- seconds to keep an MPE channel reserved after
          note-off so pitch bend from new notes does not bleed into release
          tails (default 1.0)
        - ``mpe`` -- when True (default), send MCM to enable MPE Lower Zone
          so pitch bend is per-note with sub-cent accuracy.  When False,
          use global-bend chord mode.
        - ``global_glide_time`` -- seconds for the pitch-bend glide between
          consecutive chords in global-bend mode (default 0.4).
        - ``cc_curves`` -- list of MIDI CC automation curve dicts
        - ``buffer_size`` -- pedalboard processing block size in samples
          (default 256).
    """
    resolved_params = params or {}

    if not has_external_plugin("vital"):
        logger.warning(
            "SKIPPING Vital voice render (%d notes, %.1fs): Vital VST3 "
            "is not installed on this machine. The voice will be silent. "
            "Install Vital and place at ~/.vst3/Vital.vst3",
            len(notes),
            total_duration,
        )
        return np.zeros((2, int(total_duration * sample_rate)), dtype=np.float64)

    plugin = load_external_plugin(plugin_name="vital")

    if not getattr(plugin, "is_instrument", False):
        raise RuntimeError("Loaded Vital plugin is not an instrument")

    # Restore preset/state if provided
    preset_path = resolved_params.get("preset_path")
    raw_state = resolved_params.get("raw_state")
    if preset_path is not None:
        plugin.load_preset(str(preset_path))
    elif raw_state is not None:
        plugin.raw_state = raw_state

    use_mpe = bool(resolved_params.get("mpe", True))

    # Configure Vital's MPE and pitch bend range via its parameter API.
    # Vital ignores standard MPE RPN messages for bend range, so we set these
    # directly.  pitch_bend_range maps 0.0-1.0 to 0-48 semitones; 0.5 = 24
    # semitones gives sub-cent microtonal accuracy (one LSB ~ 0.29 cents).
    if use_mpe and "mpe_enabled" in plugin.parameters:
        plugin.parameters["mpe_enabled"].raw_value = 1.0
    if "pitch_bend_range" in plugin.parameters:
        plugin.parameters["pitch_bend_range"].raw_value = 0.5  # 24 semitones

    # Apply direct parameter overrides
    vital_params: dict[str, float] = resolved_params.get("vital_params", {})
    for param_name, raw_value in vital_params.items():
        if param_name in plugin.parameters:
            plugin.parameters[param_name].raw_value = float(raw_value)
        else:
            logger.warning("Unknown Vital parameter %r -- skipping", param_name)

    tail_seconds = float(resolved_params.get("tail_seconds", 2.0))
    render_duration = total_duration + tail_seconds
    release_padding = float(resolved_params.get("release_padding", 1.0))

    # -- Build MIDI messages -------------------------------------------------
    messages: list[Message] = build_mpe_config_messages() if use_mpe else []

    if use_mpe:
        messages.extend(build_mpe_note_messages(notes, release_padding))
    else:
        global_glide_time = float(
            resolved_params.get("global_glide_time", DEFAULT_GLOBAL_GLIDE_TIME_SECONDS)
        )
        messages.extend(build_global_bend_messages(notes, global_glide_time))

    # -- CC automation curves ------------------------------------------------
    cc_curves: list[dict[str, Any]] = resolved_params.get("cc_curves", [])
    if cc_curves:
        messages.extend(build_cc_messages(cc_curves, total_duration))

    messages.sort(key=lambda m: m.time)  # type: ignore[reportAttributeAccessIssue]

    logger.info(
        "Rendering %d notes through Vital (%.1fs + %.1fs tail)",
        len(notes),
        total_duration,
        tail_seconds,
    )

    buffer_size = int(resolved_params.get("buffer_size", 256))

    audio = plugin(
        messages,
        sample_rate=sample_rate,
        duration=render_duration,
        num_channels=2,
        buffer_size=buffer_size,
    )

    # pedalboard returns shape (channels, samples) -- trim silent tail.
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        tail_start = int(total_duration * sample_rate)
        if tail_start < audio.shape[1]:
            tail = audio[:, tail_start:]
            tail_energy = np.max(np.abs(tail), axis=0)
            silent_threshold = 1e-6
            non_silent = np.where(tail_energy > silent_threshold)[0]
            if len(non_silent) > 0:
                trim_point = tail_start + non_silent[-1] + 1
                audio = audio[:, :trim_point]
            else:
                audio = audio[:, :tail_start]

    return np.asarray(audio, dtype=np.float64)
