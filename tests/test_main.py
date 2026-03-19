"""CLI argument parsing tests."""

import pytest

from main import _build_render_window, parse_args


def test_build_render_window_from_centered_snippet_args() -> None:
    args = parse_args(["chord_4567", "--snippet-at", "2:10", "--snippet-window", "12"])

    render_window = _build_render_window(args)

    assert render_window is not None
    assert render_window.start_seconds == pytest.approx(124.0)
    assert render_window.duration_seconds == pytest.approx(12.0)


def test_build_render_window_from_explicit_window_args() -> None:
    args = parse_args(["chord_4567", "--window-start", "1:30", "--window-dur", "4.5"])

    render_window = _build_render_window(args)

    assert render_window is not None
    assert render_window.start_seconds == pytest.approx(90.0)
    assert render_window.duration_seconds == pytest.approx(4.5)


def test_build_render_window_rejects_mixed_snippet_modes() -> None:
    args = parse_args(
        [
            "chord_4567",
            "--snippet-at",
            "2:10",
            "--window-start",
            "130",
            "--window-dur",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="Use either --snippet-at/--snippet-window"):
        _build_render_window(args)


def test_build_render_window_requires_window_pair() -> None:
    args = parse_args(["chord_4567", "--window-start", "130"])

    with pytest.raises(ValueError, match="must be provided together"):
        _build_render_window(args)


def test_build_render_window_requires_snippet_at_when_window_is_overridden() -> None:
    args = parse_args(["chord_4567", "--snippet-window", "8"])

    with pytest.raises(ValueError, match="requires --snippet-at"):
        _build_render_window(args)
