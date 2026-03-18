"""Synth utility and plugin-loading tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from code_musics.synth import (
    ExternalPluginSpec,
    _load_external_plugin,
    _loaded_external_plugins,
)


def test_plugin_cache_key_includes_bundle_plugin_name() -> None:
    """Regression test: two plugins from the same bundle file (e.g. LSP compressor
    and limiter) must NOT share a cache slot.  The original bug used only
    (host, format, path) as the key, so loading the compressor first and then
    asking for the limiter returned the compressor plugin, causing AttributeError
    on parameters that only the limiter has (e.g. 'threshold_db').
    """
    bundle_path = Path("/fake/bundle.vst3")
    fake_compressor = MagicMock(name="CompressorPlugin")
    fake_limiter = MagicMock(name="LimiterPlugin")

    spec_compressor = ExternalPluginSpec(
        name="fake_compressor",
        path=bundle_path,
        format="vst3",
        bundle_plugin_name="Compressor Stereo",
    )
    spec_limiter = ExternalPluginSpec(
        name="fake_limiter",
        path=bundle_path,
        format="vst3",
        bundle_plugin_name="Limiter Stereo",
    )

    def fake_load_plugin(_path: str, plugin_name: str | None = None) -> MagicMock:
        return fake_compressor if plugin_name == "Compressor Stereo" else fake_limiter

    with (
        patch("code_musics.synth._get_external_plugin_spec") as mock_get_spec,
        patch("code_musics.synth.Path.exists", return_value=True),
        patch("pedalboard.load_plugin", side_effect=fake_load_plugin),
    ):
        _loaded_external_plugins.clear()

        mock_get_spec.return_value = spec_compressor
        got_compressor = _load_external_plugin(plugin_name="fake_compressor")

        mock_get_spec.return_value = spec_limiter
        got_limiter = _load_external_plugin(plugin_name="fake_limiter")

    # The two loads must return distinct plugin objects.
    assert got_compressor is not got_limiter, (
        "Plugin cache collision: compressor and limiter share the same cache slot "
        "because bundle_plugin_name was not part of the cache key."
    )
    assert got_compressor is fake_compressor
    assert got_limiter is fake_limiter
