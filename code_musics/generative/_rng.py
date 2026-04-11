from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from code_musics.composition import HarmonicContext, line, ratio_line
from code_musics.score import Phrase

if TYPE_CHECKING:
    pass


def make_rng(seed: int) -> random.Random:
    """Create a deterministic Random instance from a seed."""
    return random.Random(seed)


def _ratios_to_phrase(
    ratios: Sequence[float],
    rhythm: Sequence[float] | Any,
    context: HarmonicContext | None = None,
    **kwargs: Any,
) -> Phrase:
    """Convert a sequence of ratios to a Phrase via line/ratio_line."""
    if context is not None:
        kw = {k: v for k, v in kwargs.items() if k != "pitch_kind"}
        return ratio_line(ratios, rhythm, context=context, **kw)
    return line(ratios, rhythm, **kwargs)
