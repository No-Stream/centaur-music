"""Rubric definitions and prompt templates for piece evaluation.

Defines the evaluation dimensions, scoring scale, and prompt templates used
by LLM judges to assess musical pieces.  The rubric is designed around broad
qualitative dimensions (rather than narrow, easily-gameable metrics) with a
heavily-weighted open-subjective field.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """A single rubric dimension."""

    key: str
    name: str
    weight: float
    description: str


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        key="musical_substance",
        name="Musical Substance",
        weight=0.25,
        description=(
            "Harmony, melody, rhythm, motifs, voice leading.  Is there actual"
            " musical content worth hearing?  Are the tuning and harmonic choices"
            " intentional and effective?  Are there ideas that reward attention?"
        ),
    ),
    Dimension(
        key="structure_form",
        name="Structure & Form",
        weight=0.20,
        description=(
            "Arc, pacing, contrast, development, arrival.  Does it feel composed"
            " rather than generative?  Are there sections, transitions, and a"
            " sense of direction?  Would you know where you are if dropped into"
            " the middle?"
        ),
    ),
    Dimension(
        key="texture_expression",
        name="Texture & Expression",
        weight=0.15,
        description=(
            "Orchestration, automation, humanization, velocity, sound choices."
            "  Does it feel alive and deliberately shaped?  Do voices interact"
            " or just stack?  Is there dynamic range and timbral variety?"
        ),
    ),
    Dimension(
        key="completeness",
        name="Completeness",
        weight=0.10,
        description=(
            "Does this feel finished?  Could you listen to it as a piece, not"
            " just a sketch or technical demo?  Does it have a beginning,"
            " development, and ending?"
        ),
    ),
    Dimension(
        key="open_subjective",
        name="Open Subjective",
        weight=0.30,
        description=(
            "Your unconstrained holistic assessment.  What is your overall"
            " impression?  What stands out — positively or negatively?  Is it"
            " interesting, surprising, moving, tedious, confused?  Score this"
            " based on your genuine reaction, not on whether it satisfies the"
            " other dimensions."
        ),
    ),
)

SCALE_ANCHORS = """\
Score each dimension 0-100 (integers only):
   0-15: Fundamentally broken or empty.
  16-35: Has basic content but major problems.
  36-55: Competent but unremarkable — works as a sketch or study.
  56-75: Genuinely good — musical, shaped, worth revisiting.
  76-90: Very strong — memorable, complete, could stand alongside composed works.
  91-100: Exceptional — remarkable craft and artistry."""


def build_judge_system_prompt() -> str:
    """Build the full system prompt sent to each judge."""
    dimension_block = "\n\n".join(
        f"### {dim.name}\n{dim.description}" for dim in DIMENSIONS
    )
    return f"""\
You are a music critic evaluating a xenharmonic composition.  You will receive
a detailed packet describing a piece — score data (notes, voices, effects,
automation), analysis metrics, and visual plots.  The music uses just intonation
and harmonic-series tuning; it will not sound like 12-TET music.

Evaluate the piece across the following dimensions.

{dimension_block}

{SCALE_ANCHORS}

Respond with ONLY a JSON object in exactly this format (no markdown fencing,
no extra text):

{{
  "dimensions": {{
    "musical_substance": {{"score": <int 0-100>, "notes": "<2-3 sentences>"}},
    "structure_form": {{"score": <int 0-100>, "notes": "<2-3 sentences>"}},
    "texture_expression": {{"score": <int 0-100>, "notes": "<2-3 sentences>"}},
    "completeness": {{"score": <int 0-100>, "notes": "<2-3 sentences>"}},
    "open_subjective": {{"score": <int 0-100>, "notes": "<2-3 sentences>"}}
  }},
  "overall_notes": "<1-2 sentence strongest impression>"
}}"""


JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "dimensions": {
            "type": "object",
            "properties": {
                dim.key: {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 100},
                        "notes": {"type": "string"},
                    },
                    "required": ["score", "notes"],
                }
                for dim in DIMENSIONS
            },
            "required": [dim.key for dim in DIMENSIONS],
        },
        "overall_notes": {"type": "string"},
    },
    "required": ["dimensions", "overall_notes"],
}

DEFAULT_JUDGE_MODELS: tuple[str, ...] = (
    "opus",
    "sonnet",
    "haiku",
)
