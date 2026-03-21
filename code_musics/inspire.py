"""Inspiration card generator for composition sessions.

Usage (via make):
    make inspire                     # 4 cards from 4 random buckets
    make inspire N=3                 # 3 random cards from any bucket
    make inspire OBLIQUE=2 IMAGE=1   # explicit per-bucket counts

Usage (direct):
    uv run python -m code_musics.inspire
    uv run python -m code_musics.inspire --n=3
    uv run python -m code_musics.inspire --oblique=2 --image=1
"""

from __future__ import annotations

import argparse
import random
import textwrap

# ---------------------------------------------------------------------------
# Card pools
# ---------------------------------------------------------------------------

OBLIQUE: list[str] = [
    # --- original Eno / Schmidt deck ---
    "Abandon normal instruments",
    "Accept advice",
    "Accretion",
    "A line has two sides",
    "Allow an easement (the abandonment of a stricture)",
    "Are there sections? Consider transitions",
    "Ask people to work against their better judgment",
    "Ask your body",
    "Balance the consistency principle with the inconsistency principle",
    "Be dirty",
    "Breathe more deeply",
    "Bridges — build, burn",
    "Cascades",
    "Change instrument roles",
    "Change nothing and continue with immaculate consistency",
    "Consider different fading systems",
    "Convert a melodic element into a rhythmic element",
    "Courage!",
    "Cut a vital connection",
    "Decorate, decorate",
    "Define an area as 'safe' and use it as an anchor",
    "Destroy — nothing / the most important thing",
    "Discard an axiom",
    "Disconnect from desire",
    "Discover the recipes you are using and abandon them",
    "Distorting time",
    "Do nothing for as long as possible",
    "Don't be afraid of things because they're easy to do",
    "Don't be frightened of clichés",
    "Don't break the silence",
    "Don't stress one thing more than another",
    "Do something boring",
    "Do the words need changing?",
    "Do we need holes?",
    "Emphasize differences",
    "Emphasize repetitions",
    "Emphasize the flaws",
    "Faced with a choice, do both",
    "Fill every beat with something",
    "Ghost echoes",
    "Give the game away",
    "Give way to your worst impulse",
    "Go slowly all the way round the outside",
    "Honor thy error as a hidden intention",
    "How would you have done it?",
    "Humanize something free of error",
    "Imagine the music as a moving chain or caterpillar",
    "Imagine the music as a set of disconnected events",
    "Infinitesimal gradations",
    "Into the impossible",
    "Is it finished?",
    "Is there something missing?",
    "Is the tuning appropriate?",
    "Just carry on",
    "Listen in total darkness, or in a very large room, very quietly",
    "Listen to the quiet voice",
    "Look at a very small object; look at its center",
    "Look at the order in which you do things",
    "Look closely at the most embarrassing details and amplify them",
    "Make a blank valuable by putting it in an exquisite frame",
    "Make a sudden, destructive, unpredictable action; incorporate",
    "Mechanicalize something idiosyncratic",
    "Mute and continue",
    "Only one element of each kind",
    "Overtly resist change",
    "Remember those quiet evenings",
    "Remove ambiguities and convert to specifics",
    "Remove specifics and convert to ambiguities",
    "Repetition is a form of change",
    "Reverse",
    "Short circuit",
    "Simple subtraction",
    "Take a break",
    "Take away the elements in order of apparent non-importance",
    "The tape is now the music",
    "Think of the radio",
    "Tidy up",
    "Trust in the you of now",
    "Turn it upside down",
    "Twist the spine",
    "Use an old idea",
    "Use an unacceptable color",
    "Use fewer notes",
    "Use filters",
    "Water",
    "What are you really thinking about just now? Incorporate",
    "What is the reality of the situation?",
    "What mistakes did you make last time?",
    "What wouldn't you do?",
    "Work at a different speed",
    "You can only make one dot at a time",
    "You don't have to be ashamed of using your own ideas",
    "[blank]",
    # --- repo-specific additions ---
    "Is the comma audible, or only theoretical?",
    "What is the utonal shadow of this?",
    "Let the sequence drift; don't correct it",
    "Strip the temperament entirely",
    "What if the 'wrong' partial were the root?",
    "Tune to the ear, not the lattice",
    "Let two voices disagree about the root",
    "What does the 11th harmonic want to do here?",
    "Build from one ratio and let everything else follow",
    "The melody is just voice leading made visible",
    "What if you kept the process and changed the material?",
    "What if you kept the material and changed the process?",
    "Remove the most 'compositional' element",
    "More sustain, less attack",
    "Less sustain, more attack",
    "What does this want to become?",
]

MUSICAL: list[str] = [
    # Bach techniques
    "Bach — arpeggiate a voice-leading skeleton: the melody is the top notes",
    "Bach — define a 2-bar cell and sequence it down by thirds",
    "Bach — write invertible counterpoint: both voice orientations must work",
    "Bach — state a subject, then invert it, augment it, run it in stretto",
    "Bach — build a ground bass and spin free variations above it; the bass IS the composition",
    "Bach — canon: one voice imitates the other with a fixed time and pitch delay",
    "Bach — retrograde: one voice is the other's exact reversal in time",
    "Bach — pedal point: hold a note while harmonies move above it into tension",
    "Bach — chaconne: a repeating harmonic skeleton; the piece lives in the variations, not the theme",
    "Bach — the sense of something communal, even sacred, reached through structure",
    # Pärt
    "Pärt tintinnabuli — one voice arpegiates a just chord, one moves by step; that's the whole system",
    # Reich / Riley
    "Reich — two copies of the same loop at slightly different speeds; the phasing IS the composition",
    "Reich — a process that is fully audible, not hidden beneath 'music'",
    "Riley — a short loop that accumulates variations across a very long time",
    # Nancarrow
    "Nancarrow — canon at an irrational tempo ratio; the voices converge only at a single computed point",
    # Spectral / JI
    "Murail / Grisey — derive the entire harmony from one timbre's overtone series",
    "Feldman — near-repetition: it recurs, but never quite the same; duration is the subject",
    "Feldman — let one thing last longer than seems reasonable",
    "Ligeti micropolyphony — dense overlapping lines creating texture, not melody",
    "Satie — harmonic stasis; one chord held until it becomes an environment, not an event",
    "Messiaen — a mode that only transposes a limited number of times before repeating",
    "Pauline Oliveros deep listening — the room is the instrument; what is already present?",
    # Pop / electronic references
    "Aphex Twin (ambient mode) — pad textures with the intimacy of headphones at 3am",
    "Aphex Twin (IDM mode) — the rhythm is broken machinery; the melody floats above it, untouched",
    "Aphex Twin (Avril 14th) — bittersweet and tender; minor-key, ending before you're ready",
    "MBV Loveless — guitars as texture fields, not instruments; melody drowns in its own reverb",
    "Boards of Canada — tape-worn, slightly out of tune in a way that feels organic, not sloppy",
    "Burial Untrue — the beat is half-submerged; the vocals are ghosts of vocals",
    "Four Tet — acoustic material treated electronically; warmth inside the machine",
    "M83 (Dead Cities) — spacious, reverb-soaked, lofi; nostalgia for a future that didn't happen",
    "Coltrane Giant Steps — rapid harmonic motion; every chord a destination, not a passing moment",
    "Modal jazz — time dilates; one harmony is a whole world, not a single step",
    "Jazz broadly — the space between notes; the rhythm section breathing together",
]

VISUAL: list[str] = [
    # Film directors
    "Ozu — stillness as shot language; the camera doesn't move; the emotion arrives anyway",
    "Kubrick — mathematical geometry containing violent emotion; symmetry with something cold inside",
    "Kubrick — the tracking shot: slow, inevitable, you cannot look away",
    # Painters
    "Rothko color field — a chord held until it becomes environment, not event; you stand inside it",
    "Monet — the subject is the light, not the thing the light falls on",
    "Van Gogh — the brushstroke has its own movement, independent of what it depicts",
    "Calder mobile — balance without stasis; things that move when touched, rest when not",
    "Mondrian — strict proportions, primary materials, zero decoration",
    "Agnes Martin — the grid as meditative surface; precision that is also quietude",
    "Caravaggio chiaroscuro — extreme contrast; very dark and very bright with nothing between",
    "Morandi still life — the same objects, the same light, infinite subtle variation",
    "Giacometti — elongated, skeletal, barely there; how little material can hold a form?",
    "Cy Twombly — marks that are almost writing but not quite; gesture over statement",
    "Richard Serra — weight, mass, scale; the material is the subject",
    # Photography / architecture
    "Hiroshi Sugimoto — long exposure; the theater is empty, only the screen still glows",
    "Brutalism — massive, exposed, structural seams visible and unapologetic",
    "Japanese ma — the negative space is not empty; it is where the composition breathes",
    "Arte Povera — make it from what you would normally discard",
    "Escher impossible structure — implies resolution but loops back through itself",
]

IMAGE: list[str] = [
    # Emotional / bittersweet
    "Bittersweet — the specific emotion of Avril 14th; tender, minor-key, ending too soon",
    "Brokedown Palace — the long exhale of something that cannot be recovered",
    "A sense of communion — the feeling Bach gives; reaching toward something larger, collectively",
    "Anticipatory grief — mourning something still present",
    "The feeling of a word you almost remember",
    "Nostalgia for a place that never existed",
    "The specific weight of a childhood memory that may not be accurate",
    "The feeling of falling in a dream, just before you wake",
    # Liminal / temporal
    "3am",
    "Sunday afternoon dread",
    "The room after a party",
    "The moment before a storm when everything goes still",
    "An airport terminal at 4am — nowhere, between everywhere",
    "The quality of light in an old photograph",
    "The sound of a place you can no longer return to",
    # Literary
    "Proust — memory as involuntary, triggered, arriving full of present-tense detail",
    "Woolf — consciousness as texture, not argument; the mind moving through a room like water",
    # Natural phenomena
    "Murmuration — thousands of agents following three simple rules; global order from local behavior",
    "Tidal cycle — not the wave; the slower breathing underneath it",
    "Crystal growth — local rules, global structure, irreversible",
    "Bioluminescence — light as a byproduct of living, not intended for vision",
    "Erosion — patient, material; time as the primary compositional force",
    "Canopy noise in wind — independent elements sharing a medium",
    "Ice forming on a window — fractal, spreading from the edges inward",
    "Fog — not the objects, but the space between them",
    "A geode — rough exterior, crystalline interior",
    "The moment a school of fish all turns at once",
    "Deep ocean pressure zones — a world with completely different rules",
    "Old-growth forest floor — layers upon layers of decay becoming structure",
    "The sound of a cave — resonance you didn't choose; the room chose it",
    "Looking at something very far away for a very long time",
    "Underwater — time moves differently",
    "Lukewarm water",
]

CONSTRAINT: list[str] = [
    "Set a duration limit. Do not exceed it.",
    "This piece may only grow — nothing removed once added",
    "Every element must contradict the previous one",
    "Compose from the ending backward",
    "The piece contains exactly one mistake, deliberately placed",
    "Use at most three distinct pitch ratios for the entire piece",
    "No repetition of any kind — if something appears twice, remove one instance",
    "Everything is a variation of the opening five seconds",
    "The piece must contain a silence longer than you are comfortable with",
    "Begin in the middle — no introduction",
    "One voice only. No harmony, no unison.",
    "Tempo is fixed. Pitch is free.",
    "Pitch is fixed. Rhythm is free.",
    "The climax must arrive in the first third",
    "Write the structure as a sentence first, then make it sound like that sentence",
    "You may not use any technique you used in the last piece",
    "The piece must be defensible as a single gesture",
    "No note may last longer than one second",
    "Every note lasts at least three seconds",
    "The dynamic range is fixed: nothing may be louder or softer than the opening note",
]

BUCKETS: dict[str, list[str]] = {
    "oblique": OBLIQUE,
    "musical": MUSICAL,
    "visual": VISUAL,
    "image": IMAGE,
    "constraint": CONSTRAINT,
}

BUCKET_LABELS: dict[str, str] = {
    "oblique": "OBLIQUE",
    "musical": "MUSICAL",
    "visual": "VISUAL / FILM",
    "image": "IMAGE / ATMOSPHERE",
    "constraint": "CONSTRAINT",
}

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

CARD_WIDTH = 54  # inner content width (excluding border chars)


def _format_card(bucket: str, text: str) -> str:
    label = BUCKET_LABELS[bucket]
    inner_w = CARD_WIDTH
    wrapped = textwrap.wrap(text, width=inner_w - 4)  # 2 spaces padding each side

    top = f"┌─ {label} " + "─" * (inner_w - len(label) - 3) + "┐"
    blank = "│" + " " * inner_w + "│"
    lines = [top, blank]
    for line in wrapped:
        padded = f"  {line}"
        lines.append("│" + padded.ljust(inner_w) + "│")
    lines.append(blank)
    lines.append("└" + "─" * inner_w + "┘")
    return "\n".join(lines)


def draw(bucket_counts: dict[str, int]) -> list[tuple[str, str]]:
    """Draw cards according to per-bucket counts. Returns (bucket, text) pairs."""
    results: list[tuple[str, str]] = []
    for bucket, count in bucket_counts.items():
        pool = BUCKETS[bucket]
        drawn = random.sample(pool, min(count, len(pool)))
        results.extend((bucket, card) for card in drawn)
    return results


def draw_random(n: int) -> list[tuple[str, str]]:
    """Draw n cards randomly across all buckets, no two from the same bucket if avoidable."""
    bucket_names = list(BUCKETS.keys())
    if n <= len(bucket_names):
        chosen_buckets = random.sample(bucket_names, n)
    else:
        chosen_buckets = random.choices(bucket_names, k=n)
    results: list[tuple[str, str]] = []
    for bucket in chosen_buckets:
        card = random.choice(BUCKETS[bucket])
        results.append((bucket, card))
    return results


def print_session(cards: list[tuple[str, str]]) -> None:
    print()
    for bucket, text in cards:
        print(_format_card(bucket, text))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Draw inspiration cards for a composition session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              inspire                          4 random cards, one per bucket
              inspire --n=3                    3 random cards from any bucket
              inspire --oblique=2 --image=1    2 oblique + 1 image card
            """
        ),
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="draw N cards randomly across all buckets",
    )
    p.add_argument("--oblique", type=int, default=0, metavar="N")
    p.add_argument("--musical", type=int, default=0, metavar="N")
    p.add_argument("--visual", type=int, default=0, metavar="N")
    p.add_argument("--image", type=int, default=0, metavar="N")
    p.add_argument("--constraint", type=int, default=0, metavar="N")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    explicit_counts = {
        "oblique": args.oblique,
        "musical": args.musical,
        "visual": args.visual,
        "image": args.image,
        "constraint": args.constraint,
    }
    any_explicit = any(v > 0 for v in explicit_counts.values())

    if args.n is not None and any_explicit:
        print("error: use either --n or per-bucket flags, not both")
        raise SystemExit(1)

    if args.n is not None:
        cards = draw_random(args.n)
    elif any_explicit:
        cards = draw({k: v for k, v in explicit_counts.items() if v > 0})
    else:
        # default: one card from each of 4 randomly chosen buckets
        cards = draw_random(4)

    print_session(cards)


if __name__ == "__main__":
    main()
