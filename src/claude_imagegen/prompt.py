from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PromptSpec:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    color_words: tuple[str, ...]
    objects: tuple[str, ...]
    style_words: tuple[str, ...]
    mood_words: tuple[str, ...]


COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "gray",
    "green",
    "grey",
    "indigo",
    "magenta",
    "orange",
    "pink",
    "purple",
    "red",
    "teal",
    "violet",
    "white",
    "yellow",
}

OBJECT_ALIASES = {
    "abstract": {"abstract", "geometric", "pattern", "poster"},
    "building": {"building", "buildings", "city", "cityscape", "skyline", "tower", "towers"},
    "cloud": {"cloud", "clouds"},
    "diagram": {
        "architecture",
        "arrow",
        "arrows",
        "badge",
        "badges",
        "connector",
        "connectors",
        "diagram",
        "flow",
        "flowchart",
        "icon",
        "icons",
        "infographic",
        "pipeline",
        "reference",
        "schematic",
        "service",
        "services",
        "tile",
        "tiles",
        "ui",
    },
    "floor": {"floor", "floors", "stone", "stones", "tiles", "wet"},
    "flower": {"flower", "flowers", "botanical", "garden", "petal", "petals"},
    "forest": {"forest", "woods", "jungle", "trees", "tree"},
    "greenhouse": {"greenhouse", "glasshouse", "conservatory", "atrium", "glass", "glazing"},
    "lamp": {"lamp", "lamps", "lights", "pendant", "pendants", "hanging"},
    "moon": {"moon", "lunar"},
    "mountain": {"mountain", "mountains", "peak", "peaks", "alpine"},
    "ocean": {"ocean", "sea", "water", "lake", "river", "waves", "wave"},
    "plant": {"plant", "plants", "tropical", "foliage", "leaf", "leaves", "monstera", "fern", "ferns"},
    "portrait": {"portrait", "person", "face", "human"},
    "robot": {"robot", "android", "mech", "machine"},
    "sun": {"sun", "sunrise", "sunset", "solar"},
}

STYLE_WORDS = {
    "cinematic",
    "dream",
    "dreamy",
    "ink",
    "minimal",
    "noir",
    "oil",
    "pixel",
    "sketch",
    "watercolor",
}

MOOD_WORDS = {
    "bright",
    "calm",
    "dark",
    "dramatic",
    "quiet",
    "soft",
    "stormy",
    "warm",
}

DEFAULT_COLORS = ("blue", "gold")
TOKEN_RE = re.compile(r"[a-z0-9]+")
GRAPHIC_CONTEXT_WORDS = frozenset(
    {
        "architecture",
        "arrow",
        "arrows",
        "badge",
        "badges",
        "box",
        "boxes",
        "connector",
        "connectors",
        "diagram",
        "flow",
        "flowchart",
        "icon",
        "icons",
        "image",
        "infographic",
        "label",
        "labels",
        "pipeline",
        "reference",
        "rounded",
        "schematic",
        "service",
        "services",
        "tile",
        "tiles",
        "ui",
    }
)
EXPLICIT_FLOOR_WORDS = frozenset({"floor", "floors", "stone", "stones", "wet"})


def parse_prompt(prompt: str) -> PromptSpec:
    normalized = " ".join(TOKEN_RE.findall(prompt.lower()))
    tokens = tuple(TOKEN_RE.findall(prompt.lower()))
    token_set = set(tokens)

    colors = tuple(dict.fromkeys(token for token in tokens if token in COLOR_WORDS))
    styles = tuple(dict.fromkeys(token for token in tokens if token in STYLE_WORDS))
    moods = tuple(dict.fromkeys(token for token in tokens if token in MOOD_WORDS))

    objects: list[str] = []
    for canonical, aliases in OBJECT_ALIASES.items():
        if aliases & token_set:
            if canonical == "floor" and _is_graphic_tile_context(token_set):
                continue
            objects.append(canonical)

    if not objects:
        objects = ["abstract"]
    if not colors:
        colors = DEFAULT_COLORS

    return PromptSpec(
        original=prompt,
        normalized=normalized,
        tokens=tokens,
        color_words=colors,
        objects=tuple(objects),
        style_words=styles,
        mood_words=moods,
    )


def _is_graphic_tile_context(token_set: set[str]) -> bool:
    if not {"tile", "tiles"} & token_set:
        return False
    if EXPLICIT_FLOOR_WORDS & token_set:
        return False
    return bool(GRAPHIC_CONTEXT_WORDS & token_set)
