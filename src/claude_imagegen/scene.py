from __future__ import annotations

from dataclasses import dataclass, replace
import random

from .palette import RGB, blend, palette_from_words
from .prompt import PromptSpec


@dataclass(frozen=True)
class SceneCandidate:
    spec: PromptSpec
    palette: tuple[RGB, ...]
    seed: int
    horizon: float
    shape_count: int
    variation: float


def build_initial_candidate(
    spec: PromptSpec,
    seed: int = 0,
    reference_palette: tuple[RGB, ...] | None = None,
    initial_palette: tuple[RGB, ...] | None = None,
) -> SceneCandidate:
    prompt_palette = palette_from_words(spec.color_words)
    palette = _merge_palettes(reference_palette, prompt_palette, initial_palette)
    rng = random.Random(seed)

    if "ocean" in spec.objects or "mountain" in spec.objects:
        horizon = 0.56
    elif "building" in spec.objects:
        horizon = 0.68
    else:
        horizon = 0.50

    return SceneCandidate(
        spec=spec,
        palette=palette,
        seed=seed,
        horizon=max(0.30, min(0.78, horizon + rng.uniform(-0.04, 0.04))),
        shape_count=8 + len(spec.objects) * 3,
        variation=rng.random(),
    )


def mutate_candidate(candidate: SceneCandidate, iteration: int) -> SceneCandidate:
    rng = random.Random(candidate.seed + iteration * 7919)
    shifted_palette = tuple(
        blend(color, rng.choice(candidate.palette), rng.uniform(0.05, 0.22))
        for color in candidate.palette
    )
    return replace(
        candidate,
        seed=candidate.seed + iteration,
        palette=shifted_palette,
        horizon=max(0.25, min(0.80, candidate.horizon + rng.uniform(-0.035, 0.035))),
        shape_count=max(5, candidate.shape_count + rng.choice((-1, 0, 1))),
        variation=rng.random(),
    )


def _merge_palettes(
    reference_palette: tuple[RGB, ...] | None,
    prompt_palette: tuple[RGB, ...],
    initial_palette: tuple[RGB, ...] | None,
) -> tuple[RGB, ...]:
    colors: list[RGB] = []
    if reference_palette:
        colors.extend(reference_palette[:3])
    colors.extend(prompt_palette)
    if initial_palette:
        colors.extend(initial_palette[:2])

    deduped: list[RGB] = []
    for color in colors:
        if color not in deduped:
            deduped.append(color)

    return tuple(deduped[:7])
