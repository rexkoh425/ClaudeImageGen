from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .palette import COLOR_RGB, RGB, average_color
from .prompt import PromptSpec


@dataclass(frozen=True)
class ScoreResult:
    total_score: float
    text_score: float
    reference_score: float
    details: dict[str, float]


def score_image(image: Image.Image, spec: PromptSpec, reference_image: Path | None = None) -> ScoreResult:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)

    color_score = _color_score(array, spec.color_words)
    object_score = _object_score(array, spec.objects)
    contrast_score = _contrast_score(array)
    mood_score = _mood_score(array, spec.mood_words)

    text_score = _clamp01(
        0.38 * color_score
        + 0.40 * object_score
        + 0.14 * contrast_score
        + 0.08 * mood_score
    )
    reference_score = _reference_score(rgb, reference_image) if reference_image else 0.0
    total_score = _clamp01(0.82 * text_score + 0.18 * reference_score)

    return ScoreResult(
        total_score=total_score,
        text_score=text_score,
        reference_score=reference_score,
        details={
            "color_score": color_score,
            "object_score": object_score,
            "contrast_score": contrast_score,
            "mood_score": mood_score,
        },
    )


def _color_score(array: np.ndarray, color_words: tuple[str, ...]) -> float:
    scores: list[float] = []
    for word in color_words:
        target = np.array(COLOR_RGB.get(word, COLOR_RGB["blue"]), dtype=np.float32)
        distances = np.linalg.norm(array - target, axis=2)
        closest = float(np.percentile(distances, 8))
        scores.append(_clamp01(1.0 - closest / 235.0))
    return float(np.mean(scores)) if scores else 0.0


def _object_score(array: np.ndarray, objects: tuple[str, ...]) -> float:
    if not objects:
        return 0.0

    height = array.shape[0]
    scores: list[float] = []
    upper = array[: max(1, height // 2)]
    lower = array[height // 2 :]
    middle = array[height // 3 : max(height // 3 + 1, int(height * 0.72))]

    for obj in objects:
        if obj == "sun":
            scores.append(_warm_presence(upper))
        elif obj == "moon":
            scores.append(_bright_neutral_presence(upper))
        elif obj == "ocean":
            scores.append(_blue_presence(lower))
        elif obj == "mountain":
            scores.append(max(_edge_density(middle), _green_presence(middle) * 0.7))
        elif obj in {"forest", "flower"}:
            scores.append(_green_presence(lower))
        elif obj == "cloud":
            scores.append(_bright_neutral_presence(upper))
        elif obj == "building":
            scores.append(max(_edge_density(lower), _dark_presence(middle)))
        elif obj in {"portrait", "robot"}:
            scores.append(max(_edge_density(middle), _contrast_score(middle)))
        elif obj == "abstract":
            scores.append(max(_contrast_score(array), _edge_density(array)))
        else:
            scores.append(_contrast_score(array))

    return _clamp01(float(np.mean(scores)))


def _reference_score(image: Image.Image, reference_image: Path) -> float:
    with Image.open(reference_image) as reference:
        reference_average = average_color(reference)
    output_average = average_color(image)
    return _rgb_similarity(output_average, reference_average)


def _rgb_similarity(a: RGB, b: RGB) -> float:
    distance = sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
    return _clamp01(1.0 - distance / 441.7)


def _warm_presence(array: np.ndarray) -> float:
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    mask = (red > 135) & (green > 45) & (red >= blue * 1.05)
    return _presence(mask)


def _blue_presence(array: np.ndarray) -> float:
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    mask = (blue > 95) & (blue > red * 1.08) & (blue >= green * 0.78)
    return _presence(mask)


def _green_presence(array: np.ndarray) -> float:
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    mask = (green > 90) & (green > red * 1.08) & (green >= blue * 0.75)
    return _presence(mask)


def _bright_neutral_presence(array: np.ndarray) -> float:
    spread = array.max(axis=2) - array.min(axis=2)
    mean = array.mean(axis=2)
    return _presence((mean > 165) & (spread < 75))


def _dark_presence(array: np.ndarray) -> float:
    return _presence(array.mean(axis=2) < 80)


def _presence(mask: np.ndarray) -> float:
    fraction = float(mask.mean())
    return _clamp01(fraction * 4.2)


def _contrast_score(array: np.ndarray) -> float:
    luminance = array.mean(axis=2)
    return _clamp01(float(luminance.std()) / 72.0)


def _edge_density(array: np.ndarray) -> float:
    luminance = array.mean(axis=2)
    dy = np.abs(np.diff(luminance, axis=0)).mean() if luminance.shape[0] > 1 else 0.0
    dx = np.abs(np.diff(luminance, axis=1)).mean() if luminance.shape[1] > 1 else 0.0
    return _clamp01(float(dx + dy) / 26.0)


def _mood_score(array: np.ndarray, mood_words: tuple[str, ...]) -> float:
    if not mood_words:
        return 0.55
    mean = float(array.mean())
    contrast = _contrast_score(array)
    scores: list[float] = []
    for mood in mood_words:
        if mood in {"bright", "soft", "quiet", "calm"}:
            scores.append(_clamp01(mean / 190.0))
        elif mood in {"dark", "stormy"}:
            scores.append(_clamp01(1.0 - mean / 210.0))
        elif mood in {"dramatic", "warm"}:
            scores.append(max(contrast, _warm_presence(array)))
    return float(np.mean(scores))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
