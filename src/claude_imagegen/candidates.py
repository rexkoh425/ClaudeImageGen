from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
from PIL import Image, ImageFilter


AESTHETIC_NEUTRAL_SCORE = 0.5


def annotate_candidate_selection(candidate: dict[str, object]) -> dict[str, object]:
    score, reasons = compute_candidate_selection(candidate)
    candidate["selection_score"] = score
    candidate["selection_reasons"] = reasons
    return candidate


def compute_candidate_aesthetic_score(image: Image.Image) -> tuple[float, dict[str, float]]:
    sample = image.convert("RGB")
    sample.thumbnail((256, 256))
    rgb = np.asarray(sample, dtype=np.float32) / 255.0
    luminance = (0.2126 * rgb[:, :, 0]) + (0.7152 * rgb[:, :, 1]) + (0.0722 * rgb[:, :, 2])
    luminance_mean = float(luminance.mean())
    luminance_std = float(luminance.std())

    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    saturation = np.divide(
        channel_max - channel_min,
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 0,
    )
    saturation_mean = float(saturation.mean())

    edge_image = sample.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_density = float(np.asarray(edge_image, dtype=np.float32).mean() / 255.0)
    quantized = sample.quantize(colors=16)
    color_bucket_count = sum(1 for count in quantized.histogram() if count)

    brightness_score = _clamp(1.0 - abs(luminance_mean - 0.52) / 0.48)
    contrast_score = _clamp(luminance_std / 0.25)
    saturation_score = _clamp(saturation_mean / 0.38)
    if saturation_mean > 0.82:
        saturation_score = min(saturation_score, _clamp(1.0 - (saturation_mean - 0.82) / 0.18))
    edge_score = _clamp(1.0 - abs(edge_density - 0.10) / 0.10)
    color_variety_score = _clamp(color_bucket_count / 10.0)

    aesthetic_score = _clamp(
        (0.22 * brightness_score)
        + (0.25 * contrast_score)
        + (0.22 * saturation_score)
        + (0.18 * edge_score)
        + (0.13 * color_variety_score)
    )
    details = {
        "brightness_score": round(brightness_score, 6),
        "contrast_score": round(contrast_score, 6),
        "saturation_score": round(saturation_score, 6),
        "edge_score": round(edge_score, 6),
        "color_variety_score": round(color_variety_score, 6),
        "luminance_mean": round(luminance_mean, 6),
        "luminance_std": round(luminance_std, 6),
        "saturation_mean": round(saturation_mean, 6),
        "edge_density": round(edge_density, 6),
        "color_bucket_count": float(color_bucket_count),
    }
    return round(aesthetic_score, 6), details


def select_recommended_candidate(candidates: Iterable[object]) -> dict[str, object]:
    best: dict[str, object] | None = None
    best_score = -1.0
    best_rank = 0

    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            continue
        candidate = dict(raw_candidate)
        score = _float_or_none(candidate.get("selection_score"))
        if score is None:
            annotate_candidate_selection(candidate)
            score = _float(candidate.get("selection_score"))
        else:
            score = _clamp(score)
            candidate["selection_score"] = round(score, 6)
            candidate["selection_reasons"] = _string_list(candidate.get("selection_reasons")) or [
                f"precomputed selection_score={score:.3f}"
            ]

        rank = _rank(candidate)
        if best is None or score > best_score or (score == best_score and rank and rank < best_rank):
            best = candidate
            best_score = score
            best_rank = rank

    if best is None:
        raise ValueError("Candidate index does not contain any selectable candidates.")
    return best


def compute_candidate_selection(candidate: Mapping[str, object]) -> tuple[float, list[str]]:
    total_score = _float(candidate.get("total_score"))
    caption_similarity_score = _float(candidate.get("caption_similarity_score"))
    reference_score = _float(candidate.get("reference_score"))
    aesthetic_value = _float_or_none(candidate.get("aesthetic_score"))
    aesthetic_score = _clamp(aesthetic_value if aesthetic_value is not None else AESTHETIC_NEUTRAL_SCORE)
    missing_objects = _string_list(candidate.get("caption_missing_objects"))
    missing_colors = _string_list(candidate.get("caption_missing_colors"))
    unexpected_objects = _string_list(candidate.get("caption_unexpected_objects"))
    unexpected_colors = _string_list(candidate.get("caption_unexpected_colors"))

    missing_object_penalty = 0.06 * len(missing_objects)
    missing_color_penalty = 0.04 * len(missing_colors)
    unexpected_object_penalty = 0.02 * len(unexpected_objects)
    unexpected_color_penalty = 0.01 * len(unexpected_colors)
    raw_score = (
        (0.58 * total_score)
        + (0.24 * caption_similarity_score)
        + (0.08 * reference_score)
        + (0.10 * aesthetic_score)
        - missing_object_penalty
        - missing_color_penalty
        - unexpected_object_penalty
        - unexpected_color_penalty
    )
    score = round(_clamp(raw_score), 6)

    reasons = [
        f"total_score={total_score:.3f} weight=0.58",
        f"caption_similarity_score={caption_similarity_score:.3f} weight=0.24",
        f"reference_score={reference_score:.3f} weight=0.08",
        f"aesthetic_score={aesthetic_score:.3f} weight=0.10",
    ]
    if aesthetic_value is None:
        reasons[-1] += " default=neutral"
    if missing_objects:
        reasons.append(f"missing_objects_penalty={missing_object_penalty:.3f}")
    if missing_colors:
        reasons.append(f"missing_colors_penalty={missing_color_penalty:.3f}")
    if unexpected_objects:
        reasons.append(f"unexpected_objects_penalty={unexpected_object_penalty:.3f}")
    if unexpected_colors:
        reasons.append(f"unexpected_colors_penalty={unexpected_color_penalty:.3f}")
    return score, reasons


def _float(value: object) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else 0.0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _rank(candidate: Mapping[str, object]) -> int:
    try:
        return int(candidate.get("rank", 0))
    except (TypeError, ValueError):
        return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
