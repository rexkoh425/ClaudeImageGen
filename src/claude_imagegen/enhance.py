from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .critique import write_pair_evaluation_request


@dataclass(frozen=True)
class EnhanceNightOptions:
    input_image: Path
    prompt: str
    output_dir: Path
    quality_target: float = 0.9
    night_luma_ceiling: float = 0.34
    mist_cap: float = 0.22
    highlight_rolloff: float = 0.35
    local_contrast: float = 0.9
    shadow_lift: float = 0.0
    foliage_clarity: float = 0.0
    mist_beam_strength: float = 0.0


@dataclass(frozen=True)
class EnhanceNightResult:
    image: Image.Image
    metadata: dict[str, object]
    image_path: Path
    metadata_path: Path
    pair_evaluation_request_path: Path


def enhance_night_image(options: EnhanceNightOptions) -> EnhanceNightResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    if not options.input_image.exists():
        raise FileNotFoundError(f"input image does not exist: {options.input_image}")

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"

    with Image.open(options.input_image) as source:
        before = source.convert("RGB")

    before_stats = _luma_stats(before)
    arr = np.asarray(before, dtype=np.float32) / 255.0
    arr = _apply_local_contrast(arr, amount=options.local_contrast)
    arr = _reduce_mist_veil(arr, cap=options.mist_cap)
    arr = _rolloff_highlights(arr, rolloff=options.highlight_rolloff)
    arr = _lift_crushed_shadows(arr, amount=options.shadow_lift)
    arr = _apply_foliage_clarity(arr, amount=options.foliage_clarity)
    arr = _enforce_luma_ceiling(arr, ceiling=options.night_luma_ceiling)
    arr = _boost_lower_half_contrast(arr, amount=options.local_contrast * 0.45)
    arr, mist_beam_source_count = _apply_mist_beams(arr, amount=options.mist_beam_strength)
    arr = _enforce_luma_ceiling(arr, ceiling=options.night_luma_ceiling)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    image.save(image_path)

    after_stats = _luma_stats(image)
    metadata: dict[str, object] = {
        "engine": "night-preserving-postprocess-v1",
        "backend": "local-postprocess",
        "prompt": options.prompt,
        "input_image": str(options.input_image),
        "image": str(image_path),
        "width": image.width,
        "height": image.height,
        "quality_target": options.quality_target,
        "night_luma_ceiling": options.night_luma_ceiling,
        "mist_cap": options.mist_cap,
        "highlight_rolloff": options.highlight_rolloff,
        "local_contrast": options.local_contrast,
        "shadow_lift": options.shadow_lift,
        "foliage_clarity": options.foliage_clarity,
        "mist_beam_strength": options.mist_beam_strength,
        "mist_beam_source_count": mist_beam_source_count,
        "before_mean_luma": before_stats["mean_luma"],
        "before_max_luma": before_stats["max_luma"],
        "before_lower_luma_std": before_stats["lower_luma_std"],
        "before_lower_luma_p10": before_stats["lower_luma_p10"],
        "after_mean_luma": after_stats["mean_luma"],
        "after_max_luma": after_stats["max_luma"],
        "after_lower_luma_std": after_stats["lower_luma_std"],
        "after_lower_luma_p10": after_stats["lower_luma_p10"],
        "acceptance_requires_pair_evaluation": True,
        "revision_hints": [
            "Use pair-evaluation-request.json with Claude vision before accepting a 0.9 target.",
            "If Claude still reports night-mood drift, lower night_luma_ceiling or mist_cap.",
            "If floor or leaf detail remains weak, raise local_contrast, shadow_lift, or foliage_clarity conservatively.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    pair_evaluation_request_path = write_pair_evaluation_request(
        output_dir,
        prompt=options.prompt,
        pairs=[
            {
                "id": "enhance-night",
                "before_image": str(options.input_image),
                "after_image": str(image_path),
            }
        ],
        quality_target=options.quality_target,
        notes=(
            "Judge whether night-preserving enhancement improved detail and atmosphere without "
            "turning deep night into dusk."
        ),
    )
    metadata["pair_evaluation_request"] = str(pair_evaluation_request_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return EnhanceNightResult(
        image=image,
        metadata=metadata,
        image_path=image_path,
        metadata_path=metadata_path,
        pair_evaluation_request_path=pair_evaluation_request_path,
    )


def _apply_local_contrast(arr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0:
        return np.clip(arr, 0.0, 1.0)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32) / 255.0
    return np.clip(arr + ((arr - blurred) * float(amount)), 0.0, 1.0)


def _reduce_mist_veil(arr: np.ndarray, *, cap: float) -> np.ndarray:
    if cap <= 0:
        return np.clip(arr, 0.0, 1.0)
    luma = _luma(arr)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    haze_mask = ((chroma < 0.16) & (luma > 0.28) & (luma < 0.82)).astype(np.float32)
    reduction = np.clip(float(cap), 0.0, 0.65) * haze_mask[..., None]
    return np.clip(arr * (1.0 - reduction), 0.0, 1.0)


def _rolloff_highlights(arr: np.ndarray, *, rolloff: float) -> np.ndarray:
    luma = _luma(arr)
    threshold = 0.78
    mask = luma > threshold
    if not np.any(mask):
        return np.clip(arr, 0.0, 1.0)
    compressed = threshold + ((luma - threshold) * np.clip(float(rolloff), 0.05, 1.0))
    scale = np.ones_like(luma)
    scale[mask] = compressed[mask] / np.maximum(luma[mask], 1e-6)
    return np.clip(arr * scale[..., None], 0.0, 1.0)


def _lift_crushed_shadows(arr: np.ndarray, *, amount: float) -> np.ndarray:
    lift = max(0.0, min(0.25, float(amount)))
    if lift <= 0:
        return np.clip(arr, 0.0, 1.0)
    luma = _luma(arr)
    threshold = 0.18
    mask = np.clip((threshold - luma) / threshold, 0.0, 1.0)
    target_luma = np.clip(luma + (lift * mask), 0.0, 1.0)
    neutral_delta = target_luma - luma
    lifted = np.clip(arr + neutral_delta[..., None], 0.0, 1.0)
    desaturation = np.clip(mask * lift * 4.0, 0.0, 0.9)
    neutral = np.repeat(target_luma[..., None], 3, axis=2)
    return np.clip((lifted * (1.0 - desaturation[..., None])) + (neutral * desaturation[..., None]), 0.0, 1.0)


def _apply_foliage_clarity(arr: np.ndarray, *, amount: float) -> np.ndarray:
    clarity = max(0.0, min(1.25, float(amount)))
    if clarity <= 0:
        return np.clip(arr, 0.0, 1.0)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    luma = _luma(arr)
    green_dominance = green - np.maximum(red, blue)
    green_mask = np.clip(green_dominance / 0.12, 0.0, 1.0)
    luma_mask = np.clip((luma - 0.05) / 0.12, 0.0, 1.0) * np.clip((0.62 - luma) / 0.24, 0.0, 1.0)
    mask = green_mask * luma_mask
    if not np.any(mask > 0.01):
        return np.clip(arr, 0.0, 1.0)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=1.1)), dtype=np.float32) / 255.0
    sharpened = np.clip(arr + ((arr - blurred) * clarity * 1.15), 0.0, 1.0)
    return np.clip((arr * (1.0 - mask[..., None])) + (sharpened * mask[..., None]), 0.0, 1.0)


def _apply_mist_beams(arr: np.ndarray, *, amount: float) -> tuple[np.ndarray, int]:
    strength = max(0.0, min(1.0, float(amount)))
    if strength <= 0:
        return np.clip(arr, 0.0, 1.0), 0
    sources = _warm_light_sources(arr)
    if not sources:
        return np.clip(arr, 0.0, 1.0), 0

    height, width, _ = arr.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    beam_mask = np.zeros((height, width), dtype=np.float32)
    shaft_offsets = (-0.16, 0.0, 0.16)

    for index, (source_x, source_y, _weight) in enumerate(sources[:3]):
        for offset in shaft_offsets:
            target_x = source_x + (width * offset)
            target_y = min(height * 0.76, source_y + (height * (0.42 + (index * 0.04))))
            vx = target_x - source_x
            vy = target_y - source_y
            length_sq = max((vx * vx) + (vy * vy), 1.0)
            dx = xx - source_x
            dy = yy - source_y
            projection = ((dx * vx) + (dy * vy)) / length_sq
            distance = np.abs((dx * vy) - (dy * vx)) / max(length_sq**0.5, 1.0)
            spread = width * (0.018 + (strength * 0.024))
            core = np.exp(-((distance / max(spread, 1.0)) ** 2))
            taper = np.sin(np.clip(projection, 0.0, 1.0) * np.pi)
            shaft = core * taper
            shaft[(projection <= 0.02) | (projection >= 1.0) | (yy < source_y) | (yy > height * 0.80)] = 0.0
            beam_mask = np.maximum(beam_mask, shaft.astype(np.float32))

    if not np.any(beam_mask > 0.001):
        return np.clip(arr, 0.0, 1.0), len(sources)

    mask_image = Image.fromarray(np.uint8(np.clip(beam_mask, 0.0, 1.0) * 255), "L").filter(
        ImageFilter.GaussianBlur(radius=max(1.0, min(width, height) * 0.025))
    )
    beam_mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    luma = _luma(arr)
    visibility = np.clip((0.58 - luma) / 0.5, 0.18, 1.0)
    alpha = np.clip(beam_mask * visibility * strength * 0.42, 0.0, 0.55)
    warm = np.array([1.0, 0.72, 0.24], dtype=np.float32)
    beamed = arr + ((1.0 - arr) * warm[None, None, :] * alpha[..., None])
    warmth_mix = np.clip(alpha * 0.22, 0.0, 0.16)
    beamed = (beamed * (1.0 - warmth_mix[..., None])) + (warm[None, None, :] * warmth_mix[..., None])
    return np.clip(beamed, 0.0, 1.0), len(sources)


def _warm_light_sources(arr: np.ndarray) -> list[tuple[float, float, float]]:
    height, width, _ = arr.shape
    luma = _luma(arr)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    upper = yy < (height * 0.62)
    warm_mask = (
        upper
        & (luma > 0.38)
        & (red > 0.62)
        & (green > 0.36)
        & (blue < 0.48)
        & ((red - blue) > 0.22)
        & (red > (blue * 1.45))
        & (green > (blue * 1.18))
    )
    weights = np.where(warm_mask, luma * ((red + green) * 0.5), 0.0)
    total = float(np.sum(weights))
    if total <= 1e-6:
        return []

    sources: list[tuple[float, float, float]] = []
    bins = min(8, max(1, width // 12))
    min_weight = total * 0.035
    for index in range(bins):
        x0 = int(round(index * width / bins))
        x1 = int(round((index + 1) * width / bins))
        section = weights[:, x0:x1]
        section_weight = float(np.sum(section))
        if section_weight <= min_weight:
            continue
        section_x = xx[:, x0:x1]
        section_y = yy[:, x0:x1]
        source_x = float(np.sum(section_x * section) / section_weight)
        source_y = float(np.sum(section_y * section) / section_weight)
        sources.append((source_x, source_y, section_weight))

    sources = _merge_nearby_light_sources(sources, width=width, height=height)
    sources.sort(key=lambda item: item[2], reverse=True)
    return sources[:4]


def _merge_nearby_light_sources(
    sources: list[tuple[float, float, float]], *, width: int, height: int
) -> list[tuple[float, float, float]]:
    merged: list[tuple[float, float, float]] = []
    for source_x, source_y, source_weight in sorted(sources, key=lambda item: item[0]):
        if merged:
            last_x, last_y, last_weight = merged[-1]
            if abs(source_x - last_x) <= width * 0.16 and abs(source_y - last_y) <= height * 0.18:
                total = last_weight + source_weight
                merged[-1] = (
                    ((last_x * last_weight) + (source_x * source_weight)) / total,
                    ((last_y * last_weight) + (source_y * source_weight)) / total,
                    total,
                )
                continue
        merged.append((source_x, source_y, source_weight))
    return merged


def _enforce_luma_ceiling(arr: np.ndarray, *, ceiling: float) -> np.ndarray:
    target = max(0.05, min(0.95, float(ceiling)))
    mean_luma = float(np.mean(_luma(arr)))
    if mean_luma <= target:
        return np.clip(arr, 0.0, 1.0)
    return np.clip(arr * (target / max(mean_luma, 1e-6)), 0.0, 1.0)


def _boost_lower_half_contrast(arr: np.ndarray, *, amount: float) -> np.ndarray:
    if amount <= 0:
        return np.clip(arr, 0.0, 1.0)
    result = np.array(arr, copy=True)
    start = result.shape[0] // 2
    lower = result[start:, :, :]
    luma = _luma(lower)
    mean = float(np.mean(luma))
    factor = 1.0 + min(0.85, float(amount))
    luma_target = np.clip(mean + ((luma - mean) * factor), 0.0, 1.0)
    scale = luma_target / np.maximum(luma, 1e-6)
    result[start:, :, :] = np.clip(lower * scale[..., None], 0.0, 1.0)
    return result


def _luma_stats(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = _luma(arr)
    lower = luma[luma.shape[0] // 2 :, :]
    return {
        "mean_luma": round(float(np.mean(luma)), 6),
        "max_luma": round(float(np.max(luma)), 6),
        "lower_luma_std": round(float(np.std(lower)), 6),
        "lower_luma_p10": round(float(np.percentile(lower, 10)), 6),
    }


def _luma(arr: np.ndarray) -> np.ndarray:
    return (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
