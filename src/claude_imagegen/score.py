from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .embedding import image_embedding_similarity
from .palette import COLOR_RGB, RGB, average_color
from .prompt import PromptSpec

COLOR_FEATURES = tuple(COLOR_RGB)
OBJECT_FEATURES = (
    "sun",
    "moon",
    "ocean",
    "mountain",
    "forest",
    "flower",
    "cloud",
    "building",
    "portrait",
    "robot",
    "diagram",
    "abstract",
)
STYLE_FEATURES = ("cinematic", "watercolor", "dreamy")
MOOD_FEATURES = ("bright", "soft", "quiet", "calm", "dark", "stormy", "dramatic", "warm")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
IMAGE_SIMILARITY_SIZE = (128, 128)
REGION_GRID = (
    ("top_left", 0, 1, 0, 1),
    ("top_center", 0, 1, 1, 2),
    ("top_right", 0, 1, 2, 3),
    ("middle_left", 1, 2, 0, 1),
    ("middle_center", 1, 2, 1, 2),
    ("middle_right", 1, 2, 2, 3),
    ("bottom_left", 2, 3, 0, 1),
    ("bottom_center", 2, 3, 1, 2),
    ("bottom_right", 2, 3, 2, 3),
)


@dataclass(frozen=True)
class ScoreResult:
    total_score: float
    text_score: float
    reference_score: float
    details: dict[str, float]


def image_similarity_score(
    image: Image.Image,
    comparison_image: Path,
    *,
    similarity_backend: str = "local",
    similarity_model: str | None = None,
    similarity_device: str = "auto",
) -> float:
    return float(
        image_similarity_details(
            image,
            comparison_image,
            similarity_backend=similarity_backend,
            similarity_model=similarity_model,
            similarity_device=similarity_device,
        )["continuity_score"]
    )


def image_similarity_details(
    image: Image.Image,
    comparison_image: Path,
    *,
    similarity_backend: str = "local",
    similarity_model: str | None = None,
    similarity_device: str = "auto",
) -> dict[str, object]:
    image_rgb = image.convert("RGB")
    with Image.open(comparison_image) as comparison:
        comparison_rgb = comparison.convert("RGB")

    image_array, comparison_array = _image_similarity_arrays(image_rgb, comparison_rgb)
    image_cosine_score = _cosine_similarity(
        _normalized_pixel_vector(image_array),
        _normalized_pixel_vector(comparison_array),
    )
    embedding_cosine_score = image_embedding_similarity(image_rgb, comparison_rgb)
    luminance_ssim_score = _luminance_ssim_score(image_array, comparison_array)
    multiscale_luminance_ssim_score = _multiscale_luminance_ssim_score(image_array, comparison_array)
    edge_cosine_score = _cosine_similarity(
        _edge_feature_vector(image_array),
        _edge_feature_vector(comparison_array),
    )
    color_histogram_score = _color_histogram_similarity(image_array, comparison_array)
    region_similarity_scores = _region_similarity_scores(image_array, comparison_array)
    regional_continuity_score = _clamp01(float(np.mean(list(region_similarity_scores.values()))))
    weakest_region, weakest_region_score = min(region_similarity_scores.items(), key=lambda item: item[1])
    local_continuity_score = _clamp01(
        (0.24 * embedding_cosine_score)
        + (0.15 * luminance_ssim_score)
        + (0.16 * multiscale_luminance_ssim_score)
        + (0.17 * edge_cosine_score)
        + (0.12 * color_histogram_score)
        + (0.06 * image_cosine_score)
        + (0.10 * regional_continuity_score)
    )

    details = {
        "image_embedding_cosine_score": round(embedding_cosine_score, 6),
        "image_cosine_score": round(image_cosine_score, 6),
        "luminance_ssim_score": round(luminance_ssim_score, 6),
        "multiscale_luminance_ssim_score": round(multiscale_luminance_ssim_score, 6),
        "edge_cosine_score": round(edge_cosine_score, 6),
        "color_histogram_score": round(color_histogram_score, 6),
        "regional_continuity_score": round(regional_continuity_score, 6),
        "region_similarity_scores": {key: round(value, 6) for key, value in region_similarity_scores.items()},
        "weakest_continuity_region": weakest_region,
        "weakest_continuity_region_score": round(weakest_region_score, 6),
        "local_continuity_score": round(local_continuity_score, 6),
    }

    normalized_backend = similarity_backend.strip().lower()
    if _is_clip_backend(normalized_backend):
        clip_score = _clip_image_image_score(
            image_rgb,
            comparison_rgb,
            model_name=similarity_model or DEFAULT_CLIP_MODEL,
            device=similarity_device,
        )
        details["clip_image_cosine_score"] = round(clip_score, 6)
        continuity_score = _clamp01((0.72 * local_continuity_score) + (0.28 * clip_score))
    elif _is_siglip_backend(normalized_backend):
        siglip_score = _siglip_image_image_score(
            image_rgb,
            comparison_rgb,
            model_name=similarity_model or DEFAULT_SIGLIP_MODEL,
            device=similarity_device,
        )
        details["siglip_image_cosine_score"] = round(siglip_score, 6)
        continuity_score = _clamp01((0.72 * local_continuity_score) + (0.28 * siglip_score))
    elif _is_dinov2_backend(normalized_backend):
        dinov2_score = _dinov2_image_image_score(
            image_rgb,
            comparison_rgb,
            model_name=similarity_model or DEFAULT_DINOV2_MODEL,
            device=similarity_device,
        )
        details["dinov2_image_cosine_score"] = round(dinov2_score, 6)
        continuity_score = _clamp01((0.68 * local_continuity_score) + (0.32 * dinov2_score))
    else:
        continuity_score = local_continuity_score

    details["continuity_score"] = round(continuity_score, 6)
    return details


def score_image(
    image: Image.Image,
    spec: PromptSpec,
    reference_image: Path | None = None,
    *,
    similarity_backend: str = "local",
    similarity_model: str | None = None,
    similarity_device: str = "auto",
) -> ScoreResult:
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)

    color_score = _color_score(array, spec.color_words)
    object_score = _object_score(array, spec.objects)
    contrast_score = _contrast_score(array)
    mood_score = _mood_score(array, spec.mood_words)
    cosine_score = _similarity_score(
        rgb,
        array,
        spec,
        backend=similarity_backend,
        model_name=similarity_model,
        device=similarity_device,
    )

    text_score = _clamp01(
        0.30 * color_score
        + 0.32 * object_score
        + 0.10 * contrast_score
        + 0.06 * mood_score
        + 0.22 * cosine_score
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
            "cosine_score": cosine_score,
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

    scores = [_object_presence(array, obj) for obj in objects]

    return _clamp01(float(np.mean(scores)))


def _reference_score(image: Image.Image, reference_image: Path) -> float:
    with Image.open(reference_image) as reference:
        reference_average = average_color(reference)
        reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    output_average = average_color(image)
    output_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    average_similarity = _rgb_similarity(output_average, reference_average)
    feature_similarity = _cosine_similarity(
        _image_feature_vector(output_array),
        _image_feature_vector(reference_array),
    )
    return _clamp01(0.65 * average_similarity + 0.35 * feature_similarity)


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


def _hard_edge_density(array: np.ndarray) -> float:
    luminance = array.mean(axis=2)
    if luminance.shape[0] <= 1 or luminance.shape[1] <= 1:
        return 0.0
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    density = ((dx > 18.0).mean() + (dy > 18.0).mean()) / 2.0
    return _clamp01(float(density / 0.16))


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


def _text_image_cosine_score(array: np.ndarray, spec: PromptSpec) -> float:
    text_vector = _text_feature_vector(spec)
    image_vector = _image_feature_vector(array)
    return _cosine_similarity(text_vector, image_vector)


def _similarity_score(
    image: Image.Image,
    array: np.ndarray,
    spec: PromptSpec,
    *,
    backend: str,
    model_name: str | None,
    device: str,
) -> float:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "local":
        return _text_image_cosine_score(array, spec)
    if _is_clip_backend(normalized_backend):
        return _clip_text_image_score(
            image,
            spec.normalized,
            model_name=model_name or DEFAULT_CLIP_MODEL,
            device=device,
        )
    if _is_siglip_backend(normalized_backend):
        return _siglip_text_image_score(
            image,
            spec.normalized,
            model_name=model_name or DEFAULT_SIGLIP_MODEL,
            device=device,
        )
    raise ValueError(f"Unsupported similarity backend: {backend}")


def _text_feature_vector(spec: PromptSpec) -> np.ndarray:
    color_words = set(spec.color_words)
    object_words = set(spec.objects)
    style_words = set(spec.style_words)
    mood_words = set(spec.mood_words)

    values: list[float] = []
    values.extend(1.0 if color in color_words else 0.0 for color in COLOR_FEATURES)
    values.extend(1.0 if obj in object_words else 0.0 for obj in OBJECT_FEATURES)
    values.extend(1.0 if style in style_words else 0.0 for style in STYLE_FEATURES)
    values.extend(1.0 if mood in mood_words else 0.0 for mood in MOOD_FEATURES)
    values.append(1.0 if style_words or mood_words else 0.35)
    return np.asarray(values, dtype=np.float32)


def _image_feature_vector(array: np.ndarray) -> np.ndarray:
    values: list[float] = []
    values.extend(_color_presence(array, COLOR_RGB[color]) for color in COLOR_FEATURES)
    values.extend(_object_presence(array, obj) for obj in OBJECT_FEATURES)
    values.extend(
        (
            _clamp01((_contrast_score(array) + _warm_presence(array)) / 2.0),
            _clamp01(1.0 - _edge_density(array) * 0.45),
            _clamp01((_bright_neutral_presence(array) + _contrast_score(array)) / 2.0),
        )
    )
    values.extend(
        (
            _clamp01(float(array.mean()) / 190.0),
            _clamp01(float(array.mean()) / 210.0),
            _clamp01(float(array.mean()) / 220.0),
            _clamp01(float(array.mean()) / 210.0),
            _clamp01(1.0 - float(array.mean()) / 210.0),
            _clamp01(1.0 - float(array.mean()) / 220.0 + _contrast_score(array) * 0.25),
            _contrast_score(array),
            _warm_presence(array),
        )
    )
    values.append(_contrast_score(array))
    return np.asarray(values, dtype=np.float32)


def _color_presence(array: np.ndarray, target: RGB) -> float:
    target_array = np.array(target, dtype=np.float32)
    distances = np.linalg.norm(array - target_array, axis=2)
    closest = float(np.percentile(distances, 10))
    return _clamp01(1.0 - closest / 255.0)


def _object_presence(array: np.ndarray, obj: str) -> float:
    height = array.shape[0]
    upper = array[: max(1, height // 2)]
    lower = array[height // 2 :]
    middle = array[height // 3 : max(height // 3 + 1, int(height * 0.72))]

    if obj == "sun":
        return _warm_presence(upper)
    if obj == "moon":
        return _bright_neutral_presence(upper)
    if obj == "ocean":
        return _blue_presence(lower)
    if obj == "mountain":
        return max(_edge_density(middle), _green_presence(middle) * 0.7)
    if obj in {"forest", "flower"}:
        return _green_presence(lower)
    if obj == "cloud":
        return _bright_neutral_presence(upper)
    if obj == "building":
        return max(_edge_density(lower), _dark_presence(middle))
    if obj in {"portrait", "robot"}:
        return max(_edge_density(middle), _contrast_score(middle))
    if obj == "diagram":
        return _clamp01(
            0.36 * _contrast_score(array)
            + 0.34 * _hard_edge_density(array)
            + 0.18 * max(_blue_presence(array), _warm_presence(array))
            + 0.12 * _edge_density(array)
        )
    if obj == "abstract":
        return max(_contrast_score(array), _edge_density(array))
    return _contrast_score(array)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return _clamp01((float(np.dot(left, right)) / denominator + 1.0) / 2.0)


def _clip_text_image_score(image: Image.Image, text: str, *, model_name: str, device: str) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-clip similarity backend requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_clip_model(model_name, resolved_device)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        outputs = model(**inputs)
        similarity = torch.nn.functional.cosine_similarity(outputs.text_embeds, outputs.image_embeds).item()
    return _clamp01((similarity + 1.0) / 2.0)


def _clip_image_image_score(
    image: Image.Image,
    comparison_image: Image.Image,
    *,
    model_name: str,
    device: str,
) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-clip image similarity requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_clip_model(model_name, resolved_device)
    inputs = processor(images=[comparison_image, image], return_tensors="pt")
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        embeddings = model.get_image_features(pixel_values=inputs["pixel_values"])
        similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    return _clamp01((similarity + 1.0) / 2.0)


def _siglip_text_image_score(image: Image.Image, text: str, *, model_name: str, device: str) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-siglip similarity backend requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_siglip_model(model_name, resolved_device)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding="max_length")
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits_per_image[0, 0]).item()
    return _clamp01(score)


def _siglip_image_image_score(
    image: Image.Image,
    comparison_image: Image.Image,
    *,
    model_name: str,
    device: str,
) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-siglip image similarity requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_siglip_model(model_name, resolved_device)
    inputs = processor(images=[comparison_image, image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(resolved_device)
    with torch.no_grad():
        embeddings = model.get_image_features(pixel_values=pixel_values)
        similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    return _clamp01((similarity + 1.0) / 2.0)


def _dinov2_image_image_score(
    image: Image.Image,
    comparison_image: Image.Image,
    *,
    model_name: str,
    device: str,
) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-dinov2 image similarity requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_dinov2_model(model_name, resolved_device)
    inputs = processor(images=[comparison_image, image], return_tensors="pt")
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = getattr(outputs, "pooler_output", None)
        if embeddings is None:
            embeddings = outputs.last_hidden_state[:, 0]
        similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    return _clamp01((similarity + 1.0) / 2.0)


def _image_similarity_arrays(image: Image.Image, comparison_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    image_resized = image.resize(IMAGE_SIMILARITY_SIZE, Image.Resampling.BICUBIC)
    comparison_resized = comparison_image.resize(IMAGE_SIMILARITY_SIZE, Image.Resampling.BICUBIC)
    return (
        np.asarray(image_resized, dtype=np.float32),
        np.asarray(comparison_resized, dtype=np.float32),
    )


def _normalized_pixel_vector(array: np.ndarray) -> np.ndarray:
    return (array.reshape(-1) / 255.0).astype(np.float32)


def _luminance_ssim_score(left: np.ndarray, right: np.ndarray) -> float:
    left_luminance = left.mean(axis=2)
    right_luminance = right.mean(axis=2)
    left_mean = float(left_luminance.mean())
    right_mean = float(right_luminance.mean())
    left_variance = float(((left_luminance - left_mean) ** 2).mean())
    right_variance = float(((right_luminance - right_mean) ** 2).mean())
    covariance = float(((left_luminance - left_mean) * (right_luminance - right_mean)).mean())
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denominator = (left_mean**2 + right_mean**2 + c1) * (left_variance + right_variance + c2)
    if denominator <= 0:
        return 0.0
    score = ((2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2)) / denominator
    return _clamp01(score)


def _multiscale_luminance_ssim_score(left: np.ndarray, right: np.ndarray) -> float:
    weights = (0.42, 0.28, 0.18, 0.12)
    scores: list[float] = []
    left_scale = left
    right_scale = right
    for _ in weights:
        scores.append(_luminance_ssim_score(left_scale, right_scale))
        if min(left_scale.shape[0], left_scale.shape[1]) < 16:
            break
        left_scale = _downsample_mean_2x(left_scale)
        right_scale = _downsample_mean_2x(right_scale)

    active_weights = weights[: len(scores)]
    total_weight = sum(active_weights)
    if total_weight <= 0:
        return 0.0
    score = sum(score * weight for score, weight in zip(scores, active_weights)) / total_weight
    return _clamp01(score)


def _downsample_mean_2x(array: np.ndarray) -> np.ndarray:
    height = array.shape[0] - (array.shape[0] % 2)
    width = array.shape[1] - (array.shape[1] % 2)
    if height <= 0 or width <= 0:
        return array
    cropped = array[:height, :width, :]
    return cropped.reshape(height // 2, 2, width // 2, 2, array.shape[2]).mean(axis=(1, 3))


def _edge_feature_vector(array: np.ndarray) -> np.ndarray:
    luminance = array.mean(axis=2)
    horizontal = np.abs(np.diff(luminance, axis=1)).reshape(-1) / 255.0
    vertical = np.abs(np.diff(luminance, axis=0)).reshape(-1) / 255.0
    return np.concatenate([horizontal, vertical]).astype(np.float32)


def _color_histogram_similarity(left: np.ndarray, right: np.ndarray) -> float:
    scores: list[float] = []
    for channel in range(3):
        left_hist, _ = np.histogram(left[:, :, channel], bins=16, range=(0, 255))
        right_hist, _ = np.histogram(right[:, :, channel], bins=16, range=(0, 255))
        left_total = float(left_hist.sum())
        right_total = float(right_hist.sum())
        if left_total <= 0 or right_total <= 0:
            scores.append(0.0)
            continue
        left_normalized = left_hist.astype(np.float32) / left_total
        right_normalized = right_hist.astype(np.float32) / right_total
        scores.append(float(np.minimum(left_normalized, right_normalized).sum()))
    return _clamp01(float(np.mean(scores)) if scores else 0.0)


def _region_similarity_scores(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    height, width = left.shape[:2]
    scores: dict[str, float] = {}
    for name, row_start, row_end, col_start, col_end in REGION_GRID:
        y0 = (height * row_start) // 3
        y1 = (height * row_end) // 3
        x0 = (width * col_start) // 3
        x1 = (width * col_end) // 3
        left_region = left[y0:y1, x0:x1, :]
        right_region = right[y0:y1, x0:x1, :]
        scores[name] = _regional_similarity_score(left_region, right_region)
    return scores


def _regional_similarity_score(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    ssim_score = _luminance_ssim_score(left, right)
    edge_score = _cosine_similarity(_edge_feature_vector(left), _edge_feature_vector(right))
    color_score = _color_histogram_similarity(left, right)
    pixel_score = _cosine_similarity(_normalized_pixel_vector(left), _normalized_pixel_vector(right))
    return _clamp01((0.36 * ssim_score) + (0.26 * edge_score) + (0.24 * color_score) + (0.14 * pixel_score))


def _resolve_torch_device(torch_module: object, device: str) -> str:
    normalized = device.strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError("similarity_device='cuda' requested, but torch reports CUDA is unavailable.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported similarity device: {device}")
    return normalized


def _is_clip_backend(normalized_backend: str) -> bool:
    return normalized_backend in {"clip", "transformers-clip"}


def _is_siglip_backend(normalized_backend: str) -> bool:
    return normalized_backend in {"siglip", "transformers-siglip"}


def _is_dinov2_backend(normalized_backend: str) -> bool:
    return normalized_backend in {"dinov2", "transformers-dinov2"}


@lru_cache(maxsize=2)
def _load_clip_model(model_name: str, device: str) -> tuple[object, object]:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-clip similarity backend requires torch and transformers.") from exc

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=2)
def _load_siglip_model(model_name: str, device: str) -> tuple[object, object]:
    try:
        from transformers import SiglipModel, SiglipProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-siglip similarity backend requires torch and transformers.") from exc

    processor = SiglipProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@lru_cache(maxsize=2)
def _load_dinov2_model(model_name: str, device: str) -> tuple[object, object]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-dinov2 image similarity requires torch and transformers.") from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
