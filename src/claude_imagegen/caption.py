from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .palette import COLOR_RGB, RGB
from .prompt import parse_prompt

DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    prompt_similarity_score: float
    backend: str
    model_name: str | None
    requested_device: str
    effective_device: str
    tokens: tuple[str, ...]


def caption_image(
    image: Image.Image,
    *,
    prompt: str,
    backend: str = "local",
    model_name: str | None = None,
    device: str = "auto",
) -> CaptionResult:
    normalized_backend = backend.strip().lower()
    requested_device = device.strip().lower()

    if normalized_backend == "none":
        caption = ""
        effective_device = "none"
        effective_model = None
    elif normalized_backend == "local":
        caption = _local_caption(image.convert("RGB"))
        effective_device = "cpu"
        effective_model = None
    elif normalized_backend in {"blip", "transformers-blip"}:
        effective_model = model_name or DEFAULT_BLIP_MODEL
        caption, effective_device = _blip_caption(
            image.convert("RGB"),
            model_name=effective_model,
            device=requested_device,
        )
    else:
        raise ValueError(f"Unsupported caption backend: {backend}")

    similarity = caption_prompt_similarity(prompt, caption)
    caption_spec = parse_prompt(caption)
    return CaptionResult(
        caption=caption,
        prompt_similarity_score=similarity,
        backend=normalized_backend,
        model_name=effective_model,
        requested_device=requested_device,
        effective_device=effective_device,
        tokens=caption_spec.tokens,
    )


def caption_prompt_similarity(prompt: str, caption: str) -> float:
    if not caption.strip():
        return 0.0

    prompt_spec = parse_prompt(prompt)
    caption_spec = parse_prompt(caption)

    prompt_objects = set(prompt_spec.objects)
    caption_objects = set(caption_spec.objects)
    prompt_colors = set(prompt_spec.color_words)
    caption_colors = set(caption_spec.color_words)
    prompt_tokens = set(prompt_spec.tokens)
    caption_tokens = set(caption_spec.tokens)

    object_recall = _set_recall(prompt_objects, caption_objects)
    color_recall = _set_recall(prompt_colors, caption_colors)
    token_jaccard = _set_jaccard(prompt_tokens, caption_tokens)

    return _clamp01(0.56 * object_recall + 0.24 * color_recall + 0.20 * token_jaccard)


def _local_caption(image: Image.Image) -> str:
    array = np.asarray(image, dtype=np.float32)
    height = array.shape[0]
    upper = array[: max(1, height // 2)]
    middle = array[height // 3 : max(height // 3 + 1, int(height * 0.72))]
    lower = array[height // 2 :]

    phrases: list[str] = []
    if _warm_presence(upper) >= 0.10:
        phrases.append(f"{_dominant_color_name(upper)} sun")
    elif _bright_neutral_presence(upper) >= 0.16:
        phrases.append("white moon")

    if _bright_neutral_presence(upper) >= 0.28:
        phrases.append("soft clouds")

    if _edge_density(middle) >= 0.16:
        phrases.append(f"{_dominant_color_name(middle)} mountains")

    if _blue_presence(lower) >= 0.10:
        phrases.append("blue ocean")

    if _green_presence(lower) >= 0.12:
        phrases.append("green forest")

    if _dark_presence(middle) >= 0.32 and _edge_density(middle) >= 0.09:
        phrases.append("dark city skyline")

    if not phrases:
        phrases.append(f"{_dominant_color_name(array)} abstract composition")

    return "a local image showing " + _join_phrases(_dedupe(phrases))


def _blip_caption(image: Image.Image, *, model_name: str, device: str) -> tuple[str, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-blip caption backend requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    processor, model = _load_blip_model(model_name, resolved_device)
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=32)
    caption = processor.decode(generated[0], skip_special_tokens=True).strip()
    return caption, resolved_device


@lru_cache(maxsize=2)
def _load_blip_model(model_name: str, device: str) -> tuple[object, object]:
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-blip caption backend requires torch and transformers.") from exc

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def _dominant_color_name(array: np.ndarray) -> str:
    average = tuple(int(channel) for channel in array.mean(axis=(0, 1)))  # type: ignore[misc]
    scored = sorted(
        ((name, _rgb_similarity(average, rgb)) for name, rgb in COLOR_RGB.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return _normalize_color_name(scored[0][0])


def _normalize_color_name(name: str) -> str:
    return "gray" if name == "grey" else name


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


def _edge_density(array: np.ndarray) -> float:
    luminance = array.mean(axis=2)
    dy = np.abs(np.diff(luminance, axis=0)).mean() if luminance.shape[0] > 1 else 0.0
    dx = np.abs(np.diff(luminance, axis=1)).mean() if luminance.shape[1] > 1 else 0.0
    return _clamp01(float(dx + dy) / 26.0)


def _presence(mask: np.ndarray) -> float:
    fraction = float(mask.mean())
    return _clamp01(fraction * 4.2)


def _rgb_similarity(left: RGB, right: RGB) -> float:
    distance = sum((left[index] - right[index]) ** 2 for index in range(3)) ** 0.5
    return _clamp01(1.0 - distance / 441.7)


def _resolve_torch_device(torch_module: object, device: str) -> str:
    normalized = device.strip().lower()
    cuda_available = bool(torch_module.cuda.is_available())
    if normalized == "auto":
        return "cuda" if cuda_available else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise RuntimeError("caption_device='cuda' requested, but torch reports CUDA is unavailable.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported caption device: {device}")
    return normalized


def _set_recall(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    return _clamp01(len(expected & actual) / len(expected))


def _set_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return _clamp01(len(left & right) / len(left | right))


def _join_phrases(phrases: list[str]) -> str:
    if len(phrases) <= 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
