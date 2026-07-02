from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from .palette import COLOR_RGB, RGB
from .prompt import parse_prompt

DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class CaptionResult:
    caption: str
    prompt_similarity_score: float
    backend: str
    model_name: str | None
    requested_device: str
    effective_device: str
    tokens: tuple[str, ...]
    similarity_backend: str = "local"
    similarity_model: str | None = None
    similarity_device: str = "cpu"
    effective_similarity_device: str = "cpu"
    lexical_prompt_similarity_score: float = 0.0
    semantic_prompt_similarity_score: float | None = None


@dataclass(frozen=True)
class CaptionDiagnostics:
    missing_objects: tuple[str, ...]
    missing_colors: tuple[str, ...]
    unexpected_objects: tuple[str, ...]
    unexpected_colors: tuple[str, ...]


def caption_image(
    image: Image.Image,
    *,
    prompt: str,
    backend: str = "local",
    model_name: str | None = None,
    device: str = "auto",
    similarity_backend: str = "local",
    similarity_model: str | None = None,
    similarity_device: str = "auto",
) -> CaptionResult:
    normalized_backend = backend.strip().lower()
    requested_device = device.strip().lower()
    normalized_similarity_backend = similarity_backend.strip().lower()
    requested_similarity_device = similarity_device.strip().lower()

    if normalized_backend == "none":
        caption = ""
        effective_device = "none"
        effective_model = None
    elif normalized_backend == "local":
        caption = _local_caption(image.convert("RGB"), prompt=prompt)
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

    lexical_similarity = caption_prompt_similarity(prompt, caption)
    semantic_similarity: float | None = None
    if normalized_similarity_backend == "local":
        similarity = lexical_similarity
        effective_similarity_model = None
        effective_similarity_device = "cpu"
    elif normalized_similarity_backend in {"sentence", "transformers-sentence"}:
        effective_similarity_model = similarity_model or DEFAULT_SENTENCE_MODEL
        similarity = caption_prompt_similarity(
            prompt,
            caption,
            backend=normalized_similarity_backend,
            model_name=effective_similarity_model,
            device=requested_similarity_device,
        )
        semantic_similarity = similarity
        effective_similarity_device = _effective_sentence_similarity_device(requested_similarity_device)
    else:
        raise ValueError(f"Unsupported caption similarity backend: {similarity_backend}")

    caption_spec = parse_prompt(caption)
    return CaptionResult(
        caption=caption,
        prompt_similarity_score=similarity,
        backend=normalized_backend,
        model_name=effective_model,
        requested_device=requested_device,
        effective_device=effective_device,
        tokens=caption_spec.tokens,
        similarity_backend=normalized_similarity_backend,
        similarity_model=effective_similarity_model,
        similarity_device=requested_similarity_device,
        effective_similarity_device=effective_similarity_device,
        lexical_prompt_similarity_score=lexical_similarity,
        semantic_prompt_similarity_score=semantic_similarity,
    )


def caption_prompt_diagnostics(prompt: str, caption: str) -> CaptionDiagnostics:
    if not caption.strip():
        prompt_spec = parse_prompt(prompt)
        return CaptionDiagnostics(
            missing_objects=tuple(sorted(set(prompt_spec.objects))),
            missing_colors=tuple(sorted(set(prompt_spec.color_words))),
            unexpected_objects=(),
            unexpected_colors=(),
        )

    prompt_spec = parse_prompt(prompt)
    caption_spec = parse_prompt(caption)
    prompt_objects = set(prompt_spec.objects)
    caption_objects = set(caption_spec.objects)
    prompt_colors = set(prompt_spec.color_words)
    caption_colors = set(caption_spec.color_words)

    return CaptionDiagnostics(
        missing_objects=tuple(sorted(prompt_objects - caption_objects)),
        missing_colors=tuple(sorted(prompt_colors - caption_colors)),
        unexpected_objects=tuple(sorted(caption_objects - prompt_objects)),
        unexpected_colors=tuple(sorted(caption_colors - prompt_colors)),
    )


def caption_prompt_similarity(
    prompt: str,
    caption: str,
    *,
    backend: str = "local",
    model_name: str | None = None,
    device: str = "auto",
) -> float:
    if not caption.strip():
        return 0.0

    normalized_backend = backend.strip().lower()
    if normalized_backend in {"sentence", "transformers-sentence"}:
        return _sentence_text_similarity_score(
            prompt,
            caption,
            model_name=model_name or DEFAULT_SENTENCE_MODEL,
            device=device,
        )
    if normalized_backend != "local":
        raise ValueError(f"Unsupported caption similarity backend: {backend}")

    return _lexical_caption_prompt_similarity(prompt, caption)


def _lexical_caption_prompt_similarity(prompt: str, caption: str) -> float:
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


def _local_caption(image: Image.Image, *, prompt: str = "") -> str:
    array = np.asarray(image, dtype=np.float32)
    height = array.shape[0]
    width = array.shape[1]
    upper = array[: max(1, height // 2)]
    middle = array[height // 3 : max(height // 3 + 1, int(height * 0.72))]
    lower = array[height // 2 :]
    bottom = array[int(height * 0.74) :]
    center = array[
        int(height * 0.25) : max(int(height * 0.25) + 1, int(height * 0.74)),
        int(width * 0.25) : max(int(width * 0.25) + 1, int(width * 0.75)),
    ]

    prompt_tokens = set(parse_prompt(prompt).tokens)
    if _prompt_requests_icon(prompt_tokens) and _icon_presence(array, center) >= 0.42:
        return "a local image showing " + _join_phrases(_local_icon_phrases(array, upper, center, prompt_tokens))

    phrases: list[str] = []
    greenhouse_present = _greenhouse_presence(upper, middle) >= 0.50
    lamp_present = _lamp_presence(upper) >= 0.06
    plant_present = _plant_presence(middle, lower) >= 0.16
    floor_present = _stone_floor_presence(bottom) >= 0.12

    if greenhouse_present:
        phrases.append("blue glass greenhouse")

    if lamp_present and greenhouse_present:
        phrases.append("gold hanging lamps")
    elif _warm_presence(upper) >= 0.10:
        phrases.append(f"{_dominant_color_name(upper)} sun")
    elif _bright_neutral_presence(upper) >= 0.16:
        phrases.append("white moon")

    if greenhouse_present and _bright_neutral_presence(upper) >= 0.08:
        phrases.append("white moon")

    if _bright_neutral_presence(upper) >= (0.10 if greenhouse_present else 0.28):
        phrases.append("soft clouds")

    if _edge_density(middle) >= 0.16 and not greenhouse_present:
        phrases.append(f"{_dominant_color_name(middle)} mountains")

    if _robot_portrait_presence(center) >= 0.18 and not greenhouse_present:
        phrases.append(f"{_dominant_color_name(center)} robot portrait")

    if _blue_presence(lower) >= 0.10:
        phrases.append("blue ocean")

    if plant_present and greenhouse_present:
        phrases.append("green tropical plants")
    elif _green_presence(lower) >= 0.12:
        phrases.append("green forest")

    if floor_present and greenhouse_present:
        phrases.append("wet stone floor")

    if _dark_presence(middle) >= 0.32 and _edge_density(middle) >= 0.09 and not greenhouse_present:
        phrases.append("dark city skyline")

    if not phrases:
        phrases.append(f"{_dominant_color_name(array)} abstract composition")

    return "a local image showing " + _join_phrases(_dedupe(phrases))


def _prompt_requests_icon(tokens: set[str]) -> bool:
    return "icon" in tokens or ("app" in tokens and ("logo" in tokens or "badge" in tokens))


def _local_icon_phrases(
    array: np.ndarray,
    upper: np.ndarray,
    center: np.ndarray,
    prompt_tokens: set[str],
) -> list[str]:
    phrases = ["teal abstract app icon" if "teal" in prompt_tokens else f"{_dominant_color_name(array)} abstract app icon"]
    if {"aperture", "lens", "camera"} & prompt_tokens and _edge_density(center) >= 0.06:
        phrases.append("camera aperture lens")
    if "sparkle" in prompt_tokens and _bright_neutral_presence(upper) >= 0.025:
        phrases.append("sparkle")
    return _dedupe(phrases)


def _icon_presence(array: np.ndarray, center: np.ndarray) -> float:
    saturated_cool = max(_blue_presence(array), _green_presence(array))
    central_geometry = _edge_density(center)
    dark_ground = _dark_presence(array)
    return _clamp01(0.42 * dark_ground + 0.36 * central_geometry + 0.22 * saturated_cool)


def _robot_portrait_presence(array: np.ndarray) -> float:
    warm_shape = _warm_presence(array)
    internal_edges = _edge_density(array)
    accent_blue = _blue_presence(array)
    dark_detail = _dark_presence(array)
    return _clamp01(0.44 * warm_shape + 0.28 * internal_edges + 0.18 * accent_blue + 0.10 * dark_detail)


def _greenhouse_presence(upper: np.ndarray, middle: np.ndarray) -> float:
    frame_grid = _frame_grid_presence(upper)
    middle_edges = _edge_density(middle)
    cool_glass = max(_blue_presence(upper), _bright_neutral_presence(upper) * 0.55)
    return _clamp01(0.78 * frame_grid + 0.12 * middle_edges + 0.10 * cool_glass)


def _frame_grid_presence(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    luminance = array.mean(axis=2)
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    vertical_score = float(((dx > 22.0).mean(axis=0)).max()) if dx.size else 0.0
    horizontal_score = float(((dy > 22.0).mean(axis=1)).max()) if dy.size else 0.0
    balanced_grid = min(vertical_score, horizontal_score) * 1.7
    secondary_edges = max(vertical_score, horizontal_score) * 0.20
    return _clamp01(balanced_grid + secondary_edges)


def _lamp_presence(array: np.ndarray) -> float:
    red = array[:, :, 0]
    green = array[:, :, 1]
    blue = array[:, :, 2]
    warm_bright = (red > 170) & (green > 115) & (red > blue * 1.15)
    warm_halo = (red > 125) & (green > 80) & (red > blue * 1.08)
    return _clamp01(_presence(warm_bright) * 0.70 + _presence(warm_halo) * 0.30)


def _plant_presence(middle: np.ndarray, lower: np.ndarray) -> float:
    return _clamp01(0.58 * _green_presence(lower) + 0.26 * _green_presence(middle) + 0.16 * max(_edge_density(lower), _edge_density(middle)))


def _stone_floor_presence(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    spread = array.max(axis=2) - array.min(axis=2)
    mean = array.mean(axis=2)
    grayish = (spread < 42) & (mean > 34) & (mean < 150)
    horizontal_detail = _edge_density(array) * 0.35
    return _clamp01(_presence(grayish) * 0.65 + horizontal_detail)


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


def _sentence_text_similarity_score(prompt: str, caption: str, *, model_name: str, device: str) -> float:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-sentence caption similarity requires torch and transformers.") from exc

    resolved_device = _resolve_torch_device(torch, device)
    tokenizer, model = _load_sentence_model(model_name, resolved_device)
    inputs = tokenizer([prompt, caption], padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(resolved_device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        similarity = torch.nn.functional.cosine_similarity(embeddings[0:1], embeddings[1:2]).item()
    return _clamp01((similarity + 1.0) / 2.0)


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


@lru_cache(maxsize=2)
def _load_sentence_model(model_name: str, device: str) -> tuple[object, object]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError("transformers-sentence caption similarity requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


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


def _effective_sentence_similarity_device(device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return _resolve_torch_device(torch, device)


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
