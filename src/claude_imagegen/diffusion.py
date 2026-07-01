from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .candidates import compute_candidate_aesthetic_score
from .critique import write_critique_request
from .quality import apply_quality_report, image_detail_metrics
from .render import cap_dimensions

DEFAULT_DIFFUSION_MODEL = "stabilityai/sdxl-turbo"
PHOTOREAL_DIFFUSION_MODEL = "SG161222/RealVisXL_V5.0"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, CGI, vector art, low detail, blurry, flat lighting, "
    "deformed architecture, people, text, watermark, extra furniture, chairs, tables, clutter, "
    "washed-out blacks, muddy details"
)
PHOTOREAL_PROMPT_PREFIX = (
    "photorealistic high-detail DSLR image, crisp micro texture, natural materials, "
    "physically plausible lighting"
)
NIGHT_PHOTOREAL_PROMPT_PREFIX = (
    "photorealistic high-detail DSLR image, deep night exposure, crisp micro texture, "
    "warm practical lights, controlled blacks, physically plausible volumetric atmosphere"
)

DIFFUSION_PROFILES: dict[str, dict[str, object]] = {
    "turbo": {
        "model": DEFAULT_DIFFUSION_MODEL,
        "steps": 4,
        "guidance_scale": 0.0,
        "prompt_prefix": "",
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "focus_terms": (),
    },
    "photoreal": {
        "model": PHOTOREAL_DIFFUSION_MODEL,
        "steps": 24,
        "guidance_scale": 6.5,
        "prompt_prefix": PHOTOREAL_PROMPT_PREFIX,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "focus_terms": ("photoreal", "detail"),
    },
    "night-photoreal": {
        "model": PHOTOREAL_DIFFUSION_MODEL,
        "steps": 28,
        "guidance_scale": 7.0,
        "prompt_prefix": NIGHT_PHOTOREAL_PROMPT_PREFIX,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "focus_terms": ("night", "lamp", "mist", "reflection", "detail"),
    },
}
DIFFUSION_PROFILE_NAMES = tuple(DIFFUSION_PROFILES)


@dataclass(frozen=True)
class DiffusionOptions:
    prompt: str
    output_dir: Path
    negative_prompt: str | None = None
    model: str | None = None
    profile: str = "turbo"
    width: int = 1024
    height: int = 768
    steps: int | None = None
    guidance_scale: float | None = None
    seeds: tuple[int, ...] = (101, 202, 303, 404)
    device: str = "auto"
    quality_target: float | None = None
    prompt_focus: tuple[str, ...] = ("auto",)


@dataclass(frozen=True)
class DiffusionResult:
    image: Image.Image
    metadata: dict[str, object]
    image_path: Path
    metadata_path: Path
    quality_report_path: Path
    critique_request_path: Path
    candidates_path: Path
    contact_sheet_path: Path


class DiffusionDependencyError(RuntimeError):
    """Raised when optional local diffusion dependencies are not installed."""


def generate_diffusion_image(options: DiffusionOptions) -> DiffusionResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    config = _resolve_diffusion_config(options)
    steps = int(config["steps"])
    guidance_scale = float(config["guidance_scale"])
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not options.seeds:
        raise ValueError("at least one seed is required")

    width, height = _diffusion_dimensions(options.width, options.height)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    pipeline, effective_device = _load_pipeline(model=str(config["model"]), device=options.device)

    entries: list[dict[str, object]] = []
    images: list[Image.Image] = []
    for seed in options.seeds:
        image = _run_pipeline(
            pipeline,
            prompt=str(config["prompt"]),
            negative_prompt=str(config["negative_prompt"]),
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            device=effective_device,
        )
        image_path = candidates_dir / f"diffusion-seed-{seed}.png"
        image.save(image_path)
        entry = _candidate_entry(
            image,
            image_path=image_path,
            seed=seed,
            prompt_focus_terms=tuple(config["prompt_focus_terms"]),
        )
        entries.append(entry)
        images.append(image)

    entries.sort(key=lambda item: float(item["selection_score"]), reverse=True)
    selected = entries[0]
    selected_seed = int(selected["seed"])
    selected_image = images[list(options.seeds).index(selected_seed)]

    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"
    candidates_path = output_dir / "candidates.json"
    contact_sheet_path = _write_contact_sheet(candidates_dir, entries)
    selected_image.save(image_path)
    candidates_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    prompt_token_estimate = _prompt_token_estimate(str(config["prompt"]))
    metadata: dict[str, object] = {
        "engine": "diffusers-text-to-image-v1",
        "backend": "diffusers",
        "diffusion_profile": config["profile"],
        "model": config["model"],
        "prompt": options.prompt,
        "normalized_prompt": config["prompt"],
        "prompt_token_estimate": prompt_token_estimate,
        "prompt_length_warning": _prompt_length_warning(prompt_token_estimate),
        "negative_prompt": config["negative_prompt"],
        "prompt_focus_terms": list(config["prompt_focus_terms"]),
        "selection_strategy": "prompt-aware-detail-aesthetic-v1",
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "device": options.device,
        "effective_device": effective_device,
        "seeds": list(options.seeds),
        "selected_seed": selected_seed,
        "candidate_count": len(entries),
        "candidate_index": str(candidates_path),
        "candidate_contact_sheet": str(contact_sheet_path),
        "candidate_images": [entry["image"] for entry in entries],
        "recommended_candidate_rank": selected["rank"],
        "recommended_candidate_image": selected["image"],
        "recommended_candidate_score": selected["selection_score"],
        "recommended_candidate_aesthetic_score": selected["aesthetic_score"],
        "recommended_candidate_aesthetic_details": selected["aesthetic_details"],
        "recommended_candidate_prompt_signal_score": selected["prompt_signal_score"],
        "recommended_candidate_prompt_signal_details": selected["prompt_signal_details"],
        "total_score": selected["selection_score"],
        "threshold": 0.58,
        "quality_target": options.quality_target,
        "image_detail_score": selected["image_detail_score"],
        "image_detail_metrics": selected["image_detail_metrics"],
        "prompt_signal_score": selected["prompt_signal_score"],
        "prompt_signal_details": selected["prompt_signal_details"],
        "caption_similarity_score": 0.0,
        "image_caption": "",
        "visual_critique_required": True,
        "revision_hints": [
            "Ask Claude to inspect image.png and fill critique-request.json before accepting high quality targets.",
            "Use the contact sheet to pick the strongest seed before another diffusion run.",
            "For GPT/Sora parity requests, compare before/after pairs with pair-eval and do not accept below 0.9.",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    quality_report_path = apply_quality_report(output_dir, metadata)
    critique_request_path = write_critique_request(
        output_dir,
        image_path=image_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return DiffusionResult(
        image=selected_image,
        metadata=metadata,
        image_path=image_path,
        metadata_path=metadata_path,
        quality_report_path=quality_report_path,
        critique_request_path=critique_request_path,
        candidates_path=candidates_path,
        contact_sheet_path=contact_sheet_path,
    )


def _candidate_entry(
    image: Image.Image,
    *,
    image_path: Path,
    seed: int,
    prompt_focus_terms: tuple[str, ...],
) -> dict[str, object]:
    detail_metrics = image_detail_metrics(image)
    aesthetic_score, aesthetic_details = compute_candidate_aesthetic_score(image)
    prompt_signal_score, prompt_signal_details = _prompt_signal_score(image, prompt_focus_terms=prompt_focus_terms)
    selection_score = round(
        max(
            0.0,
            min(
                1.0,
                (0.48 * float(detail_metrics["detail_score"]))
                + (0.28 * aesthetic_score)
                + (0.24 * prompt_signal_score),
            ),
        ),
        6,
    )
    return {
        "rank": 0,
        "seed": seed,
        "image": str(image_path),
        "selection_score": selection_score,
        "selection_reasons": [
            f"image_detail_score={detail_metrics['detail_score']:.3f} weight=0.48",
            f"aesthetic_score={aesthetic_score:.3f} weight=0.28",
            f"prompt_signal_score={prompt_signal_score:.3f} weight=0.24",
        ],
        "image_detail_score": detail_metrics["detail_score"],
        "image_detail_metrics": detail_metrics,
        "aesthetic_score": aesthetic_score,
        "aesthetic_details": aesthetic_details,
        "prompt_signal_score": prompt_signal_score,
        "prompt_signal_details": prompt_signal_details,
    }


def _resolve_diffusion_config(options: DiffusionOptions) -> dict[str, object]:
    profile_name = options.profile.strip().lower()
    if profile_name not in DIFFUSION_PROFILES:
        raise ValueError(f"unknown diffusion profile: {options.profile}")
    profile = DIFFUSION_PROFILES[profile_name]
    prompt = _apply_prompt_prefix(options.prompt.strip(), str(profile.get("prompt_prefix") or ""))
    focus_terms = _prompt_focus_terms(
        options.prompt,
        explicit=options.prompt_focus,
        profile_terms=tuple(str(term) for term in profile.get("focus_terms", ())),
    )
    return {
        "profile": profile_name,
        "model": options.model or str(profile["model"]),
        "prompt": prompt,
        "negative_prompt": options.negative_prompt or str(profile["negative_prompt"]),
        "steps": options.steps if options.steps is not None else int(profile["steps"]),
        "guidance_scale": options.guidance_scale
        if options.guidance_scale is not None
        else float(profile["guidance_scale"]),
        "prompt_focus_terms": focus_terms,
    }


def _apply_prompt_prefix(prompt: str, prefix: str) -> str:
    if not prefix:
        return prompt
    if prompt.lower().startswith(prefix.lower()):
        return prompt
    return f"{prefix}, {prompt}"


def _prompt_focus_terms(
    prompt: str,
    *,
    explicit: tuple[str, ...],
    profile_terms: tuple[str, ...],
) -> tuple[str, ...]:
    normalized_explicit = tuple(term.strip().lower() for term in explicit if term.strip())
    if normalized_explicit and normalized_explicit != ("auto",):
        return tuple(dict.fromkeys(normalized_explicit))

    lower_prompt = prompt.lower()
    discovered: list[str] = list(profile_terms)
    keyword_groups = (
        ("night", ("night", "midnight", "nocturne", "dark")),
        ("greenhouse", ("greenhouse", "glasshouse", "conservatory")),
        ("plant", ("plant", "plants", "tropical", "foliage", "leaves")),
        ("leaf detail", ("leaf vein", "leaf veins", "veined leaves", "sharp leaf")),
        ("lamp", ("lamp", "lamps", "lantern", "tungsten", "practical light")),
        ("mist", ("mist", "fog", "haze", "volumetric", "atmosphere")),
        ("ray", ("ray", "rays", "beam", "beams", "shaft", "shafts", "god ray")),
        ("reflection", ("reflection", "reflections", "mirror", "wet", "glossy")),
        ("floor", ("floor", "stone", "tile")),
        ("glass", ("glass", "mullion", "mullions", "window", "windows")),
        ("photoreal", ("photoreal", "photorealistic", "realistic", "dslr")),
        ("detail", ("detail", "detailed", "micro texture", "sharp")),
    )
    for term, needles in keyword_groups:
        if any(needle in lower_prompt for needle in needles):
            discovered.append(term)
    return tuple(dict.fromkeys(discovered))


def _prompt_signal_score(image: Image.Image, *, prompt_focus_terms: tuple[str, ...]) -> tuple[float, dict[str, object]]:
    if not prompt_focus_terms:
        return 0.5, {"terms": [], "term_scores": {}, "note": "neutral score because no prompt focus terms were provided"}

    import numpy as np

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    luma = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    horizontal_edges = np.abs(np.diff(luma, axis=1))
    vertical_edges = np.abs(np.diff(luma, axis=0))
    edge_density = float(
        (
            np.mean(horizontal_edges > 0.08)
            + np.mean(vertical_edges > 0.08)
        )
        / 2.0
    )
    mean_luma = float(np.mean(luma))
    highlight_ratio = float(np.mean(luma > 0.72))
    green_ratio = float(np.mean((green > red * 1.08) & (green > blue * 1.08) & (green > 0.22)))
    warm_highlight_ratio = float(np.mean((red > 0.65) & (green > 0.38) & (blue < 0.35) & (luma > 0.45)))
    bottom = luma[int(luma.shape[0] * 0.58) :, :]
    bottom_highlight_ratio = float(np.mean(bottom > 0.55)) if bottom.size else 0.0
    low_chroma_midtones = float(np.mean((chroma < 0.16) & (luma > 0.22) & (luma < 0.72)))

    term_scores: dict[str, float] = {}
    for term in prompt_focus_terms:
        term_scores[term] = _term_prompt_signal(
            term,
            mean_luma=mean_luma,
            highlight_ratio=highlight_ratio,
            green_ratio=green_ratio,
            warm_highlight_ratio=warm_highlight_ratio,
            bottom_highlight_ratio=bottom_highlight_ratio,
            low_chroma_midtones=low_chroma_midtones,
            edge_density=edge_density,
        )

    score = round(sum(term_scores.values()) / max(1, len(term_scores)), 6)
    return score, {
        "terms": list(prompt_focus_terms),
        "term_scores": term_scores,
        "mean_luma": round(mean_luma, 6),
        "highlight_ratio": round(highlight_ratio, 6),
        "green_ratio": round(green_ratio, 6),
        "warm_highlight_ratio": round(warm_highlight_ratio, 6),
        "bottom_highlight_ratio": round(bottom_highlight_ratio, 6),
        "low_chroma_midtones": round(low_chroma_midtones, 6),
        "edge_density": round(edge_density, 6),
    }


def _term_prompt_signal(
    term: str,
    *,
    mean_luma: float,
    highlight_ratio: float,
    green_ratio: float,
    warm_highlight_ratio: float,
    bottom_highlight_ratio: float,
    low_chroma_midtones: float,
    edge_density: float,
) -> float:
    normalized = term.strip().lower()
    if normalized in {"night", "dark", "deep night"}:
        return _clamp01(((1.0 - mean_luma) * 0.72) + min(1.0, highlight_ratio * 10.0) * 0.28)
    if normalized in {"plant", "greenhouse"}:
        return _clamp01((green_ratio * 8.0) + (edge_density * 0.45))
    if normalized in {"leaf detail", "detail", "photoreal"}:
        return _clamp01((edge_density * 3.2) + (highlight_ratio * 0.55))
    if normalized in {"lamp", "light", "tungsten"}:
        return _clamp01((warm_highlight_ratio * 70.0) + (highlight_ratio * 0.35))
    if normalized in {"mist", "fog", "haze", "volumetric"}:
        return _clamp01((low_chroma_midtones * 0.85) + (highlight_ratio * 2.5))
    if normalized in {"ray", "rays", "beam", "beams"}:
        return _clamp01((highlight_ratio * 12.0) + (edge_density * 0.7))
    if normalized in {"reflection", "wet", "mirror", "floor"}:
        return _clamp01((bottom_highlight_ratio * 8.0) + (edge_density * 0.35))
    if normalized in {"glass", "mullion", "mullions"}:
        return _clamp01((edge_density * 2.5) + (highlight_ratio * 1.5))
    return 0.5


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


def _prompt_token_estimate(prompt: str) -> int:
    return len([token for token in prompt.replace(",", " ").split() if token])


def _prompt_length_warning(token_estimate: int) -> str | None:
    if token_estimate <= 77:
        return None
    return (
        f"Prompt is approximately {token_estimate} tokens; SDXL Turbo/CLIP commonly uses a 77-token text window, "
        "so later details may be truncated. Shorten the prompt or put the most important details first."
    )


def _write_contact_sheet(candidates_dir: Path, entries: list[dict[str, object]]) -> Path:
    thumbs: list[tuple[dict[str, object], Image.Image]] = []
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
        with Image.open(str(entry["image"])) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((360, 240))
            thumbs.append((entry, thumbnail.copy()))

    columns = min(2, max(1, len(thumbs)))
    rows = (len(thumbs) + columns - 1) // columns
    padding = 12
    label_height = 38
    tile_width = max(thumb.width for _, thumb in thumbs)
    tile_height = max(thumb.height for _, thumb in thumbs) + label_height
    sheet = Image.new(
        "RGB",
        (
            (columns * tile_width) + ((columns + 1) * padding),
            (rows * tile_height) + ((rows + 1) * padding),
        ),
        (245, 245, 242),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (entry, thumb) in enumerate(thumbs):
        row = index // columns
        column = index % columns
        x = padding + column * (tile_width + padding)
        y = padding + row * (tile_height + padding)
        sheet.paste(thumb, (x, y))
        label = f"#{entry['rank']} seed {entry['seed']} sel {float(entry['selection_score']):.3f}"
        draw.text((x, y + thumb.height + 6), label, fill=(25, 28, 32))
        draw.text((x, y + thumb.height + 22), f"detail {float(entry['image_detail_score']):.3f}", fill=(55, 58, 64))

    path = candidates_dir / "contact-sheet.png"
    sheet.save(path)
    return path


def _run_pipeline(
    pipeline: object,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
) -> Image.Image:
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=_torch_generator(seed=seed, device=device),
        height=height,
        width=width,
    )
    images = getattr(result, "images", None)
    if not images:
        raise RuntimeError("diffusion pipeline did not return an image")
    return images[0].convert("RGB")


def _load_pipeline(*, model: str, device: str) -> tuple[object, str]:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise DiffusionDependencyError(
            "Optional diffusion dependencies are missing. Install them with: "
            "python -m pip install -e .[diffusion]"
        ) from exc

    effective_device = _effective_device(torch, requested=device)
    dtype = torch.float16 if effective_device == "cuda" else torch.float32
    kwargs: dict[str, object] = {"torch_dtype": dtype}
    if effective_device == "cuda":
        kwargs["variant"] = "fp16"
    pipeline = AutoPipelineForText2Image.from_pretrained(model, **kwargs)
    return pipeline.to(effective_device), effective_device


def _torch_generator(*, seed: int, device: str) -> object | None:
    try:
        import torch
    except ImportError:
        return None
    generator_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _effective_device(torch_module: object, *, requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        return normalized
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _diffusion_dimensions(width: int, height: int) -> tuple[int, int]:
    capped_width, capped_height = cap_dimensions(width, height)
    return max(64, capped_width - (capped_width % 8)), max(64, capped_height - (capped_height % 8))
