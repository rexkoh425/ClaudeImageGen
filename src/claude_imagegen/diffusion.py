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
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, CGI, vector art, low detail, blurry, flat lighting, "
    "deformed architecture, people, text, watermark"
)


@dataclass(frozen=True)
class DiffusionOptions:
    prompt: str
    output_dir: Path
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    model: str = DEFAULT_DIFFUSION_MODEL
    width: int = 1024
    height: int = 768
    steps: int = 4
    guidance_scale: float = 0.0
    seeds: tuple[int, ...] = (101, 202, 303, 404)
    device: str = "auto"
    quality_target: float | None = None


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
    if options.steps <= 0:
        raise ValueError("steps must be positive")
    if not options.seeds:
        raise ValueError("at least one seed is required")

    width, height = _diffusion_dimensions(options.width, options.height)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    pipeline, effective_device = _load_pipeline(model=options.model, device=options.device)

    entries: list[dict[str, object]] = []
    images: list[Image.Image] = []
    for seed in options.seeds:
        image = _run_pipeline(
            pipeline,
            prompt=options.prompt,
            negative_prompt=options.negative_prompt,
            width=width,
            height=height,
            steps=options.steps,
            guidance_scale=options.guidance_scale,
            seed=seed,
            device=effective_device,
        )
        image_path = candidates_dir / f"diffusion-seed-{seed}.png"
        image.save(image_path)
        entry = _candidate_entry(image, image_path=image_path, seed=seed)
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

    prompt_token_estimate = _prompt_token_estimate(options.prompt)
    metadata: dict[str, object] = {
        "engine": "diffusers-text-to-image-v1",
        "backend": "diffusers",
        "model": options.model,
        "prompt": options.prompt,
        "normalized_prompt": options.prompt.strip(),
        "prompt_token_estimate": prompt_token_estimate,
        "prompt_length_warning": _prompt_length_warning(prompt_token_estimate),
        "negative_prompt": options.negative_prompt,
        "width": width,
        "height": height,
        "steps": options.steps,
        "guidance_scale": options.guidance_scale,
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
        "total_score": selected["selection_score"],
        "threshold": 0.58,
        "quality_target": options.quality_target,
        "image_detail_score": selected["image_detail_score"],
        "image_detail_metrics": selected["image_detail_metrics"],
        "caption_similarity_score": 0.0,
        "image_caption": "",
        "visual_critique_required": True,
        "revision_hints": [
            "Ask Claude to inspect image.png and fill critique-request.json before accepting high quality targets.",
            "Use the contact sheet to pick the strongest seed before another diffusion run.",
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


def _candidate_entry(image: Image.Image, *, image_path: Path, seed: int) -> dict[str, object]:
    detail_metrics = image_detail_metrics(image)
    aesthetic_score, aesthetic_details = compute_candidate_aesthetic_score(image)
    selection_score = round(
        max(
            0.0,
            min(
                1.0,
                (0.58 * float(detail_metrics["detail_score"])) + (0.42 * aesthetic_score),
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
            f"image_detail_score={detail_metrics['detail_score']:.3f} weight=0.58",
            f"aesthetic_score={aesthetic_score:.3f} weight=0.42",
        ],
        "image_detail_score": detail_metrics["detail_score"],
        "image_detail_metrics": detail_metrics,
        "aesthetic_score": aesthetic_score,
        "aesthetic_details": aesthetic_details,
    }


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
