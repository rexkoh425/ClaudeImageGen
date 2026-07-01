from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from .generator import GenerateOptions, GenerateResult, generate_image
from .refine import RefineOptions, refine_image
from .render import cap_dimensions
from .score import DEFAULT_CLIP_MODEL, DEFAULT_DINOV2_MODEL, DEFAULT_SIGLIP_MODEL

DEFAULT_VERIFY_SIZES = ((320, 192), (768, 432), (1024, 640))
DEFAULT_VERIFY_PROMPT = "cinematic red robot portrait over blue ocean with clouds, reflections, and atmospheric light"
DEFAULT_REFINE_PROMPT = "cinematic red robot portrait over blue ocean with brighter clouds and stronger water reflections"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"


@dataclass(frozen=True)
class VerifyOptions:
    output_dir: Path
    sizes: tuple[tuple[int, int], ...] = DEFAULT_VERIFY_SIZES
    prompt: str = DEFAULT_VERIFY_PROMPT
    refine_prompt: str = DEFAULT_REFINE_PROMPT
    max_iterations: int = 3
    threshold: float = 0.99
    save_candidates: int = 2
    strong_model: bool = False
    strong_similarity_backend: str = "transformers-clip"
    strong_model_device: str = "auto"
    similarity_model: str | None = None
    strong_continuity_backend: str = "local"
    continuity_model: str | None = None
    caption_model: str | None = None
    caption_similarity_backend: str = "local"
    caption_similarity_model: str | None = None


def run_verification(options: VerifyOptions) -> dict[str, object]:
    if not options.sizes:
        raise ValueError("at least one verification size is required")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if options.save_candidates <= 0:
        raise ValueError("save_candidates must be positive so refinement can verify auto candidate selection")

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    generated_results: list[GenerateResult] = []
    for index, (width, height) in enumerate(options.sizes, start=1):
        case_dir = output_dir / f"generate-{index:02d}-{width}x{height}"
        result = generate_image(
            GenerateOptions(
                prompt=options.prompt,
                output_dir=case_dir,
                width=width,
                height=height,
                max_iterations=options.max_iterations,
                threshold=options.threshold,
                seed=index,
                save_candidates=options.save_candidates,
                similarity_backend="local",
                similarity_device="cpu",
                caption_backend="local",
                caption_device="cpu",
                caption_similarity_backend="local",
                caption_similarity_device="cpu",
            )
        )
        generated_results.append(result)
        cases.append(_case_report("generate", result, requested_size=cap_dimensions(width, height)))

    refine_source = generated_results[0]
    refine_dir = output_dir / "refine-auto-candidate"
    refine_result = refine_image(
        RefineOptions(
            from_dir=refine_source.metadata_path.parent,
            candidate_rank="auto",
            prompt=options.refine_prompt,
            output_dir=refine_dir,
            max_iterations=1,
            threshold=0.1,
            similarity_backend="local",
            similarity_device="cpu",
            caption_backend="local",
            caption_device="cpu",
            caption_similarity_backend="local",
            caption_similarity_device="cpu",
        )
    )
    cases.append(_case_report("refine", refine_result, requested_size=_metadata_size(refine_result.metadata)))

    strong_model_status = "not-requested"
    if options.strong_model:
        strong_model_status = _run_strong_model_case(options, output_dir, cases)

    report = {
        "status": "pass" if all(str(case.get("status")) == "pass" for case in cases) else "fail",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "prompt": options.prompt,
        "refine_prompt": options.refine_prompt,
        "sizes": [f"{width}x{height}" for width, height in options.sizes],
        "strong_model": strong_model_status,
        "strong_similarity_backend": options.strong_similarity_backend if options.strong_model else None,
        "strong_continuity_backend": options.strong_continuity_backend if options.strong_model else None,
        "caption_similarity_backend": options.caption_similarity_backend if options.strong_model else None,
        "cases": cases,
    }
    report_path = output_dir / "verification-report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("*", "x").replace(",", "x").strip()
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid size '{value}'. Use WIDTHxHEIGHT, for example 1024x640.")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid size '{value}'. Width and height must be integers.") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{value}'. Width and height must be positive.")
    return width, height


def _run_strong_model_case(options: VerifyOptions, output_dir: Path, cases: list[dict[str, object]]) -> str:
    width, height = options.sizes[0]
    strong_dir = output_dir / f"strong-model-{width}x{height}"
    try:
        result = generate_image(
            GenerateOptions(
                prompt=options.prompt,
                output_dir=strong_dir,
                width=width,
                height=height,
                max_iterations=1,
                threshold=0.1,
                seed=911,
                save_candidates=1,
                similarity_backend=options.strong_similarity_backend,
                similarity_model=options.similarity_model or _default_similarity_model(options.strong_similarity_backend),
                similarity_device=options.strong_model_device,
                caption_backend="transformers-blip",
                caption_model=options.caption_model or DEFAULT_BLIP_MODEL,
                caption_device=options.strong_model_device,
                caption_similarity_backend=options.caption_similarity_backend,
                caption_similarity_model=options.caption_similarity_model,
                caption_similarity_device=options.strong_model_device,
            )
        )
    except Exception as exc:  # pragma: no cover - depends on optional local model installs
        cases.append(
            {
                "type": "strong-model",
                "status": "fail",
                "size": f"{width}x{height}",
                "output_dir": str(strong_dir),
                "error": str(exc),
            }
        )
        return "fail"

    cases.append(_case_report("strong-model", result, requested_size=cap_dimensions(width, height)))
    if options.strong_continuity_backend.strip().lower() != "local":
        try:
            continuity_result = refine_image(
                RefineOptions(
                    from_dir=result.metadata_path.parent,
                    prompt=options.refine_prompt,
                    output_dir=output_dir / f"strong-continuity-{width}x{height}",
                    max_iterations=1,
                    threshold=0.1,
                    similarity_backend=options.strong_similarity_backend,
                    similarity_model=options.similarity_model or _default_similarity_model(options.strong_similarity_backend),
                    similarity_device=options.strong_model_device,
                    continuity_backend=options.strong_continuity_backend,
                    continuity_model=options.continuity_model or _default_continuity_model(options.strong_continuity_backend),
                    continuity_device=options.strong_model_device,
                    caption_backend="transformers-blip",
                    caption_model=options.caption_model or DEFAULT_BLIP_MODEL,
                    caption_device=options.strong_model_device,
                    caption_similarity_backend=options.caption_similarity_backend,
                    caption_similarity_model=options.caption_similarity_model,
                    caption_similarity_device=options.strong_model_device,
                )
            )
        except Exception as exc:  # pragma: no cover - depends on optional local model installs
            cases.append(
                {
                    "type": "strong-continuity",
                    "status": "fail",
                    "size": f"{width}x{height}",
                    "output_dir": str(output_dir / f"strong-continuity-{width}x{height}"),
                    "error": str(exc),
                }
            )
            return "fail"
        cases.append(_case_report("strong-continuity", continuity_result, requested_size=cap_dimensions(width, height)))
    return "pass"


def _default_similarity_model(similarity_backend: str) -> str:
    normalized = similarity_backend.strip().lower()
    if normalized in {"siglip", "transformers-siglip"}:
        return DEFAULT_SIGLIP_MODEL
    return DEFAULT_CLIP_MODEL


def _default_continuity_model(continuity_backend: str) -> str | None:
    normalized = continuity_backend.strip().lower()
    if normalized in {"dinov2", "transformers-dinov2"}:
        return DEFAULT_DINOV2_MODEL
    if normalized in {"siglip", "transformers-siglip"}:
        return DEFAULT_SIGLIP_MODEL
    if normalized in {"clip", "transformers-clip"}:
        return DEFAULT_CLIP_MODEL
    return None


def _case_report(case_type: str, result: GenerateResult, *, requested_size: tuple[int, int]) -> dict[str, object]:
    metadata = result.metadata
    width, height = _metadata_size(metadata)
    files_ok = all(
        path.exists()
        for path in (
            result.image_path,
            result.metadata_path,
            result.progress_path,
            result.metadata_path.parent / "quality-report.json",
        )
    )
    size_ok = (width, height) == requested_size
    candidates_ok = metadata.get("candidate_count", 0) == 0 or bool(metadata.get("candidate_index"))
    refine_ok = case_type != "refine" or metadata.get("parent_candidate_selection") == "auto"
    status = "pass" if files_ok and size_ok and candidates_ok and refine_ok else "fail"
    return {
        "type": case_type,
        "status": status,
        "size": f"{requested_size[0]}x{requested_size[1]}",
        "output_dir": str(result.metadata_path.parent),
        "image": str(result.image_path),
        "metadata": str(result.metadata_path),
        "quality_report": str(metadata.get("quality_report")),
        "quality_status": metadata.get("quality_status"),
        "quality_score": metadata.get("quality_score"),
        "total_score": metadata.get("total_score"),
        "caption_similarity_score": metadata.get("caption_similarity_score"),
        "caption_similarity_backend": metadata.get("caption_similarity_backend"),
        "caption_similarity_model": metadata.get("caption_similarity_model"),
        "effective_caption_similarity_device": metadata.get("effective_caption_similarity_device"),
        "initial_similarity_score": metadata.get("initial_similarity_score"),
        "similarity_backend": metadata.get("similarity_backend"),
        "similarity_model": metadata.get("similarity_model"),
        "continuity_backend": metadata.get("continuity_backend"),
        "continuity_model": metadata.get("continuity_model"),
        "effective_similarity_device": metadata.get("effective_similarity_device"),
        "effective_continuity_device": metadata.get("effective_continuity_device"),
        "effective_caption_device": metadata.get("effective_caption_device"),
        "parent_candidate_selection": metadata.get("parent_candidate_selection"),
    }


def _metadata_size(metadata: dict[str, object]) -> tuple[int, int]:
    return int(metadata.get("width", 0)), int(metadata.get("height", 0))
