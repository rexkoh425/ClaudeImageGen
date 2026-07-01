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
SCENE_PLAN_COUNT_FIELDS = (
    "scene_plan_background_stop_count",
    "scene_plan_element_count",
    "scene_plan_gradient_count",
    "scene_plan_motif_count",
    "scene_plan_texture_count",
    "scene_plan_material_count",
    "scene_plan_terrain_count",
    "scene_plan_reflection_count",
    "scene_plan_warp_count",
    "scene_plan_veil_count",
    "scene_plan_light_count",
    "scene_plan_beam_count",
    "scene_plan_cloud_count",
    "scene_plan_shadow_count",
)


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
    strong_sizes: tuple[tuple[int, int], ...] | None = None
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

    complex_width, complex_height = options.sizes[-1]
    complex_result = _run_complex_plan_case(
        options,
        output_dir,
        width=complex_width,
        height=complex_height,
    )
    cases.append(_case_report("complex-plan", complex_result, requested_size=cap_dimensions(complex_width, complex_height)))

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
        "strong_sizes": [f"{width}x{height}" for width, height in _strong_verification_sizes(options)] if options.strong_model else None,
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
    for index, (raw_width, raw_height) in enumerate(_strong_verification_sizes(options), start=1):
        width, height = cap_dimensions(raw_width, raw_height)
        strong_dir = output_dir / f"strong-model-{width}x{height}"
        try:
            result = generate_image(
                GenerateOptions(
                    prompt=options.prompt,
                    output_dir=strong_dir,
                    width=raw_width,
                    height=raw_height,
                    max_iterations=1,
                    threshold=0.1,
                    seed=911 + index,
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

        cases.append(_case_report("strong-model", result, requested_size=(width, height)))
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
            cases.append(_case_report("strong-continuity", continuity_result, requested_size=(width, height)))
    return "pass"


def _strong_verification_sizes(options: VerifyOptions) -> tuple[tuple[int, int], ...]:
    return options.strong_sizes or (options.sizes[0],)


def _run_complex_plan_case(options: VerifyOptions, output_dir: Path, *, width: int, height: int) -> GenerateResult:
    capped_width, capped_height = cap_dimensions(width, height)
    case_dir = output_dir / f"complex-plan-{capped_width}x{capped_height}"
    case_dir.mkdir(parents=True, exist_ok=True)
    scene_plan_path = case_dir / "scene-plan.json"
    scene_plan_path.write_text(json.dumps(_complex_scene_plan(), indent=2), encoding="utf-8")
    return generate_image(
        GenerateOptions(
            prompt=options.prompt,
            output_dir=case_dir,
            scene_plan=scene_plan_path,
            width=width,
            height=height,
            max_iterations=1,
            threshold=0.1,
            seed=707,
            save_candidates=options.save_candidates,
            similarity_backend="local",
            similarity_device="cpu",
            caption_backend="local",
            caption_device="cpu",
            caption_similarity_backend="local",
            caption_similarity_device="cpu",
        )
    )


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
    critique_request_path = result.metadata_path.parent / "critique-request.json"
    files_ok = all(
        path.exists()
        for path in (
            result.image_path,
            result.metadata_path,
            result.progress_path,
            result.metadata_path.parent / "quality-report.json",
            critique_request_path,
        )
    )
    size_ok = (width, height) == requested_size
    candidates_ok = metadata.get("candidate_count", 0) == 0 or bool(metadata.get("candidate_index"))
    refine_ok = case_type != "refine" or metadata.get("parent_candidate_selection") == "auto"
    complex_ok = case_type != "complex-plan" or _complex_scene_plan_ok(metadata)
    status = "pass" if files_ok and size_ok and candidates_ok and refine_ok and complex_ok else "fail"
    return {
        "type": case_type,
        "status": status,
        "size": f"{requested_size[0]}x{requested_size[1]}",
        "output_dir": str(result.metadata_path.parent),
        "image": str(result.image_path),
        "metadata": str(result.metadata_path),
        "quality_report": str(metadata.get("quality_report")),
        "critique_request": str(metadata.get("critique_request") or critique_request_path),
        "comparison_request": metadata.get("comparison_request"),
        "quality_status": metadata.get("quality_status"),
        "quality_score": metadata.get("quality_score"),
        "total_score": metadata.get("total_score"),
        "caption_backend": metadata.get("caption_backend"),
        "caption_model": metadata.get("caption_model"),
        "caption_similarity_score": metadata.get("caption_similarity_score"),
        "caption_similarity_backend": metadata.get("caption_similarity_backend"),
        "caption_similarity_model": metadata.get("caption_similarity_model"),
        "effective_caption_similarity_device": metadata.get("effective_caption_similarity_device"),
        "initial_similarity_score": metadata.get("initial_similarity_score"),
        "refinement_delta": metadata.get("refinement_delta"),
        "similarity_backend": metadata.get("similarity_backend"),
        "similarity_model": metadata.get("similarity_model"),
        "continuity_backend": metadata.get("continuity_backend"),
        "continuity_model": metadata.get("continuity_model"),
        "effective_similarity_device": metadata.get("effective_similarity_device"),
        "effective_continuity_device": metadata.get("effective_continuity_device"),
        "effective_caption_device": metadata.get("effective_caption_device"),
        "parent_candidate_selection": metadata.get("parent_candidate_selection"),
        "scene_plan_used": metadata.get("scene_plan_used"),
        "scene_plan_title": metadata.get("scene_plan_title"),
        "scene_plan_feature_count": _scene_plan_feature_count(metadata),
        **{field: metadata.get(field) for field in SCENE_PLAN_COUNT_FIELDS},
        "scene_plan_atmosphere_used": metadata.get("scene_plan_atmosphere_used"),
        "scene_plan_focus_used": metadata.get("scene_plan_focus_used"),
    }


def _metadata_size(metadata: dict[str, object]) -> tuple[int, int]:
    return int(metadata.get("width", 0)), int(metadata.get("height", 0))


def _complex_scene_plan_ok(metadata: dict[str, object]) -> bool:
    return (
        bool(metadata.get("scene_plan_used"))
        and _scene_plan_feature_count(metadata) >= 12
        and _int_metadata(metadata, "scene_plan_material_count") >= 1
        and _int_metadata(metadata, "scene_plan_terrain_count") >= 1
        and _int_metadata(metadata, "scene_plan_reflection_count") >= 1
        and _int_metadata(metadata, "scene_plan_warp_count") >= 1
        and _int_metadata(metadata, "scene_plan_beam_count") >= 1
        and _int_metadata(metadata, "scene_plan_cloud_count") >= 1
        and _int_metadata(metadata, "scene_plan_shadow_count") >= 1
        and bool(metadata.get("scene_plan_focus_used"))
    )


def _scene_plan_feature_count(metadata: dict[str, object]) -> int:
    count = sum(_int_metadata(metadata, field) for field in SCENE_PLAN_COUNT_FIELDS)
    if metadata.get("scene_plan_atmosphere_used"):
        count += 1
    if metadata.get("scene_plan_focus_used"):
        count += 1
    return count


def _int_metadata(metadata: dict[str, object], field: str) -> int:
    try:
        return int(metadata.get(field, 0))
    except (TypeError, ValueError):
        return 0


def _complex_scene_plan() -> dict[str, object]:
    return {
        "title": "verification complex coastal robot scene",
        "palette": ["#102040", "#ff5533", "#286fc4", "#123d2a", "#fff1dd"],
        "background": {
            "top": "#102040",
            "bottom": "#205080",
            "direction": "vertical",
            "stops": [
                {"at": 0.0, "color": "#102040"},
                {"at": 0.42, "color": "#ffcf8a"},
                {"at": 1.0, "color": "#205080"},
            ],
        },
        "objects": [
            {"type": "sun", "label": "large warm focal sun", "x": 0.24, "y": 0.24, "size": 0.18, "color": "#ff5533"},
            {"type": "robot", "label": "central red robot portrait", "x": 0.50, "y": 0.48, "size": 0.22, "color": "#c83a3a"},
            {"type": "ocean", "label": "reflective lower ocean", "y": 0.58, "color": "#286fc4"},
            {"type": "foreground", "label": "dark grassy foreground", "y": 0.82, "color": "#123d2a"},
        ],
        "elements": [
            {"type": "glow", "label": "sun bloom", "x": 0.24, "y": 0.24, "width": 0.24, "height": 0.24, "fill": "#ffcf8a", "opacity": 0.42, "z": 1},
            {
                "type": "rectangle",
                "label": "deep water gradient",
                "x": 0.50,
                "y": 0.70,
                "width": 1.0,
                "height": 0.26,
                "gradient": {"type": "linear", "colors": ["#2e8ddb", "#0a3d72"], "direction": "vertical"},
                "opacity": 0.48,
                "blend": "multiply",
                "z": 5,
            },
            {"type": "polyline", "label": "water highlight", "points": [[0.12, 0.66], [0.40, 0.69], [0.86, 0.65]], "stroke": "#f6e2b5", "width": 0.01, "opacity": 0.72, "blur": 0.012, "blend": "screen", "z": 7},
        ],
        "motifs": [
            {"type": "starfield", "label": "small upper sky points", "count": 24, "region": [0.0, 0.02, 1.0, 0.28], "color": "#fff5cc", "size": 0.006, "opacity": 0.60, "seed": 12, "z": 8},
            {"type": "grass", "label": "foreground grass texture", "count": 80, "region": [0.0, 0.78, 1.0, 1.0], "color": "#1a5c36", "size": 0.045, "opacity": 0.65, "seed": 21, "z": 12},
        ],
        "textures": [
            {"type": "ripple", "label": "water surface bands", "count": 30, "region": [0.0, 0.56, 1.0, 0.80], "color": "#d8f3ff", "density": 0.6, "scale": 0.025, "opacity": 0.36, "blend": "screen", "seed": 31, "z": 9}
        ],
        "materials": [
            {"type": "water", "label": "reflective ocean material", "region": [0.0, 0.56, 1.0, 0.82], "colors": ["#8bdcff", "#0b3b71"], "intensity": 0.72, "scale": 0.035, "opacity": 0.58, "seed": 41, "z": 8}
        ],
        "terrains": [
            {"type": "mountain", "label": "faceted distant ridge", "points": [[0.02, 0.56], [0.22, 0.24], [0.42, 0.56], [0.64, 0.34], [0.96, 0.56]], "base": 0.78, "fill": "#405070", "shade": "#182030", "highlight": "#7890b0", "opacity": 0.76, "facets": True, "z": 4}
        ],
        "reflections": [
            {"type": "vertical", "label": "mirrored sky and ridge", "source": [0.0, 0.16, 1.0, 0.56], "target": [0.0, 0.56, 1.0, 0.80], "opacity": 0.36, "blur": 0.025, "fade": 0.62, "tint": "#2d88d8", "blend": "screen", "z": 8}
        ],
        "warps": [
            {"type": "wave", "label": "water reflection displacement", "region": [0.0, 0.56, 1.0, 0.82], "direction": "horizontal", "amplitude": 0.018, "wavelength": 0.38, "phase": 0.20, "seed": 43, "z": 10}
        ],
        "atmosphere": {"type": "horizon_fog", "label": "cool horizon haze", "color": "#d8e8f0", "horizon": 0.56, "height": 0.22, "strength": 0.32},
        "veils": [
            {"type": "mist", "label": "localized sea mist", "region": [0.04, 0.48, 0.96, 0.68], "color": "#d8e8f0", "opacity": 0.22, "blur": 0.026, "blend": "screen", "falloff": 0.18, "direction": "vertical", "z": 8}
        ],
        "lights": [
            {"type": "radial", "label": "warm focal light", "x": 0.24, "y": 0.24, "radius": 0.35, "color": "#ffcf8a", "intensity": 0.48, "z": 10}
        ],
        "beams": [
            {"type": "sunbeam", "label": "diagonal shafts through haze", "x": 0.24, "y": 0.24, "angle": 70.0, "length": 0.70, "spread": 24.0, "color": "#ffcf8a", "opacity": 0.22, "blur": 0.035, "blend": "screen", "count": 2, "seed": 44, "z": 9}
        ],
        "clouds": [
            {"type": "cumulus", "label": "upper cloud bank", "region": [0.05, 0.08, 0.95, 0.34], "color": "#fff5dd", "shadow": "#8aa0b8", "opacity": 0.38, "blur": 0.025, "count": 3, "lobes": 5, "scale": 0.13, "blend": "screen", "seed": 45, "z": 4}
        ],
        "shadows": [
            {"type": "ellipse", "label": "robot grounding shadow", "x": 0.50, "y": 0.76, "width": 0.42, "height": 0.10, "color": "#101820", "opacity": 0.34, "blur": 0.035, "blend": "multiply", "z": 9}
        ],
        "focus": {"type": "depth", "label": "focal composition band", "region": [0.04, 0.10, 0.84, 0.82], "blur": 0.018, "falloff": 0.12, "mode": "outside"},
        "style": {"grain": 0.08, "vignette": 0.12, "saturation": 0.32, "contrast": 0.22, "warmth": 0.18, "bloom": 0.22, "antialias": 1.0},
    }
