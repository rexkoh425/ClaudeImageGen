from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .candidates import annotate_candidate_selection, compute_candidate_aesthetic_score, select_recommended_candidate
from .caption import CaptionDiagnostics, caption_image, caption_prompt_diagnostics
from .critique import write_critique_request
from .palette import COLOR_RGB, RGB, extract_reference_palette
from .pixels import export_pixel_csv
from .prompt import parse_prompt
from .quality import apply_quality_report, image_detail_metrics
from .render import cap_dimensions, render_candidate, render_scene_plan
from .scene import SceneCandidate, build_initial_candidate, mutate_candidate
from .scene_plan import PlannedCloud, PlannedObject, PlannedTexture, ScenePlan, parse_scene_plan
from .score import ScoreResult, image_similarity_details, score_image


@dataclass(frozen=True)
class GenerateOptions:
    prompt: str
    output_dir: Path
    reference_image: Path | None = None
    initial_image: Path | None = None
    width: int = 720
    height: int = 480
    max_iterations: int = 32
    threshold: float = 0.58
    quality_target: float | None = None
    seed: int = 0
    pixel_csv: bool = False
    scene_plan: Path | None = None
    auto_refine: bool = True
    similarity_backend: str = "local"
    similarity_model: str | None = None
    similarity_device: str = "auto"
    continuity_backend: str | None = None
    continuity_model: str | None = None
    continuity_device: str | None = None
    caption_backend: str = "local"
    caption_model: str | None = None
    caption_device: str = "auto"
    caption_similarity_backend: str = "local"
    caption_similarity_model: str | None = None
    caption_similarity_device: str = "auto"
    save_candidates: int = 0


@dataclass(frozen=True)
class GenerateResult:
    image: Image.Image
    metadata: dict[str, object]
    image_path: Path
    metadata_path: Path
    progress_path: Path
    pixels_path: Path | None
    candidates_path: Path | None


@dataclass(frozen=True)
class CandidateSnapshot:
    iteration: int
    image: Image.Image
    score: ScoreResult
    met_threshold: bool


GRAPHIC_PROMPT_TOKENS = {
    "architecture",
    "arrow",
    "arrows",
    "badge",
    "badges",
    "box",
    "boxes",
    "cpu",
    "diagram",
    "flow",
    "flowchart",
    "gpu",
    "icon",
    "infographic",
    "label",
    "labels",
    "logo",
    "pipeline",
    "service",
    "services",
    "schematic",
    "tile",
    "tiles",
    "ui",
}
HARD_EDGE_ELEMENT_KINDS = {
    "aperture",
    "arc",
    "arrow",
    "circle",
    "ellipse",
    "label",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "rectangle",
    "rounded-rectangle",
    "rounded_rectangle",
    "roundrect",
    "sparkle",
    "text",
}


def generate_image(options: GenerateOptions) -> GenerateResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if options.save_candidates < 0:
        raise ValueError("save_candidates must not be negative")

    width, height = cap_dimensions(options.width, options.height)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_palette = _palette_from_optional_image(options.reference_image)
    initial_palette = _palette_from_optional_image(options.initial_image)
    spec = parse_prompt(options.prompt)
    scene_plan = parse_scene_plan(options.scene_plan) if options.scene_plan else None
    _validate_prompt_scene_plan_semantics(options, spec=spec, scene_plan=scene_plan)
    initial_blend_applied, initial_blend_reason = _initial_image_blend_decision(
        options.initial_image,
        spec=spec,
        scene_plan=scene_plan,
    )
    candidate = build_initial_candidate(
        spec,
        seed=options.seed,
        reference_palette=reference_palette,
        initial_palette=initial_palette,
    )

    best_image: Image.Image | None = None
    best_score: ScoreResult | None = None
    best_iteration = 0
    best_scene_plan: ScenePlan | None = scene_plan
    refinement_actions: list[str] = []
    refinement_rounds = 0
    progress: list[dict[str, object]] = []
    top_candidates: list[CandidateSnapshot] = []

    for iteration in range(1, options.max_iterations + 1):
        image = (
            render_scene_plan(scene_plan, width=width, height=height, seed=options.seed + iteration)
            if scene_plan
            else render_candidate(candidate, width=width, height=height)
        )
        image = _blend_initial_image(image, options.initial_image, blend=initial_blend_applied)
        score = score_image(
            image,
            spec,
            reference_image=options.reference_image,
            similarity_backend=options.similarity_backend,
            similarity_model=options.similarity_model,
            similarity_device=options.similarity_device,
        )
        met_threshold = score.total_score >= options.threshold

        progress.append(
            {
                "iteration": iteration,
                "total_score": f"{score.total_score:.6f}",
                "text_score": f"{score.text_score:.6f}",
                "reference_score": f"{score.reference_score:.6f}",
                "cosine_score": f"{score.details.get('cosine_score', 0.0):.6f}",
                "met_threshold": str(met_threshold).lower(),
            }
        )

        if best_score is None or score.total_score > best_score.total_score:
            best_image = image
            best_score = score
            best_iteration = iteration
            best_scene_plan = scene_plan

        if options.save_candidates:
            _remember_candidate(
                top_candidates,
                CandidateSnapshot(
                    iteration=iteration,
                    image=image.copy(),
                    score=score,
                    met_threshold=met_threshold,
                ),
                limit=options.save_candidates,
            )

        if met_threshold:
            break

        if scene_plan and options.auto_refine and iteration < options.max_iterations:
            refined_scene_plan, actions = _refine_scene_plan(scene_plan, spec=spec, score=score)
            if actions:
                refinement_rounds += 1
                refinement_actions.extend(f"iteration {iteration}: {action}" for action in actions)
                scene_plan = refined_scene_plan
        elif not scene_plan:
            candidate = mutate_candidate(candidate, iteration)

    assert best_image is not None
    assert best_score is not None

    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.csv"
    pixels_path = output_dir / "pixels.csv" if options.pixel_csv else None

    best_image.save(image_path)
    _write_progress(progress_path, progress)
    candidates_path, candidate_contact_sheet, candidate_entries = _write_candidate_artifacts(
        output_dir,
        top_candidates,
        prompt=options.prompt,
        caption_backend=options.caption_backend,
        caption_model=options.caption_model,
        caption_device=options.caption_device,
        caption_similarity_backend=options.caption_similarity_backend,
        caption_similarity_model=options.caption_similarity_model,
        caption_similarity_device=options.caption_similarity_device,
    )
    metadata_scene_plan = best_scene_plan or scene_plan
    caption_result = caption_image(
        best_image,
        prompt=options.prompt,
        backend=options.caption_backend,
        model_name=options.caption_model,
        device=options.caption_device,
        similarity_backend=options.caption_similarity_backend,
        similarity_model=options.caption_similarity_model,
        similarity_device=options.caption_similarity_device,
    )
    caption_diagnostics = (
        CaptionDiagnostics((), (), (), ())
        if caption_result.backend == "none"
        else caption_prompt_diagnostics(options.prompt, caption_result.caption)
    )
    recommended_candidate = select_recommended_candidate(candidate_entries) if candidate_entries else None
    continuity_backend = options.continuity_backend or options.similarity_backend
    continuity_model = options.continuity_model if options.continuity_backend else options.similarity_model
    continuity_device = options.continuity_device or options.similarity_device
    initial_similarity = (
        image_similarity_details(
            best_image,
            options.initial_image,
            similarity_backend=continuity_backend,
            similarity_model=continuity_model,
            similarity_device=continuity_device,
        )
        if options.initial_image
        else None
    )
    detail_metrics = image_detail_metrics(best_image)

    metadata: dict[str, object] = {
        "prompt": options.prompt,
        "normalized_prompt": spec.normalized,
        "width": width,
        "height": height,
        "iterations": best_iteration,
        "max_iterations": options.max_iterations,
        "threshold": options.threshold,
        "quality_target": options.quality_target,
        "met_threshold": best_score.total_score >= options.threshold,
        "total_score": round(best_score.total_score, 6),
        "text_score": round(best_score.text_score, 6),
        "reference_score": round(best_score.reference_score, 6),
        "initial_similarity_score": initial_similarity["continuity_score"] if initial_similarity else None,
        "initial_similarity_details": initial_similarity,
        "initial_image_blend_applied": initial_blend_applied,
        "initial_image_blend_reason": initial_blend_reason,
        "image_detail_score": detail_metrics["detail_score"],
        "image_detail_metrics": detail_metrics,
        "score_details": {key: round(value, 6) for key, value in best_score.details.items()},
        "auto_refine": options.auto_refine,
        "refinement_rounds": refinement_rounds,
        "refinement_actions": refinement_actions,
        "similarity_backend": options.similarity_backend,
        "similarity_model": options.similarity_model,
        "similarity_device": options.similarity_device,
        "effective_similarity_device": _effective_similarity_device(
            backend=options.similarity_backend,
            requested_device=options.similarity_device,
        ),
        "continuity_backend": continuity_backend,
        "continuity_model": continuity_model,
        "continuity_device": continuity_device,
        "effective_continuity_device": _effective_similarity_device(
            backend=continuity_backend,
            requested_device=continuity_device,
        ),
        "caption_backend": caption_result.backend,
        "caption_model": caption_result.model_name,
        "caption_device": caption_result.requested_device,
        "effective_caption_device": caption_result.effective_device,
        "caption_similarity_backend": caption_result.similarity_backend,
        "caption_similarity_model": caption_result.similarity_model,
        "caption_similarity_device": caption_result.similarity_device,
        "effective_caption_similarity_device": caption_result.effective_similarity_device,
        "image_caption": caption_result.caption,
        "caption_similarity_score": round(caption_result.prompt_similarity_score, 6),
        "lexical_caption_similarity_score": round(caption_result.lexical_prompt_similarity_score, 6),
        "semantic_caption_similarity_score": (
            round(caption_result.semantic_prompt_similarity_score, 6)
            if caption_result.semantic_prompt_similarity_score is not None
            else None
        ),
        "caption_tokens": list(caption_result.tokens),
        "caption_missing_objects": list(caption_diagnostics.missing_objects),
        "caption_missing_colors": list(caption_diagnostics.missing_colors),
        "caption_unexpected_objects": list(caption_diagnostics.unexpected_objects),
        "caption_unexpected_colors": list(caption_diagnostics.unexpected_colors),
        "candidate_count": len(candidate_entries),
        "candidate_index": str(candidates_path) if candidates_path else None,
        "candidate_contact_sheet": str(candidate_contact_sheet) if candidate_contact_sheet else None,
        "candidate_images": [entry["image"] for entry in candidate_entries],
        "recommended_candidate_rank": recommended_candidate.get("rank") if recommended_candidate else None,
        "recommended_candidate_image": recommended_candidate.get("image") if recommended_candidate else None,
        "recommended_candidate_score": recommended_candidate.get("selection_score") if recommended_candidate else None,
        "recommended_candidate_aesthetic_score": recommended_candidate.get("aesthetic_score") if recommended_candidate else None,
        "recommended_candidate_aesthetic_details": recommended_candidate.get("aesthetic_details") if recommended_candidate else None,
        "recommended_candidate_reasons": recommended_candidate.get("selection_reasons") if recommended_candidate else [],
        "revision_hints": _revision_hints(
            spec=spec,
            score=best_score,
            threshold=options.threshold,
            scene_plan=metadata_scene_plan,
            reference_image=options.reference_image,
            caption_diagnostics=caption_diagnostics,
            caption_similarity_score=caption_result.prompt_similarity_score,
        ),
        "seed": options.seed,
        "objects": list(spec.objects),
        "color_words": list(spec.color_words),
        "style_words": list(spec.style_words),
        "mood_words": list(spec.mood_words),
        "reference_image": str(options.reference_image) if options.reference_image else None,
        "initial_image": str(options.initial_image) if options.initial_image else None,
        "reference_palette": _palette_to_hex(reference_palette),
        "initial_palette": _palette_to_hex(initial_palette),
        "scene_plan": str(options.scene_plan) if options.scene_plan else None,
        "scene_plan_used": metadata_scene_plan is not None,
        "scene_plan_title": metadata_scene_plan.title if metadata_scene_plan else None,
        "scene_plan_objects": [obj.kind for obj in metadata_scene_plan.objects] if metadata_scene_plan else [],
        "scene_plan_background_stop_count": len(metadata_scene_plan.background.stops) if metadata_scene_plan else 0,
        "scene_plan_element_count": len(metadata_scene_plan.elements) if metadata_scene_plan else 0,
        "scene_plan_gradient_count": sum(1 for element in metadata_scene_plan.elements if element.gradient) if metadata_scene_plan else 0,
        "scene_plan_motif_count": len(metadata_scene_plan.motifs) if metadata_scene_plan else 0,
        "scene_plan_texture_count": len(metadata_scene_plan.textures) if metadata_scene_plan else 0,
        "scene_plan_material_count": len(metadata_scene_plan.materials) if metadata_scene_plan else 0,
        "scene_plan_terrain_count": len(metadata_scene_plan.terrains) if metadata_scene_plan else 0,
        "scene_plan_reflection_count": len(metadata_scene_plan.reflections) if metadata_scene_plan else 0,
        "scene_plan_warp_count": len(metadata_scene_plan.warps) if metadata_scene_plan else 0,
        "scene_plan_atmosphere_used": metadata_scene_plan.atmosphere is not None if metadata_scene_plan else False,
        "scene_plan_veil_count": len(metadata_scene_plan.veils) if metadata_scene_plan else 0,
        "scene_plan_light_count": len(metadata_scene_plan.lights) if metadata_scene_plan else 0,
        "scene_plan_beam_count": len(metadata_scene_plan.beams) if metadata_scene_plan else 0,
        "scene_plan_cloud_count": len(metadata_scene_plan.clouds) if metadata_scene_plan else 0,
        "scene_plan_shadow_count": len(metadata_scene_plan.shadows) if metadata_scene_plan else 0,
        "scene_plan_focus_used": metadata_scene_plan.focus is not None if metadata_scene_plan else False,
        "scene_plan_focus_blur": metadata_scene_plan.focus.blur if metadata_scene_plan and metadata_scene_plan.focus else 0.0,
        "scene_plan_antialias": metadata_scene_plan.style.get("antialias", 0.0) if metadata_scene_plan else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "claude-planned-cpu-renderer-v1" if metadata_scene_plan else "cpu-surrogate-iterative-v0",
    }
    apply_quality_report(output_dir, metadata)
    write_critique_request(output_dir, image_path=image_path, metadata_path=metadata_path, metadata=metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if pixels_path:
        export_pixel_csv(best_image, pixels_path)

    return GenerateResult(
        image=best_image,
        metadata=metadata,
        image_path=image_path,
        metadata_path=metadata_path,
        progress_path=progress_path,
        pixels_path=pixels_path,
        candidates_path=candidates_path,
    )


def _palette_from_optional_image(path: Path | None) -> tuple[tuple[int, int, int], ...] | None:
    if path is None:
        return None
    return extract_reference_palette(path)


def _palette_to_hex(palette: tuple[RGB, ...] | None) -> list[str]:
    if not palette:
        return []
    return [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette]


def _validate_prompt_scene_plan_semantics(options: GenerateOptions, *, spec: object, scene_plan: ScenePlan | None) -> None:
    if scene_plan is None:
        return

    prompt_tokens = set(getattr(spec, "tokens", ()))
    if len(prompt_tokens) >= 3:
        return

    plan_tokens = _scene_plan_semantic_tokens(scene_plan)
    missing_semantic_terms = sorted(plan_tokens - prompt_tokens)
    if len(missing_semantic_terms) < 2:
        return

    preview = ", ".join(missing_semantic_terms[:8])
    raise ValueError(
        "prompt is too short for the supplied scene plan at a high quality target; "
        "pass the full user prompt to --prompt so scoring, caption backchecks, and "
        f"quality gates evaluate the intended image. Detected scene-plan terms include: {preview}."
    )


def _scene_plan_semantic_tokens(scene_plan: ScenePlan) -> set[str]:
    semantic_text: list[str] = [scene_plan.title]
    semantic_text.extend(obj.kind for obj in scene_plan.objects)
    semantic_text.extend(obj.label for obj in scene_plan.objects)
    for element in scene_plan.elements:
        semantic_text.append(element.kind)
        semantic_text.append(element.label)
        text = element.extra.get("text")
        if isinstance(text, str):
            semantic_text.append(text)

    tokens = set(parse_prompt(" ".join(semantic_text)).tokens)
    return {
        token
        for token in tokens
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "for",
            "with",
            "premium",
            "local",
            "final",
            "image",
        }
    }


def _initial_image_blend_decision(
    initial_image: Path | None,
    *,
    spec: object,
    scene_plan: ScenePlan | None,
) -> tuple[bool, str | None]:
    if not initial_image:
        return False, None
    if not initial_image.exists():
        raise FileNotFoundError(f"Initial image not found: {initial_image}")
    if scene_plan is None:
        return True, "initial image blend applied for candidate continuity"

    prompt_tokens = set(getattr(spec, "tokens", ()))
    hard_edge_kinds = {element.kind for element in scene_plan.elements} & HARD_EDGE_ELEMENT_KINDS
    contains_text = any(
        element.kind in {"text", "label"} or isinstance(element.extra.get("text"), str)
        for element in scene_plan.elements
    )
    if hard_edge_kinds or contains_text or (prompt_tokens & GRAPHIC_PROMPT_TOKENS):
        reason_bits = []
        if hard_edge_kinds:
            reason_bits.append("hard-edge scene-plan elements")
        if contains_text:
            reason_bits.append("text labels")
        if prompt_tokens & GRAPHIC_PROMPT_TOKENS:
            reason_bits.append("graphic prompt terms")
        return False, "skipped initial image blend for " + ", ".join(reason_bits)

    return True, "initial image blend applied for scene continuity"


def _blend_initial_image(image: Image.Image, initial_image: Path | None, *, blend: bool) -> Image.Image:
    if not initial_image:
        return image
    if not initial_image.exists():
        raise FileNotFoundError(f"Initial image not found: {initial_image}")
    if not blend:
        return image
    with Image.open(initial_image) as existing:
        base = existing.convert("RGB").resize(image.size)
    return Image.blend(base, image, 0.56)


def _refine_scene_plan(
    scene_plan: ScenePlan,
    *,
    spec: object,
    score: ScoreResult,
) -> tuple[ScenePlan, list[str]]:
    actions: list[str] = []
    plan_objects = {obj.kind for obj in scene_plan.objects}
    requested_objects = tuple(getattr(spec, "objects", ()))

    objects = list(scene_plan.objects)
    clouds = list(scene_plan.clouds)
    textures = list(scene_plan.textures)
    for missing_object in requested_objects:
        if missing_object in plan_objects or _scene_plan_satisfies_requested_object(scene_plan, missing_object):
            continue
        planned_object = _default_planned_object(missing_object, index=len(objects), scene_plan=scene_plan)
        objects.append(planned_object)
        plan_objects.add(missing_object)
        actions.append(f"added missing object '{missing_object}'")
        if missing_object == "cloud" and not clouds:
            clouds.append(_default_cloud_layer(scene_plan))
            actions.append("added cloud layer for prompt cloud evidence")

    style = dict(scene_plan.style)
    if score.details.get("contrast_score", 1.0) < 0.35:
        style["contrast"] = min(1.0, style.get("contrast", 0.0) + 0.18)
        style["vignette"] = min(1.0, style.get("vignette", 0.0) + 0.08)
        actions.append("increased contrast and vignette")

    if tuple(getattr(spec, "color_words", ())) and score.details.get("color_score", 1.0) < 0.55:
        style["saturation"] = min(1.0, style.get("saturation", 0.0) + 0.16)
        actions.append("increased saturation for requested colors")

    if (
        "diagram" in requested_objects
        and _scene_plan_satisfies_requested_object(scene_plan, "diagram")
        and _diagram_detail_refinement_needed(scene_plan, score)
    ):
        textures.extend(_default_diagram_detail_textures(scene_plan, start_index=len(textures)))
        style["detail"] = max(style.get("detail", 0.0), 0.62)
        style["sharpen"] = max(style.get("sharpen", 0.0), 0.58)
        style["contrast"] = min(1.0, max(style.get("contrast", 0.0), 0.36))
        actions.append("added diagram detail texture")

    if not actions:
        return scene_plan, []

    return (
        replace(
            scene_plan,
            objects=tuple(objects),
            clouds=tuple(clouds),
            textures=tuple(sorted(textures, key=lambda texture: texture.z)),
            style=style,
        ),
        actions,
    )


def _scene_plan_satisfies_requested_object(scene_plan: ScenePlan, requested_object: str) -> bool:
    if requested_object != "diagram":
        return False

    element_kinds = {_normalized_element_kind(element.kind) for element in scene_plan.elements}
    hard_edge_count = len(element_kinds & {_normalized_element_kind(kind) for kind in HARD_EDGE_ELEMENT_KINDS})
    if hard_edge_count >= 2:
        return True

    semantic_text: list[str] = [scene_plan.title]
    semantic_text.extend(element.kind for element in scene_plan.elements)
    semantic_text.extend(element.label for element in scene_plan.elements)
    for element in scene_plan.elements:
        text = element.extra.get("text")
        if isinstance(text, str):
            semantic_text.append(text)
    tokens = set(parse_prompt(" ".join(semantic_text)).tokens)
    return bool(element_kinds and tokens & GRAPHIC_PROMPT_TOKENS)


def _normalized_element_kind(kind: str) -> str:
    return kind.replace("-", "_").strip().lower()


def _diagram_detail_refinement_needed(scene_plan: ScenePlan, score: ScoreResult) -> bool:
    if any(
        texture.extra.get("auto_refined") is True and texture.extra.get("purpose") == "diagram_detail"
        for texture in scene_plan.textures
    ):
        return False
    if len(scene_plan.textures) >= 2 and scene_plan.style.get("detail", 0.0) >= 0.55:
        return False
    return (
        score.details.get("object_score", 1.0) < 0.58
        or score.details.get("color_score", 1.0) < 0.58
        or score.details.get("contrast_score", 1.0) < 0.48
    )


def _default_diagram_detail_textures(scene_plan: ScenePlan, *, start_index: int) -> tuple[PlannedTexture, ...]:
    palette = scene_plan.palette
    cool = palette[1] if len(palette) > 1 else COLOR_RGB["cyan"]
    warm = palette[2] if len(palette) > 2 else COLOR_RGB["gold"]
    return (
        PlannedTexture(
            kind="hatching",
            label="auto-refined subtle diagram hatch detail",
            count=160,
            region=(0.04, 0.12, 0.96, 0.88),
            color=cool,
            density=0.42,
            scale=0.026,
            opacity=0.16,
            blend="screen",
            seed=131 + start_index,
            z=12 + start_index,
            extra={"auto_refined": True, "purpose": "diagram_detail"},
        ),
        PlannedTexture(
            kind="speckles",
            label="auto-refined premium diagram micro highlights",
            count=260,
            region=(0.04, 0.10, 0.96, 0.86),
            color=warm,
            density=0.28,
            scale=0.012,
            opacity=0.18,
            blend="screen",
            seed=197 + start_index,
            z=13 + start_index,
            extra={"auto_refined": True, "purpose": "diagram_detail"},
        ),
    )


def _default_planned_object(kind: str, *, index: int, scene_plan: ScenePlan) -> PlannedObject:
    color = _object_color(kind, scene_plan)
    defaults: dict[str, tuple[float, float, float]] = {
        "sun": (0.28, 0.25, 0.16),
        "moon": (0.72, 0.22, 0.12),
        "cloud": (0.62, 0.25, 0.12),
        "greenhouse": (0.50, 0.16, 0.88),
        "lamp": (0.50, 0.22, 0.08),
        "plant": (0.50, 0.70, 0.30),
        "floor": (0.50, 0.78, 0.22),
        "mountain": (0.50, 0.55, 0.28),
        "ocean": (0.50, 0.58, 0.18),
        "forest": (0.50, 0.78, 0.22),
        "flower": (0.50, 0.78, 0.18),
        "building": (0.50, 0.64, 0.25),
        "portrait": (0.50, 0.48, 0.24),
        "robot": (0.50, 0.50, 0.22),
        "abstract": (0.50, 0.50, 0.24),
    }
    x, y, size = defaults.get(kind, (0.50, 0.50, 0.18))
    if kind in {"ocean", "mountain", "forest", "flower", "building"}:
        extra = {"layers": 3} if kind == "mountain" else {}
    elif kind == "lamp":
        extra = {"count": 3, "spread": 0.36}
    elif kind == "plant":
        extra = {"count": 20}
    else:
        extra = {}
    return PlannedObject(
        kind=kind,
        label=f"auto-refined {kind}",
        x=x,
        y=y,
        size=size,
        color=color,
        opacity=0.92,
        extra={**extra, "auto_refined": True, "source_index": index},
    )


def _default_cloud_layer(scene_plan: ScenePlan) -> PlannedCloud:
    return PlannedCloud(
        kind="cumulus",
        label="auto-refined soft cloud bank",
        region=(0.08, 0.08, 0.94, 0.36),
        color=_object_color("cloud", scene_plan),
        shadow=(120, 136, 160),
        opacity=0.42,
        blur=0.026,
        count=4,
        lobes=5,
        scale=0.12,
        blend="screen",
        seed=97,
        z=6,
        extra={"auto_refined": True},
    )


def _object_color(kind: str, scene_plan: ScenePlan) -> RGB:
    palette = scene_plan.palette
    if kind == "sun":
        return COLOR_RGB["red"]
    if kind == "moon":
        return (232, 230, 210)
    if kind in {"ocean", "water", "lake"}:
        return COLOR_RGB["blue"]
    if kind == "greenhouse":
        return (118, 142, 150)
    if kind == "lamp":
        return COLOR_RGB["gold"]
    if kind == "floor":
        return (78, 86, 88)
    if kind == "plant":
        return COLOR_RGB["green"]
    if kind in {"forest", "flower"}:
        return COLOR_RGB["green"]
    if kind == "cloud":
        return (245, 248, 250)
    if kind == "mountain":
        return (74, 86, 112)
    if kind == "building":
        return (44, 50, 64)
    return palette[min(len(palette) - 1, 0)]


def _effective_similarity_device(*, backend: str, requested_device: str) -> str:
    normalized_backend = backend.strip().lower()
    normalized_device = requested_device.strip().lower()
    if normalized_backend == "local":
        return "cpu"
    if normalized_device != "auto":
        return normalized_device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _revision_hints(
    *,
    spec: object,
    score: ScoreResult,
    threshold: float,
    scene_plan: ScenePlan | None,
    reference_image: Path | None,
    caption_diagnostics: CaptionDiagnostics,
    caption_similarity_score: float,
) -> list[str]:
    caption_object_gap = bool(caption_diagnostics.missing_objects) and caption_similarity_score < 0.46
    caption_color_gap = bool(caption_diagnostics.missing_colors) and caption_similarity_score < 0.40
    if score.total_score >= threshold and not caption_object_gap and not caption_color_gap:
        return []

    hints: list[str] = []
    spec_objects = tuple(getattr(spec, "objects", ()))
    spec_color_words = tuple(getattr(spec, "color_words", ()))
    spec_mood_words = tuple(getattr(spec, "mood_words", ()))

    if caption_object_gap:
        hints.append(
            "The image caption missed requested objects: "
            f"{', '.join(caption_diagnostics.missing_objects)}. "
            "Revise the scene plan so those objects read clearly in the rendered image."
        )

    if caption_color_gap:
        hints.append(
            "The image caption missed requested colors: "
            f"{', '.join(caption_diagnostics.missing_colors)}. "
            "Use larger, clearer color regions or lighting accents for those colors."
        )

    if scene_plan and spec_objects:
        plan_objects = {obj.kind for obj in scene_plan.objects}
        missing_objects = [
            obj
            for obj in spec_objects
            if obj not in plan_objects and not _scene_plan_satisfies_requested_object(scene_plan, obj)
        ]
        if missing_objects:
            hints.append(f"Add missing scene-plan objects: {', '.join(missing_objects)}.")

    if spec_color_words and score.details.get("color_score", 1.0) < 0.55:
        hints.append(
            "Strengthen requested colors: "
            f"{', '.join(spec_color_words)}. "
            "Use palette entries, background stops, fills, materials, or lights that visibly contain them."
        )

    if spec_objects and score.details.get("object_score", 1.0) < 0.55:
        hints.append(
            "Increase prompt-object evidence with explicit shapes, terrain, materials, motifs, or elements for: "
            f"{', '.join(spec_objects)}."
        )

    if score.details.get("contrast_score", 1.0) < 0.35:
        hints.append("Increase tonal separation with clearer foreground/background contrast, shadows, lights, or silhouettes.")

    if spec_mood_words and score.details.get("mood_score", 1.0) < 0.45:
        hints.append(f"Make the scene mood read more clearly as: {', '.join(spec_mood_words)}.")

    if reference_image and score.reference_score < 0.45:
        hints.append("Move palette and composition closer to the reference image before rerunning.")

    return hints[:6]


def _remember_candidate(candidates: list[CandidateSnapshot], snapshot: CandidateSnapshot, *, limit: int) -> None:
    candidates.append(snapshot)
    candidates.sort(key=lambda candidate: candidate.score.total_score, reverse=True)
    del candidates[limit:]


def _write_candidate_artifacts(
    output_dir: Path,
    candidates: list[CandidateSnapshot],
    *,
    prompt: str,
    caption_backend: str,
    caption_model: str | None,
    caption_device: str,
    caption_similarity_backend: str,
    caption_similarity_model: str | None,
    caption_similarity_device: str,
) -> tuple[Path | None, Path | None, list[dict[str, object]]]:
    if not candidates:
        return None, None, []

    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.score.total_score, reverse=True)
    entries: list[dict[str, object]] = []

    for rank, candidate in enumerate(sorted_candidates, start=1):
        image_path = candidates_dir / f"candidate-{rank:03d}-iter-{candidate.iteration:03d}.png"
        candidate.image.save(image_path)
        caption_result = caption_image(
            candidate.image,
            prompt=prompt,
            backend=caption_backend,
            model_name=caption_model,
            device=caption_device,
            similarity_backend=caption_similarity_backend,
            similarity_model=caption_similarity_model,
            similarity_device=caption_similarity_device,
        )
        caption_diagnostics = (
            CaptionDiagnostics((), (), (), ())
            if caption_result.backend == "none"
            else caption_prompt_diagnostics(prompt, caption_result.caption)
        )
        aesthetic_score, aesthetic_details = compute_candidate_aesthetic_score(candidate.image)
        entry: dict[str, object] = {
            "rank": rank,
            "iteration": candidate.iteration,
            "image": str(image_path),
            "total_score": round(candidate.score.total_score, 6),
            "text_score": round(candidate.score.text_score, 6),
            "reference_score": round(candidate.score.reference_score, 6),
            "score_details": {key: round(value, 6) for key, value in candidate.score.details.items()},
            "met_threshold": candidate.met_threshold,
            "caption": caption_result.caption,
            "caption_similarity_score": round(caption_result.prompt_similarity_score, 6),
            "caption_similarity_backend": caption_result.similarity_backend,
            "caption_similarity_model": caption_result.similarity_model,
            "effective_caption_similarity_device": caption_result.effective_similarity_device,
            "lexical_caption_similarity_score": round(caption_result.lexical_prompt_similarity_score, 6),
            "semantic_caption_similarity_score": (
                round(caption_result.semantic_prompt_similarity_score, 6)
                if caption_result.semantic_prompt_similarity_score is not None
                else None
            ),
            "caption_missing_objects": list(caption_diagnostics.missing_objects),
            "caption_missing_colors": list(caption_diagnostics.missing_colors),
            "caption_unexpected_objects": list(caption_diagnostics.unexpected_objects),
            "caption_unexpected_colors": list(caption_diagnostics.unexpected_colors),
            "aesthetic_score": aesthetic_score,
            "aesthetic_details": aesthetic_details,
        }
        entries.append(annotate_candidate_selection(entry))

    contact_sheet_path = _write_candidate_contact_sheet(candidates_dir, sorted_candidates, entries)
    candidates_path = output_dir / "candidates.json"
    candidates_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return candidates_path, contact_sheet_path, entries


def _write_candidate_contact_sheet(
    candidates_dir: Path,
    candidates: list[CandidateSnapshot],
    entries: list[dict[str, object]],
) -> Path:
    max_thumb_width = 360
    max_thumb_height = 240
    thumbnails: list[Image.Image] = []
    for candidate in candidates:
        thumbnail = candidate.image.copy()
        thumbnail.thumbnail((max_thumb_width, max_thumb_height))
        thumbnails.append(thumbnail)

    padding = 12
    label_height = 46
    columns = min(3, max(1, len(thumbnails)))
    rows = (len(thumbnails) + columns - 1) // columns
    tile_width = max(thumbnail.width for thumbnail in thumbnails)
    tile_height = max(thumbnail.height for thumbnail in thumbnails) + label_height
    sheet = Image.new(
        "RGB",
        (
            columns * tile_width + (columns + 1) * padding,
            rows * tile_height + (rows + 1) * padding,
        ),
        (245, 245, 242),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (thumbnail, entry) in enumerate(zip(thumbnails, entries)):
        row = index // columns
        column = index % columns
        x = padding + column * (tile_width + padding)
        y = padding + row * (tile_height + padding)
        sheet.paste(thumbnail, (x, y))
        label_y = y + thumbnail.height + 5
        label = _truncate_text_to_width(draw, _candidate_contact_sheet_label(entry), tile_width)
        caption = _truncate_text_to_width(draw, str(entry.get("caption", "")), tile_width)
        draw.text((x, label_y), label, fill=(25, 28, 32))
        if caption:
            draw.text((x, label_y + 17), caption, fill=(55, 58, 64))

    contact_sheet_path = candidates_dir / "contact-sheet.png"
    sheet.save(contact_sheet_path)
    return contact_sheet_path


def _candidate_contact_sheet_label(entry: dict[str, object]) -> str:
    total_score = float(entry.get("total_score", 0.0))
    selection_score = float(entry.get("selection_score", 0.0))
    aesthetic_score = float(entry.get("aesthetic_score", 0.0))
    return f"#{entry['rank']} iter {entry['iteration']} score {total_score:.3f} sel {selection_score:.3f} aes {aesthetic_score:.3f}"


def _truncate_text_to_width(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> str:
    if not text:
        return ""
    if draw.textbbox((0, 0), text)[2] <= max_width:
        return text

    suffix = "..."
    if draw.textbbox((0, 0), suffix)[2] > max_width:
        return ""

    low = 0
    high = len(text)
    best = suffix
    while low <= high:
        midpoint = (low + high) // 2
        candidate = text[:midpoint].rstrip() + suffix
        if draw.textbbox((0, 0), candidate)[2] <= max_width:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return best


def _write_progress(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "total_score", "text_score", "reference_score", "cosine_score", "met_threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)
