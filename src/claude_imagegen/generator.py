from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from .palette import RGB, extract_reference_palette
from .pixels import export_pixel_csv
from .prompt import parse_prompt
from .render import cap_dimensions, render_candidate, render_scene_plan
from .scene import SceneCandidate, build_initial_candidate, mutate_candidate
from .scene_plan import ScenePlan, parse_scene_plan
from .score import ScoreResult, score_image


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
    seed: int = 0
    pixel_csv: bool = False
    scene_plan: Path | None = None


@dataclass(frozen=True)
class GenerateResult:
    image: Image.Image
    metadata: dict[str, object]
    image_path: Path
    metadata_path: Path
    progress_path: Path
    pixels_path: Path | None


def generate_image(options: GenerateOptions) -> GenerateResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    if options.max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    width, height = cap_dimensions(options.width, options.height)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_palette = _palette_from_optional_image(options.reference_image)
    initial_palette = _palette_from_optional_image(options.initial_image)
    spec = parse_prompt(options.prompt)
    scene_plan = parse_scene_plan(options.scene_plan) if options.scene_plan else None
    candidate = build_initial_candidate(
        spec,
        seed=options.seed,
        reference_palette=reference_palette,
        initial_palette=initial_palette,
    )

    best_image: Image.Image | None = None
    best_score: ScoreResult | None = None
    best_iteration = 0
    progress: list[dict[str, object]] = []

    for iteration in range(1, options.max_iterations + 1):
        image = (
            render_scene_plan(scene_plan, width=width, height=height, seed=options.seed + iteration)
            if scene_plan
            else render_candidate(candidate, width=width, height=height)
        )
        image = _blend_initial_image(image, options.initial_image)
        score = score_image(image, spec, reference_image=options.reference_image)
        met_threshold = score.total_score >= options.threshold

        progress.append(
            {
                "iteration": iteration,
                "total_score": f"{score.total_score:.6f}",
                "text_score": f"{score.text_score:.6f}",
                "reference_score": f"{score.reference_score:.6f}",
                "met_threshold": str(met_threshold).lower(),
            }
        )

        if best_score is None or score.total_score > best_score.total_score:
            best_image = image
            best_score = score
            best_iteration = iteration

        if met_threshold:
            break

        if not scene_plan:
            candidate = mutate_candidate(candidate, iteration)

    assert best_image is not None
    assert best_score is not None

    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.csv"
    pixels_path = output_dir / "pixels.csv" if options.pixel_csv else None

    best_image.save(image_path)
    _write_progress(progress_path, progress)

    metadata: dict[str, object] = {
        "prompt": options.prompt,
        "normalized_prompt": spec.normalized,
        "width": width,
        "height": height,
        "iterations": best_iteration,
        "max_iterations": options.max_iterations,
        "threshold": options.threshold,
        "met_threshold": best_score.total_score >= options.threshold,
        "total_score": round(best_score.total_score, 6),
        "text_score": round(best_score.text_score, 6),
        "reference_score": round(best_score.reference_score, 6),
        "score_details": {key: round(value, 6) for key, value in best_score.details.items()},
        "revision_hints": _revision_hints(
            spec=spec,
            score=best_score,
            threshold=options.threshold,
            scene_plan=scene_plan,
            reference_image=options.reference_image,
        ),
        "seed": options.seed,
        "objects": list(spec.objects),
        "color_words": list(spec.color_words),
        "style_words": list(spec.style_words),
        "reference_image": str(options.reference_image) if options.reference_image else None,
        "initial_image": str(options.initial_image) if options.initial_image else None,
        "reference_palette": _palette_to_hex(reference_palette),
        "initial_palette": _palette_to_hex(initial_palette),
        "scene_plan": str(options.scene_plan) if options.scene_plan else None,
        "scene_plan_used": scene_plan is not None,
        "scene_plan_title": scene_plan.title if scene_plan else None,
        "scene_plan_objects": [obj.kind for obj in scene_plan.objects] if scene_plan else [],
        "scene_plan_background_stop_count": len(scene_plan.background.stops) if scene_plan else 0,
        "scene_plan_element_count": len(scene_plan.elements) if scene_plan else 0,
        "scene_plan_gradient_count": sum(1 for element in scene_plan.elements if element.gradient) if scene_plan else 0,
        "scene_plan_motif_count": len(scene_plan.motifs) if scene_plan else 0,
        "scene_plan_texture_count": len(scene_plan.textures) if scene_plan else 0,
        "scene_plan_material_count": len(scene_plan.materials) if scene_plan else 0,
        "scene_plan_terrain_count": len(scene_plan.terrains) if scene_plan else 0,
        "scene_plan_reflection_count": len(scene_plan.reflections) if scene_plan else 0,
        "scene_plan_warp_count": len(scene_plan.warps) if scene_plan else 0,
        "scene_plan_atmosphere_used": scene_plan.atmosphere is not None if scene_plan else False,
        "scene_plan_veil_count": len(scene_plan.veils) if scene_plan else 0,
        "scene_plan_light_count": len(scene_plan.lights) if scene_plan else 0,
        "scene_plan_beam_count": len(scene_plan.beams) if scene_plan else 0,
        "scene_plan_cloud_count": len(scene_plan.clouds) if scene_plan else 0,
        "scene_plan_shadow_count": len(scene_plan.shadows) if scene_plan else 0,
        "scene_plan_focus_used": scene_plan.focus is not None if scene_plan else False,
        "scene_plan_focus_blur": scene_plan.focus.blur if scene_plan and scene_plan.focus else 0.0,
        "scene_plan_antialias": scene_plan.style.get("antialias", 0.0) if scene_plan else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "claude-planned-cpu-renderer-v1" if scene_plan else "cpu-surrogate-iterative-v0",
    }
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
    )


def _palette_from_optional_image(path: Path | None) -> tuple[tuple[int, int, int], ...] | None:
    if path is None:
        return None
    return extract_reference_palette(path)


def _palette_to_hex(palette: tuple[RGB, ...] | None) -> list[str]:
    if not palette:
        return []
    return [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in palette]


def _blend_initial_image(image: Image.Image, initial_image: Path | None) -> Image.Image:
    if not initial_image:
        return image
    if not initial_image.exists():
        raise FileNotFoundError(f"Initial image not found: {initial_image}")
    with Image.open(initial_image) as existing:
        base = existing.convert("RGB").resize(image.size)
    return Image.blend(base, image, 0.56)


def _revision_hints(
    *,
    spec: object,
    score: ScoreResult,
    threshold: float,
    scene_plan: ScenePlan | None,
    reference_image: Path | None,
) -> list[str]:
    if score.total_score >= threshold:
        return []

    hints: list[str] = []
    spec_objects = tuple(getattr(spec, "objects", ()))
    spec_color_words = tuple(getattr(spec, "color_words", ()))
    spec_mood_words = tuple(getattr(spec, "mood_words", ()))

    if scene_plan and spec_objects:
        plan_objects = {obj.kind for obj in scene_plan.objects}
        missing_objects = [obj for obj in spec_objects if obj not in plan_objects]
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


def _write_progress(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["iteration", "total_score", "text_score", "reference_score", "met_threshold"],
        )
        writer.writeheader()
        writer.writerows(rows)
