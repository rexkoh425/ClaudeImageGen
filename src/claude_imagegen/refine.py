from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .generator import GenerateOptions, GenerateResult, generate_image
from .prompt import parse_prompt


@dataclass(frozen=True)
class RefineOptions:
    from_dir: Path
    prompt: str
    output_dir: Path
    reference_image: Path | None = None
    scene_plan: Path | None = None
    width: int | None = None
    height: int | None = None
    max_iterations: int = 32
    threshold: float = 0.58
    seed: int = 0
    pixel_csv: bool = False
    auto_refine: bool = True
    similarity_backend: str = "local"
    similarity_model: str | None = None
    similarity_device: str = "auto"


def refine_image(options: RefineOptions) -> GenerateResult:
    parent_image = options.from_dir / "image.png"
    parent_metadata_path = options.from_dir / "metadata.json"
    if not parent_image.exists():
        raise FileNotFoundError(f"Parent image not found: {parent_image}")

    parent_metadata = _read_parent_metadata(parent_metadata_path)
    width = options.width or _int_metadata(parent_metadata, "width", 720)
    height = options.height or _int_metadata(parent_metadata, "height", 480)
    scene_plan = options.scene_plan or _discover_scene_plan(options.from_dir, parent_metadata)
    scene_plan, scene_plan_actions, scene_plan_source = _prepare_refined_scene_plan(
        scene_plan,
        prompt=options.prompt,
        output_dir=options.output_dir,
    )

    result = generate_image(
        GenerateOptions(
            prompt=options.prompt,
            output_dir=options.output_dir,
            reference_image=options.reference_image,
            initial_image=parent_image,
            scene_plan=scene_plan,
            width=width,
            height=height,
            max_iterations=options.max_iterations,
            threshold=options.threshold,
            seed=options.seed,
            pixel_csv=options.pixel_csv,
            auto_refine=options.auto_refine,
            similarity_backend=options.similarity_backend,
            similarity_model=options.similarity_model,
            similarity_device=options.similarity_device,
        )
    )

    lineage_depth = _int_metadata(parent_metadata, "refinement_lineage_depth", 0) + 1
    result.metadata.update(
        {
            "refined_from": str(options.from_dir),
            "parent_image": str(parent_image),
            "parent_metadata": str(parent_metadata_path) if parent_metadata_path.exists() else None,
            "parent_prompt": parent_metadata.get("prompt"),
            "parent_total_score": parent_metadata.get("total_score"),
            "parent_similarity_backend": parent_metadata.get("similarity_backend"),
            "refinement_lineage_depth": lineage_depth,
            "scene_plan_refined_from": str(scene_plan_source) if scene_plan_source else None,
            "scene_plan_refine_actions": scene_plan_actions,
        }
    )
    result.metadata_path.write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    return result


def _read_parent_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def _discover_scene_plan(parent_dir: Path, parent_metadata: dict[str, object]) -> Path | None:
    local_plan = parent_dir / "scene-plan.json"
    if local_plan.exists():
        return local_plan

    raw_scene_plan = parent_metadata.get("scene_plan")
    if not isinstance(raw_scene_plan, str) or not raw_scene_plan:
        return None

    metadata_plan = Path(raw_scene_plan)
    if metadata_plan.exists():
        return metadata_plan

    parent_relative = parent_dir / metadata_plan
    return parent_relative if parent_relative.exists() else None


def _prepare_refined_scene_plan(
    scene_plan: Path | None,
    *,
    prompt: str,
    output_dir: Path,
) -> tuple[Path | None, list[str], Path | None]:
    if scene_plan is None or not scene_plan.exists():
        return scene_plan, [], scene_plan

    data = json.loads(scene_plan.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return scene_plan, [], scene_plan

    actions = _apply_prompt_delta_edits(data, prompt)
    output_dir.mkdir(parents=True, exist_ok=True)
    refined_plan = output_dir / "scene-plan.json"
    refined_plan.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return refined_plan, actions, scene_plan


def _apply_prompt_delta_edits(data: dict[str, object], prompt: str) -> list[str]:
    spec = parse_prompt(prompt)
    tokens = set(spec.tokens)
    normalized = spec.normalized
    actions: list[str] = []

    if "cloud" in spec.objects:
        clouds = _list_field(data, "clouds")
        if not clouds:
            clouds.append(
                {
                    "type": "cumulus",
                    "label": "refine-added cloud bank",
                    "region": [0.08, 0.08, 0.94, 0.36],
                    "color": "#fff1dd",
                    "shadow": "#788ca8",
                    "opacity": 0.42,
                    "blur": 0.026,
                    "count": 4,
                    "lobes": 5,
                    "scale": 0.12,
                    "blend": "screen",
                    "seed": 107,
                    "z": 6,
                }
            )
            actions.append("added cloud bank from revised prompt")
        else:
            for cloud in clouds:
                if isinstance(cloud, dict):
                    cloud["count"] = min(48, int(cloud.get("count", 3)) + 2)
                    cloud["opacity"] = min(1.0, float(cloud.get("opacity", 0.36)) + 0.08)
            actions.append("increased cloud density and opacity")
        _ensure_object(data, "cloud", "#fff1dd", actions)

    if "cyan" in spec.color_words or "neon" in tokens:
        motifs = _list_field(data, "motifs")
        window_lights = [motif for motif in motifs if isinstance(motif, dict) and str(motif.get("type", "")).lower() in {"window_lights", "windows"}]
        if not window_lights:
            motifs.append(
                {
                    "type": "window_lights",
                    "label": "refine-added cyan window lights",
                    "count": 80,
                    "region": [0.45, 0.28, 0.95, 0.64],
                    "color": "#6ee7ff",
                    "size": 0.008,
                    "opacity": 0.74,
                    "seed": 109,
                    "z": 10,
                }
            )
            actions.append("added cyan window light motif")
        else:
            for motif in window_lights:
                motif["count"] = min(800, int(motif.get("count", 80)) + 80)
                motif["opacity"] = min(1.0, float(motif.get("opacity", 0.7)) + 0.08)
                motif["color"] = "#6ee7ff"
            actions.append("brightened cyan window lights")
        _bump_style(data, "saturation", 0.08)
        _bump_style(data, "bloom", 0.06)

    if "grass" in tokens or "foreground grass" in normalized:
        motifs = _list_field(data, "motifs")
        grass_motifs = [motif for motif in motifs if isinstance(motif, dict) and str(motif.get("type", "")).lower() == "grass"]
        for motif in grass_motifs:
            motif["count"] = min(800, int(motif.get("count", 100)) + 80)
            motif["opacity"] = min(1.0, float(motif.get("opacity", 0.65)) + 0.05)
        if grass_motifs:
            actions.append("sharpened foreground grass detail")

    if "reflection" in tokens or "reflections" in tokens or "stronger water" in normalized:
        for reflection in _list_field(data, "reflections"):
            if isinstance(reflection, dict):
                reflection["opacity"] = min(1.0, float(reflection.get("opacity", 0.34)) + 0.08)
        for texture in _list_field(data, "textures"):
            if isinstance(texture, dict) and str(texture.get("type", "")).lower() == "ripple":
                texture["count"] = min(1400, int(texture.get("count", 24)) + 12)
                texture["opacity"] = min(1.0, float(texture.get("opacity", 0.34)) + 0.04)
        actions.append("strengthened water reflections and ripples")

    return actions


def _list_field(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if isinstance(value, list):
        return value
    data[key] = []
    return data[key]  # type: ignore[return-value]


def _ensure_object(data: dict[str, object], kind: str, color: str, actions: list[str]) -> None:
    objects = _list_field(data, "objects")
    for item in objects:
        if isinstance(item, dict) and str(item.get("type", item.get("kind", ""))).lower() == kind:
            return
    objects.append({"type": kind, "label": f"refine-added {kind}", "x": 0.62, "y": 0.24, "size": 0.12, "color": color, "opacity": 0.48})
    actions.append(f"added {kind} object from revised prompt")


def _bump_style(data: dict[str, object], key: str, amount: float) -> None:
    style = data.get("style")
    if not isinstance(style, dict):
        style = {}
        data["style"] = style
    try:
        current = float(style.get(key, 0.0))
    except (TypeError, ValueError):
        current = 0.0
    style[key] = min(1.0, current + amount)


def _int_metadata(metadata: dict[str, object], key: str, default: int) -> int:
    try:
        return int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
