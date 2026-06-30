from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .palette import COLOR_RGB, RGB

PathCommand = tuple[str, tuple[tuple[float, float], ...]]


@dataclass(frozen=True)
class PlannedColorStop:
    position: float
    color: RGB


@dataclass(frozen=True)
class PlannedBackground:
    top: RGB
    bottom: RGB
    direction: str
    stops: tuple[PlannedColorStop, ...]


@dataclass(frozen=True)
class PlannedObject:
    kind: str
    label: str
    x: float
    y: float
    size: float
    color: RGB
    opacity: float
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedGradient:
    kind: str
    colors: tuple[RGB, ...]
    direction: str
    center: tuple[float, float]
    radius: float


@dataclass(frozen=True)
class PlannedElement:
    kind: str
    label: str
    x: float
    y: float
    width: float
    height: float
    points: tuple[tuple[float, float], ...]
    commands: tuple[PathCommand, ...]
    gradient: PlannedGradient | None
    fill: RGB | None
    stroke: RGB | None
    opacity: float
    blur: float
    blend: str
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedMotif:
    kind: str
    label: str
    count: int
    region: tuple[float, float, float, float]
    color: RGB
    size: float
    opacity: float
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedTexture:
    kind: str
    label: str
    count: int
    region: tuple[float, float, float, float]
    color: RGB
    density: float
    scale: float
    opacity: float
    blend: str
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedMaterial:
    kind: str
    label: str
    region: tuple[float, float, float, float]
    colors: tuple[RGB, ...]
    intensity: float
    scale: float
    opacity: float
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedTerrain:
    kind: str
    label: str
    points: tuple[tuple[float, float], ...]
    base: float
    fill: RGB
    shade: RGB | None
    highlight: RGB | None
    opacity: float
    blur: float
    blend: str
    facets: bool
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedReflection:
    kind: str
    label: str
    source: tuple[float, float, float, float]
    target: tuple[float, float, float, float]
    opacity: float
    blur: float
    fade: float
    tint: RGB | None
    blend: str
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedWarp:
    kind: str
    label: str
    region: tuple[float, float, float, float]
    direction: str
    amplitude: float
    wavelength: float
    phase: float
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedAtmosphere:
    kind: str
    label: str
    color: RGB
    horizon: float
    height: float
    strength: float


@dataclass(frozen=True)
class PlannedVeil:
    kind: str
    label: str
    region: tuple[float, float, float, float]
    color: RGB
    opacity: float
    blur: float
    blend: str
    falloff: float
    direction: str
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedFocus:
    kind: str
    label: str
    region: tuple[float, float, float, float]
    blur: float
    falloff: float
    mode: str


@dataclass(frozen=True)
class PlannedLight:
    kind: str
    label: str
    x: float
    y: float
    radius: float
    color: RGB
    intensity: float
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedBeam:
    kind: str
    label: str
    x: float
    y: float
    angle: float
    length: float
    spread: float
    color: RGB
    opacity: float
    blur: float
    blend: str
    count: int
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedCloud:
    kind: str
    label: str
    region: tuple[float, float, float, float]
    color: RGB
    shadow: RGB | None
    opacity: float
    blur: float
    count: int
    lobes: int
    scale: float
    blend: str
    seed: int
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class PlannedShadow:
    kind: str
    label: str
    x: float
    y: float
    width: float
    height: float
    points: tuple[tuple[float, float], ...]
    color: RGB
    opacity: float
    blur: float
    blend: str
    z: int
    extra: dict[str, Any]


@dataclass(frozen=True)
class ScenePlan:
    title: str
    palette: tuple[RGB, ...]
    background: PlannedBackground
    objects: tuple[PlannedObject, ...]
    elements: tuple[PlannedElement, ...]
    motifs: tuple[PlannedMotif, ...]
    textures: tuple[PlannedTexture, ...]
    materials: tuple[PlannedMaterial, ...]
    terrains: tuple[PlannedTerrain, ...]
    reflections: tuple[PlannedReflection, ...]
    warps: tuple[PlannedWarp, ...]
    atmosphere: PlannedAtmosphere | None
    veils: tuple[PlannedVeil, ...]
    focus: PlannedFocus | None
    lights: tuple[PlannedLight, ...]
    beams: tuple[PlannedBeam, ...]
    clouds: tuple[PlannedCloud, ...]
    shadows: tuple[PlannedShadow, ...]
    style: dict[str, float]
    source_path: Path


def parse_scene_plan(path: Path) -> ScenePlan:
    if not path.exists():
        raise FileNotFoundError(f"Scene plan not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("scene plan must be a JSON object")

    palette = tuple(_parse_color(value) for value in data.get("palette", []))
    if not palette:
        palette = (COLOR_RGB["blue"], COLOR_RGB["gold"])

    background_data = _as_dict(data.get("background"))
    top = _parse_color(background_data.get("top", palette[0]))
    bottom = _parse_color(background_data.get("bottom", palette[min(1, len(palette) - 1)]))
    background_direction = _direction_value(background_data.get("direction", "vertical"))
    background_stops = _parse_color_stops(background_data.get("stops"), top=top, bottom=bottom)

    objects: list[PlannedObject] = []
    for raw_object in data.get("objects", []):
        obj = _as_dict(raw_object)
        kind = str(obj.get("type", obj.get("kind", "shape"))).strip().lower() or "shape"
        color = _parse_color(obj.get("color", palette[len(objects) % len(palette)]))
        objects.append(
            PlannedObject(
                kind=kind,
                label=str(obj.get("label", kind)),
                x=_unit_float(obj.get("x", 0.5), default=0.5),
                y=_unit_float(obj.get("y", 0.5), default=0.5),
                size=_unit_float(obj.get("size", 0.18), default=0.18),
                color=color,
                opacity=_unit_float(obj.get("opacity", 1.0), default=1.0),
                extra={key: value for key, value in obj.items() if key not in {"type", "kind", "label", "x", "y", "size", "color", "opacity"}},
            )
        )

    elements: list[PlannedElement] = []
    for raw_element in data.get("elements", []):
        element = _as_dict(raw_element)
        kind = str(element.get("type", element.get("kind", "shape"))).strip().lower() or "shape"
        elements.append(
            PlannedElement(
                kind=kind,
                label=str(element.get("label", kind)),
                x=_unit_float(element.get("x", 0.5), default=0.5),
                y=_unit_float(element.get("y", 0.5), default=0.5),
                width=_unit_float(element.get("width", element.get("size", 0.16)), default=0.16),
                height=_unit_float(element.get("height", element.get("size", 0.16)), default=0.16),
                points=_parse_points(element.get("points", [])),
                commands=_parse_path_commands(element.get("commands", [])),
                gradient=_parse_gradient(element.get("gradient")),
                fill=_parse_optional_color(element.get("fill")),
                stroke=_parse_optional_color(element.get("stroke")),
                opacity=_unit_float(element.get("opacity", 1.0), default=1.0),
                blur=_unit_float(element.get("blur", 0.0), default=0.0),
                blend=_blend_value(element.get("blend", "normal")),
                z=_int_value(element.get("z", len(elements)), default=len(elements)),
                extra={
                    key: value
                    for key, value in element.items()
                    if key
                    not in {
                        "type",
                        "kind",
                        "label",
                        "x",
                        "y",
                        "width",
                        "height",
                        "size",
                        "points",
                        "commands",
                        "gradient",
                        "fill",
                        "stroke",
                        "opacity",
                        "blur",
                        "blend",
                        "z",
                    }
                },
            )
        )

    motifs: list[PlannedMotif] = []
    for raw_motif in data.get("motifs", []):
        motif = _as_dict(raw_motif)
        kind = str(motif.get("type", motif.get("kind", "dots"))).strip().lower() or "dots"
        motifs.append(
            PlannedMotif(
                kind=kind,
                label=str(motif.get("label", kind)),
                count=max(0, min(800, _int_value(motif.get("count", 12), default=12))),
                region=_parse_region(motif.get("region", [0.0, 0.0, 1.0, 1.0])),
                color=_parse_color(motif.get("color", palette[len(motifs) % len(palette)])),
                size=_unit_float(motif.get("size", 0.02), default=0.02),
                opacity=_unit_float(motif.get("opacity", 1.0), default=1.0),
                seed=_int_value(motif.get("seed", len(motifs)), default=len(motifs)),
                z=_int_value(motif.get("z", 10 + len(motifs)), default=10 + len(motifs)),
                extra={
                    key: value
                    for key, value in motif.items()
                    if key not in {"type", "kind", "label", "count", "region", "color", "size", "opacity", "seed", "z"}
                },
            )
        )

    textures: list[PlannedTexture] = []
    for raw_texture in data.get("textures", []):
        texture = _as_dict(raw_texture)
        kind = str(texture.get("type", texture.get("kind", "speckles"))).strip().lower() or "speckles"
        textures.append(
            PlannedTexture(
                kind=kind,
                label=str(texture.get("label", kind)),
                count=max(0, min(1400, _int_value(texture.get("count", 0), default=0))),
                region=_parse_region(texture.get("region", [0.0, 0.0, 1.0, 1.0])),
                color=_parse_color(texture.get("color", palette[len(textures) % len(palette)])),
                density=_unit_float(texture.get("density", 0.45), default=0.45),
                scale=_unit_float(texture.get("scale", 0.035), default=0.035),
                opacity=_unit_float(texture.get("opacity", 0.5), default=0.5),
                blend=_blend_value(texture.get("blend", "normal")),
                seed=_int_value(texture.get("seed", len(textures)), default=len(textures)),
                z=_int_value(texture.get("z", 12 + len(textures)), default=12 + len(textures)),
                extra={
                    key: value
                    for key, value in texture.items()
                    if key
                    not in {
                        "type",
                        "kind",
                        "label",
                        "count",
                        "region",
                        "color",
                        "density",
                        "scale",
                        "opacity",
                        "blend",
                        "seed",
                        "z",
                    }
                },
            )
        )

    materials: list[PlannedMaterial] = []
    for raw_material in data.get("materials", []):
        material = _as_dict(raw_material)
        kind = str(material.get("type", material.get("kind", "surface"))).strip().lower() or "surface"
        colors = _parse_color_sequence(material.get("colors"), fallback=palette)
        materials.append(
            PlannedMaterial(
                kind=kind,
                label=str(material.get("label", kind)),
                region=_parse_region(material.get("region", [0.0, 0.0, 1.0, 1.0])),
                colors=colors,
                intensity=_unit_float(material.get("intensity", 0.55), default=0.55),
                scale=_unit_float(material.get("scale", 0.04), default=0.04),
                opacity=_unit_float(material.get("opacity", 0.55), default=0.55),
                seed=_int_value(material.get("seed", len(materials)), default=len(materials)),
                z=_int_value(material.get("z", 14 + len(materials)), default=14 + len(materials)),
                extra={
                    key: value
                    for key, value in material.items()
                    if key not in {"type", "kind", "label", "region", "colors", "intensity", "scale", "opacity", "seed", "z"}
                },
            )
        )

    terrains: list[PlannedTerrain] = []
    for raw_terrain in data.get("terrains", []):
        terrain = _as_dict(raw_terrain)
        kind = str(terrain.get("type", terrain.get("kind", "ridge"))).strip().lower() or "ridge"
        terrains.append(
            PlannedTerrain(
                kind=kind,
                label=str(terrain.get("label", kind)),
                points=_parse_points(terrain.get("points", [])),
                base=_unit_float(terrain.get("base", 0.78), default=0.78),
                fill=_parse_color(terrain.get("fill", terrain.get("color", palette[len(terrains) % len(palette)]))),
                shade=_parse_optional_color(terrain.get("shade")),
                highlight=_parse_optional_color(terrain.get("highlight")),
                opacity=_unit_float(terrain.get("opacity", 1.0), default=1.0),
                blur=_unit_float(terrain.get("blur", 0.0), default=0.0),
                blend=_blend_value(terrain.get("blend", "normal")),
                facets=_bool_value(terrain.get("facets", True), default=True),
                z=_int_value(terrain.get("z", 8 + len(terrains)), default=8 + len(terrains)),
                extra={
                    key: value
                    for key, value in terrain.items()
                    if key not in {"type", "kind", "label", "points", "base", "fill", "color", "shade", "highlight", "opacity", "blur", "blend", "facets", "z"}
                },
            )
        )

    reflections: list[PlannedReflection] = []
    for raw_reflection in data.get("reflections", []):
        reflection = _as_dict(raw_reflection)
        kind = str(reflection.get("type", reflection.get("kind", "vertical"))).strip().lower() or "vertical"
        reflections.append(
            PlannedReflection(
                kind=kind,
                label=str(reflection.get("label", kind)),
                source=_parse_region(reflection.get("source", [0.0, 0.0, 1.0, 0.5])),
                target=_parse_region(reflection.get("target", [0.0, 0.5, 1.0, 1.0])),
                opacity=_unit_float(reflection.get("opacity", 0.45), default=0.45),
                blur=_unit_float(reflection.get("blur", 0.02), default=0.02),
                fade=_unit_float(reflection.get("fade", 0.55), default=0.55),
                tint=_parse_optional_color(reflection.get("tint")),
                blend=_blend_value(reflection.get("blend", "normal")),
                z=_int_value(reflection.get("z", 13 + len(reflections)), default=13 + len(reflections)),
                extra={
                    key: value
                    for key, value in reflection.items()
                    if key not in {"type", "kind", "label", "source", "target", "opacity", "blur", "fade", "tint", "blend", "z"}
                },
            )
        )

    warps: list[PlannedWarp] = []
    for raw_warp in data.get("warps", []):
        warp = _as_dict(raw_warp)
        kind = str(warp.get("type", warp.get("kind", "wave"))).strip().lower() or "wave"
        direction = str(warp.get("direction", "horizontal")).strip().lower()
        if direction not in {"horizontal", "vertical"}:
            direction = "horizontal"
        warps.append(
            PlannedWarp(
                kind=kind,
                label=str(warp.get("label", kind)),
                region=_parse_region(warp.get("region", [0.0, 0.0, 1.0, 1.0])),
                direction=direction,
                amplitude=_unit_float(warp.get("amplitude", 0.025), default=0.025),
                wavelength=_unit_float(warp.get("wavelength", 0.35), default=0.35),
                phase=_unit_float(warp.get("phase", 0.0), default=0.0),
                seed=_int_value(warp.get("seed", len(warps)), default=len(warps)),
                z=_int_value(warp.get("z", 15 + len(warps)), default=15 + len(warps)),
                extra={
                    key: value
                    for key, value in warp.items()
                    if key not in {"type", "kind", "label", "region", "direction", "amplitude", "wavelength", "phase", "seed", "z"}
                },
            )
        )

    atmosphere = _parse_atmosphere(data.get("atmosphere"))
    focus = _parse_focus(data.get("focus"))

    veils: list[PlannedVeil] = []
    for raw_veil in data.get("veils", []):
        veil = _as_dict(raw_veil)
        kind = str(veil.get("type", veil.get("kind", "mist"))).strip().lower() or "mist"
        veils.append(
            PlannedVeil(
                kind=kind,
                label=str(veil.get("label", kind)),
                region=_parse_region(veil.get("region", [0.0, 0.0, 1.0, 1.0])),
                color=_parse_color(veil.get("color", (216, 232, 240))),
                opacity=_unit_float(veil.get("opacity", 0.32), default=0.32),
                blur=_unit_float(veil.get("blur", 0.02), default=0.02),
                blend=_blend_value(veil.get("blend", "screen")),
                falloff=_unit_float(veil.get("falloff", 0.16), default=0.16),
                direction=_direction_value(veil.get("direction", "vertical")),
                z=_int_value(veil.get("z", 12 + len(veils)), default=12 + len(veils)),
                extra={
                    key: value
                    for key, value in veil.items()
                    if key not in {"type", "kind", "label", "region", "color", "opacity", "blur", "blend", "falloff", "direction", "z"}
                },
            )
        )

    lights: list[PlannedLight] = []
    for raw_light in data.get("lights", []):
        light = _as_dict(raw_light)
        kind = str(light.get("type", light.get("kind", "radial"))).strip().lower() or "radial"
        lights.append(
            PlannedLight(
                kind=kind,
                label=str(light.get("label", kind)),
                x=_unit_float(light.get("x", 0.5), default=0.5),
                y=_unit_float(light.get("y", 0.5), default=0.5),
                radius=_unit_float(light.get("radius", 0.35), default=0.35),
                color=_parse_color(light.get("color", (255, 255, 255))),
                intensity=_unit_float(light.get("intensity", 0.5), default=0.5),
                z=_int_value(light.get("z", 20 + len(lights)), default=20 + len(lights)),
                extra={
                    key: value
                    for key, value in light.items()
                    if key not in {"type", "kind", "label", "x", "y", "radius", "color", "intensity", "z"}
                },
            )
        )

    beams: list[PlannedBeam] = []
    for raw_beam in data.get("beams", []):
        beam = _as_dict(raw_beam)
        kind = str(beam.get("type", beam.get("kind", "beam"))).strip().lower() or "beam"
        beams.append(
            PlannedBeam(
                kind=kind,
                label=str(beam.get("label", kind)),
                x=_unit_float(beam.get("x", 0.5), default=0.5),
                y=_unit_float(beam.get("y", 0.25), default=0.25),
                angle=_float_value(beam.get("angle", 90.0), default=90.0),
                length=_unit_float(beam.get("length", 0.6), default=0.6),
                spread=max(1.0, min(120.0, _float_value(beam.get("spread", 18.0), default=18.0))),
                color=_parse_color(beam.get("color", (255, 255, 255))),
                opacity=_unit_float(beam.get("opacity", 0.3), default=0.3),
                blur=_unit_float(beam.get("blur", 0.02), default=0.02),
                blend=_blend_value(beam.get("blend", "screen")),
                count=max(1, min(24, _int_value(beam.get("count", 1), default=1))),
                seed=_int_value(beam.get("seed", len(beams)), default=len(beams)),
                z=_int_value(beam.get("z", 18 + len(beams)), default=18 + len(beams)),
                extra={
                    key: value
                    for key, value in beam.items()
                    if key not in {"type", "kind", "label", "x", "y", "angle", "length", "spread", "color", "opacity", "blur", "blend", "count", "seed", "z"}
                },
            )
        )

    clouds: list[PlannedCloud] = []
    for raw_cloud in data.get("clouds", []):
        cloud = _as_dict(raw_cloud)
        kind = str(cloud.get("type", cloud.get("kind", "cumulus"))).strip().lower() or "cumulus"
        clouds.append(
            PlannedCloud(
                kind=kind,
                label=str(cloud.get("label", kind)),
                region=_parse_region(cloud.get("region", [0.0, 0.0, 1.0, 0.4])),
                color=_parse_color(cloud.get("color", (245, 248, 250))),
                shadow=_parse_optional_color(cloud.get("shadow")),
                opacity=_unit_float(cloud.get("opacity", 0.6), default=0.6),
                blur=_unit_float(cloud.get("blur", 0.02), default=0.02),
                count=max(1, min(48, _int_value(cloud.get("count", 1), default=1))),
                lobes=max(2, min(12, _int_value(cloud.get("lobes", 5), default=5))),
                scale=_unit_float(cloud.get("scale", 0.14), default=0.14),
                blend=_blend_value(cloud.get("blend", "screen")),
                seed=_int_value(cloud.get("seed", len(clouds)), default=len(clouds)),
                z=_int_value(cloud.get("z", 6 + len(clouds)), default=6 + len(clouds)),
                extra={
                    key: value
                    for key, value in cloud.items()
                    if key not in {"type", "kind", "label", "region", "color", "shadow", "opacity", "blur", "count", "lobes", "scale", "blend", "seed", "z"}
                },
            )
        )

    shadows: list[PlannedShadow] = []
    for raw_shadow in data.get("shadows", []):
        shadow = _as_dict(raw_shadow)
        kind = str(shadow.get("type", shadow.get("kind", "ellipse"))).strip().lower() or "ellipse"
        shadows.append(
            PlannedShadow(
                kind=kind,
                label=str(shadow.get("label", kind)),
                x=_unit_float(shadow.get("x", 0.5), default=0.5),
                y=_unit_float(shadow.get("y", 0.7), default=0.7),
                width=_unit_float(shadow.get("width", 0.25), default=0.25),
                height=_unit_float(shadow.get("height", 0.08), default=0.08),
                points=_parse_points(shadow.get("points", [])),
                color=_parse_color(shadow.get("color", (20, 24, 30))),
                opacity=_unit_float(shadow.get("opacity", 0.35), default=0.35),
                blur=_unit_float(shadow.get("blur", 0.025), default=0.025),
                blend=_blend_value(shadow.get("blend", "multiply")),
                z=_int_value(shadow.get("z", 9 + len(shadows)), default=9 + len(shadows)),
                extra={
                    key: value
                    for key, value in shadow.items()
                    if key not in {"type", "kind", "label", "x", "y", "width", "height", "points", "color", "opacity", "blur", "blend", "z"}
                },
            )
        )

    style_data = _as_dict(data.get("style"))
    style = {
        "grain": _unit_float(style_data.get("grain", 0.0), default=0.0),
        "vignette": _unit_float(style_data.get("vignette", 0.0), default=0.0),
        "saturation": _unit_float(style_data.get("saturation", 0.0), default=0.0),
        "contrast": _unit_float(style_data.get("contrast", 0.0), default=0.0),
        "warmth": _unit_float(style_data.get("warmth", 0.0), default=0.0),
        "bloom": _unit_float(style_data.get("bloom", 0.0), default=0.0),
        "antialias": _unit_float(style_data.get("antialias", 0.0), default=0.0),
    }

    return ScenePlan(
        title=str(data.get("title", path.stem)),
        palette=palette,
        background=PlannedBackground(top=top, bottom=bottom, direction=background_direction, stops=background_stops),
        objects=tuple(objects),
        elements=tuple(sorted(elements, key=lambda element: element.z)),
        motifs=tuple(sorted(motifs, key=lambda motif: motif.z)),
        textures=tuple(sorted(textures, key=lambda texture: texture.z)),
        materials=tuple(sorted(materials, key=lambda material: material.z)),
        terrains=tuple(sorted(terrains, key=lambda terrain: terrain.z)),
        reflections=tuple(sorted(reflections, key=lambda reflection: reflection.z)),
        warps=tuple(sorted(warps, key=lambda warp: warp.z)),
        atmosphere=atmosphere,
        veils=tuple(sorted(veils, key=lambda veil: veil.z)),
        focus=focus,
        lights=tuple(sorted(lights, key=lambda light: light.z)),
        beams=tuple(sorted(beams, key=lambda beam: beam.z)),
        clouds=tuple(sorted(clouds, key=lambda cloud: cloud.z)),
        shadows=tuple(sorted(shadows, key=lambda shadow: shadow.z)),
        style=style,
        source_path=path,
    )


def _parse_color(value: Any) -> RGB:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in COLOR_RGB:
            return COLOR_RGB[lowered]
        if lowered.startswith("#") and len(lowered) == 7:
            return (int(lowered[1:3], 16), int(lowered[3:5], 16), int(lowered[5:7], 16))
    if isinstance(value, list | tuple) and len(value) == 3:
        return tuple(_channel(component) for component in value)  # type: ignore[return-value]
    raise ValueError(f"Unsupported color value: {value!r}")


def _parse_optional_color(value: Any) -> RGB | None:
    if value is None:
        return None
    return _parse_color(value)


def _parse_gradient(value: Any) -> PlannedGradient | None:
    gradient = _as_dict(value)
    if not gradient:
        return None

    raw_colors = gradient.get("colors")
    colors: list[RGB] = []
    if isinstance(raw_colors, list | tuple):
        for raw_color in raw_colors:
            try:
                colors.append(_parse_color(raw_color))
            except ValueError:
                continue
    elif "from" in gradient and "to" in gradient:
        colors = [_parse_color(gradient["from"]), _parse_color(gradient["to"])]
    if len(colors) < 2:
        return None

    kind = str(gradient.get("type", gradient.get("kind", "linear"))).strip().lower()
    if kind not in {"linear", "radial"}:
        kind = "linear"

    direction = str(gradient.get("direction", "vertical")).strip().lower()
    if direction not in {"vertical", "horizontal", "diagonal", "reverse-diagonal"}:
        direction = "vertical"

    center = (0.5, 0.5)
    raw_center = gradient.get("center")
    if isinstance(raw_center, list | tuple) and len(raw_center) >= 2:
        center = (_unit_float(raw_center[0], default=0.5), _unit_float(raw_center[1], default=0.5))

    return PlannedGradient(
        kind=kind,
        colors=tuple(colors[:4]),
        direction=direction,
        center=center,
        radius=_unit_float(gradient.get("radius", 1.0), default=1.0),
    )


def _parse_color_stops(value: Any, *, top: RGB, bottom: RGB) -> tuple[PlannedColorStop, ...]:
    stops: list[PlannedColorStop] = []
    if isinstance(value, list | tuple):
        color_count = len(value)
        for index, raw_stop in enumerate(value):
            try:
                if isinstance(raw_stop, dict):
                    position = _unit_float(raw_stop.get("at", raw_stop.get("position", index / max(1, color_count - 1))), default=0.0)
                    color = _parse_color(raw_stop.get("color"))
                else:
                    position = index / max(1, color_count - 1)
                    color = _parse_color(raw_stop)
            except (TypeError, ValueError):
                continue
            stops.append(PlannedColorStop(position=position, color=color))

    if len(stops) < 2:
        stops = [PlannedColorStop(position=0.0, color=top), PlannedColorStop(position=1.0, color=bottom)]

    return tuple(sorted(stops, key=lambda stop: stop.position)[:8])


def _parse_atmosphere(value: Any) -> PlannedAtmosphere | None:
    atmosphere = _as_dict(value)
    if not atmosphere:
        return None
    kind = str(atmosphere.get("type", atmosphere.get("kind", "horizon_fog"))).strip().lower() or "horizon_fog"
    return PlannedAtmosphere(
        kind=kind,
        label=str(atmosphere.get("label", kind)),
        color=_parse_color(atmosphere.get("color", (216, 232, 240))),
        horizon=_unit_float(atmosphere.get("horizon", 0.5), default=0.5),
        height=_unit_float(atmosphere.get("height", 0.22), default=0.22),
        strength=_unit_float(atmosphere.get("strength", 0.35), default=0.35),
    )


def _parse_focus(value: Any) -> PlannedFocus | None:
    focus = _as_dict(value)
    if not focus:
        return None
    kind = str(focus.get("type", focus.get("kind", "depth"))).strip().lower() or "depth"
    mode = str(focus.get("mode", "outside")).strip().lower()
    if mode not in {"outside", "inside"}:
        mode = "outside"
    return PlannedFocus(
        kind=kind,
        label=str(focus.get("label", kind)),
        region=_parse_region(focus.get("region", [0.0, 0.0, 1.0, 1.0])),
        blur=_unit_float(focus.get("blur", 0.0), default=0.0),
        falloff=_unit_float(focus.get("falloff", 0.08), default=0.08),
        mode=mode,
    )


def _parse_color_sequence(value: Any, fallback: tuple[RGB, ...]) -> tuple[RGB, ...]:
    colors: list[RGB] = []
    if isinstance(value, list | tuple):
        for raw_color in value:
            try:
                colors.append(_parse_color(raw_color))
            except ValueError:
                continue
    if not colors:
        colors = list(fallback[:2])
    if len(colors) == 1:
        colors.append(colors[0])
    return tuple(colors[:4])


def _parse_points(value: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list | tuple):
        return ()
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, list | tuple) and len(item) >= 2:
            points.append((_unit_float(item[0], default=0.0), _unit_float(item[1], default=0.0)))
    return tuple(points)


def _parse_path_commands(value: Any) -> tuple[PathCommand, ...]:
    if not isinstance(value, list | tuple):
        return ()
    commands: list[PathCommand] = []
    expected_points = {"M": 1, "L": 1, "Q": 2, "C": 3, "Z": 0}
    for raw_command in value:
        if not isinstance(raw_command, list | tuple) or not raw_command:
            continue
        command = str(raw_command[0]).strip().upper()
        if command not in expected_points:
            continue
        points = _parse_command_points(raw_command[1:], expected_points[command])
        if len(points) == expected_points[command]:
            commands.append((command, points))
    return tuple(commands)


def _parse_command_points(value: Any, expected_count: int) -> tuple[tuple[float, float], ...]:
    if expected_count == 0:
        return ()
    if not isinstance(value, list | tuple):
        return ()
    if len(value) == expected_count and all(isinstance(item, list | tuple) for item in value):
        return _parse_points(value)
    if len(value) >= expected_count * 2:
        return tuple(
            (_unit_float(value[index], default=0.0), _unit_float(value[index + 1], default=0.0))
            for index in range(0, expected_count * 2, 2)
        )
    return ()


def _parse_region(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, list | tuple) and len(value) >= 4:
        x0 = _unit_float(value[0], default=0.0)
        y0 = _unit_float(value[1], default=0.0)
        x1 = _unit_float(value[2], default=1.0)
        y1 = _unit_float(value[3], default=1.0)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    return 0.0, 0.0, 1.0, 1.0


def _channel(value: Any) -> int:
    return max(0, min(255, int(value)))


def _unit_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _blend_value(value: Any) -> str:
    blend = str(value).strip().lower()
    return blend if blend in {"normal", "screen", "multiply", "overlay", "soft-light"} else "normal"


def _direction_value(value: Any) -> str:
    direction = str(value).strip().lower()
    return direction if direction in {"vertical", "horizontal", "diagonal", "reverse-diagonal"} else "vertical"


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
