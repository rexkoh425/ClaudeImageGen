from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .palette import COLOR_RGB, RGB, blend
from .scene import SceneCandidate
from .scene_plan import (
    PlannedAtmosphere,
    PlannedBeam,
    PlannedCloud,
    PlannedElement,
    PlannedFocus,
    PlannedGradient,
    PlannedLight,
    PlannedMaterial,
    PlannedMotif,
    PlannedObject,
    PlannedReflection,
    PlannedShadow,
    PlannedTerrain,
    PlannedTexture,
    PlannedVeil,
    PlannedWarp,
    ScenePlan,
)

MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MAX_PIXELS = MAX_WIDTH * MAX_HEIGHT


def cap_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    scale = min(1.0, MAX_WIDTH / width, MAX_HEIGHT / height)
    capped_width = max(1, int(width * scale))
    capped_height = max(1, int(height * scale))

    pixels = capped_width * capped_height
    if pixels > MAX_PIXELS:
        pixel_scale = math.sqrt(MAX_PIXELS / pixels)
        capped_width = max(1, int(capped_width * pixel_scale))
        capped_height = max(1, int(capped_height * pixel_scale))

    return capped_width, capped_height


def render_candidate(candidate: SceneCandidate, width: int = MAX_WIDTH, height: int = MAX_HEIGHT) -> Image.Image:
    width, height = cap_dimensions(width, height)
    image = _gradient(width, height, candidate)
    draw = ImageDraw.Draw(image, "RGBA")
    objects = candidate.spec.objects

    if "cloud" in objects:
        _draw_clouds(draw, candidate, width, height)
    if "sun" in objects:
        _draw_sun(draw, candidate, width, height)
    if "moon" in objects:
        _draw_moon(draw, candidate, width, height)
    if "mountain" in objects:
        _draw_mountains(draw, candidate, width, height)
    if "ocean" in objects:
        _draw_ocean(draw, candidate, width, height)
    if "forest" in objects or "flower" in objects:
        _draw_botanical(draw, candidate, width, height)
    if "building" in objects:
        _draw_city(draw, candidate, width, height)
    if "portrait" in objects:
        _draw_portrait(draw, candidate, width, height)
    if "robot" in objects:
        _draw_robot(draw, candidate, width, height)
    if "abstract" in objects or len(objects) == 1:
        _draw_abstract(draw, candidate, width, height)

    if "cinematic" in candidate.spec.style_words:
        bar = max(4, height // 18)
        draw.rectangle((0, 0, width, bar), fill=(8, 8, 10, 210))
        draw.rectangle((0, height - bar, width, height), fill=(8, 8, 10, 210))
    if "watercolor" in candidate.spec.style_words or "dreamy" in candidate.spec.style_words:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.45))

    return image.convert("RGB")


def render_scene_plan(plan: ScenePlan, width: int = MAX_WIDTH, height: int = MAX_HEIGHT, seed: int = 0) -> Image.Image:
    width, height = cap_dimensions(width, height)
    antialias_scale = _antialias_scale(plan.style.get("antialias", 0.0))
    if antialias_scale > 1:
        image = _render_scene_plan_base(plan, width * antialias_scale, height * antialias_scale, seed)
        return image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")

    return _render_scene_plan_base(plan, width, height, seed).convert("RGB")


def _render_scene_plan_base(plan: ScenePlan, width: int, height: int, seed: int) -> Image.Image:
    image = _planned_gradient(width, height, plan)
    draw = ImageDraw.Draw(image, "RGBA")

    for obj in plan.objects:
        _draw_planned_object(draw, obj, width, height, seed)

    detail_layers: list[tuple[int, int, int, str, object]] = []
    detail_layers.extend((terrain.z, 0, index, "terrain", terrain) for index, terrain in enumerate(plan.terrains))
    detail_layers.extend((element.z, 1, index, "element", element) for index, element in enumerate(plan.elements))
    detail_layers.extend((motif.z, 2, index, "motif", motif) for index, motif in enumerate(plan.motifs))
    detail_layers.extend((texture.z, 3, index, "texture", texture) for index, texture in enumerate(plan.textures))
    detail_layers.extend((material.z, 4, index, "material", material) for index, material in enumerate(plan.materials))
    detail_layers.extend((reflection.z, 5, index, "reflection", reflection) for index, reflection in enumerate(plan.reflections))
    detail_layers.extend((warp.z, 6, index, "warp", warp) for index, warp in enumerate(plan.warps))
    detail_layers.extend((beam.z, 7, index, "beam", beam) for index, beam in enumerate(plan.beams))
    detail_layers.extend((cloud.z, 8, index, "cloud", cloud) for index, cloud in enumerate(plan.clouds))
    detail_layers.extend((shadow.z, 9, index, "shadow", shadow) for index, shadow in enumerate(plan.shadows))
    detail_layers.extend((veil.z, 10, index, "veil", veil) for index, veil in enumerate(plan.veils))

    for _, _, _, layer_kind, layer in sorted(detail_layers):
        if layer_kind == "terrain" and isinstance(layer, PlannedTerrain):
            image = _draw_planned_terrain(image, layer, width, height)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "element" and isinstance(layer, PlannedElement):
            image = _draw_planned_element(image, layer, width, height)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "motif" and isinstance(layer, PlannedMotif):
            _draw_planned_motif(draw, layer, width, height, seed)
        elif layer_kind == "texture" and isinstance(layer, PlannedTexture):
            image = _draw_planned_texture(image, layer, width, height, seed)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "reflection" and isinstance(layer, PlannedReflection):
            image = _draw_planned_reflection(image, layer, width, height)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "material" and isinstance(layer, PlannedMaterial):
            image = _draw_planned_material(image, layer, width, height, seed)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "warp" and isinstance(layer, PlannedWarp):
            image = _draw_planned_warp(image, layer, width, height, seed)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "beam" and isinstance(layer, PlannedBeam):
            image = _draw_planned_beam(image, layer, width, height, seed)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "cloud" and isinstance(layer, PlannedCloud):
            image = _draw_planned_cloud_layer(image, layer, width, height, seed)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "shadow" and isinstance(layer, PlannedShadow):
            image = _draw_planned_shadow(image, layer, width, height)
            draw = ImageDraw.Draw(image, "RGBA")
        elif layer_kind == "veil" and isinstance(layer, PlannedVeil):
            image = _draw_planned_veil(image, layer, width, height)
            draw = ImageDraw.Draw(image, "RGBA")

    if plan.atmosphere:
        image = _apply_atmosphere(image, plan.atmosphere)

    if plan.lights:
        image = _apply_lights(image, plan.lights)

    if plan.focus and plan.focus.blur > 0:
        image = _apply_focus_blur(image, plan.focus)

    if plan.style.get("bloom", 0.0) > 0:
        image = _apply_bloom(image, plan.style["bloom"])
    if any(plan.style.get(key, 0.0) > 0 for key in ("saturation", "contrast", "warmth")):
        image = _apply_color_grade(
            image,
            saturation=plan.style.get("saturation", 0.0),
            contrast=plan.style.get("contrast", 0.0),
            warmth=plan.style.get("warmth", 0.0),
        )
    if any(plan.style.get(key, 0.0) > 0 for key in ("detail", "sharpen")):
        image = _apply_detail_enhancement(
            image,
            detail=plan.style.get("detail", 0.0),
            sharpen=plan.style.get("sharpen", 0.0),
        )
    if plan.style.get("grain", 0.0) > 0:
        image = _add_grain(image, plan.style["grain"], seed)
    if plan.style.get("vignette", 0.0) > 0:
        image = _apply_vignette(image, plan.style["vignette"])

    return image.convert("RGB")


def _antialias_scale(amount: float) -> int:
    return 2 if amount > 0 else 1


def _apply_detail_enhancement(image: Image.Image, *, detail: float, sharpen: float) -> Image.Image:
    detail_amount = max(0.0, min(1.0, detail))
    sharpen_amount = max(0.0, min(1.0, sharpen))
    enhanced = image.convert("RGB")
    if detail_amount > 0:
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(
                radius=0.8 + detail_amount * 1.4,
                percent=int(80 + detail_amount * 240),
                threshold=max(0, int(round(5 - detail_amount * 4))),
            )
        )
    if sharpen_amount > 0:
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(
                radius=0.6 + sharpen_amount * 0.8,
                percent=int(70 + sharpen_amount * 180),
                threshold=max(0, int(round(4 - sharpen_amount * 3))),
            )
        )
    return enhanced.convert("RGB")


def _gradient(width: int, height: int, candidate: SceneCandidate) -> Image.Image:
    palette = candidate.palette
    top = blend(palette[0], (238, 244, 250), 0.42)
    bottom = palette[1 % len(palette)]
    if "ocean" in candidate.spec.objects:
        bottom = blend(COLOR_RGB["blue"], bottom, 0.25)
    if "forest" in candidate.spec.objects or "flower" in candidate.spec.objects:
        bottom = blend(COLOR_RGB["green"], bottom, 0.35)

    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / max(1, height - 1)
        color = blend(top, bottom, t)
        draw.line((0, y, width, y), fill=color)
    return image


def _planned_gradient(width: int, height: int, plan: ScenePlan) -> Image.Image:
    stops = plan.background.stops
    positions = np.array([stop.position for stop in stops], dtype=np.float32)
    colors = np.array([stop.color for stop in stops], dtype=np.float32)
    if plan.background.direction == "horizontal":
        t = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        t = np.repeat(t, height, axis=0)
    elif plan.background.direction == "diagonal":
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        t = (x + y) / 2.0
    elif plan.background.direction == "reverse-diagonal":
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(1.0, 0.0, width, dtype=np.float32)[None, :]
        t = (x + y) / 2.0
    else:
        t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        t = np.repeat(t, width, axis=1)

    flat_t = t.reshape(-1)
    channels = [np.interp(flat_t, positions, colors[:, channel]) for channel in range(3)]
    array = np.stack(channels, axis=1).reshape(height, width, 3)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _draw_planned_object(
    draw: ImageDraw.ImageDraw,
    obj: PlannedObject,
    width: int,
    height: int,
    seed: int,
) -> None:
    kind = obj.kind
    alpha = max(0, min(255, int(round(obj.opacity * 255))))
    color = (*obj.color, alpha)

    if kind == "sun":
        radius = max(6, int(min(width, height) * obj.size))
        cx, cy = int(width * obj.x), int(height * obj.y)
        draw.ellipse((cx - radius * 1.45, cy - radius * 1.45, cx + radius * 1.45, cy + radius * 1.45), fill=(*blend(obj.color, COLOR_RGB["yellow"], 0.55), min(150, alpha)))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        return

    if kind == "moon":
        radius = max(5, int(min(width, height) * obj.size))
        cx, cy = int(width * obj.x), int(height * obj.y)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        draw.ellipse((cx - radius // 3, cy - radius, cx + radius, cy + radius), fill=(28, 38, 62, min(180, alpha)))
        return

    if kind in {"greenhouse", "glasshouse", "conservatory", "atrium"}:
        _draw_planned_greenhouse(draw, obj, width, height)
        return

    if kind in {"lamp", "lamps", "light", "lights", "pendant"}:
        _draw_planned_lamps(draw, obj, width, height)
        return

    if kind in {"plant", "plants", "foliage", "leaf", "leaves", "tropical"}:
        _draw_planned_plants(draw, obj, width, height, seed)
        return

    if kind in {"floor", "stone", "tile", "tiles", "wet-floor", "wet_floor"}:
        _draw_planned_floor(draw, obj, width, height)
        return

    if kind in {"ocean", "water", "lake"}:
        y0 = int(height * obj.y)
        draw.rectangle((0, y0, width, height), fill=(*blend(obj.color, (18, 70, 125), 0.20), alpha))
        for i in range(10):
            y = y0 + int((height - y0) * (i + 1) / 12)
            draw.arc((-width * 0.08, y - 18, width * 1.08, y + 22), 0, 180, fill=(235, 245, 250, 85), width=2)
        return

    if kind in {"foreground", "ground"}:
        y0 = int(height * obj.y)
        draw.rectangle((0, y0, width, height), fill=color)
        return

    if kind in {"mountain", "mountains"}:
        _draw_planned_mountains(draw, obj, width, height, seed)
        return

    if kind in {"cloud", "clouds"}:
        _draw_planned_cloud(draw, obj, width, height)
        return

    if kind in {"tree", "forest"}:
        _draw_planned_trees(draw, obj, width, height, seed)
        return

    if kind in {"building", "city", "skyline"}:
        _draw_planned_buildings(draw, obj, width, height, seed)
        return

    if kind in {"circle", "ellipse"}:
        radius = max(4, int(min(width, height) * obj.size))
        cx, cy = int(width * obj.x), int(height * obj.y)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        return

    if kind in {"rect", "rectangle"}:
        w = max(6, int(width * obj.size))
        h = max(6, int(height * obj.size))
        cx, cy = int(width * obj.x), int(height * obj.y)
        draw.rectangle((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), fill=color)
        return

    if kind == "triangle":
        radius = max(6, int(min(width, height) * obj.size))
        cx, cy = int(width * obj.x), int(height * obj.y)
        draw.polygon([(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)], fill=color)


def _draw_planned_element(
    image: Image.Image,
    element: PlannedElement,
    width: int,
    height: int,
) -> Image.Image:
    if element.gradient or element.blur > 0 or element.blend != "normal":
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        _draw_planned_element_direct(overlay, element, width, height)
        if element.blur > 0:
            overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * element.blur)))
        return _composite_overlay(image, overlay, element.blend)

    _draw_planned_element_direct(image, element, width, height)
    return image


def _draw_planned_element_direct(
    image: Image.Image,
    element: PlannedElement,
    width: int,
    height: int,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    fill = _rgba(element.fill, element.opacity) if element.fill else None
    stroke = _rgba(element.stroke, element.opacity) if element.stroke else fill
    line_width = _element_line_width(element, width, height)

    if element.kind in {"polyline", "line"}:
        points = _scaled_points(element.points, width, height)
        if len(points) >= 2 and stroke:
            draw.line(points, fill=stroke, width=line_width, joint="curve")
        return

    if element.kind == "arrow":
        points = _scaled_points(element.points, width, height)
        if len(points) >= 2 and stroke:
            _draw_element_arrow(draw, points, stroke=stroke, line_width=line_width)
        return

    if element.kind in {"text", "label"}:
        _draw_element_text(draw, element, width, height, fill=fill or stroke)
        return

    if element.kind == "path":
        points = _sample_path_commands(element.commands, width, height)
        if len(points) >= 2:
            if _path_is_closed(element.commands):
                if element.gradient:
                    _apply_element_gradient(image, element.gradient, _shape_mask(image.size, element.opacity, polygon=points), _point_bounds(points))
                elif element.fill:
                    draw.polygon(points, fill=fill)
            if stroke:
                draw.line(points, fill=stroke, width=line_width, joint="curve")
        return

    if element.kind == "polygon":
        points = _scaled_points(element.points, width, height)
        if len(points) >= 3:
            if element.gradient:
                _apply_element_gradient(image, element.gradient, _shape_mask(image.size, element.opacity, polygon=points), _point_bounds(points))
            else:
                draw.polygon(points, fill=fill)
            if stroke:
                draw.line([*points, points[0]], fill=stroke, width=line_width)
        return

    if element.kind in {"ellipse", "circle"}:
        bbox = _element_bbox(element, width, height)
        if element.gradient:
            _apply_element_gradient(image, element.gradient, _shape_mask(image.size, element.opacity, ellipse=bbox), bbox)
            if stroke:
                draw.ellipse(bbox, outline=stroke, width=line_width)
        else:
            draw.ellipse(bbox, fill=fill, outline=stroke, width=line_width if stroke else 1)
        return

    if element.kind in {"rounded_rectangle", "rounded-rectangle", "roundrect"}:
        bbox = _element_bbox(element, width, height)
        radius = _element_corner_radius(element, bbox, width, height)
        if element.gradient:
            _apply_element_gradient(
                image,
                element.gradient,
                _shape_mask(image.size, element.opacity, rounded_rectangle=(bbox, radius)),
                bbox,
            )
            if stroke:
                draw.rounded_rectangle(bbox, radius=radius, outline=stroke, width=line_width)
        else:
            draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=stroke, width=line_width if stroke else 1)
        return

    if element.kind == "aperture":
        _draw_element_aperture(draw, element, width, height, fill=fill, stroke=stroke, line_width=line_width)
        return

    if element.kind in {"rectangle", "rect"}:
        bbox = _element_bbox(element, width, height)
        if element.gradient:
            _apply_element_gradient(image, element.gradient, _shape_mask(image.size, element.opacity, rectangle=bbox), bbox)
            if stroke:
                draw.rectangle(bbox, outline=stroke, width=line_width)
        else:
            draw.rectangle(bbox, fill=fill, outline=stroke, width=line_width if stroke else 1)
        return

    if element.kind == "glow":
        _draw_element_glow(draw, element, width, height)
        return

    if element.kind == "sparkle":
        _draw_element_sparkle(draw, element, width, height, fill=fill, stroke=stroke, line_width=line_width)
        return

    if element.kind == "arc":
        bbox = _element_bbox(element, width, height)
        start = float(element.extra.get("start", 0))
        end = float(element.extra.get("end", 180))
        if stroke:
            draw.arc(bbox, start=start, end=end, fill=stroke, width=line_width)


def _draw_element_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    stroke: tuple[int, int, int, int],
    line_width: int,
) -> None:
    draw.line(points, fill=stroke, width=line_width, joint="curve")
    end = points[-1]
    start = points[-2]
    for candidate_start, candidate_end in zip(reversed(points[:-1]), reversed(points[1:])):
        if candidate_start != candidate_end:
            start = candidate_start
            end = candidate_end
            break
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_length = max(line_width * 3.5, 8.0)
    head_spread = math.radians(28.0)
    left = (
        int(round(end[0] - (head_length * math.cos(angle - head_spread)))),
        int(round(end[1] - (head_length * math.sin(angle - head_spread)))),
    )
    right = (
        int(round(end[0] - (head_length * math.cos(angle + head_spread)))),
        int(round(end[1] - (head_length * math.sin(angle + head_spread)))),
    )
    draw.polygon([end, left, right], fill=stroke)


def _draw_element_text(
    draw: ImageDraw.ImageDraw,
    element: PlannedElement,
    width: int,
    height: int,
    *,
    fill: tuple[int, int, int, int] | None,
) -> None:
    text = str(element.extra.get("text") or element.label or "").strip()
    if not text or fill is None:
        return
    font_size = max(8, int(round(min(width, height) * max(0.035, element.height))))
    font = _load_diagram_font(font_size)
    x = int(round(element.x * width))
    y = int(round(element.y * height))
    stroke_fill = _rgba(element.stroke, element.opacity) if element.stroke else None
    stroke_width = max(0, int(round(font_size * 0.08))) if stroke_fill else 0
    anchor = str(element.extra.get("anchor", "mm"))
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _load_diagram_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_element_sparkle(
    draw: ImageDraw.ImageDraw,
    element: PlannedElement,
    width: int,
    height: int,
    *,
    fill: tuple[int, int, int, int] | None,
    stroke: tuple[int, int, int, int] | None,
    line_width: int,
) -> None:
    color = fill or stroke
    if color is None:
        return

    cx = int(round(element.x * width))
    cy = int(round(element.y * height))
    outer_x = max(2, int(round(element.width * width / 2)))
    outer_y = max(2, int(round(element.height * height / 2)))
    inner_ratio = _float_extra(element.extra, "inner", 0.34)
    inner_ratio = max(0.12, min(0.65, inner_ratio))
    inner_x = max(1, int(round(outer_x * inner_ratio)))
    inner_y = max(1, int(round(outer_y * inner_ratio)))
    points = [
        (cx, cy - outer_y),
        (cx + inner_x, cy - inner_y),
        (cx + outer_x, cy),
        (cx + inner_x, cy + inner_y),
        (cx, cy + outer_y),
        (cx - inner_x, cy + inner_y),
        (cx - outer_x, cy),
        (cx - inner_x, cy - inner_y),
    ]
    draw.polygon(points, fill=color)
    if stroke and stroke != color:
        draw.line([*points, points[0]], fill=stroke, width=line_width, joint="curve")


def _draw_element_aperture(
    draw: ImageDraw.ImageDraw,
    element: PlannedElement,
    width: int,
    height: int,
    *,
    fill: tuple[int, int, int, int] | None,
    stroke: tuple[int, int, int, int] | None,
    line_width: int,
) -> None:
    color = stroke or fill
    if color is None:
        return

    bbox = _element_bbox(element, width, height)
    draw.ellipse(bbox, fill=fill, outline=color, width=line_width)
    left, top, right, bottom = bbox
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    outer_rx = max(2.0, (right - left) / 2 - line_width)
    outer_ry = max(2.0, (bottom - top) / 2 - line_width)
    inner_ratio = max(0.22, min(0.52, _float_extra(element.extra, "inner", 0.34)))
    inner_rx = outer_rx * inner_ratio
    inner_ry = outer_ry * inner_ratio
    inner_bbox = (
        int(round(cx - inner_rx)),
        int(round(cy - inner_ry)),
        int(round(cx + inner_rx)),
        int(round(cy + inner_ry)),
    )

    try:
        blades = int(element.extra.get("blades", 6))
    except (TypeError, ValueError):
        blades = 6
    blades = max(5, min(9, blades))
    try:
        rotation = math.radians(float(element.extra.get("rotation", -90.0)))
    except (TypeError, ValueError):
        rotation = math.radians(-90.0)
    step = (math.pi * 2) / blades
    for index in range(blades):
        angle = rotation + step * index
        outer_angle = angle + step * 0.68
        inner_point = (
            int(round(cx + math.cos(angle) * inner_rx)),
            int(round(cy + math.sin(angle) * inner_ry)),
        )
        outer_point = (
            int(round(cx + math.cos(outer_angle) * outer_rx)),
            int(round(cy + math.sin(outer_angle) * outer_ry)),
        )
        draw.line([inner_point, outer_point], fill=color, width=line_width, joint="curve")

    draw.ellipse(inner_bbox, fill=fill, outline=color, width=line_width)


def _draw_planned_terrain(
    image: Image.Image,
    terrain: PlannedTerrain,
    width: int,
    height: int,
) -> Image.Image:
    ridge_points = sorted(_scaled_points(terrain.points, width, height))
    if len(ridge_points) < 2 or terrain.opacity <= 0:
        return image

    base_y = int(terrain.base * height)
    left_x = ridge_points[0][0]
    right_x = ridge_points[-1][0]
    silhouette = [*ridge_points, (right_x, base_y), (left_x, base_y)]

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    fill = _rgba(terrain.fill, terrain.opacity)
    if fill:
        draw.polygon(silhouette, fill=fill)

    if terrain.facets:
        shade = terrain.shade or blend(terrain.fill, (0, 0, 0), 0.45)
        highlight = terrain.highlight or blend(terrain.fill, (255, 255, 255), 0.32)
        for index, (start, end) in enumerate(zip(ridge_points, ridge_points[1:])):
            color = shade if index % 2 == 0 else highlight
            facet = [start, end, (end[0], base_y), (start[0], base_y)]
            draw.polygon(facet, fill=_rgba(color, terrain.opacity))

    if terrain.blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * terrain.blur)))
    return _composite_overlay(image, overlay, terrain.blend)


def _draw_planned_motif(
    draw: ImageDraw.ImageDraw,
    motif: PlannedMotif,
    width: int,
    height: int,
    seed: int,
) -> None:
    rng = random.Random(seed * 1009 + motif.seed * 9176 + motif.z)
    x0, y0, x1, y1 = motif.region
    left, top = int(x0 * width), int(y0 * height)
    right, bottom = int(x1 * width), int(y1 * height)
    color = _rgba(motif.color, motif.opacity) or (*motif.color, 255)

    if motif.kind in {"starfield", "stars", "sparkles"}:
        radius = max(1, int(min(width, height) * motif.size))
        for _ in range(motif.count):
            x = rng.randint(left, max(left, right - 1))
            y = rng.randint(top, max(top, bottom - 1))
            draw.line((x - radius, y, x + radius, y), fill=color, width=1)
            draw.line((x, y - radius, x, y + radius), fill=color, width=1)
            draw.point((x, y), fill=(*motif.color, 255))
        return

    if motif.kind in {"grass", "grass_blades", "reeds"}:
        blade_height = max(3, int(height * motif.size))
        line_width = max(1, int(min(width, height) * motif.size * 0.12))
        for _ in range(motif.count):
            base_x = rng.randint(left, max(left, right - 1))
            base_y = rng.randint(top, max(top, bottom - 1))
            lean = rng.randint(-blade_height // 2, blade_height // 2)
            tip_y = max(top, base_y - rng.randint(blade_height // 2, blade_height))
            draw.line((base_x, base_y, base_x + lean, tip_y), fill=color, width=line_width)
        return

    if motif.kind in {"rain", "streaks"}:
        length = max(4, int(height * motif.size))
        line_width = max(1, int(min(width, height) * motif.size * 0.08))
        slant = int(length * 0.25)
        for _ in range(motif.count):
            x = rng.randint(left, max(left, right - 1))
            y = rng.randint(top, max(top, bottom - 1))
            draw.line((x, y, x + slant, y + length), fill=color, width=line_width)
        return

    if motif.kind in {"window_lights", "windows"}:
        box_w = max(2, int(width * motif.size))
        box_h = max(2, int(height * motif.size * 0.65))
        for _ in range(motif.count):
            x = rng.randint(left, max(left, right - box_w))
            y = rng.randint(top, max(top, bottom - box_h))
            draw.rectangle((x, y, x + box_w, y + box_h), fill=color)
        return

    dot_radius = max(1, int(min(width, height) * motif.size))
    for _ in range(motif.count):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=color)


def _draw_planned_texture(
    image: Image.Image,
    texture: PlannedTexture,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    left, top, right, bottom = _scaled_region(texture.region, width, height)
    if right <= left or bottom <= top or texture.opacity <= 0:
        return image

    if texture.kind in {"paper", "grain", "noise", "mist"}:
        return _apply_noise_texture(image, texture, (left, top, right, bottom), seed)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rng = random.Random(seed * 1301 + texture.seed * 7919 + texture.z)

    if texture.kind in {"hatching", "crosshatch", "contour"}:
        _draw_texture_hatching(draw, texture, (left, top, right, bottom), rng)
        if texture.kind == "crosshatch":
            _draw_texture_hatching(draw, texture, (left, top, right, bottom), rng, angle_offset=78.0)
    elif texture.kind in {"ripple", "ripples", "wavelets"}:
        _draw_texture_ripples(draw, texture, (left, top, right, bottom), rng)
    else:
        _draw_texture_speckles(draw, texture, (left, top, right, bottom), rng)

    return _composite_overlay(image, overlay, texture.blend)


def _draw_planned_material(
    image: Image.Image,
    material: PlannedMaterial,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    left, top, right, bottom = _scaled_region(material.region, width, height)
    if right <= left or bottom <= top or material.opacity <= 0:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    _draw_material_gradient(overlay, material, (left, top, right, bottom))
    draw = ImageDraw.Draw(overlay, "RGBA")
    rng = random.Random(seed * 1777 + material.seed * 2801 + material.z)

    if material.kind in {"water", "ocean", "lake", "river"}:
        _draw_material_water(draw, material, (left, top, right, bottom), rng)
    elif material.kind in {"foliage", "grass", "forest", "leaf", "leaves"}:
        _draw_material_foliage(draw, material, (left, top, right, bottom), rng)
    elif material.kind in {"metal", "steel", "chrome"}:
        _draw_material_metal(draw, material, (left, top, right, bottom), rng)
    elif material.kind in {"stone", "floor", "tile", "tiles", "wet-stone", "wet_stone"} or "stone" in material.label.lower() or "floor" in material.label.lower():
        _draw_material_stone(draw, material, (left, top, right, bottom), rng)
    else:
        _draw_material_surface(draw, material, (left, top, right, bottom), rng)

    return _composite_overlay(image, overlay, "normal")


def _draw_planned_reflection(
    image: Image.Image,
    reflection: PlannedReflection,
    width: int,
    height: int,
) -> Image.Image:
    source_box = _scaled_region(reflection.source, width, height)
    target_box = _scaled_region(reflection.target, width, height)
    source_left, source_top, source_right, source_bottom = source_box
    target_left, target_top, target_right, target_bottom = target_box
    target_width = target_right - target_left
    target_height = target_bottom - target_top
    if source_right <= source_left or source_bottom <= source_top or target_width <= 0 or target_height <= 0 or reflection.opacity <= 0:
        return image

    patch = image.convert("RGB").crop(source_box)
    if reflection.kind in {"horizontal", "left_right", "side"}:
        patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    patch = patch.resize((target_width, target_height), Image.Resampling.BICUBIC)
    if reflection.tint is not None:
        tint_strength = _float_extra(reflection.extra, "tint_strength", default=0.35)
        tint = Image.new("RGB", patch.size, reflection.tint)
        patch = Image.blend(patch, tint, tint_strength)
    if reflection.blur > 0:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * reflection.blur)))

    alpha = np.full((target_height, target_width), reflection.opacity * 255.0, dtype=np.float32)
    if reflection.fade > 0:
        if reflection.kind in {"horizontal", "left_right", "side"}:
            ramp = np.linspace(1.0, 1.0 - reflection.fade, target_width, dtype=np.float32)[None, :]
        else:
            ramp = np.linspace(1.0, 1.0 - reflection.fade, target_height, dtype=np.float32)[:, None]
        alpha *= np.clip(ramp, 0.0, 1.0)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay.paste(patch.convert("RGBA"), (target_left, target_top), Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8), "L"))
    return _composite_overlay(image, overlay, reflection.blend)


def _draw_planned_warp(
    image: Image.Image,
    warp: PlannedWarp,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    left, top, right, bottom = _scaled_region(warp.region, width, height)
    region_width = right - left
    region_height = bottom - top
    if region_width <= 1 or region_height <= 1 or warp.amplitude <= 0:
        return image

    region = np.asarray(image.convert("RGB").crop((left, top, right, bottom)), dtype=np.uint8)
    warped = np.empty_like(region)
    jitter = _float_extra(warp.extra, "jitter", default=0.0)

    if warp.direction == "vertical":
        line_count = region_width
        amplitude_px = max(1, int(round(region_height * warp.amplitude)))
        offsets = _warp_offsets(line_count, amplitude_px, warp, seed, jitter)
        for x, offset in enumerate(offsets):
            warped[:, x, :] = np.roll(region[:, x, :], int(offset), axis=0)
    else:
        line_count = region_height
        amplitude_px = max(1, int(round(region_width * warp.amplitude)))
        offsets = _warp_offsets(line_count, amplitude_px, warp, seed, jitter)
        for y, offset in enumerate(offsets):
            warped[y, :, :] = np.roll(region[y, :, :], int(offset), axis=0)

    output = image.convert("RGB").copy()
    output.paste(Image.fromarray(warped, "RGB"), (left, top))
    return output


def _warp_offsets(line_count: int, amplitude_px: int, warp: PlannedWarp, seed: int, jitter: float) -> np.ndarray:
    wavelength_px = max(2.0, line_count * max(0.05, warp.wavelength))
    positions = np.arange(line_count, dtype=np.float32)
    offsets = np.sin((positions / wavelength_px) * math.tau + warp.phase * math.tau) * amplitude_px
    if jitter > 0:
        rng = np.random.default_rng(seed * 1997 + warp.seed * 3253 + warp.z)
        offsets += rng.normal(0.0, max(0.25, amplitude_px * jitter), line_count)
    return np.rint(offsets).astype(np.int16)


def _draw_planned_beam(
    image: Image.Image,
    beam: PlannedBeam,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    if beam.opacity <= 0 or beam.count <= 0 or beam.length <= 0:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    origin = (beam.x * width, beam.y * height)
    length_px = max(1.0, beam.length * max(width, height))
    centers = _beam_centers(beam, seed)
    beam_width = max(1.0, beam.spread / max(1, beam.count) * 0.72)
    alpha = max(0, min(255, int(round(beam.opacity * 255))))

    for center_angle in centers:
        half = math.radians(beam_width / 2.0)
        angle = math.radians(center_angle)
        left = (
            origin[0] + math.cos(angle - half) * length_px,
            origin[1] + math.sin(angle - half) * length_px,
        )
        right = (
            origin[0] + math.cos(angle + half) * length_px,
            origin[1] + math.sin(angle + half) * length_px,
        )
        draw.polygon([origin, left, right], fill=(*beam.color, alpha))

    _apply_beam_occlusions(overlay, beam, width, height)

    if beam.blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * beam.blur)))
    return _composite_overlay(image, overlay, beam.blend)


def _apply_beam_occlusions(overlay: Image.Image, beam: PlannedBeam, width: int, height: int) -> None:
    raw_occlusions = beam.extra.get("occlusions")
    if not isinstance(raw_occlusions, list | tuple):
        return

    draw = ImageDraw.Draw(overlay, "RGBA")
    for raw_occlusion in raw_occlusions:
        if not isinstance(raw_occlusion, dict):
            continue
        kind = str(raw_occlusion.get("type", raw_occlusion.get("kind", "rectangle"))).strip().lower()
        if kind in {"polygon", "shape"}:
            points = _normalized_points(raw_occlusion.get("points"), width, height)
            if len(points) >= 3:
                draw.polygon(points, fill=(0, 0, 0, 0))
            continue

        region = _normalized_region(raw_occlusion.get("region"))
        if region is None:
            continue
        bounds = _scaled_region(region, width, height)
        if kind in {"ellipse", "circle"}:
            draw.ellipse(bounds, fill=(0, 0, 0, 0))
        else:
            draw.rectangle(bounds, fill=(0, 0, 0, 0))


def _beam_centers(beam: PlannedBeam, seed: int) -> list[float]:
    if beam.count == 1:
        return [beam.angle]
    rng = random.Random(seed * 2371 + beam.seed * 3067 + beam.z)
    start = beam.angle - beam.spread / 2.0
    step = beam.spread / max(1, beam.count - 1)
    jitter = float(beam.extra.get("jitter", 0.0)) if isinstance(beam.extra.get("jitter", 0.0), int | float) else 0.0
    return [start + index * step + rng.uniform(-jitter, jitter) for index in range(beam.count)]


def _draw_planned_cloud_layer(
    image: Image.Image,
    cloud: PlannedCloud,
    width: int,
    height: int,
    seed: int,
) -> Image.Image:
    left, top, right, bottom = _scaled_region(cloud.region, width, height)
    region_width = right - left
    region_height = bottom - top
    if region_width <= 1 or region_height <= 1 or cloud.opacity <= 0:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    centers = _cloud_centers((left, top, right, bottom), cloud, seed)
    base_radius = max(3, int(min(width, height) * max(0.02, cloud.scale) * 0.42))
    alpha = max(0, min(255, int(round(cloud.opacity * 255))))
    shadow = cloud.shadow or blend(cloud.color, (60, 72, 96), 0.45)

    for cloud_index, (cx, cy) in enumerate(centers):
        rng = random.Random(seed * 3413 + cloud.seed * 1699 + cloud_index * 97 + cloud.z)
        width_scale = rng.uniform(1.25, 1.75)
        height_scale = rng.uniform(0.62, 0.92)
        shadow_alpha = max(0, min(255, int(alpha * 0.55)))
        if shadow_alpha:
            shadow_box = (
                cx - base_radius * width_scale,
                cy - base_radius * 0.04,
                cx + base_radius * width_scale,
                cy + base_radius * height_scale,
            )
            draw.ellipse(shadow_box, fill=(*shadow, shadow_alpha))

        for lobe in range(cloud.lobes):
            if cloud.lobes == 1:
                offset_t = 0.0
            else:
                offset_t = (lobe / (cloud.lobes - 1)) * 2.0 - 1.0
            lobe_x = cx + offset_t * base_radius * width_scale * rng.uniform(0.42, 0.72)
            lobe_y = cy + rng.uniform(-0.32, 0.22) * base_radius
            rx = base_radius * rng.uniform(0.78, 1.28)
            ry = base_radius * rng.uniform(0.50, 0.95)
            draw.ellipse((lobe_x - rx, lobe_y - ry, lobe_x + rx, lobe_y + ry), fill=(*cloud.color, alpha))

    if cloud.blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * cloud.blur)))
    return _composite_overlay(image, overlay, cloud.blend)


def _draw_planned_shadow(
    image: Image.Image,
    shadow: PlannedShadow,
    width: int,
    height: int,
) -> Image.Image:
    if shadow.opacity <= 0:
        return image

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    fill = _rgba(shadow.color, shadow.opacity)
    if not fill:
        return image

    if shadow.kind in {"polygon", "shape"} and len(shadow.points) >= 3:
        draw.polygon(_scaled_points(shadow.points, width, height), fill=fill)
    elif shadow.kind in {"rectangle", "rect"}:
        draw.rectangle(_shadow_bbox(shadow, width, height), fill=fill)
    else:
        draw.ellipse(_shadow_bbox(shadow, width, height), fill=fill)

    if shadow.blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * shadow.blur)))
    return _composite_overlay(image, overlay, shadow.blend)


def _draw_planned_veil(
    image: Image.Image,
    veil: PlannedVeil,
    width: int,
    height: int,
) -> Image.Image:
    if veil.opacity <= 0:
        return image

    left, top, right, bottom = _scaled_region(veil.region, width, height)
    region_w = max(0, right - left)
    region_h = max(0, bottom - top)
    if region_w <= 1 or region_h <= 1:
        return image

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[top:bottom, left:right, :3] = np.array(veil.color, dtype=np.uint8)
    overlay[top:bottom, left:right, 3] = _veil_alpha(region_w, region_h, veil)

    overlay_image = Image.fromarray(overlay, "RGBA")
    if veil.blur > 0:
        overlay_image = overlay_image.filter(ImageFilter.GaussianBlur(radius=max(0.0, min(width, height) * veil.blur)))
    return _composite_overlay(image, overlay_image, veil.blend)


def _veil_alpha(region_w: int, region_h: int, veil: PlannedVeil) -> np.ndarray:
    base_alpha = max(0, min(255, int(round(veil.opacity * 255))))
    if base_alpha <= 0:
        return np.zeros((region_h, region_w), dtype=np.uint8)

    if veil.falloff <= 0:
        return np.full((region_h, region_w), base_alpha, dtype=np.uint8)

    x = np.linspace(0.0, 1.0, region_w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, region_h, dtype=np.float32)[:, None]
    if veil.direction == "horizontal":
        axis = np.broadcast_to(x, (region_h, region_w))
    elif veil.direction == "diagonal":
        axis = (x + y) / 2.0
    elif veil.direction == "reverse-diagonal":
        axis = ((1.0 - x) + y) / 2.0
    else:
        axis = np.broadcast_to(y, (region_h, region_w))

    profile = np.clip(np.minimum(axis, 1.0 - axis) / max(0.001, veil.falloff), 0.0, 1.0)
    return np.clip(profile * base_alpha, 0, 255).astype(np.uint8)


def _shadow_bbox(shadow: PlannedShadow, width: int, height: int) -> tuple[int, int, int, int]:
    cx = int(shadow.x * width)
    cy = int(shadow.y * height)
    half_w = max(1, int(shadow.width * width / 2))
    half_h = max(1, int(shadow.height * height / 2))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _cloud_centers(bounds: tuple[int, int, int, int], cloud: PlannedCloud, seed: int) -> list[tuple[float, float]]:
    left, top, right, bottom = bounds
    if cloud.count == 1:
        return [((left + right) / 2.0, (top + bottom) / 2.0)]

    rng = random.Random(seed * 2659 + cloud.seed * 1237 + cloud.z)
    width = max(1, right - left)
    height = max(1, bottom - top)
    centers: list[tuple[float, float]] = []
    for index in range(cloud.count):
        x_t = (index + 0.5) / cloud.count
        x = left + width * x_t + rng.uniform(-0.18, 0.18) * width / cloud.count
        y = top + height * rng.uniform(0.28, 0.72)
        centers.append((x, y))
    return centers


def _draw_material_gradient(
    overlay: Image.Image,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = bounds
    region_h = max(1, bottom - top)
    draw = ImageDraw.Draw(overlay, "RGBA")
    for y in range(top, bottom):
        t = (y - top) / max(1, region_h - 1)
        color = _multi_color_blend(material.colors, t)
        alpha = int(255 * material.opacity * (0.58 + material.intensity * 0.34))
        draw.line((left, y, right, y), fill=(*color, max(0, min(255, alpha))))


def _draw_material_water(
    draw: ImageDraw.ImageDraw,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    ripple_count = max(6, int(region_h * (0.18 + material.intensity * 0.55)))
    segment = max(8, int(region_w * max(0.015, material.scale) * 2.5))
    amplitude = max(1, int(region_h * max(0.008, material.scale) * 0.65))
    light = blend(material.colors[0], COLOR_RGB["cyan"], 0.20)
    alpha = max(40, min(210, int(255 * material.opacity * (0.26 + material.intensity * 0.28))))

    for index in range(ripple_count):
        y = top + int((index + 0.5) * region_h / ripple_count) + rng.randint(-amplitude, amplitude)
        points: list[tuple[int, int]] = []
        phase = rng.random() * math.tau
        for x in range(left, right + segment, segment):
            wave = int(math.sin((x - left) / max(1, segment) + phase) * amplitude)
            points.append((max(left, min(right - 1, x)), max(top, min(bottom - 1, y + wave))))
        if len(points) >= 2:
            draw.line(points, fill=(*light, alpha), width=max(1, int(material.scale * min(region_w, region_h) * 0.45)), joint="curve")

    glints = max(8, int(region_w * region_h * material.intensity * 0.006))
    for _ in range(min(glints, 240)):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        radius = max(1, int(min(region_w, region_h) * material.scale * rng.uniform(0.08, 0.18)))
        draw.line((x - radius * 3, y, x + radius * 3, y), fill=(255, 246, 202, 245), width=1)


def _draw_material_foliage(
    draw: ImageDraw.ImageDraw,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = max(10, int(region_w * region_h * material.intensity * 0.006))
    stem = max(3, int(region_h * max(0.02, material.scale) * 0.9))
    color_a = material.colors[0]
    color_b = material.colors[min(1, len(material.colors) - 1)]
    alpha = max(45, min(220, int(255 * material.opacity * 0.8)))
    line_width = max(1, int(min(region_w, region_h) * material.scale * 0.10))

    for _ in range(min(count, 900)):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        lean = rng.randint(-stem, stem)
        color = blend(color_a, color_b, rng.random())
        draw.line((x, y, max(left, min(right - 1, x + lean)), max(top, y - rng.randint(stem // 2, stem))), fill=(*color, alpha), width=line_width)


def _draw_material_metal(
    draw: ImageDraw.ImageDraw,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = max(4, int(region_h * material.intensity * 0.35))
    highlight = blend(material.colors[0], (255, 255, 255), 0.62)
    shadow = material.colors[min(1, len(material.colors) - 1)]
    for _ in range(count):
        y = rng.randint(top, max(top, bottom - 1))
        color = highlight if rng.random() > 0.45 else shadow
        alpha = int(255 * material.opacity * rng.uniform(0.18, 0.52))
        draw.line((left, y, right, y + rng.randint(-2, 2)), fill=(*color, alpha), width=max(1, int(region_w * material.scale * 0.02)))


def _draw_material_stone(
    draw: ImageDraw.ImageDraw,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    line = blend(material.colors[0], (220, 230, 232), 0.28)
    shadow = blend(material.colors[min(1, len(material.colors) - 1)], (0, 0, 0), 0.25)
    alpha = max(45, min(210, int(255 * material.opacity * (0.34 + material.intensity * 0.32))))
    tile_h = max(8, int(region_h * max(0.12, material.scale * 4.0)))
    tile_w = max(14, int(region_w * max(0.10, material.scale * 5.0)))

    for y in range(top + tile_h, bottom, tile_h):
        draw.line((left, y, right, y + rng.randint(-1, 1)), fill=(*line, alpha), width=1)
    offset = 0
    for x in range(left + tile_w, right, tile_w):
        draw.line((x + offset, top, x + offset + rng.randint(-1, 1), bottom), fill=(*shadow, max(30, alpha // 2)), width=1)
        offset = 0 if offset else tile_w // 2

    glints = max(8, int(region_w * region_h * material.intensity * 0.004))
    for _ in range(min(glints, 200)):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        length = rng.randint(max(4, region_w // 80), max(8, region_w // 24))
        draw.line((x, y, min(right - 1, x + length), y + rng.randint(-1, 1)), fill=(230, 240, 235, max(55, alpha)), width=1)


def _draw_material_surface(
    draw: ImageDraw.ImageDraw,
    material: PlannedMaterial,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = max(8, int(region_w * region_h * material.intensity * 0.003))
    radius = max(1, int(min(region_w, region_h) * max(0.004, material.scale) * 0.25))
    alpha = max(24, min(180, int(255 * material.opacity * 0.45)))
    for _ in range(min(count, 700)):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        color = material.colors[rng.randrange(len(material.colors))]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))


def _draw_texture_hatching(
    draw: ImageDraw.ImageDraw,
    texture: PlannedTexture,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
    angle_offset: float = 0.0,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = texture.count or max(8, int((region_w + region_h) * texture.density * 0.45))
    stroke_len = max(4, int(min(region_w, region_h) * max(0.02, texture.scale) * 3.0))
    line_width = max(1, int(min(region_w, region_h) * max(0.004, texture.scale) * 0.16))
    color = _rgba(texture.color, texture.opacity) or (*texture.color, 255)
    angle = math.radians(float(texture.extra.get("angle", -18.0)) + angle_offset)
    dx = max(1, int(math.cos(angle) * stroke_len))
    dy = int(math.sin(angle) * stroke_len)

    for _ in range(count):
        x0 = rng.randint(left, max(left, right - 1))
        y0 = rng.randint(top, max(top, bottom - 1))
        x1 = max(left, min(right - 1, x0 + dx))
        y1 = max(top, min(bottom - 1, y0 + dy))
        draw.line((x0, y0, x1, y1), fill=color, width=line_width)


def _draw_texture_ripples(
    draw: ImageDraw.ImageDraw,
    texture: PlannedTexture,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = texture.count or max(4, int(region_h * texture.density * 0.65))
    amplitude = max(1, int(region_h * max(0.01, texture.scale) * 0.8))
    segment = max(8, int(region_w * max(0.02, texture.scale) * 2.8))
    color = _rgba(texture.color, texture.opacity) or (*texture.color, 255)
    line_width = max(1, int(min(region_w, region_h) * max(0.004, texture.scale) * 0.12))

    for index in range(count):
        y = top + int((index + 0.5) * region_h / max(1, count))
        y += rng.randint(-amplitude, amplitude)
        points: list[tuple[int, int]] = []
        phase = rng.random() * math.tau
        for x in range(left, right + segment, segment):
            wave = int(math.sin((x - left) / max(1, segment) + phase) * amplitude)
            points.append((max(left, min(right - 1, x)), max(top, min(bottom - 1, y + wave))))
        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width, joint="curve")


def _draw_texture_speckles(
    draw: ImageDraw.ImageDraw,
    texture: PlannedTexture,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    left, top, right, bottom = bounds
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    count = texture.count or max(12, int(region_w * region_h * texture.density * 0.018))
    radius = max(1, int(min(region_w, region_h) * max(0.004, texture.scale) * 0.5))
    color = _rgba(texture.color, texture.opacity) or (*texture.color, 255)
    for _ in range(min(count, 1400)):
        x = rng.randint(left, max(left, right - 1))
        y = rng.randint(top, max(top, bottom - 1))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _apply_noise_texture(
    image: Image.Image,
    texture: PlannedTexture,
    bounds: tuple[int, int, int, int],
    seed: int,
) -> Image.Image:
    left, top, right, bottom = bounds
    rng = np.random.default_rng(seed * 1697 + texture.seed * 2039 + texture.z)
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    region_h = max(1, bottom - top)
    region_w = max(1, right - left)
    noise = rng.random((region_h, region_w))
    threshold = 1.0 - max(0.05, texture.density) * 0.65
    alpha = np.where(noise >= threshold, int(texture.opacity * 255), int(texture.opacity * 72))
    color = np.array(texture.color, dtype=np.uint8)
    overlay[top:bottom, left:right, :3] = color
    overlay[top:bottom, left:right, 3] = alpha.astype(np.uint8)
    return _composite_overlay(image, Image.fromarray(overlay, "RGBA"), texture.blend)


def _composite_overlay(image: Image.Image, overlay: Image.Image, blend: str) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    layer = np.asarray(overlay.convert("RGBA"), dtype=np.float32)
    alpha = layer[:, :, 3:4] / 255.0
    source = layer[:, :, :3]

    if blend == "screen":
        blended = 255.0 - (255.0 - base) * (255.0 - source) / 255.0
    elif blend == "multiply":
        blended = base * source / 255.0
    elif blend == "overlay":
        base_norm = base / 255.0
        source_norm = source / 255.0
        blended = np.where(
            base_norm <= 0.5,
            2.0 * base_norm * source_norm,
            1.0 - 2.0 * (1.0 - base_norm) * (1.0 - source_norm),
        ) * 255.0
    elif blend == "soft-light":
        base_norm = base / 255.0
        source_norm = source / 255.0
        blended = np.where(
            source_norm <= 0.5,
            base_norm - (1.0 - 2.0 * source_norm) * base_norm * (1.0 - base_norm),
            base_norm + (2.0 * source_norm - 1.0) * (np.sqrt(np.clip(base_norm, 0.0, 1.0)) - base_norm),
        ) * 255.0
    else:
        blended = source

    output = base * (1.0 - alpha) + blended * alpha
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")


def _shape_mask(
    size: tuple[int, int],
    opacity: float,
    *,
    rectangle: tuple[int, int, int, int] | None = None,
    rounded_rectangle: tuple[tuple[int, int, int, int], int] | None = None,
    ellipse: tuple[int, int, int, int] | None = None,
    polygon: list[tuple[int, int]] | None = None,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    alpha = max(0, min(255, int(round(opacity * 255))))
    if rectangle is not None:
        draw.rectangle(rectangle, fill=alpha)
    elif rounded_rectangle is not None:
        bbox, radius = rounded_rectangle
        draw.rounded_rectangle(bbox, radius=radius, fill=alpha)
    elif ellipse is not None:
        draw.ellipse(ellipse, fill=alpha)
    elif polygon:
        draw.polygon(polygon, fill=alpha)
    return mask


def _apply_element_gradient(
    image: Image.Image,
    gradient: PlannedGradient,
    mask: Image.Image,
    bounds: tuple[int, int, int, int],
) -> None:
    layer = _element_gradient_layer(image.size, gradient, bounds)
    layer.putalpha(mask)
    if image.mode == "RGBA":
        image.alpha_composite(layer)
    else:
        image.paste(layer.convert("RGB"), mask=layer.getchannel("A"))


def _element_gradient_layer(
    size: tuple[int, int],
    gradient: PlannedGradient,
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    width, height = size
    left, top, right, bottom = _clamp_bounds(bounds, width, height)
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    if gradient.kind == "radial":
        region_w = max(1, right - left)
        region_h = max(1, bottom - top)
        cx = left + gradient.center[0] * region_w
        cy = top + gradient.center[1] * region_h
        radius = max(1.0, min(region_w, region_h) * max(0.01, gradient.radius))
        t = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / radius
    elif gradient.direction == "horizontal":
        t = (x_grid - left) / max(1, right - left)
    elif gradient.direction == "diagonal":
        t = ((x_grid - left) / max(1, right - left) + (y_grid - top) / max(1, bottom - top)) / 2.0
    elif gradient.direction == "reverse-diagonal":
        t = ((right - x_grid) / max(1, right - left) + (y_grid - top) / max(1, bottom - top)) / 2.0
    else:
        t = (y_grid - top) / max(1, bottom - top)

    return Image.fromarray(_interpolate_gradient_colors(gradient.colors, np.clip(t, 0.0, 1.0)), "RGB").convert("RGBA")


def _interpolate_gradient_colors(colors: tuple[RGB, ...], t: np.ndarray) -> np.ndarray:
    color_array = np.array(colors, dtype=np.float32)
    scaled = t * (len(colors) - 1)
    lower = np.clip(np.floor(scaled).astype(np.int16), 0, len(colors) - 2)
    local_t = (scaled - lower)[..., None]
    blended = color_array[lower] * (1.0 - local_t) + color_array[lower + 1] * local_t
    return np.clip(blended, 0, 255).astype(np.uint8)


def _multi_color_blend(colors: tuple[RGB, ...], t: float) -> RGB:
    if len(colors) == 1:
        return colors[0]
    scaled = max(0.0, min(1.0, t)) * (len(colors) - 1)
    lower = min(len(colors) - 2, int(math.floor(scaled)))
    return blend(colors[lower], colors[lower + 1], scaled - lower)


def _apply_lights(image: Image.Image, lights: tuple[PlannedLight, ...]) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    y_grid, x_grid = np.ogrid[0:height, 0:width]

    for light in lights:
        cx = light.x * max(1, width - 1)
        cy = light.y * max(1, height - 1)
        radius = max(1.0, light.radius * min(width, height))
        distance = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        mask = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 1.8
        strength = mask[..., None] * light.intensity

        if light.kind in {"shadow", "shade", "darken"}:
            array *= 1.0 - strength * 0.85
            continue

        color = np.array(light.color, dtype=np.float32)
        if light.kind in {"rim", "cool", "tint"}:
            array = array * (1.0 - strength * 0.35) + color * (strength * 0.35)
        else:
            array = array + (255.0 - array) * strength * (color / 255.0)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _apply_atmosphere(image: Image.Image, atmosphere: PlannedAtmosphere) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    fog_height = max(0.01, atmosphere.height)
    distance = np.abs(y - atmosphere.horizon) / fog_height
    smooth = np.where(distance < 1.0, 0.5 + 0.5 * np.cos(distance * math.pi), 0.0)
    mask = (smooth * atmosphere.strength)[..., None]
    color = np.array(atmosphere.color, dtype=np.float32).reshape(1, 1, 3)
    output = array * (1.0 - mask) + color * mask
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")


def _apply_focus_blur(image: Image.Image, focus: PlannedFocus) -> Image.Image:
    radius = max(0.0, min(image.size) * focus.blur)
    if radius <= 0:
        return image

    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = np.asarray(image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    mask = _focus_mask(image.size, focus)[..., None]
    output = base * (1.0 - mask) + blurred * mask
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")


def _focus_mask(size: tuple[int, int], focus: PlannedFocus) -> np.ndarray:
    width, height = size
    x0, y0, x1, y1 = focus.region
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    if focus.falloff <= 0:
        return (inside if focus.mode == "inside" else ~inside).astype(np.float32)

    falloff = max(0.001, focus.falloff)
    if focus.mode == "inside":
        edge_distance = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
        return np.where(inside, np.clip(edge_distance / falloff, 0.0, 1.0), 0.0).astype(np.float32)

    outside_x = np.maximum.reduce([x0 - x, x - x1, np.zeros_like(x)])
    outside_y = np.maximum.reduce([y0 - y, y - y1, np.zeros_like(y)])
    distance = np.maximum(outside_x, outside_y)
    return np.clip(distance / falloff, 0.0, 1.0).astype(np.float32)


def _draw_planned_mountains(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed + 231)
    base_y = int(height * obj.y)
    layers = max(1, min(5, int(obj.extra.get("layers", 3))))
    for layer in range(layers):
        points = [(0, base_y + layer * 18)]
        step = max(55, width // 6)
        for x in range(-step, width + step, step):
            peak = base_y - int(height * obj.size * rng.uniform(0.65, 1.25)) + layer * 22
            points.append((x + rng.randint(-20, 20), peak))
            points.append((x + step // 2, base_y + rng.randint(-5, 24) + layer * 16))
        points.append((width, height))
        draw.polygon(points, fill=(*blend(obj.color, (45, 50, 60), 0.18 * layer), 210))


def _draw_planned_cloud(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int) -> None:
    radius = max(8, int(min(width, height) * obj.size))
    cx, cy = int(width * obj.x), int(height * obj.y)
    for offset, scale in [(-1.1, 0.78), (-0.35, 1.0), (0.45, 0.88), (1.15, 0.70)]:
        r = int(radius * scale)
        ox = int(offset * radius)
        draw.ellipse((cx + ox - r, cy - r, cx + ox + r, cy + r), fill=(*obj.color, 150))


def _draw_planned_trees(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed + int(obj.x * 1000) + 307)
    count = max(1, min(40, int(obj.extra.get("count", 10 if obj.kind == "forest" else 1))))
    for i in range(count):
        x = int(width * obj.x) if count == 1 else rng.randint(0, width)
        base = int(height * obj.y) + rng.randint(-8, 8)
        tree_h = max(14, int(height * obj.size * rng.uniform(0.75, 1.25)))
        trunk = blend(obj.color, (90, 58, 35), 0.55)
        draw.rectangle((x - 2, base - tree_h // 2, x + 2, base), fill=(*trunk, 220))
        draw.polygon([(x, base - tree_h), (x - tree_h // 3, base - tree_h // 3), (x + tree_h // 3, base - tree_h // 3)], fill=(*obj.color, 220))


def _draw_planned_buildings(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed + 401)
    base = int(height * obj.y)
    count = max(3, min(24, int(obj.extra.get("count", 10))))
    building_w = max(10, width // (count + 5))
    for index in range(count):
        x = int(index * width / count)
        h = rng.randint(max(20, int(height * obj.size * 0.65)), max(24, int(height * obj.size * 1.5)))
        draw.rectangle((x, base - h, x + building_w, base), fill=(*obj.color, 220))
        for y in range(base - h + 8, base - 5, 14):
            draw.rectangle((x + 4, y, x + min(building_w - 3, 10), y + 5), fill=(245, 214, 115, 105))


def _draw_planned_greenhouse(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int) -> None:
    span = max(0.20, min(0.96, obj.size)) * width * 0.50
    cx = int(width * obj.x)
    apex_y = int(height * max(0.02, obj.y - obj.size * 0.08))
    eave_y = int(height * min(0.62, obj.y + obj.size * 0.06))
    base_y = int(height * min(0.72, obj.y + obj.size * 0.34))
    left = int(max(0, cx - span))
    right = int(min(width, cx + span))
    frame = (*blend(obj.color, (220, 240, 245), 0.20), max(80, int(obj.opacity * 235)))
    glass = (*blend(obj.color, (45, 86, 112), 0.45), max(18, int(obj.opacity * 38)))
    line_w = max(1, int(min(width, height) * max(0.008, obj.size * 0.006)))

    draw.polygon([(left, eave_y), (cx, apex_y), (right, eave_y), (right, base_y), (left, base_y)], fill=glass)
    draw.line((left, eave_y, cx, apex_y, right, eave_y), fill=frame, width=line_w)
    draw.line((left, eave_y, left, base_y, right, base_y, right, eave_y), fill=frame, width=line_w)
    pane_count = max(3, min(9, int(obj.extra.get("panes", 6))))
    for index in range(1, pane_count):
        x = left + int((right - left) * index / pane_count)
        roof_y = int(eave_y - (eave_y - apex_y) * (1.0 - abs((x - cx) / max(1, span))))
        draw.line((x, roof_y, x, base_y), fill=(*obj.color, max(55, int(obj.opacity * 145))), width=max(1, line_w - 1))
    for t in (0.36, 0.62):
        y = int(eave_y + (base_y - eave_y) * t)
        draw.line((left, y, right, y), fill=(*obj.color, max(45, int(obj.opacity * 115))), width=max(1, line_w - 1))


def _draw_planned_lamps(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int) -> None:
    count = max(1, min(8, int(obj.extra.get("count", 1))))
    spread = float(obj.extra.get("spread", max(0.10, obj.size * 2.4)))
    radius = max(4, int(min(width, height) * max(0.025, obj.size)))
    for index in range(count):
        offset = 0.0 if count == 1 else (index / (count - 1) - 0.5) * spread
        cx = int(width * max(0.04, min(0.96, obj.x + offset)))
        cy = int(height * obj.y)
        cord_top = max(0, cy - int(height * max(0.08, obj.size * 1.9)))
        draw.line((cx, cord_top, cx, cy - radius), fill=(12, 14, 12, 190), width=max(1, width // 420))
        for step in range(4, 0, -1):
            current = radius * (1.0 + step * 0.75)
            alpha = int(obj.opacity * (18 + step * 18))
            draw.ellipse((cx - current, cy - current, cx + current, cy + current), fill=(*obj.color, alpha))
        core = blend(obj.color, (255, 248, 220), 0.58)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*core, max(155, int(obj.opacity * 255))))


def _draw_planned_plants(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int, seed: int) -> None:
    rng = random.Random(seed + int(obj.x * 1000) + int(obj.y * 1000) + 613)
    count = max(4, min(120, int(obj.extra.get("count", 14))))
    center_x = int(width * obj.x)
    base_y = int(height * obj.y)
    reach_x = max(12, int(width * max(0.08, obj.size * 0.85)))
    reach_y = max(14, int(height * max(0.12, obj.size * 0.95)))
    stem_color = blend(obj.color, (20, 58, 38), 0.45)
    highlight = blend(obj.color, (90, 190, 120), 0.35)

    anchor_specs = [(-0.33, -0.06, 1.25), (0.0, -0.13, 1.10), (0.33, -0.04, 1.18)]
    for offset_x, offset_y, scale in anchor_specs:
        leaf_x = center_x + int(reach_x * offset_x)
        leaf_y = base_y + int(reach_y * offset_y)
        leaf_w = int(reach_x * 0.34 * scale)
        leaf_h = int(reach_y * 0.30 * scale)
        _draw_leaf(draw, leaf_x, leaf_y, leaf_w, leaf_h, obj.color, highlight, alpha=max(120, int(obj.opacity * 235)))

    for _ in range(count):
        leaf_x = center_x + rng.randint(-reach_x, reach_x)
        leaf_y = base_y + rng.randint(-reach_y, max(2, reach_y // 4))
        leaf_w = max(4, int(reach_x * rng.uniform(0.12, 0.28)))
        leaf_h = max(6, int(reach_y * rng.uniform(0.12, 0.34)))
        color = blend(obj.color, (8, 38, 22), rng.uniform(0.0, 0.45))
        draw.line((leaf_x, base_y + rng.randint(-3, 8), leaf_x, leaf_y + leaf_h // 2), fill=(*stem_color, 170), width=max(1, width // 380))
        _draw_leaf(draw, leaf_x, leaf_y, leaf_w, leaf_h, color, highlight, alpha=max(90, int(obj.opacity * rng.uniform(130, 225))))


def _draw_leaf(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    half_w: int,
    half_h: int,
    color: RGB,
    vein: RGB,
    *,
    alpha: int,
) -> None:
    points = [
        (cx, cy - half_h),
        (cx - half_w, cy),
        (cx, cy + half_h),
        (cx + half_w, cy),
    ]
    draw.polygon(points, fill=(*color, alpha))
    draw.line((cx, cy - half_h, cx, cy + half_h), fill=(*vein, min(245, alpha + 15)), width=1)
    draw.line((cx, cy, cx - half_w // 2, cy - half_h // 3), fill=(*vein, min(210, alpha)), width=1)
    draw.line((cx, cy, cx + half_w // 2, cy - half_h // 3), fill=(*vein, min(210, alpha)), width=1)


def _draw_planned_floor(draw: ImageDraw.ImageDraw, obj: PlannedObject, width: int, height: int) -> None:
    y0 = int(height * obj.y)
    base = blend(obj.color, (28, 34, 36), 0.28)
    alpha = max(80, int(obj.opacity * 235))
    draw.rectangle((0, y0, width, height), fill=(*base, alpha))
    line = blend(obj.color, (205, 215, 214), 0.22)
    tile_h = max(8, int(height * max(0.055, obj.size * 0.20)))
    tile_w = max(18, int(width * max(0.12, obj.size * 0.70)))
    for y in range(y0 + tile_h, height, tile_h):
        draw.line((0, y, width, y), fill=(*line, 105), width=1)
    for row, y in enumerate(range(y0, height, tile_h)):
        offset = 0 if row % 2 == 0 else tile_w // 2
        for x in range(-offset, width, tile_w):
            draw.line((x, y, x, min(height, y + tile_h)), fill=(12, 18, 18, 90), width=1)
    for i in range(8):
        y = y0 + int((height - y0) * (i + 1) / 10)
        draw.line((int(width * 0.18), y, int(width * 0.82), y - 1), fill=(220, 230, 220, 50 + i * 8), width=1)


def _scaled_points(points: tuple[tuple[float, float], ...], width: int, height: int) -> list[tuple[int, int]]:
    return [(int(x * width), int(y * height)) for x, y in points]


def _normalized_points(value: object, width: int, height: int) -> list[tuple[int, int]]:
    if not isinstance(value, list | tuple):
        return []
    points: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, list | tuple) and len(item) >= 2:
            points.append((int(_clamp_unit(item[0]) * width), int(_clamp_unit(item[1]) * height)))
    return points


def _normalized_region(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    x0 = _clamp_unit(value[0])
    y0 = _clamp_unit(value[1])
    x1 = _clamp_unit(value[2])
    y1 = _clamp_unit(value[3])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _clamp_unit(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _scaled_region(region: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = region
    left = max(0, min(width, int(round(x0 * width))))
    top = max(0, min(height, int(round(y0 * height))))
    right = max(0, min(width, int(round(x1 * width))))
    bottom = max(0, min(height, int(round(y1 * height))))
    return left, top, right, bottom


def _point_bounds(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def _clamp_bounds(bounds: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    return max(0, min(width, left)), max(0, min(height, top)), max(0, min(width, right)), max(0, min(height, bottom))


def _sample_path_commands(
    commands: tuple[tuple[str, tuple[tuple[float, float], ...]], ...],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    points: list[tuple[float, float]] = []
    current: tuple[float, float] | None = None
    start: tuple[float, float] | None = None
    for command, command_points in commands:
        if command == "M":
            current = command_points[0]
            start = current
            points.append(current)
        elif command == "L" and current is not None:
            current = command_points[0]
            points.append(current)
        elif command == "Q" and current is not None:
            control, end = command_points
            points.extend(_sample_quadratic(current, control, end))
            current = end
        elif command == "C" and current is not None:
            control_a, control_b, end = command_points
            points.extend(_sample_cubic(current, control_a, control_b, end))
            current = end
        elif command == "Z" and start is not None:
            current = start
            points.append(start)
    return _scaled_points(tuple(points), width, height)


def _sample_quadratic(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    steps: int = 24,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        inv = 1.0 - t
        result.append(
            (
                inv * inv * start[0] + 2 * inv * t * control[0] + t * t * end[0],
                inv * inv * start[1] + 2 * inv * t * control[1] + t * t * end[1],
            )
        )
    return result


def _sample_cubic(
    start: tuple[float, float],
    control_a: tuple[float, float],
    control_b: tuple[float, float],
    end: tuple[float, float],
    steps: int = 32,
) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for index in range(1, steps + 1):
        t = index / steps
        inv = 1.0 - t
        result.append(
            (
                inv**3 * start[0]
                + 3 * inv * inv * t * control_a[0]
                + 3 * inv * t * t * control_b[0]
                + t**3 * end[0],
                inv**3 * start[1]
                + 3 * inv * inv * t * control_a[1]
                + 3 * inv * t * t * control_b[1]
                + t**3 * end[1],
            )
        )
    return result


def _path_is_closed(commands: tuple[tuple[str, tuple[tuple[float, float], ...]], ...]) -> bool:
    return bool(commands and commands[-1][0] == "Z")


def _element_bbox(element: PlannedElement, width: int, height: int) -> tuple[int, int, int, int]:
    cx, cy = int(element.x * width), int(element.y * height)
    half_w = max(1, int(element.width * width / 2))
    half_h = max(1, int(element.height * height / 2))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _element_line_width(element: PlannedElement, width: int, height: int) -> int:
    raw_width = element.extra.get("stroke_width", element.extra.get("line_width"))
    if raw_width is not None:
        try:
            relative = float(raw_width)
        except (TypeError, ValueError):
            relative = 0.006
    elif element.kind in {"polyline", "line", "arrow", "path", "polygon", "arc"}:
        relative = element.width
    else:
        relative = 0.006
    return max(1, int(round(min(width, height) * max(0.001, min(1.0, relative)))))


def _element_corner_radius(
    element: PlannedElement,
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> int:
    left, top, right, bottom = bbox
    max_radius = max(1, min(right - left, bottom - top) // 2)
    radius = max(0.0, min(1.0, _float_extra(element.extra, "radius", 0.18)))
    return max(1, min(max_radius, int(round(min(width, height) * radius))))


def _rgba(color: RGB | None, opacity: float) -> tuple[int, int, int, int] | None:
    if color is None:
        return None
    return (*color, max(0, min(255, int(round(opacity * 255)))))


def _float_extra(values: dict[str, object], key: str, default: float) -> float:
    try:
        parsed = float(values.get(key, default))
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _draw_element_glow(draw: ImageDraw.ImageDraw, element: PlannedElement, width: int, height: int) -> None:
    color = element.fill or element.stroke or COLOR_RGB["yellow"]
    cx, cy = int(element.x * width), int(element.y * height)
    radius = max(4, int(min(width, height) * max(element.width, element.height)))
    for step in range(5, 0, -1):
        current = radius * step / 3
        alpha = int(42 * element.opacity * step / 5)
        draw.ellipse(
            (cx - current, cy - current, cx + current, cy + current),
            fill=(*color, alpha),
        )


def _add_grain(image: Image.Image, amount: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    noise = rng.normal(0, amount * 24.0, array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")


def _apply_bloom(image: Image.Image, amount: float) -> Image.Image:
    amount = max(0.0, min(1.0, amount))
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = array[:, :, 0] * 0.2126 + array[:, :, 1] * 0.7152 + array[:, :, 2] * 0.0722
    mask_strength = np.clip((luminance - 145.0) / 110.0, 0.0, 1.0)[..., None]
    bright = np.clip(array * mask_strength, 0, 255).astype(np.uint8)
    radius = 2.0 + amount * 7.5
    blurred = np.asarray(Image.fromarray(bright, "RGB").filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    output = array + blurred * (0.24 + amount * 0.82)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")


def _apply_color_grade(image: Image.Image, saturation: float, contrast: float, warmth: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    contrast = max(0.0, min(1.0, contrast))
    saturation = max(0.0, min(1.0, saturation))
    warmth = max(0.0, min(1.0, warmth))

    if contrast > 0:
        factor = 1.0 + contrast * 0.78
        array = (array - 127.5) * factor + 127.5
    if warmth > 0:
        array[:, :, 0] += warmth * 78.0
        array[:, :, 1] += warmth * 10.0
        array[:, :, 2] -= warmth * 82.0
    if saturation > 0:
        luminance = (array[:, :, 0] * 0.2126 + array[:, :, 1] * 0.7152 + array[:, :, 2] * 0.0722)[..., None]
        array = luminance + (array - luminance) * (1.0 + saturation * 3.0)

    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def _apply_vignette(image: Image.Image, amount: float) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = array.shape[:2]
    y, x = np.ogrid[-1.0:1.0:height * 1j, -1.0:1.0:width * 1j]
    distance = np.sqrt(x * x + y * y)
    falloff = np.clip((distance - 0.35) / 0.85, 0.0, 1.0)
    factor = 1.0 - falloff[..., None] * max(0.0, min(1.0, amount)) * 0.55
    return Image.fromarray(np.clip(array * factor, 0, 255).astype(np.uint8), "RGB")


def _draw_sun(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    warm = _first_matching(candidate.palette, fallback=COLOR_RGB["yellow"], prefer="warm")
    size = min(width, height) * 0.18
    cx = int(width * (0.30 + candidate.variation * 0.35))
    cy = int(height * 0.26)
    bbox = (cx - size, cy - size, cx + size, cy + size)
    draw.ellipse(bbox, fill=(*blend(warm, COLOR_RGB["yellow"], 0.50), 235))
    draw.ellipse(_shrink(bbox, 0.62), fill=(*warm, 245))


def _draw_moon(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    size = min(width, height) * 0.12
    cx, cy = int(width * 0.72), int(height * 0.22)
    draw.ellipse((cx - size, cy - size, cx + size, cy + size), fill=(232, 230, 210, 230))
    draw.ellipse((cx - size * 0.45, cy - size, cx + size * 1.25, cy + size), fill=(35, 48, 72, 190))


def _draw_ocean(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    y0 = int(height * candidate.horizon)
    blue = _first_matching(candidate.palette, fallback=COLOR_RGB["blue"], prefer="blue")
    draw.rectangle((0, y0, width, height), fill=(*blend(blue, (18, 70, 125), 0.32), 210))
    for i in range(9):
        y = y0 + int((height - y0) * (i + 1) / 11)
        alpha = 90 + i * 10
        draw.arc((-width * 0.1, y - 22, width * 1.1, y + 24), 0, 180, fill=(235, 245, 250, alpha), width=2)


def _draw_mountains(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    rng = random.Random(candidate.seed + 51)
    y_base = int(height * candidate.horizon)
    for layer in range(3):
        points = [(0, y_base + layer * 22)]
        step = max(60, width // 5)
        for x in range(-step, width + step, step):
            peak_y = y_base - rng.randint(height // 7, height // 3) + layer * 30
            points.append((x + rng.randint(-20, 20), peak_y))
            points.append((x + step // 2, y_base + rng.randint(-5, 30) + layer * 22))
        points.append((width, height))
        color = blend(candidate.palette[layer % len(candidate.palette)], (55, 62, 74), 0.42 + layer * 0.12)
        draw.polygon(points, fill=(*color, 185))


def _draw_clouds(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    rng = random.Random(candidate.seed + 73)
    for _ in range(5):
        cx = rng.randint(0, width)
        cy = rng.randint(height // 10, height // 2)
        radius = rng.randint(max(10, width // 35), max(20, width // 16))
        for offset in (-1, 0, 1):
            draw.ellipse(
                (cx + offset * radius - radius, cy - radius, cx + offset * radius + radius, cy + radius),
                fill=(245, 248, 250, 115),
            )


def _draw_botanical(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    rng = random.Random(candidate.seed + 101)
    green = _first_matching(candidate.palette, fallback=COLOR_RGB["green"], prefer="green")
    accent = candidate.palette[-1]
    for _ in range(max(8, candidate.shape_count)):
        x = rng.randint(0, width)
        y = rng.randint(int(height * 0.45), height)
        stem = rng.randint(height // 8, height // 3)
        draw.line((x, y, x + rng.randint(-15, 15), y - stem), fill=(*green, 210), width=max(1, width // 180))
        leaf_w = rng.randint(max(5, width // 70), max(10, width // 34))
        leaf_h = rng.randint(max(4, height // 80), max(9, height // 40))
        draw.ellipse((x - leaf_w, y - stem, x + leaf_w, y - stem + leaf_h * 2), fill=(*green, 175))
        if "flower" in candidate.spec.objects:
            draw.ellipse((x - leaf_w, y - stem - leaf_h, x + leaf_w, y - stem + leaf_h), fill=(*accent, 210))


def _draw_city(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    rng = random.Random(candidate.seed + 131)
    y_base = int(height * candidate.horizon)
    for x in range(0, width, max(18, width // 14)):
        w = rng.randint(max(14, width // 35), max(26, width // 16))
        h = rng.randint(height // 8, height // 2)
        color = blend(candidate.palette[(x // max(1, w)) % len(candidate.palette)], (28, 32, 42), 0.62)
        draw.rectangle((x, y_base - h, x + w, y_base), fill=(*color, 220))
        for wy in range(y_base - h + 8, y_base - 4, 14):
            draw.rectangle((x + 4, wy, x + min(w - 4, 10), wy + 5), fill=(245, 214, 115, 105))


def _draw_portrait(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    skin = blend(candidate.palette[0], (218, 168, 132), 0.55)
    cx, cy = width // 2, int(height * 0.43)
    rx, ry = int(width * 0.13), int(height * 0.21)
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(*skin, 225))
    draw.ellipse((cx - rx // 2, cy - ry // 5, cx - rx // 4, cy), fill=(25, 25, 30, 190))
    draw.ellipse((cx + rx // 4, cy - ry // 5, cx + rx // 2, cy), fill=(25, 25, 30, 190))
    draw.arc((cx - rx // 2, cy, cx + rx // 2, cy + ry // 2), 10, 170, fill=(80, 42, 48, 170), width=2)


def _draw_robot(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    metal = blend(candidate.palette[0], (180, 190, 200), 0.62)
    cx, y = width // 2, int(height * 0.36)
    size = min(width, height) // 4
    draw.rounded_rectangle((cx - size, y, cx + size, y + size), radius=max(3, size // 9), fill=(*metal, 225))
    eye = blend(candidate.palette[-1], COLOR_RGB["cyan"], 0.35)
    draw.rectangle((cx - size // 2, y + size // 3, cx - size // 4, y + size // 2), fill=(*eye, 235))
    draw.rectangle((cx + size // 4, y + size // 3, cx + size // 2, y + size // 2), fill=(*eye, 235))


def _draw_abstract(draw: ImageDraw.ImageDraw, candidate: SceneCandidate, width: int, height: int) -> None:
    rng = random.Random(candidate.seed + 171)
    for i in range(candidate.shape_count):
        color = candidate.palette[i % len(candidate.palette)]
        x0 = rng.randint(-width // 10, width)
        y0 = rng.randint(-height // 10, height)
        size = rng.randint(max(18, min(width, height) // 12), max(32, min(width, height) // 4))
        alpha = rng.randint(70, 165)
        if i % 3 == 0:
            draw.ellipse((x0, y0, x0 + size, y0 + size), fill=(*color, alpha))
        elif i % 3 == 1:
            angle = candidate.variation * math.pi
            points = [
                (x0, y0),
                (x0 + int(math.cos(angle) * size), y0 + int(math.sin(angle) * size)),
                (x0 + size // 2, y0 + size),
            ]
            draw.polygon(points, fill=(*color, alpha))
        else:
            draw.rectangle((x0, y0, x0 + size, y0 + size // 2), fill=(*color, alpha))


def _first_matching(palette: tuple[RGB, ...], fallback: RGB, prefer: str) -> RGB:
    for color in palette:
        red, green, blue = color
        if prefer == "warm" and red >= green and red >= blue:
            return color
        if prefer == "blue" and blue >= red and blue >= green:
            return color
        if prefer == "green" and green >= red and green >= blue:
            return color
    return fallback


def _shrink(bbox: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * amount / 2, (y1 - y0) * amount / 2
    return cx - w, cy - h, cx + w, cy + h
