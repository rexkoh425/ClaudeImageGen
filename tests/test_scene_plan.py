import json
from pathlib import Path

from claude_imagegen.generator import GenerateOptions, generate_image
from claude_imagegen.quality import image_detail_metrics
from claude_imagegen.render import render_scene_plan
from claude_imagegen.scene_plan import parse_scene_plan


def test_parse_scene_plan_normalizes_palette_and_objects(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude planned coastal sunrise",
                "palette": ["#112244", "coral", [40, 150, 210]],
                "background": {
                    "type": "gradient",
                    "top": "#102040",
                    "bottom": "#205080",
                    "direction": "vertical",
                    "stops": [
                        {"at": 0.0, "color": "#102040"},
                        {"at": 0.48, "color": "#ffcf8a"},
                        {"at": 1.0, "color": "#205080"},
                    ],
                },
                "objects": [
                    {
                        "type": "sun",
                        "label": "large warm sunrise",
                        "x": 0.24,
                        "y": 0.22,
                        "size": 0.18,
                        "color": "#ff5533",
                    },
                    {
                        "type": "foreground",
                        "label": "dark pine foreground",
                        "y": 0.74,
                        "color": "#123d2a",
                    },
                ],
                "elements": [
                    {
                        "type": "polyline",
                        "label": "Claude-drawn reflection highlight",
                        "points": [[0.1, 0.7], [0.5, 0.72], [0.9, 0.7]],
                        "stroke": "#f6e2b5",
                        "width": 0.02,
                        "opacity": 0.75,
                        "z": 4,
                    },
                        {
                            "type": "path",
                            "label": "Claude-drawn curved wave",
                        "commands": [
                            ["M", 0.12, 0.70],
                            ["C", 0.30, 0.55, 0.48, 0.82, 0.64, 0.66],
                            ["Q", 0.78, 0.52, 0.90, 0.70],
                        ],
                        "stroke": "#c9e8ff",
                        "width": 0.01,
                        "opacity": 0.8,
                        "blur": 0.02,
                        "blend": "screen",
                            "z": 5,
                        },
                        {
                            "type": "rectangle",
                            "label": "Claude-drawn vertical color field",
                            "x": 0.5,
                            "y": 0.5,
                            "width": 0.4,
                            "height": 0.3,
                            "gradient": {
                                "type": "linear",
                                "colors": ["#102040", "#ffcf8a"],
                                "direction": "vertical",
                            },
                            "opacity": 0.9,
                            "z": 6,
                        }
                    ],
                "motifs": [
                    {
                        "type": "starfield",
                        "label": "sparse upper stars",
                        "count": 12,
                        "region": [0.0, 0.0, 1.0, 0.35],
                        "color": "#fff5cc",
                        "size": 0.012,
                        "opacity": 0.9,
                        "seed": 3,
                        "z": 8,
                    }
                ],
                "textures": [
                    {
                        "type": "hatching",
                        "label": "Claude planned shoreline hatch marks",
                        "region": [0.0, 0.58, 1.0, 0.95],
                        "color": "#f6e2b5",
                        "density": 0.55,
                        "scale": 0.035,
                        "opacity": 0.45,
                        "blend": "screen",
                        "seed": 5,
                        "z": 9,
                    }
                ],
                "materials": [
                    {
                        "type": "water",
                        "label": "Claude planned reflective ocean material",
                        "region": [0.0, 0.56, 1.0, 0.80],
                        "colors": ["#8bdcff", "#0b3b71"],
                        "intensity": 0.7,
                        "scale": 0.04,
                        "opacity": 0.64,
                        "seed": 17,
                        "z": 10,
                    }
                ],
                "terrains": [
                    {
                        "type": "mountain",
                        "label": "Claude planned faceted mountain ridge",
                        "points": [[0.08, 0.56], [0.28, 0.24], [0.46, 0.56], [0.64, 0.32], [0.92, 0.56]],
                        "base": 0.78,
                        "fill": "#405070",
                        "shade": "#182030",
                        "highlight": "#7890b0",
                        "opacity": 0.82,
                        "blur": 0.01,
                        "blend": "normal",
                        "facets": True,
                        "z": 4,
                    }
                ],
                "reflections": [
                    {
                        "type": "vertical",
                        "label": "Claude planned reflected mountains",
                        "source": [0.0, 0.18, 1.0, 0.56],
                        "target": [0.0, 0.56, 1.0, 0.80],
                        "opacity": 0.42,
                        "blur": 0.035,
                        "fade": 0.65,
                        "tint": "#2d88d8",
                        "blend": "screen",
                        "z": 8,
                    }
                ],
                "warps": [
                    {
                        "type": "wave",
                        "label": "Claude planned water displacement",
                        "region": [0.0, 0.56, 1.0, 0.80],
                        "direction": "horizontal",
                        "amplitude": 0.035,
                        "wavelength": 0.42,
                        "phase": 0.25,
                        "seed": 23,
                        "z": 9,
                    }
                ],
                "atmosphere": {
                    "type": "horizon_fog",
                    "label": "Claude planned cool horizon haze",
                    "color": "#d8e8f0",
                    "horizon": 0.56,
                    "height": 0.24,
                    "strength": 0.46,
                },
                "veils": [
                    {
                        "type": "mist",
                        "label": "Claude planned localized sea mist",
                        "region": [0.10, 0.46, 0.92, 0.68],
                        "color": "#d8e8f0",
                        "opacity": 0.38,
                        "blur": 0.025,
                        "blend": "screen",
                        "falloff": 0.18,
                        "direction": "vertical",
                        "z": 8,
                    }
                ],
                "lights": [
                    {
                        "type": "radial",
                        "label": "warm sun illumination",
                        "x": 0.24,
                        "y": 0.22,
                        "radius": 0.35,
                        "color": "#ffcf8a",
                        "intensity": 0.55,
                        "z": 9,
                    }
                ],
                "beams": [
                    {
                        "type": "sunbeam",
                        "label": "Claude planned diagonal sun shaft",
                        "x": 0.24,
                        "y": 0.24,
                        "angle": 62.0,
                        "length": 0.78,
                        "spread": 22.0,
                        "color": "#ffcf8a",
                        "opacity": 0.32,
                        "blur": 0.035,
                        "blend": "screen",
                        "count": 2,
                        "seed": 13,
                        "z": 7,
                    }
                ],
                "clouds": [
                    {
                        "type": "cumulus",
                        "label": "Claude planned soft cloud bank",
                        "region": [0.05, 0.10, 0.95, 0.36],
                        "color": "#fff5dd",
                        "shadow": "#8aa0b8",
                        "opacity": 0.62,
                        "blur": 0.025,
                        "count": 3,
                        "lobes": 5,
                        "scale": 0.16,
                        "blend": "screen",
                        "seed": 29,
                        "z": 3,
                    }
                ],
                "shadows": [
                    {
                        "type": "ellipse",
                        "label": "Claude planned soft contact shadow",
                        "x": 0.44,
                        "y": 0.74,
                        "width": 0.26,
                        "height": 0.08,
                        "color": "#101820",
                        "opacity": 0.46,
                        "blur": 0.035,
                        "blend": "multiply",
                        "z": 9,
                    }
                ],
                "focus": {
                    "type": "depth",
                    "label": "Claude planned focal band",
                    "region": [0.08, 0.18, 0.78, 0.74],
                    "blur": 0.035,
                    "falloff": 0.12,
                    "mode": "outside",
                },
                "style": {
                    "grain": 0.15,
                    "vignette": 0.2,
                    "saturation": 0.45,
                    "contrast": 0.35,
                    "warmth": 0.25,
                    "bloom": 0.4,
                    "antialias": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    plan = parse_scene_plan(plan_path)

    assert plan.title == "Claude planned coastal sunrise"
    assert plan.palette[0] == (17, 34, 68)
    assert plan.palette[1] == (255, 127, 80)
    assert plan.palette[2] == (40, 150, 210)
    assert plan.background.top == (16, 32, 64)
    assert plan.background.direction == "vertical"
    assert plan.background.stops[0].position == 0.0
    assert plan.background.stops[0].color == (16, 32, 64)
    assert plan.background.stops[1].position == 0.48
    assert plan.background.stops[1].color == (255, 207, 138)
    assert plan.background.stops[2].position == 1.0
    assert plan.background.stops[2].color == (32, 80, 128)
    assert plan.objects[0].kind == "sun"
    assert plan.objects[0].x == 0.24
    assert plan.elements[0].kind == "polyline"
    assert plan.elements[0].points == ((0.1, 0.7), (0.5, 0.72), (0.9, 0.7))
    assert plan.elements[0].stroke == (246, 226, 181)
    assert plan.elements[0].z == 4
    assert plan.elements[1].kind == "path"
    assert plan.elements[1].commands[0] == ("M", ((0.12, 0.70),))
    assert plan.elements[1].commands[1] == ("C", ((0.30, 0.55), (0.48, 0.82), (0.64, 0.66)))
    assert plan.elements[1].commands[2] == ("Q", ((0.78, 0.52), (0.90, 0.70)))
    assert plan.elements[1].blur == 0.02
    assert plan.elements[1].blend == "screen"
    assert plan.elements[2].gradient is not None
    assert plan.elements[2].gradient.kind == "linear"
    assert plan.elements[2].gradient.colors == ((16, 32, 64), (255, 207, 138))
    assert plan.elements[2].gradient.direction == "vertical"
    assert plan.motifs[0].kind == "starfield"
    assert plan.motifs[0].count == 12
    assert plan.motifs[0].region == (0.0, 0.0, 1.0, 0.35)
    assert plan.motifs[0].color == (255, 245, 204)
    assert plan.motifs[0].z == 8
    assert plan.textures[0].kind == "hatching"
    assert plan.textures[0].region == (0.0, 0.58, 1.0, 0.95)
    assert plan.textures[0].color == (246, 226, 181)
    assert plan.textures[0].density == 0.55
    assert plan.textures[0].scale == 0.035
    assert plan.textures[0].blend == "screen"
    assert plan.textures[0].z == 9
    assert plan.materials[0].kind == "water"
    assert plan.materials[0].region == (0.0, 0.56, 1.0, 0.80)
    assert plan.materials[0].colors == ((139, 220, 255), (11, 59, 113))
    assert plan.materials[0].intensity == 0.7
    assert plan.materials[0].scale == 0.04
    assert plan.materials[0].opacity == 0.64
    assert plan.materials[0].z == 10
    assert plan.terrains[0].kind == "mountain"
    assert plan.terrains[0].label == "Claude planned faceted mountain ridge"
    assert plan.terrains[0].points == ((0.08, 0.56), (0.28, 0.24), (0.46, 0.56), (0.64, 0.32), (0.92, 0.56))
    assert plan.terrains[0].base == 0.78
    assert plan.terrains[0].fill == (64, 80, 112)
    assert plan.terrains[0].shade == (24, 32, 48)
    assert plan.terrains[0].highlight == (120, 144, 176)
    assert plan.terrains[0].opacity == 0.82
    assert plan.terrains[0].blur == 0.01
    assert plan.terrains[0].blend == "normal"
    assert plan.terrains[0].facets is True
    assert plan.terrains[0].z == 4
    assert plan.reflections[0].kind == "vertical"
    assert plan.reflections[0].label == "Claude planned reflected mountains"
    assert plan.reflections[0].source == (0.0, 0.18, 1.0, 0.56)
    assert plan.reflections[0].target == (0.0, 0.56, 1.0, 0.80)
    assert plan.reflections[0].opacity == 0.42
    assert plan.reflections[0].blur == 0.035
    assert plan.reflections[0].fade == 0.65
    assert plan.reflections[0].tint == (45, 136, 216)
    assert plan.reflections[0].blend == "screen"
    assert plan.reflections[0].z == 8
    assert plan.warps[0].kind == "wave"
    assert plan.warps[0].label == "Claude planned water displacement"
    assert plan.warps[0].region == (0.0, 0.56, 1.0, 0.80)
    assert plan.warps[0].direction == "horizontal"
    assert plan.warps[0].amplitude == 0.035
    assert plan.warps[0].wavelength == 0.42
    assert plan.warps[0].phase == 0.25
    assert plan.warps[0].seed == 23
    assert plan.warps[0].z == 9
    assert plan.atmosphere is not None
    assert plan.atmosphere.kind == "horizon_fog"
    assert plan.atmosphere.color == (216, 232, 240)
    assert plan.atmosphere.horizon == 0.56
    assert plan.atmosphere.height == 0.24
    assert plan.atmosphere.strength == 0.46
    assert plan.veils[0].kind == "mist"
    assert plan.veils[0].label == "Claude planned localized sea mist"
    assert plan.veils[0].region == (0.10, 0.46, 0.92, 0.68)
    assert plan.veils[0].color == (216, 232, 240)
    assert plan.veils[0].opacity == 0.38
    assert plan.veils[0].blur == 0.025
    assert plan.veils[0].blend == "screen"
    assert plan.veils[0].falloff == 0.18
    assert plan.veils[0].direction == "vertical"
    assert plan.veils[0].z == 8
    assert plan.lights[0].kind == "radial"
    assert plan.lights[0].radius == 0.35
    assert plan.lights[0].color == (255, 207, 138)
    assert plan.lights[0].intensity == 0.55
    assert plan.beams[0].kind == "sunbeam"
    assert plan.beams[0].label == "Claude planned diagonal sun shaft"
    assert plan.beams[0].x == 0.24
    assert plan.beams[0].y == 0.24
    assert plan.beams[0].angle == 62.0
    assert plan.beams[0].length == 0.78
    assert plan.beams[0].spread == 22.0
    assert plan.beams[0].color == (255, 207, 138)
    assert plan.beams[0].opacity == 0.32
    assert plan.beams[0].blur == 0.035
    assert plan.beams[0].blend == "screen"
    assert plan.beams[0].count == 2
    assert plan.beams[0].seed == 13
    assert plan.beams[0].z == 7
    assert plan.clouds[0].kind == "cumulus"
    assert plan.clouds[0].label == "Claude planned soft cloud bank"
    assert plan.clouds[0].region == (0.05, 0.10, 0.95, 0.36)
    assert plan.clouds[0].color == (255, 245, 221)
    assert plan.clouds[0].shadow == (138, 160, 184)
    assert plan.clouds[0].opacity == 0.62
    assert plan.clouds[0].blur == 0.025
    assert plan.clouds[0].count == 3
    assert plan.clouds[0].lobes == 5
    assert plan.clouds[0].scale == 0.16
    assert plan.clouds[0].blend == "screen"
    assert plan.clouds[0].seed == 29
    assert plan.clouds[0].z == 3
    assert plan.shadows[0].kind == "ellipse"
    assert plan.shadows[0].label == "Claude planned soft contact shadow"
    assert plan.shadows[0].x == 0.44
    assert plan.shadows[0].y == 0.74
    assert plan.shadows[0].width == 0.26
    assert plan.shadows[0].height == 0.08
    assert plan.shadows[0].color == (16, 24, 32)
    assert plan.shadows[0].opacity == 0.46
    assert plan.shadows[0].blur == 0.035
    assert plan.shadows[0].blend == "multiply"
    assert plan.shadows[0].z == 9
    assert plan.focus is not None
    assert plan.focus.kind == "depth"
    assert plan.focus.label == "Claude planned focal band"
    assert plan.focus.region == (0.08, 0.18, 0.78, 0.74)
    assert plan.focus.blur == 0.035
    assert plan.focus.falloff == 0.12
    assert plan.focus.mode == "outside"
    assert plan.style["grain"] == 0.15
    assert plan.style["vignette"] == 0.2
    assert plan.style["saturation"] == 0.45
    assert plan.style["contrast"] == 0.35
    assert plan.style["warmth"] == 0.25
    assert plan.style["bloom"] == 0.4
    assert plan.style["antialias"] == 0.75


def test_parse_scene_plan_accepts_utf8_bom_from_powershell(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "BOM encoded plan",
                "palette": ["#112244"],
                "objects": [{"type": "sun", "color": "#ff5533"}],
            }
        ),
        encoding="utf-8-sig",
    )

    plan = parse_scene_plan(plan_path)

    assert plan.title == "BOM encoded plan"


def test_scene_plan_drives_rendered_composition_and_metadata(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude planned deliberate composition",
                "palette": ["#102040", "#ff5533", "#12482f"],
                "background": {
                    "top": "#102040",
                    "bottom": "#1e5f8f",
                    "stops": [
                        {"at": 0.0, "color": "#102040"},
                        {"at": 0.54, "color": "#ffcf8a"},
                        {"at": 1.0, "color": "#1e5f8f"},
                    ],
                },
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.2, "color": "#ff5533"},
                    {"type": "ocean", "y": 0.56, "color": "#286fc4"},
                    {"type": "foreground", "y": 0.78, "color": "#12482f"},
                ],
                "elements": [
                    {
                        "type": "polyline",
                        "points": [[0.15, 0.65], [0.85, 0.65]],
                        "stroke": "#f6e2b5",
                        "width": 0.02,
                        "z": 6,
                    },
                    {
                        "type": "rectangle",
                        "x": 0.5,
                        "y": 0.66,
                        "width": 1.0,
                        "height": 0.12,
                        "gradient": {
                            "type": "linear",
                            "colors": ["#6ec8ff", "#164d9f"],
                            "direction": "vertical",
                        },
                        "opacity": 0.22,
                        "blend": "screen",
                        "z": 5,
                    }
                ],
                "motifs": [
                    {
                        "type": "grass",
                        "count": 18,
                        "region": [0.0, 0.76, 1.0, 1.0],
                        "color": "#1a5c36",
                        "size": 0.08,
                        "seed": 2,
                        "z": 7,
                    }
                ],
                "textures": [
                    {
                        "type": "ripple",
                        "count": 22,
                        "region": [0.0, 0.58, 1.0, 0.78],
                        "color": "#d8f3ff",
                        "density": 0.65,
                        "scale": 0.03,
                        "opacity": 0.42,
                        "blend": "screen",
                        "seed": 6,
                        "z": 6,
                    }
                ],
                "materials": [
                    {
                        "type": "foliage",
                        "region": [0.0, 0.78, 1.0, 1.0],
                        "colors": ["#1e7a4a", "#071e18"],
                        "intensity": 0.55,
                        "scale": 0.05,
                        "opacity": 0.58,
                        "seed": 8,
                        "z": 8,
                    }
                ],
                "terrains": [
                    {
                        "type": "mountain",
                        "points": [[0.0, 0.56], [0.35, 0.28], [0.65, 0.56], [1.0, 0.36]],
                        "base": 0.78,
                        "fill": "#405070",
                        "shade": "#182030",
                        "highlight": "#7890b0",
                        "opacity": 0.7,
                        "z": 4,
                    }
                ],
                "reflections": [
                    {
                        "type": "vertical",
                        "source": [0.0, 0.18, 1.0, 0.56],
                        "target": [0.0, 0.56, 1.0, 0.78],
                        "opacity": 0.34,
                        "blur": 0.02,
                        "fade": 0.55,
                        "tint": "#286fc4",
                        "blend": "screen",
                        "z": 6,
                    }
                ],
                "warps": [
                    {
                        "type": "wave",
                        "region": [0.0, 0.56, 1.0, 0.78],
                        "direction": "horizontal",
                        "amplitude": 0.025,
                        "wavelength": 0.5,
                        "phase": 0.15,
                        "seed": 12,
                        "z": 7,
                    }
                ],
                "atmosphere": {"color": "#d8e8f0", "horizon": 0.56, "height": 0.18, "strength": 0.28},
                "veils": [
                    {
                        "type": "mist",
                        "region": [0.0, 0.48, 1.0, 0.70],
                        "color": "#d8e8f0",
                        "opacity": 0.24,
                        "blur": 0.02,
                        "blend": "screen",
                        "z": 8,
                    }
                ],
                "lights": [
                    {
                        "type": "radial",
                        "x": 0.25,
                        "y": 0.25,
                        "radius": 0.28,
                        "color": "#ffcf8a",
                        "intensity": 0.5,
                    }
                ],
                "beams": [
                    {
                        "type": "sunbeam",
                        "x": 0.25,
                        "y": 0.25,
                        "angle": 72.0,
                        "length": 0.66,
                        "spread": 20.0,
                        "color": "#ffcf8a",
                        "opacity": 0.24,
                        "blur": 0.02,
                        "blend": "screen",
                        "count": 1,
                        "z": 7,
                    }
                ],
                "clouds": [
                    {
                        "type": "cumulus",
                        "region": [0.05, 0.10, 0.95, 0.35],
                        "color": "#fff5dd",
                        "shadow": "#8aa0b8",
                        "opacity": 0.44,
                        "blur": 0.02,
                        "count": 2,
                        "lobes": 5,
                        "scale": 0.12,
                        "blend": "screen",
                        "seed": 9,
                        "z": 3,
                    }
                ],
                "shadows": [
                    {
                        "type": "ellipse",
                        "x": 0.55,
                        "y": 0.78,
                        "width": 0.52,
                        "height": 0.10,
                        "opacity": 0.36,
                        "blur": 0.025,
                        "blend": "multiply",
                        "z": 9,
                    }
                ],
                "focus": {
                    "type": "depth",
                    "region": [0.0, 0.0, 0.82, 0.80],
                    "blur": 0.02,
                    "falloff": 0.10,
                    "mode": "outside",
                },
                "style": {"grain": 0.0, "vignette": 0.0},
            }
        ),
        encoding="utf-8",
    )

    result = generate_image(
        GenerateOptions(
            prompt="cinematic red sun over blue ocean with detailed foreground",
            output_dir=tmp_path / "out",
            scene_plan=plan_path,
            width=200,
            height=120,
            max_iterations=3,
            threshold=0.1,
            seed=4,
        )
    )

    sun_pixel = result.image.getpixel((50, 30))
    foreground_pixel = result.image.getpixel((100, 110))

    assert sun_pixel[0] > 200
    assert sun_pixel[0] > sun_pixel[1] + 35
    assert sun_pixel[0] > sun_pixel[2] + 90
    assert foreground_pixel[1] > foreground_pixel[0]
    assert result.metadata["scene_plan_used"] is True
    assert result.metadata["scene_plan_title"] == "Claude planned deliberate composition"
    assert result.metadata["scene_plan_objects"] == ["sun", "ocean", "foreground"]
    assert result.metadata["scene_plan_background_stop_count"] == 3
    assert result.metadata["scene_plan_element_count"] == 2
    assert result.metadata["scene_plan_gradient_count"] == 1
    assert result.metadata["scene_plan_motif_count"] == 1
    assert result.metadata["scene_plan_texture_count"] == 1
    assert result.metadata["scene_plan_material_count"] == 1
    assert result.metadata["scene_plan_terrain_count"] == 1
    assert result.metadata["scene_plan_reflection_count"] == 1
    assert result.metadata["scene_plan_warp_count"] == 1
    assert result.metadata["scene_plan_atmosphere_used"] is True
    assert result.metadata["scene_plan_veil_count"] == 1
    assert result.metadata["scene_plan_light_count"] == 1
    assert result.metadata["scene_plan_beam_count"] == 1
    assert result.metadata["scene_plan_cloud_count"] == 1
    assert result.metadata["scene_plan_shadow_count"] == 1
    assert result.metadata["scene_plan_focus_used"] is True
    assert result.metadata["scene_plan_focus_blur"] == 0.02
    assert result.metadata["scene_plan_antialias"] == 0.0
    assert result.metadata["revision_hints"] == []


def test_scene_plan_metadata_gives_claude_revision_hints_when_alignment_is_low(tmp_path: Path):
    plan_path = tmp_path / "weak-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Weak empty plan",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
            }
        ),
        encoding="utf-8",
    )

    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean",
            output_dir=tmp_path / "out",
            scene_plan=plan_path,
            width=120,
            height=80,
            max_iterations=2,
            threshold=0.95,
            seed=3,
            auto_refine=False,
        )
    )

    assert result.metadata["met_threshold"] is False
    assert result.metadata["auto_refine"] is False
    assert result.metadata["revision_hints"] == [
        "The image caption missed requested objects: ocean, sun. Revise the scene plan so those objects read clearly in the rendered image.",
        "The image caption missed requested colors: blue, red. Use larger, clearer color regions or lighting accents for those colors.",
        "Add missing scene-plan objects: ocean, sun.",
        "Strengthen requested colors: red, blue. Use palette entries, background stops, fills, materials, or lights that visibly contain them.",
        "Increase prompt-object evidence with explicit shapes, terrain, materials, motifs, or elements for: ocean, sun.",
        "Increase tonal separation with clearer foreground/background contrast, shadows, lights, or silhouettes.",
    ]


def test_scene_plan_elements_render_with_z_order(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude detail layers",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.5,
                        "height": 0.5,
                        "fill": "#ff0000",
                        "z": 1,
                    },
                    {
                        "type": "ellipse",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.35,
                        "height": 0.35,
                        "fill": "#0000ff",
                        "z": 2,
                    },
                    {
                        "type": "polyline",
                        "points": [[0.2, 0.75], [0.8, 0.75]],
                        "stroke": "#ffffff",
                        "width": 0.05,
                        "z": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100)

    center = image.getpixel((50, 50))
    line = image.getpixel((50, 75))
    rectangle_only = image.getpixel((30, 30))

    assert center[2] > 200
    assert center[0] < 80
    assert line[0] > 220 and line[1] > 220 and line[2] > 220
    assert rectangle_only[0] > 200


def test_scene_plan_path_elements_render_curved_filled_shapes(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude curved path",
                "palette": ["#050812"],
                "background": {"top": "#050812", "bottom": "#050812"},
                "objects": [],
                "elements": [
                    {
                        "type": "path",
                        "commands": [
                            ["M", 0.08, 0.62],
                            ["C", 0.26, 0.38, 0.44, 0.88, 0.62, 0.58],
                            ["Q", 0.78, 0.36, 0.92, 0.62],
                            ["L", 0.92, 0.95],
                            ["L", 0.08, 0.95],
                            ["Z"],
                        ],
                        "fill": "#2aa6d9",
                        "stroke": "#d8f3ff",
                        "width": 0.012,
                        "opacity": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=140, height=100)
    filled_pixel = image.getpixel((70, 82))
    stroke_pixel = image.getpixel((70, 67))
    background_pixel = image.getpixel((70, 20))

    assert filled_pixel[2] > 150
    assert filled_pixel[1] > filled_pixel[0]
    assert stroke_pixel[0] > 120 and stroke_pixel[2] > 170
    assert background_pixel == (5, 8, 18)


def test_scene_plan_soft_elements_blur_and_screen_blend(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude soft element",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "elements": [
                    {
                        "type": "ellipse",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.20,
                        "height": 0.20,
                        "fill": "#ff8844",
                        "opacity": 0.72,
                        "blur": 0.06,
                        "blend": "screen",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=120, height=80)
    center = image.getpixel((60, 40))
    soft_edge = image.getpixel((74, 40))
    far_background = image.getpixel((110, 40))

    assert center[0] > 120
    assert soft_edge[0] > far_background[0] + 8
    assert soft_edge[1] > far_background[1] + 3
    assert far_background == (32, 32, 32)


def test_scene_plan_elements_support_overlay_and_soft_light_blends(tmp_path: Path):
    plan_path = tmp_path / "blend-mode-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude richer blend modes",
                "palette": ["#606060"],
                "background": {"top": "#606060", "bottom": "#606060"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.25,
                        "y": 0.5,
                        "width": 0.42,
                        "height": 1.0,
                        "fill": "#e07840",
                        "opacity": 1.0,
                        "blend": "overlay",
                    },
                    {
                        "type": "rectangle",
                        "x": 0.75,
                        "y": 0.5,
                        "width": 0.42,
                        "height": 1.0,
                        "fill": "#ffffff",
                        "opacity": 1.0,
                        "blend": "soft-light",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=40)
    overlay_pixel = image.getpixel((25, 20))
    soft_light_pixel = image.getpixel((75, 20))
    background_pixel = image.getpixel((50, 20))

    assert plan.elements[0].blend == "overlay"
    assert plan.elements[1].blend == "soft-light"
    assert overlay_pixel == (168, 90, 48)
    assert all(150 <= channel <= 160 for channel in soft_light_pixel)
    assert background_pixel == (96, 96, 96)


def test_scene_plan_gradient_element_fills_shape_without_leaking(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude gradient element",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.5,
                        "height": 0.6,
                        "gradient": {
                            "type": "linear",
                            "colors": ["#0044ff", "#ffcc44"],
                            "direction": "vertical",
                        },
                        "opacity": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100)
    upper = image.getpixel((50, 25))
    lower = image.getpixel((50, 75))
    outside = image.getpixel((10, 50))

    assert upper[2] > upper[0] + 120
    assert lower[0] > lower[2] + 120
    assert lower[1] > upper[1] + 80
    assert outside == (16, 16, 16)


def test_scene_plan_background_color_stops_create_horizon_band(tmp_path: Path):
    plan_path = tmp_path / "background-stops-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude sky stops",
                "palette": ["#000010"],
                "background": {
                    "top": "#000010",
                    "bottom": "#001040",
                    "stops": [
                        {"at": 0.0, "color": "#000010"},
                        {"at": 0.50, "color": "#ffcc66"},
                        {"at": 1.0, "color": "#001040"},
                    ],
                },
                "objects": [],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100)
    top = image.getpixel((50, 0))
    horizon = image.getpixel((50, 50))
    bottom = image.getpixel((50, 99))

    assert top == (0, 0, 16)
    assert horizon[0] > 240
    assert horizon[1] > 180
    assert horizon[2] < 120
    assert bottom[2] > bottom[0] + 55


def test_scene_plan_radial_gradient_element_supports_soft_focus(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude radial gradient",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "elements": [
                    {
                        "type": "ellipse",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.6,
                        "height": 0.6,
                        "gradient": {
                            "type": "radial",
                            "colors": ["#fff0a0", "#ff5533"],
                            "center": [0.45, 0.45],
                            "radius": 0.75,
                        },
                        "opacity": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100)
    center = image.getpixel((50, 50))
    edge = image.getpixel((75, 50))
    outside = image.getpixel((12, 50))

    assert center[0] >= edge[0]
    assert center[1] > edge[1] + 45
    assert edge[0] > edge[1]
    assert outside == (32, 32, 32)


def test_scene_plan_motifs_expand_repeated_details_deterministically(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude motif detail",
                "palette": ["#050812"],
                "background": {"top": "#050812", "bottom": "#050812"},
                "objects": [],
                "motifs": [
                    {
                        "type": "starfield",
                        "count": 20,
                        "region": [0.0, 0.0, 1.0, 0.5],
                        "color": "#fff5cc",
                        "size": 0.018,
                        "seed": 9,
                        "z": 4,
                    },
                    {
                        "type": "grass",
                        "count": 16,
                        "region": [0.0, 0.72, 1.0, 1.0],
                        "color": "#1c7a42",
                        "size": 0.09,
                        "seed": 11,
                        "z": 5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=120, height=80, seed=5)
    image_b = render_scene_plan(plan, width=120, height=80, seed=5)
    top_bright_pixels = 0
    lower_green_pixels = 0
    for y in range(0, 40):
        for x in range(120):
            pixel = image_a.getpixel((x, y))
            if pixel[0] > 180 and pixel[1] > 170 and pixel[2] > 120:
                top_bright_pixels += 1
    for y in range(58, 80):
        for x in range(120):
            pixel = image_a.getpixel((x, y))
            if pixel[1] > pixel[0] * 1.5 and pixel[1] > pixel[2] * 1.2:
                lower_green_pixels += 1

    assert image_a.tobytes() == image_b.tobytes()
    assert top_bright_pixels > 20
    assert lower_green_pixels > 20


def test_scene_plan_textures_add_region_bound_surface_detail(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude texture detail",
                "palette": ["#202830"],
                "background": {"top": "#202830", "bottom": "#202830"},
                "objects": [],
                "textures": [
                    {
                        "type": "hatching",
                        "count": 36,
                        "region": [0.0, 0.50, 1.0, 1.0],
                        "color": "#f6e2b5",
                        "density": 0.8,
                        "scale": 0.035,
                        "opacity": 0.55,
                        "blend": "screen",
                        "seed": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=120, height=80, seed=7)
    image_b = render_scene_plan(plan, width=120, height=80, seed=7)
    upper_changed = 0
    lower_changed = 0
    for y in range(80):
        for x in range(120):
            pixel = image_a.getpixel((x, y))
            if pixel != (32, 40, 48):
                if y < 40:
                    upper_changed += 1
                else:
                    lower_changed += 1

    assert image_a.tobytes() == image_b.tobytes()
    assert upper_changed == 0
    assert lower_changed > 80


def test_scene_plan_layers_elements_motifs_and_textures_by_global_z_order(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude global layer stack",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.4,
                        "height": 0.4,
                        "fill": "#ff0000",
                        "z": 5,
                    }
                ],
                "textures": [
                    {
                        "type": "paper",
                        "region": [0.0, 0.0, 1.0, 1.0],
                        "color": "#ffffff",
                        "density": 1.0,
                        "opacity": 1.0,
                        "blend": "normal",
                        "seed": 4,
                        "z": 1,
                    }
                ],
                "motifs": [
                    {
                        "type": "dots",
                        "count": 20,
                        "region": [0.45, 0.45, 0.55, 0.55],
                        "color": "#00ff00",
                        "size": 0.02,
                        "opacity": 1.0,
                        "seed": 6,
                        "z": 9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100, seed=5)
    center = image.getpixel((50, 50))
    rectangle_edge = image.getpixel((36, 36))

    assert rectangle_edge[0] > 220
    assert rectangle_edge[1] < 60
    assert center[1] > 180
    assert center[0] < 80


def test_scene_plan_materials_add_compact_region_bound_surface_quality(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude material detail",
                "palette": ["#202830"],
                "background": {"top": "#202830", "bottom": "#202830"},
                "objects": [],
                "materials": [
                    {
                        "type": "water",
                        "region": [0.0, 0.50, 1.0, 1.0],
                        "colors": ["#8bdcff", "#0b3b71"],
                        "intensity": 0.85,
                        "scale": 0.04,
                        "opacity": 0.72,
                        "seed": 13,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=120, height=80, seed=7)
    image_b = render_scene_plan(plan, width=120, height=80, seed=7)
    upper_changed = 0
    lower_blue_pixels = 0
    lower_bright_pixels = 0
    for y in range(80):
        for x in range(120):
            pixel = image_a.getpixel((x, y))
            if pixel != (32, 40, 48) and y < 40:
                upper_changed += 1
            if y >= 40 and pixel[2] > pixel[0] + 35:
                lower_blue_pixels += 1
            if y >= 40 and pixel[0] > 120 and pixel[1] > 145 and pixel[2] > 160:
                lower_bright_pixels += 1

    assert image_a.tobytes() == image_b.tobytes()
    assert upper_changed == 0
    assert lower_blue_pixels > 2500
    assert lower_bright_pixels > 40


def test_scene_plan_reflections_mirror_source_region_into_target(tmp_path: Path):
    plan_path = tmp_path / "reflection-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude mirror reflection",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.5,
                        "y": 0.25,
                        "width": 0.2,
                        "height": 0.2,
                        "fill": "#ff0000",
                        "opacity": 1.0,
                        "z": 1,
                    }
                ],
                "reflections": [
                    {
                        "type": "vertical",
                        "source": [0.4, 0.15, 0.6, 0.35],
                        "target": [0.4, 0.55, 0.6, 0.85],
                        "opacity": 0.85,
                        "blur": 0.0,
                        "fade": 0.0,
                        "blend": "normal",
                        "z": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=3)
    image_b = render_scene_plan(plan, width=100, height=100, seed=3)
    source_pixel = image_a.getpixel((50, 25))
    reflected_pixel = image_a.getpixel((50, 65))
    outside_target = image_a.getpixel((20, 65))

    assert image_a.tobytes() == image_b.tobytes()
    assert source_pixel[0] > 230
    assert reflected_pixel[0] > outside_target[0] + 160
    assert reflected_pixel[1] < 45
    assert outside_target == (16, 16, 16)


def test_scene_plan_warps_displace_region_pixels_deterministically(tmp_path: Path):
    plan_path = tmp_path / "warp-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude water warp",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "elements": [
                    {
                        "type": "rectangle",
                        "x": 0.42,
                        "y": 0.50,
                        "width": 0.18,
                        "height": 0.60,
                        "fill": "#ff0000",
                        "opacity": 1.0,
                        "z": 1,
                    }
                ],
                "warps": [
                    {
                        "type": "wave",
                        "region": [0.20, 0.20, 0.80, 0.80],
                        "direction": "horizontal",
                        "amplitude": 0.15,
                        "wavelength": 1.0,
                        "phase": 0.25,
                        "seed": 4,
                        "z": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=7)
    image_b = render_scene_plan(plan, width=100, height=100, seed=7)
    top_shifted = image_a.getpixel((58, 20))
    top_original_left = image_a.getpixel((34, 20))
    middle_shifted = image_a.getpixel((26, 50))
    middle_unshifted_right = image_a.getpixel((58, 50))
    outside_region = image_a.getpixel((10, 50))

    assert image_a.tobytes() == image_b.tobytes()
    assert top_shifted[0] > 220
    assert top_original_left == (16, 16, 16)
    assert middle_shifted[0] > 220
    assert middle_unshifted_right == (16, 16, 16)
    assert outside_region == (16, 16, 16)


def test_scene_plan_terrains_render_faceted_ridges_from_claude_points(tmp_path: Path):
    plan_path = tmp_path / "terrain-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude faceted terrain",
                "palette": ["#101010"],
                "background": {"top": "#101010", "bottom": "#101010"},
                "objects": [],
                "terrains": [
                    {
                        "type": "mountain",
                        "points": [[0.20, 0.70], [0.50, 0.20], [0.80, 0.70]],
                        "base": 0.92,
                        "fill": "#405070",
                        "shade": "#182030",
                        "highlight": "#7890b0",
                        "opacity": 1.0,
                        "facets": True,
                        "z": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=5)
    image_b = render_scene_plan(plan, width=100, height=100, seed=5)
    peak = image_a.getpixel((50, 35))
    shaded_facet = image_a.getpixel((42, 68))
    highlighted_facet = image_a.getpixel((62, 68))
    outside = image_a.getpixel((10, 80))

    assert image_a.tobytes() == image_b.tobytes()
    assert peak[2] > peak[0]
    assert shaded_facet[2] > shaded_facet[0]
    assert highlighted_facet[0] > shaded_facet[0] + 35
    assert highlighted_facet[1] > shaded_facet[1] + 45
    assert outside == (16, 16, 16)


def test_scene_plan_lights_brighten_and_tint_local_regions(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude lighting",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "lights": [
                    {
                        "type": "radial",
                        "x": 0.5,
                        "y": 0.5,
                        "radius": 0.36,
                        "color": "#ffcc66",
                        "intensity": 0.75,
                    },
                    {
                        "type": "shadow",
                        "x": 0.05,
                        "y": 0.05,
                        "radius": 0.42,
                        "intensity": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100)
    center = image.getpixel((50, 50))
    edge = image.getpixel((98, 98))
    shadow_corner = image.getpixel((4, 4))

    assert center[0] > edge[0] + 50
    assert center[1] > edge[1] + 35
    assert center[0] > center[2]
    assert shadow_corner[0] < edge[0]


def test_scene_plan_beams_render_directional_light_shafts(tmp_path: Path):
    plan_path = tmp_path / "beam-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude sun beam",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "beams": [
                    {
                        "type": "sunbeam",
                        "x": 0.5,
                        "y": 0.12,
                        "angle": 90.0,
                        "length": 0.82,
                        "spread": 20.0,
                        "color": "#ffcc66",
                        "opacity": 0.65,
                        "blur": 0.0,
                        "blend": "screen",
                        "count": 1,
                        "z": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=7)
    image_b = render_scene_plan(plan, width=100, height=100, seed=7)
    beam_center = image_a.getpixel((50, 62))
    outside = image_a.getpixel((10, 62))

    assert image_a.tobytes() == image_b.tobytes()
    assert beam_center[0] > outside[0] + 120
    assert beam_center[1] > outside[1] + 80
    assert beam_center[0] > beam_center[2]
    assert outside == (32, 32, 32)


def test_scene_plan_beams_respect_claude_authored_occlusion_masks(tmp_path: Path):
    plan_path = tmp_path / "beam-occlusion-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude occluded sun beam",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "beams": [
                    {
                        "type": "sunbeam",
                        "x": 0.5,
                        "y": 0.12,
                        "angle": 90.0,
                        "length": 0.82,
                        "spread": 20.0,
                        "color": "#ffcc66",
                        "opacity": 0.70,
                        "blur": 0.0,
                        "blend": "screen",
                        "count": 1,
                        "z": 1,
                        "occlusions": [
                            {"type": "rectangle", "region": [0.42, 0.44, 0.58, 0.72]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=100, height=100, seed=7)
    clear_beam = image.getpixel((50, 36))
    occluded_beam = image.getpixel((50, 58))
    outside = image.getpixel((10, 58))

    assert clear_beam[0] > outside[0] + 130
    assert clear_beam[1] > outside[1] + 90
    assert occluded_beam == outside


def test_scene_plan_clouds_render_soft_banks_deterministically(tmp_path: Path):
    plan_path = tmp_path / "cloud-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude cloud bank",
                "palette": ["#101020"],
                "background": {"top": "#101020", "bottom": "#101020"},
                "objects": [],
                "clouds": [
                    {
                        "type": "cumulus",
                        "region": [0.25, 0.20, 0.75, 0.42],
                        "color": "#fff5dd",
                        "shadow": "#708098",
                        "opacity": 0.9,
                        "blur": 0.0,
                        "count": 1,
                        "lobes": 5,
                        "scale": 0.20,
                        "blend": "screen",
                        "seed": 4,
                        "z": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=3)
    image_b = render_scene_plan(plan, width=100, height=100, seed=3)
    cloud_core = image_a.getpixel((50, 31))
    lower_clear = image_a.getpixel((50, 70))

    assert image_a.tobytes() == image_b.tobytes()
    assert cloud_core[0] > lower_clear[0] + 150
    assert cloud_core[1] > lower_clear[1] + 130
    assert cloud_core[2] > lower_clear[2] + 100
    assert lower_clear == (16, 16, 32)


def test_scene_plan_shadows_render_soft_grounding_shapes_deterministically(tmp_path: Path):
    plan_path = tmp_path / "shadow-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude contact shadow",
                "palette": ["#c8d0d8"],
                "background": {"top": "#c8d0d8", "bottom": "#c8d0d8"},
                "objects": [],
                "shadows": [
                    {
                        "type": "ellipse",
                        "x": 0.5,
                        "y": 0.58,
                        "width": 0.46,
                        "height": 0.18,
                        "color": "#202830",
                        "opacity": 0.72,
                        "blur": 0.03,
                        "blend": "multiply",
                        "z": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=120, height=80, seed=11)
    image_b = render_scene_plan(plan, width=120, height=80, seed=11)
    shadow_core = image_a.getpixel((60, 46))
    far_corner = image_a.getpixel((8, 8))
    soft_edge = image_a.getpixel((34, 46))

    assert image_a.tobytes() == image_b.tobytes()
    assert far_corner == (200, 208, 216)
    assert sum(shadow_core) < sum(far_corner) - 180
    assert sum(shadow_core) < sum(soft_edge) < sum(far_corner)


def test_scene_plan_veils_render_region_bound_atmospheric_depth(tmp_path: Path):
    plan_path = tmp_path / "veil-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude local mist veil",
                "palette": ["#202020"],
                "background": {"top": "#202020", "bottom": "#202020"},
                "objects": [],
                "veils": [
                    {
                        "type": "mist",
                        "region": [0.10, 0.38, 0.90, 0.66],
                        "color": "#d8e8f0",
                        "opacity": 0.62,
                        "blur": 0.0,
                        "blend": "screen",
                        "falloff": 0.20,
                        "direction": "vertical",
                        "z": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=100, height=100, seed=5)
    image_b = render_scene_plan(plan, width=100, height=100, seed=5)
    veil_core = image_a.getpixel((50, 52))
    veil_soft_edge = image_a.getpixel((50, 39))
    outside = image_a.getpixel((50, 20))
    side_outside = image_a.getpixel((4, 52))

    assert image_a.tobytes() == image_b.tobytes()
    assert outside == (32, 32, 32)
    assert side_outside == (32, 32, 32)
    assert sum(veil_core) > sum(outside) + 250
    assert sum(outside) < sum(veil_soft_edge) < sum(veil_core)


def test_scene_plan_focus_blurs_outside_claude_focal_region(tmp_path: Path):
    plan_path = tmp_path / "focus-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude focus blur",
                "palette": ["#000000"],
                "background": {"top": "#000000", "bottom": "#000000"},
                "objects": [],
                "elements": [
                    {"type": "rectangle", "x": 0.25, "y": 0.5, "width": 0.18, "height": 0.58, "fill": "#ffffff", "z": 1},
                    {"type": "rectangle", "x": 0.75, "y": 0.5, "width": 0.18, "height": 0.58, "fill": "#ffffff", "z": 1},
                ],
                "focus": {
                    "type": "depth",
                    "region": [0.0, 0.0, 0.5, 1.0],
                    "blur": 0.05,
                    "falloff": 0.0,
                    "mode": "outside",
                },
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image_a = render_scene_plan(plan, width=120, height=80, seed=9)
    image_b = render_scene_plan(plan, width=120, height=80, seed=9)
    pixels = image_a.load()
    left_intermediate = 0
    right_intermediate = 0
    for y in range(image_a.height):
        for x in range(image_a.width):
            red = pixels[x, y][0]
            if 0 < red < 255:
                if x < image_a.width // 2:
                    left_intermediate += 1
                else:
                    right_intermediate += 1

    assert image_a.tobytes() == image_b.tobytes()
    assert left_intermediate == 0
    assert right_intermediate > 80


def test_scene_plan_style_antialias_smooths_hard_edges_without_resizing(tmp_path: Path):
    hard_path = tmp_path / "hard-edge-scene-plan.json"
    soft_path = tmp_path / "soft-edge-scene-plan.json"
    base_plan = {
        "title": "Claude sharp edge test",
        "palette": ["#000000"],
        "background": {"top": "#000000", "bottom": "#000000"},
        "objects": [],
        "elements": [
            {
                "type": "polygon",
                "points": [[0.12, 0.12], [0.88, 0.22], [0.26, 0.88]],
                "fill": "#ffffff",
                "opacity": 1.0,
            }
        ],
    }
    hard_path.write_text(json.dumps(base_plan), encoding="utf-8")
    soft_path.write_text(json.dumps({**base_plan, "style": {"antialias": 1.0}}), encoding="utf-8")

    hard_image = render_scene_plan(parse_scene_plan(hard_path), width=80, height=60)
    soft_image_a = render_scene_plan(parse_scene_plan(soft_path), width=80, height=60)
    soft_image_b = render_scene_plan(parse_scene_plan(soft_path), width=80, height=60)
    hard_pixels = hard_image.load()
    soft_pixels = soft_image_a.load()
    hard_edge_pixels = sum(
        1 for y in range(hard_image.height) for x in range(hard_image.width) if 0 < hard_pixels[x, y][0] < 255
    )
    soft_edge_pixels = sum(
        1 for y in range(soft_image_a.height) for x in range(soft_image_a.width) if 0 < soft_pixels[x, y][0] < 255
    )

    assert soft_image_a.size == (80, 60)
    assert soft_image_a.tobytes() == soft_image_b.tobytes()
    assert hard_edge_pixels == 0
    assert soft_edge_pixels > 40


def test_scene_plan_vignette_uses_smooth_falloff_without_hard_rectangular_edges(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "smooth vignette",
                "palette": ["#808080"],
                "background": {"top": "#808080", "bottom": "#808080"},
                "objects": [],
                "style": {"vignette": 0.6},
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=120, height=80)
    middle_row = [sum(image.getpixel((x, 40))) / 3 for x in range(120)]
    adjacent_jumps = [abs(middle_row[x + 1] - middle_row[x]) for x in range(119)]

    assert max(adjacent_jumps) < 18
    assert sum(image.getpixel((60, 40))) > sum(image.getpixel((0, 0)))


def test_scene_plan_style_color_grade_adjusts_final_image(tmp_path: Path):
    base_path = tmp_path / "base-scene-plan.json"
    graded_path = tmp_path / "graded-scene-plan.json"
    base_plan = {
        "title": "ungraded color",
        "palette": ["#507090"],
        "background": {"top": "#507090", "bottom": "#507090"},
        "objects": [],
    }
    graded_plan = {
        **base_plan,
        "title": "graded color",
        "style": {"saturation": 0.8, "contrast": 0.5, "warmth": 0.7},
    }
    base_path.write_text(json.dumps(base_plan), encoding="utf-8")
    graded_path.write_text(json.dumps(graded_plan), encoding="utf-8")

    base = render_scene_plan(parse_scene_plan(base_path), width=80, height=60)
    graded = render_scene_plan(parse_scene_plan(graded_path), width=80, height=60)
    base_pixel = base.getpixel((40, 30))
    graded_pixel = graded.getpixel((40, 30))

    assert graded_pixel[0] > base_pixel[0] + 8
    assert graded_pixel[2] < base_pixel[2]
    assert max(graded_pixel) - min(graded_pixel) > max(base_pixel) - min(base_pixel)


def test_scene_plan_detail_style_controls_cpu_sharpening(tmp_path: Path):
    base_plan = {
        "title": "detail style test",
        "palette": ["#102040", "#d8f0ff"],
        "background": {"top": "#102040", "bottom": "#d8f0ff"},
        "objects": [{"type": "mountain", "x": 0.5, "y": 0.52, "size": 0.32, "color": "#455a78", "layers": 3}],
        "elements": [
            {
                "type": "polyline",
                "points": [[0.08, 0.72], [0.25, 0.66], [0.46, 0.74], [0.72, 0.64], [0.94, 0.70]],
                "stroke": "#f6e2b5",
                "width": 0.012,
                "opacity": 0.72,
                "z": 8,
            }
        ],
        "textures": [
            {
                "type": "hatching",
                "count": 120,
                "region": [0.05, 0.45, 0.95, 0.92],
                "color": "#ffffff",
                "density": 0.8,
                "scale": 0.025,
                "opacity": 0.28,
                "blend": "screen",
                "seed": 8,
                "z": 9,
            }
        ],
        "style": {"antialias": 1.0},
    }
    plain_path = tmp_path / "plain.json"
    detail_path = tmp_path / "detail.json"
    plain_path.write_text(json.dumps(base_plan), encoding="utf-8")
    detailed_plan = dict(base_plan)
    detailed_plan["style"] = {"antialias": 1.0, "detail": 0.8, "sharpen": 0.7}
    detail_path.write_text(json.dumps(detailed_plan), encoding="utf-8")

    plain = render_scene_plan(parse_scene_plan(plain_path), width=160, height=96, seed=4)
    detailed = render_scene_plan(parse_scene_plan(detail_path), width=160, height=96, seed=4)

    assert parse_scene_plan(detail_path).style["detail"] == 0.8
    assert parse_scene_plan(detail_path).style["sharpen"] == 0.7
    assert image_detail_metrics(detailed)["detail_score"] > image_detail_metrics(plain)["detail_score"]


def test_scene_plan_style_bloom_spreads_bright_regions_softly(tmp_path: Path):
    plan_path = tmp_path / "bloom-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "bloom style",
                "palette": ["#080808"],
                "background": {"top": "#080808", "bottom": "#080808"},
                "objects": [],
                "elements": [
                    {
                        "type": "ellipse",
                        "x": 0.5,
                        "y": 0.5,
                        "width": 0.16,
                        "height": 0.16,
                        "fill": "#fff0a0",
                        "opacity": 1.0,
                    }
                ],
                "style": {"bloom": 0.8},
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=120, height=80)
    center = image.getpixel((60, 40))
    halo = image.getpixel((73, 40))
    far = image.getpixel((112, 40))

    assert center[0] > 220
    assert halo[0] > far[0] + 20
    assert halo[1] > far[1] + 15


def test_scene_plan_atmosphere_adds_smooth_horizon_depth(tmp_path: Path):
    plan_path = tmp_path / "atmosphere-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "Claude atmosphere",
                "palette": ["#102030"],
                "background": {"top": "#102030", "bottom": "#102030"},
                "objects": [],
                "atmosphere": {
                    "type": "horizon_fog",
                    "color": "#d8e8f0",
                    "horizon": 0.50,
                    "height": 0.26,
                    "strength": 0.65,
                },
            }
        ),
        encoding="utf-8",
    )
    plan = parse_scene_plan(plan_path)

    image = render_scene_plan(plan, width=120, height=80)
    horizon = image.getpixel((60, 40))
    upper = image.getpixel((60, 12))
    lower = image.getpixel((60, 72))
    middle_row = [sum(image.getpixel((60, y))) / 3 for y in range(80)]
    adjacent_jumps = [abs(middle_row[y + 1] - middle_row[y]) for y in range(79)]

    assert horizon[0] > upper[0] + 70
    assert horizon[2] > lower[2] + 70
    assert upper == (16, 32, 48)
    assert lower == (16, 32, 48)
    assert max(adjacent_jumps) < 16
