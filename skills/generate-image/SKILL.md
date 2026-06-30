---
name: generate-image
description: Generate a local PNG from a text prompt, optionally using a reference or initial image, through the CPU-first claude-imagegen executable.
---

# Generate Image

Use this skill when the user asks Claude Code to generate an image from a prompt in the current folder, especially when they mention local generation, iterative refinement, reference images, initial generated images, cosine similarity, caption backchecking, optional CLIP/BLIP scoring, or RGB pixel export.

## Workflow

1. Confirm the prompt is present. If it is missing, ask for it.
2. Prefer a project-local output folder such as `claude-imagegen-output/<short-slug>`.
3. Write a Claude-authored `scene-plan.json` beside the output folder. This is where Claude Code should spend most of the semantic work: translate the prompt into explicit composition, palette, object placement, foreground/background depth, style hints, and reference-image observations.

```json
{
  "title": "short descriptive title",
  "palette": ["#102040", "#ff5533", "#286fc4", "#123d2a"],
  "background": {
    "top": "#102040",
    "bottom": "#205080",
    "direction": "vertical",
    "stops": [
      { "at": 0.0, "color": "#102040" },
      { "at": 0.48, "color": "#ffcf8a" },
      { "at": 1.0, "color": "#205080" }
    ]
  },
  "objects": [
    { "type": "sun", "label": "large warm focal point", "x": 0.25, "y": 0.24, "size": 0.18, "color": "#ff5533" },
    { "type": "mountain", "label": "layered distant silhouettes", "y": 0.58, "size": 0.32, "color": "#49394c", "layers": 3 },
    { "type": "ocean", "label": "blue reflective lower half", "y": 0.56, "color": "#286fc4" },
    { "type": "foreground", "label": "dark framing base", "y": 0.78, "color": "#123d2a" }
  ],
  "elements": [
    { "type": "glow", "label": "sun bloom", "x": 0.25, "y": 0.24, "width": 0.20, "height": 0.20, "fill": "#ffb45a", "opacity": 0.45, "z": 1 },
    { "type": "ellipse", "label": "soft atmospheric haze", "x": 0.28, "y": 0.26, "width": 0.38, "height": 0.24, "fill": "#ffcf8a", "opacity": 0.24, "blur": 0.08, "blend": "screen", "z": 2 },
    { "type": "rectangle", "label": "deeper water gradient", "x": 0.5, "y": 0.68, "width": 1.0, "height": 0.24, "gradient": { "type": "linear", "colors": ["#2e8ddb", "#0a3d72"], "direction": "vertical" }, "opacity": 0.46, "blend": "multiply", "z": 5 },
    { "type": "polyline", "label": "water reflection highlight", "points": [[0.12, 0.66], [0.38, 0.68], [0.62, 0.67], [0.88, 0.65]], "stroke": "#f6e2b5", "width": 0.008, "opacity": 0.72, "blur": 0.012, "blend": "screen", "z": 6 },
    { "type": "path", "label": "curved shoreline", "commands": [["M", 0.0, 0.84], ["C", 0.18, 0.76, 0.38, 0.88, 0.58, 0.78], ["Q", 0.78, 0.70, 1.0, 0.83], ["L", 1.0, 1.0], ["L", 0.0, 1.0], ["Z"]], "fill": "#0b241b", "stroke": "#174b36", "width": 0.006, "opacity": 0.78, "z": 7 },
    { "type": "polygon", "label": "foreground silhouette", "points": [[0.0, 0.84], [0.18, 0.76], [0.36, 0.86], [0.58, 0.74], [0.82, 0.85], [1.0, 0.78], [1.0, 1.0], [0.0, 1.0]], "fill": "#0b241b", "opacity": 0.78, "z": 7 }
  ],
  "motifs": [
    { "type": "starfield", "label": "small upper-sky stars", "count": 36, "region": [0.0, 0.02, 1.0, 0.38], "color": "#fff5cc", "size": 0.006, "opacity": 0.72, "seed": 12, "z": 8 },
    { "type": "grass", "label": "foreground blade texture", "count": 80, "region": [0.0, 0.78, 1.0, 1.0], "color": "#1a5c36", "size": 0.045, "opacity": 0.65, "seed": 21, "z": 9 }
  ],
  "textures": [
    { "type": "ripple", "label": "soft water surface bands", "count": 24, "region": [0.0, 0.56, 1.0, 0.78], "color": "#d8f3ff", "density": 0.6, "scale": 0.025, "opacity": 0.36, "blend": "screen", "seed": 31, "z": 9 },
    { "type": "hatching", "label": "foreground painterly surface marks", "count": 90, "region": [0.0, 0.78, 1.0, 1.0], "color": "#2a7047", "density": 0.65, "scale": 0.035, "opacity": 0.42, "blend": "normal", "seed": 32, "z": 10 }
  ],
  "materials": [
    { "type": "water", "label": "reflective ocean material", "region": [0.0, 0.56, 1.0, 0.80], "colors": ["#8bdcff", "#0b3b71"], "intensity": 0.72, "scale": 0.035, "opacity": 0.58, "seed": 41, "z": 8 },
    { "type": "foliage", "label": "dense foreground foliage material", "region": [0.0, 0.78, 1.0, 1.0], "colors": ["#1e7a4a", "#071e18"], "intensity": 0.64, "scale": 0.045, "opacity": 0.52, "seed": 42, "z": 11 }
  ],
  "terrains": [
    { "type": "mountain", "label": "faceted mountain ridge from Claude points", "points": [[0.02, 0.56], [0.22, 0.24], [0.42, 0.56], [0.64, 0.34], [0.96, 0.56]], "base": 0.78, "fill": "#405070", "shade": "#182030", "highlight": "#7890b0", "opacity": 0.76, "facets": true, "z": 4 }
  ],
  "reflections": [
    { "type": "vertical", "label": "soft mirrored sky and mountains in water", "source": [0.0, 0.16, 1.0, 0.56], "target": [0.0, 0.56, 1.0, 0.80], "opacity": 0.36, "blur": 0.025, "fade": 0.62, "tint": "#2d88d8", "blend": "screen", "z": 8 }
  ],
  "warps": [
    { "type": "wave", "label": "subtle water displacement for reflected shapes", "region": [0.0, 0.56, 1.0, 0.82], "direction": "horizontal", "amplitude": 0.018, "wavelength": 0.38, "phase": 0.20, "seed": 43, "z": 10 }
  ],
  "atmosphere": { "type": "horizon_fog", "label": "cool horizon haze for depth", "color": "#d8e8f0", "horizon": 0.56, "height": 0.22, "strength": 0.32 },
  "veils": [
    { "type": "mist", "label": "localized sea mist over the waterline", "region": [0.04, 0.48, 0.96, 0.68], "color": "#d8e8f0", "opacity": 0.22, "blur": 0.026, "blend": "screen", "falloff": 0.18, "direction": "vertical", "z": 8 }
  ],
  "lights": [
    { "type": "radial", "label": "warm focal light", "x": 0.25, "y": 0.24, "radius": 0.35, "color": "#ffcf8a", "intensity": 0.48, "z": 10 },
    { "type": "shadow", "label": "dark lower edge", "x": 0.50, "y": 0.95, "radius": 0.55, "intensity": 0.25, "z": 11 }
  ],
  "beams": [
    { "type": "sunbeam", "label": "soft diagonal shafts through haze", "x": 0.25, "y": 0.24, "angle": 70.0, "length": 0.70, "spread": 24.0, "color": "#ffcf8a", "opacity": 0.22, "blur": 0.035, "blend": "screen", "count": 2, "seed": 44, "z": 9, "occlusions": [{ "type": "polygon", "points": [[0.08, 0.55], [0.46, 0.55], [0.46, 0.63], [0.08, 0.63]] }] }
  ],
  "clouds": [
    { "type": "cumulus", "label": "soft upper cloud bank", "region": [0.05, 0.08, 0.95, 0.34], "color": "#fff5dd", "shadow": "#8aa0b8", "opacity": 0.38, "blur": 0.025, "count": 3, "lobes": 5, "scale": 0.13, "blend": "screen", "seed": 45, "z": 4 }
  ],
  "shadows": [
    { "type": "ellipse", "label": "soft grounding shadow", "x": 0.50, "y": 0.78, "width": 0.52, "height": 0.10, "color": "#101820", "opacity": 0.34, "blur": 0.035, "blend": "multiply", "z": 9 }
  ],
  "focus": { "type": "depth", "label": "focal composition band", "region": [0.04, 0.10, 0.84, 0.82], "blur": 0.018, "falloff": 0.12, "mode": "outside" },
  "style": { "grain": 0.08, "vignette": 0.12, "saturation": 0.32, "contrast": 0.22, "warmth": 0.18, "bloom": 0.22, "antialias": 1.0 }
}
```

4. Run:

```bash
claude-imagegen generate \
  --prompt "<user prompt>" \
  --scene-plan "claude-imagegen-output/<short-slug>/scene-plan.json" \
  --output-dir "claude-imagegen-output/<short-slug>" \
  --width 720 \
  --height 480 \
  --max-iterations 32 \
  --threshold 0.58 \
  --similarity-backend local
```

5. If the user provides a reference image, inspect it and encode the important palette/composition observations in `scene-plan.json`, then also add `--reference-image "/absolute/path/to/reference.png"`. After a run, use `metadata.json` field `reference_palette` as concrete extracted colors to fold back into palette entries, gradients, materials, lights, or veils.
6. If the user provides an already-generated image file to continue from, add `--initial-image "/absolute/path/to/image.png"`. If the user provides a previous claude-imagegen output folder, prefer:

```bash
claude-imagegen refine \
  --from-dir "claude-imagegen-output/<previous-slug>" \
  --prompt "<revised user prompt>" \
  --output-dir "claude-imagegen-output/<new-slug>" \
  --max-iterations 8 \
  --threshold 0.62
```

After a refine run, use `metadata.json` fields `initial_similarity_score`, `refined_from`, `parent_image`, and `refinement_lineage_depth` to report continuity with the previous image. Use `initial_palette` to preserve or intentionally shift the starting image colors in the next `scene-plan.json`.
7. Use `"background"` `"stops"` when Claude needs a richer sky, horizon glow, or color banding. Stops use normalized `"at"` positions and `"color"` values. Supported `"direction"` values are `vertical`, `horizontal`, `diagonal`, and `reverse-diagonal`. Keep `top` and `bottom` for compatibility, but prefer explicit stops for polished scenes.
8. Use `"elements"` for detail work Claude can compute directly: reflection lines, rim lights, silhouettes, windows, constellations, graphic shapes, and foreground accents. Supported element types include `polyline`, `path`, `polygon`, `ellipse`, `rectangle`, `glow`, and `arc`. `path` supports `M`, `L`, `Q`, `C`, and `Z` commands with normalized coordinates. Filled rectangles, ellipses, polygons, and closed paths can include `"gradient"` with `"type": "linear"` or `"radial"`, at least two `"colors"`, and a linear `"direction"` of `vertical`, `horizontal`, `diagonal`, or `reverse-diagonal`. Any element can include `"blur"` for soft edges and `"blend"` of `normal`, `screen`, `multiply`, `overlay`, or `soft-light`; prefer `overlay` for stronger material color and `soft-light` for subtle tonal shaping.
9. Use `"motifs"` for repeated details where Claude should specify intent compactly: stars, sparkles, grass blades, rain streaks, dots, and window lights. Supported motif types include `starfield`, `grass`, `rain`, `window_lights`, and generic dots.
10. Use `"textures"` for region-bound surface quality that should not require listing individual marks. Supported texture types include `hatching`, `crosshatch`, `ripple`, `paper`, `grain`, `noise`, `mist`, and speckles. Prefer textures for water bands, painterly foreground marks, atmospheric grain, fabric, stone, and subtle surface breakup.
11. Use `"materials"` when Claude can describe a whole region with one semantic surface directive. Supported material types include `water`, `foliage`, `metal`, and generic surfaces. Each material should include a normalized `region`, at least two `colors`, `intensity`, `scale`, `opacity`, `seed`, and `z`. Prefer materials for broad surface quality, then use elements/textures/motifs for specific accents.
12. Use `"terrains"` when Claude should control ridges, mountains, islands, or cliffs with compact normalized points instead of procedural randomness. Supported fields are `points`, `base`, `fill`, optional `shade`, optional `highlight`, `opacity`, `blur`, `blend`, `facets`, and `z`.
13. Use `"reflections"` when a rendered source region should be mirrored into another region, especially water. Supported fields are normalized `source`, normalized `target`, `opacity`, `blur`, `fade`, optional `tint`, `blend`, and `z`. Prefer reflections for broad mirrored shapes, then use textures/elements for ripples and highlights.
14. Use `"warps"` when an already-rendered region needs deterministic distortion, especially reflected water. Supported fields are normalized `region`, `direction` of `horizontal` or `vertical`, `amplitude`, `wavelength`, `phase`, `seed`, and `z`. Prefer small amplitudes for natural water; use reflections/materials first, then warp them.
15. Use `"atmosphere"` for depth and aerial perspective. Supported fields are `color`, `horizon`, `height`, and `strength`, all normalized except `color`. Prefer it for mist, haze, distance fade, and soft horizon depth.
16. Use `"veils"` for localized mist, fog sheets, smoke, rain wash, underwater haze, or soft glow layers that should affect only one region instead of the whole horizon. Supported fields are normalized `region`, `color`, `opacity`, `blur`, `blend`, `falloff`, `direction`, and `z`. Prefer `screen` veils for haze/glow, `multiply` veils for smoke or dim foreground atmosphere, and `soft-light` veils for gentle color wash.
17. Use `"lights"` for global mood and local illumination. Supported light types include `radial`, `rim`, `tint`, and `shadow`; each light should include `x`, `y`, `radius`, `color` when brightening, and `intensity`.
18. Use `"beams"` for directional shafts of light through haze, forest gaps, windows, or sunrise/sunset scenes. Supported fields are `x`, `y`, `angle` in screen degrees, `length`, `spread` in degrees, `color`, `opacity`, `blur`, `blend`, `count`, `seed`, `z`, and optional `"occlusions"`. Beam `occlusions` are Claude-authored masks with `type` of `rectangle`, `ellipse`, or `polygon`; rectangles and ellipses use normalized `region`, while polygons use normalized `points`. Use `angle: 90` for downward beams and occlusions when rays should pass behind mountains, windows, or foreground layers.
19. Use `"clouds"` for compact cloud banks, haze puffs, and soft sky massing. Supported fields are normalized `region`, `color`, optional `shadow`, `opacity`, `blur`, `count`, `lobes`, `scale`, `blend`, `seed`, and `z`. Prefer clouds over many hand-written ellipse elements when the prompt calls for sky texture.
20. Use `"shadows"` for cast shadows, contact shadows, and grounding. Supported shadow types are `ellipse`, `rectangle`, and `polygon`. Ellipse and rectangle shadows use normalized `x`, `y`, `width`, and `height`; polygon shadows use normalized `points`. All shadows support `color`, `opacity`, `blur`, `blend`, and `z`. Prefer soft `multiply` shadows to make objects feel grounded without adding local semantic inference.
21. Use `"focus"` for depth-of-field or selective softness after the scene is composed. Supported fields are normalized `region`, `blur`, `falloff`, and `mode` of `outside` or `inside`. Prefer `mode: "outside"` to keep the main subject sharp while softly blurring background or edge clutter.
22. Use `"style"` for final Claude-authored image finishing. Supported fields are `grain`, `vignette`, `saturation`, `contrast`, `warmth`, `bloom`, and `antialias`, each from `0.0` to `1.0`. Use `antialias` when Claude-authored polygons, terrain, silhouettes, or hard geometric edges need a smoother final raster without increasing the saved output size. Use the other fields sparingly for cinematic color, soft highlights, and final polish after composition is already clear.
23. Use default `--similarity-backend local` for fast deterministic scoring. If the user explicitly asks for stronger local model scoring and local `torch`/`transformers` weights are available, use `--similarity-backend transformers-clip --similarity-model openai/clip-vit-base-patch32 --similarity-device auto`. This can use CUDA for scoring when PyTorch reports it is available.
24. Leave default `--caption-backend local` enabled so `metadata.json` records `image_caption` and `caption_similarity_score`. If the user explicitly asks for stronger image-to-text checking and local `torch`/`transformers` weights are available, use `--caption-backend transformers-blip --caption-model Salesforce/blip-image-captioning-base --caption-device auto`. This can use CUDA for captioning when PyTorch reports it is available.
25. Leave local auto-refinement enabled by default. Add `--no-auto-refine` only when the user wants to compare an unchanged scene plan against the scorer.
26. Add `--pixel-csv` only when the user explicitly wants every final `x,y,r,g,b` value; the file is very large at 2048x2048.
27. Read `metadata.json`. If `met_threshold` is false after auto-refinement, inspect `revision_hints` first and revise `scene-plan.json` using those concrete missing-object, color, contrast, mood, or reference-alignment hints before rerunning. If `caption_similarity_score` is low or `image_caption` omits important requested objects, revise the next `scene-plan.json` to make those objects visually explicit. When present, also use `reference_palette` and `initial_palette` to anchor the next palette, gradients, materials, lights, or veils to user-provided imagery. Prefer semantic scene-plan changes over simply increasing local iterations.
28. Report the generated `image.png`, `metadata.json`, score, `score_details.cosine_score`, `similarity_backend`, `image_caption`, `caption_similarity_score`, any `refinement_actions`, and for refine runs the `initial_similarity_score`.

## Constraints

- This is a CPU-first renderer. Do not install GPU diffusion packages for this skill.
- Resolution is capped at 2048x2048 while preserving aspect ratio.
- The default scorer and captioner are lightweight local prompt/reference proxies with explicit cosine and caption backchecking. Optional `transformers-clip` and `transformers-blip` are only for stronger model-backed checks when dependencies and weights are already available or the user accepts downloading them.
- The best quality path is Claude-authored `scene-plan.json`; avoid relying on keyword-only generation unless the user asks for a quick rough draft.
