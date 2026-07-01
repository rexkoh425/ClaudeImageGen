---
name: generate-image
description: Generate a local PNG from a text prompt through the CPU-first claude-imagegen executable, with an optional local Diffusers backend for higher-detail GPU images.
---

# Generate Image

Use this skill when the user asks Claude Code to generate an image from a prompt in the current folder, especially when they mention local generation, iterative refinement, reference images, initial generated images, cosine similarity, caption backchecking, optional CLIP/BLIP scoring, or RGB pixel export.

## Vision-in-the-loop refinement (the core loop)

The default CPU path has no external image API. Claude Code authors the scene plan and,
crucially, judges the result. The strongest signal is Claude opening the rendered PNG with
its own vision and scoring how close it is to the prompt — the LMM-as-evaluator /
VQAScore idea, run locally with no API. The canonical CPU loop is:

1. Author `scene-plan.json` and run `generate` (see Workflow below).
2. **Open `image.png` with Claude's own vision** and judge it against the prompt. Do not rely
   only on the numeric scores; actually look at the image.
3. Read the generated `critique-request.json`, then write a `critique.json` response that
   fills its `expected_response` schema. Answer every VQAScore-style `visual_checklist`
   object/color question plus style/mood checks in `element_checks`. Keep the response
   JSON-only and use only actions listed in `allowed_edit_actions` unless you plan to
   rewrite `scene-plan.json` manually.
4. Run `refine --critique critique.json`. The critique's `edits` are applied to the scene
   plan automatically, the judgement is logged to `metadata.json` under `visual_critique`, it
   becomes a weighted `visual_judgement` check in `quality-report.json`, and its
   `missing`/`wrong`/`extra` plus failed checklist items become `Judge:` entries in
   `next_actions`. Failed object/color checks and style/mood checks also create conservative
   checklist-derived edits: missing checked objects can add visible default objects or cloud
   banks, weak checked colors can increase saturation/contrast and append the matching palette
   color, and weak style/mood checks can tune contrast, bloom, vignette, warmth, or saturation.
5. On refine runs, read `comparison-request.json`, open the parent and child images side by side,
   and judge whether the child improved while preserving identity, layout, palette, and subject
   continuity. Save its `expected_response` as `comparison.json`, then run
   `refine --comparison comparison.json` when the child regressed so `follow_up_edits` are
   applied to the next scene plan and logged under `visual_comparison`.
6. Re-open the new `image.png`, and read `initial_similarity_details.image_embedding_cosine_score`
   (image-to-image embedding cosine, 0-1) to confirm the edit stayed continuous with the parent
   while moving toward the target. Repeat until `verdict` is `accept` and closeness is high.

## High-quality multi-refinement mode

When the user asks for best quality, more detail, or a target near `0.9`, use a multi-refinement loop instead of a single render. If optional Diffusers/Torch dependencies are available and the user wants photoreal detail, prefer `claude-imagegen diffuse --profile night-photoreal` first, then use `--initial-image <previous image.png> --strength 0.16` to make conservative local image-to-image refinement passes when Claude says the composition is close but details need work. Otherwise use the CPU scene-plan loop with `--quality-target 0.9`, inspect `image.png` with Claude vision, fill `critique.json`, and only accept the image when `quality-report.json` has `target_quality_met: true`. That requires local prompt/detail evidence plus Claude visual `closeness_score >= 0.9`.

For these runs, set `"style"` with `"detail"` and `"sharpen"` in addition to color grade controls, and make the scene plan visibly dense: use foreground/midground/background separation, materials, textures, shadows, veils, reflections, and small motif detail. Prefer 2-4 refinement rounds with `--save-candidates 4`; do not keep increasing local iterations if Claude vision says the composition is wrong.

Do not claim GPT/Sora parity just because local checks improved. If the user asks for GPT/Sora-level quality, ask Claude vision to judge the PNG directly and report the score honestly; a low `closeness_score` or `sora_gpt_parity: false` means the gate failed. When comparing a raw output with a refined output, use `claude-imagegen pair-eval` to write `pair-evaluation-request.json`, open the before/after images with Claude vision, and fill its `expected_response` before accepting the result. The CPU renderer can produce structured illustrations; photoreal attempts should use an actual image model through `diffuse` or another image-generation backend.

Useful semantic object types for indoor/detail scenes include `"greenhouse"`, `"plant"`, `"lamp"`, and `"floor"` in addition to the older `"sun"`, `"moon"`, `"cloud"`, `"ocean"`, `"mountain"`, `"forest"`, `"building"`, `"portrait"`, and `"robot"` objects. Use `"veils"` for mist/fog and `"clouds"` only when the prompt explicitly asks for clouds; use `"moon"` only when a visible moon is explicitly requested.

First-time setup can be checked with these commands. The diffusion check reports missing optional packages and whether CUDA is visible:

```bash
claude-imagegen setup
claude-imagegen setup --with-diffusion
```

For photoreal local GPU attempts after diffusion setup, use the compact `night-photoreal` profile; it keeps lamp, mist-beam, floor-reflection, and leaf-detail terms early enough to avoid CLIP prompt truncation:

```bash
claude-imagegen diffuse \
  --profile night-photoreal \
  --prompt "<user prompt>" \
  --output-dir "claude-imagegen-output/<short-slug>-gpu" \
  --width 1024 \
  --height 768 \
  --seeds 101,202,303,404 \
  --device auto \
  --quality-target 0.9
```

Then inspect `image.png`, `candidates/contact-sheet.png`, and `critique-request.json` with Claude vision before accepting the result.

For conservative local image-to-image refinement of a strong photoreal attempt, add `--initial-image "claude-imagegen-output/<previous>/image.png" --strength 0.16`; lower strength preserves composition and leaf detail, higher strength changes the scene more.

For before/after scoring without generating a new image, use:

```bash
claude-imagegen pair-eval \
  --prompt "<user prompt>" \
  --before "claude-imagegen-output/<base>/image.png" \
  --after "claude-imagegen-output/<refined>/image.png" \
  --pair-id "<short-slug>" \
  --output-dir "claude-imagegen-output/<short-slug>-eval" \
  --quality-target 0.9
```

Open `pair-evaluation-request.json` and fill its `expected_response`; the gate fails unless the best after image reaches `0.9`, detail is strong, and GPT/Sora parity is visually plausible.

For non-generative local metric evidence, run `claude-imagegen audit-pair --before "claude-imagegen-output/<base>/image.png" --after "claude-imagegen-output/<refined>/image.png" --prompt "<user prompt>" --output-dir "claude-imagegen-output/<short-slug>-audit"` and pass its `pair-audit.json` to the planner with `--audit`.

After writing Claude's pair-evaluation response, run `claude-imagegen eval-plan --evaluation <response.json> --audit "claude-imagegen-output/<short-slug>-audit/pair-audit.json" --prompt "<user prompt>" --output-dir "claude-imagegen-output/<short-slug>-plan" --quality-target 0.9 --min-evaluations 2` to convert the score, local audit, and failure modes into the next local command. If you have multiple Claude judge passes, repeat `--evaluation`; the planner uses the conservative score and keeps the gate closed when judges disagree. Follow the plan only if it keeps the acceptance gate closed below `0.9`.

If Claude vision says the refined image improved detail but became too bright, hazy, or dusk-like for a deep-night prompt, run the dark-preserving local postprocess before another evaluation:

```bash
claude-imagegen enhance-night \
  --input-image "claude-imagegen-output/<refined>/image.png" \
  --prompt "<user prompt>" \
  --output-dir "claude-imagegen-output/<refined>-night" \
  --quality-target 0.9 \
  --shadow-lift 0.08 \
  --foliage-clarity 0.35
```

Then open the new `pair-evaluation-request.json` and compare the before/after images. Do not accept the enhanced image unless Claude's response says the night mood is preserved and the after image reaches the requested gate.

`critique.json` schema (all fields optional except `closeness_score`):

```json
{
  "closeness_score": 0.42,
  "verdict": "revise",
  "summary": "Sun and ocean read well, but there are no clouds and contrast is flat.",
  "present": ["sun", "ocean"],
  "missing": ["clouds"],
  "wrong": ["sun reads too small"],
  "extra": ["unrequested red blob lower-left"],
  "edits": [
    { "action": "add_object", "type": "cloud", "x": 0.6, "y": 0.2, "size": 0.14, "color": "#fff1dd" },
    { "action": "add_cloud", "color": "#fff1dd" },
    { "action": "resize_object", "type": "sun", "size": 0.2 },
    { "action": "recolor_object", "type": "ocean", "color": "#1d5fa8" },
    { "action": "move_object", "type": "sun", "x": 0.28, "y": 0.24 },
    { "action": "remove_object", "type": "robot" },
    { "action": "adjust_style", "field": "contrast", "delta": 0.15 },
    { "action": "set_style", "field": "saturation", "value": 0.5 },
    { "action": "set_palette", "colors": ["#102040", "#ff5533", "#286fc4"] }
  ]
}
```

Edit actions map to scene-plan JSON changes: `add_object`, `remove_object`, `recolor_object`,
`move_object`, `resize_object`, `set_opacity`, `set_style`, `adjust_style`, `set_palette`,
`add_element`, `add_cloud`. Unknown actions are skipped, not fatal, so Claude can also just
rewrite `scene-plan.json` directly and pass it with `--scene-plan` when a large change is needed.

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
  --quality-target 0.9 \
  --similarity-backend local \
  --save-candidates 4
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

If saved candidates exist, prefer `--candidate-rank auto` so the refine command starts from the recommended candidate recorded in `candidates.json`; use `--candidate-rank <rank>` only when visual inspection shows a specific candidate is better. After a refine run, inspect `quality-report.json` first, then open `comparison-request.json` to judge the parent and child images together, and pass the filled response to `refine --comparison comparison.json` when its `follow_up_edits` should drive the next scene-plan update. Use `metadata.json` fields `initial_similarity_score`, `initial_similarity_details`, `refinement_delta`, `refinement_guidance`, `comparison_request`, `visual_comparison`, `refined_from`, `parent_image`, `parent_candidate_selection`, `parent_candidate_rank`, `parent_candidate_selection_score`, `parent_candidate_aesthetic_score`, and `refinement_lineage_depth` to report continuity and whether the child improved or regressed against the previous image. Use `refinement_guidance.priority_axes` to choose the next prompt-alignment, quality, caption-alignment, or continuity fix before editing the next `scene-plan.json`. Use `initial_palette` to preserve or intentionally shift the starting image colors in the next `scene-plan.json`.
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
22. Use `"style"` for final Claude-authored image finishing. Supported fields are `grain`, `vignette`, `saturation`, `contrast`, `warmth`, `bloom`, `antialias`, `"detail"`, and `"sharpen"`, each from `0.0` to `1.0`. Use `antialias` when Claude-authored polygons, terrain, silhouettes, or hard geometric edges need a smoother final raster without increasing the saved output size. Use `"detail"` and `"sharpen"` for CPU-side final edge/detail enhancement after the composition is already clear; do not use them as a substitute for real scene-plan detail.
23. Use default `--similarity-backend local` for fast deterministic scoring. If the user explicitly asks for stronger local model scoring and local `torch`/`transformers` weights are available, prefer `--similarity-backend transformers-siglip --similarity-model google/siglip-base-patch16-224 --similarity-device auto` for single prompt/image scoring loops, or use `--similarity-backend transformers-clip --similarity-model openai/clip-vit-base-patch32 --similarity-device auto` for the baseline CLIPScore-style check. Both can use CUDA for prompt-image scoring when PyTorch reports it is available.
24. Use `--continuity-backend transformers-dinov2 --continuity-model facebook/dinov2-base --continuity-device auto` on refine or initial-image runs when the user asks for stronger image-to-image continuity. This keeps text alignment and parent-image preservation separate: SigLIP or CLIP can score the prompt while DINOv2 scores whether the new image stayed semantically close to the parent. DINOv2 adds `initial_similarity_details.dinov2_image_cosine_score`; SigLIP adds `siglip_image_cosine_score`; CLIP adds `clip_image_cosine_score`.
25. Leave default `--caption-backend local` enabled so `metadata.json` records `image_caption`, `caption_similarity_score`, `caption_similarity_backend`, `lexical_caption_similarity_score`, `semantic_caption_similarity_score`, `caption_missing_objects`, `caption_missing_colors`, `caption_unexpected_objects`, and `caption_unexpected_colors`. If the user explicitly asks for stronger image-to-text checking and local `torch`/`transformers` weights are available, use `--caption-backend transformers-blip --caption-model Salesforce/blip-image-captioning-base --caption-device auto`. This can use CUDA for captioning when PyTorch reports it is available. For stronger NLP comparison between the prompt and the generated caption, add `--caption-similarity-backend transformers-sentence --caption-similarity-model sentence-transformers/all-MiniLM-L6-v2 --caption-similarity-device auto`.
26. Leave local auto-refinement enabled by default. Add `--no-auto-refine` only when the user wants to compare an unchanged scene plan against the scorer.
27. Use `--save-candidates 4` for complex, subjective, or low-confidence prompts. Inspect `metadata.json` fields `recommended_candidate_rank`, `recommended_candidate_score`, `recommended_candidate_aesthetic_score`, and `recommended_candidate_reasons`, plus `candidates/contact-sheet.png`, `candidates.json`, and the `candidates/candidate-*.png` files when several candidates have close scores, when the best score looks visually worse than an alternative, or when deciding which image to refine next. Use candidate-level `selection_score`, `selection_reasons`, `aesthetic_score`, `aesthetic_details`, `caption`, `caption_similarity_score`, and caption missing/unexpected fields to compare alternatives before choosing a refinement parent. Prefer `refine --candidate-rank auto` to use the saved recommendation; call `refine --candidate-rank <rank>` when a visually inspected candidate should override the automatic choice.
28. Add `--pixel-csv` only when the user explicitly wants every final `x,y,r,g,b` value; the file is very large at 2048x2048.
29. Use `claude-imagegen verify --output-dir claude-imagegen-output/verification --size 320x192 --size 768x432 --size 1024x640` when you need to prove the installation handles multiple sizes, larger outputs, candidate saving, one complex planned scene, and automatic refinement in one run. The complex planned scene case must report nonzero scene-plan counts for materials, terrain, reflections, warps, beams, clouds, shadows, and focus, and refine cases should report `refinement_delta` so parent-vs-child scoring is visible. Inspect top-level `device_summary` to confirm CPU/GPU usage by similarity, continuity, caption, and caption-similarity role, and inspect `image_summary` plus each case's `image_nonblank` / `image_variance_sum` to confirm outputs are not blank or uniform. Add `--strong-model --strong-model-device auto` when local CLIP or SigLIP plus BLIP dependencies and weights are available and the user wants GPU/CPU-backed model verification. Add one or more `--strong-size WIDTHxHEIGHT` flags when the strong SigLIP/CLIP, BLIP, sentence-similarity, and DINOv2 cases must run at larger or different dimensions than the default first `--size`. Use `--strong-similarity-backend transformers-siglip --similarity-model google/siglip-base-patch16-224` for SigLIP strong verification cases. Add `--strong-continuity-backend transformers-dinov2 --continuity-model facebook/dinov2-base` to include matching DINOv2 refine-continuity cases. Add `--caption-similarity-backend transformers-sentence --caption-similarity-model sentence-transformers/all-MiniLM-L6-v2` for semantic prompt/caption similarity in strong cases.
30. Read `quality-report.json` first. If its `status` is `revise`, use `next_actions` as the immediate checklist for the next scene-plan edit, then open `critique-request.json` before writing the next `critique.json`; answer its `visual_checklist` items in `element_checks` so requested objects/colors are judged explicitly. For refine runs, also open `comparison-request.json` and compare parent/child images side by side before accepting a child that improved numerically but regressed visually; pass the filled comparison response to `refine --comparison comparison.json` when the next run should apply its `follow_up_edits`. Then read `metadata.json` for the underlying details: if `met_threshold` is false after auto-refinement, inspect `revision_hints`; if `visual_comparison` is present, inspect its verdict, regressions, and applied edits; if `refinement_guidance` is present, inspect `decision`, `priority_axes`, severity, and actions before choosing the next scene-plan edit; if `caption_similarity_score` is low or `image_caption` omits important requested objects, use `caption_missing_objects` and `caption_missing_colors`; when `caption_similarity_backend` is `transformers-sentence`, compare `semantic_caption_similarity_score` against `lexical_caption_similarity_score` to distinguish wording mismatch from missing visual evidence; for refine runs, inspect `refinement_delta` first to compare parent and child total score, quality score, caption similarity, and continuity, then inspect `initial_similarity_details.continuity_score`, `image_embedding_cosine_score`, `image_cosine_score`, `luminance_ssim_score`, `multiscale_luminance_ssim_score`, `region_similarity_scores`, `weakest_continuity_region`, `edge_cosine_score`, and `color_histogram_score`; if DINOv2 is active, inspect `dinov2_image_cosine_score` as the strongest image-to-image continuity check; if SigLIP is active, inspect `siglip_image_cosine_score`; if CLIP is active, inspect `clip_image_cosine_score`. When present, also use `reference_palette` and `initial_palette` to anchor the next palette, gradients, materials, lights, or veils to user-provided imagery. Prefer semantic scene-plan changes over simply increasing local iterations.
31. Report the generated `image.png`, `metadata.json`, `quality-report.json` status/score, `critique-request.json`, score, `score_details.cosine_score`, `similarity_backend`, `continuity_backend` when present, `image_caption`, `caption_similarity_score`, `caption_similarity_backend`, candidate artifact path and recommended candidate rank when present, caption missing evidence when present, any `refinement_actions`, and for refine runs the `initial_similarity_score`, `refinement_delta`, `refinement_guidance`, `comparison-request.json`, `visual_comparison`, plus the most relevant `initial_similarity_details` fields. For `verify`, report `verification-report.json`, case count, `device_summary`, `image_summary`, requested `strong_sizes`, strong-model status, case `critique_request` paths, refine-case `comparison_request` paths, refine-case `refinement_delta`, and any failed case details.

## Constraints

- This is CPU-first by default. Install or use GPU diffusion packages only when the user asks for higher-detail photoreal output or explicitly accepts the optional diffusion setup.
- Resolution is capped at 2048x2048 while preserving aspect ratio.
- The default scorer and captioner are lightweight local prompt/reference proxies with explicit cosine and caption backchecking. Optional `transformers-clip`, `transformers-siglip`, `transformers-dinov2`, `transformers-blip`, and `transformers-sentence` are only for stronger model-backed checks when dependencies and weights are already available or the user accepts downloading them.
- The best CPU quality path is Claude-authored `scene-plan.json`; for photoreal detail, use `claude-imagegen diffuse` plus Claude visual critique instead of relying on keyword-only CPU generation.
