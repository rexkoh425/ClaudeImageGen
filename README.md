# Claude ImageGen

CPU-first Claude Code plugin prototype for generating a local PNG from a text prompt and optional reference or initial image. Optional model-backed CLIP and BLIP checks can use local CPU or CUDA when PyTorch reports that CUDA is available.

The default path does not download diffusion weights or require a GPU. It follows the same broad loop used by text-guided image optimization systems:

1. Parse the prompt into compact visual targets.
2. Render a candidate image from geometric and semantic primitives.
3. Score the result against text and optional reference-image features.
4. Caption-backcheck what the image appears to contain against the prompt.
5. Mutate or locally refine the candidate and repeat until the score reaches a threshold or the iteration budget is exhausted.

The maximum output size is capped at 2048x2048 while preserving aspect ratio.

The higher-quality path is `--scene-plan`: Claude Code first writes a structured scene JSON with palette, background, object placement, depth, and style hints. Python then renders that explicit composition. This keeps the heavy semantic reasoning inside Claude Code while local work stays limited to deterministic rendering, scoring, and targeted refinement.

## Quick Start

```bash
python -m pip install -e .
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480 \
  --max-iterations 32 \
  --save-candidates 4
```

For better quality, write a scene plan first:

```json
{
  "title": "coastal sunrise composition",
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
    { "type": "sun", "x": 0.25, "y": 0.24, "size": 0.18, "color": "#ff5533" },
    { "type": "mountain", "y": 0.58, "size": 0.32, "color": "#49394c", "layers": 3 },
    { "type": "ocean", "y": 0.56, "color": "#286fc4" },
    { "type": "foreground", "y": 0.78, "color": "#123d2a" }
  ],
  "elements": [
    { "type": "glow", "x": 0.25, "y": 0.24, "width": 0.20, "height": 0.20, "fill": "#ffb45a", "opacity": 0.45, "z": 1 },
    { "type": "ellipse", "x": 0.28, "y": 0.26, "width": 0.38, "height": 0.24, "fill": "#ffcf8a", "opacity": 0.24, "blur": 0.08, "blend": "screen", "z": 2 },
    { "type": "rectangle", "x": 0.5, "y": 0.68, "width": 1.0, "height": 0.24, "gradient": { "type": "linear", "colors": ["#2e8ddb", "#0a3d72"], "direction": "vertical" }, "opacity": 0.46, "blend": "multiply", "z": 5 },
    { "type": "polyline", "points": [[0.12, 0.66], [0.38, 0.68], [0.62, 0.67], [0.88, 0.65]], "stroke": "#f6e2b5", "width": 0.008, "opacity": 0.72, "blur": 0.012, "blend": "screen", "z": 6 },
    { "type": "path", "commands": [["M", 0.0, 0.84], ["C", 0.18, 0.76, 0.38, 0.88, 0.58, 0.78], ["Q", 0.78, 0.70, 1.0, 0.83], ["L", 1.0, 1.0], ["L", 0.0, 1.0], ["Z"]], "fill": "#0b241b", "stroke": "#174b36", "width": 0.006, "opacity": 0.78, "z": 7 },
    { "type": "polygon", "points": [[0.0, 0.84], [0.18, 0.76], [0.36, 0.86], [0.58, 0.74], [0.82, 0.85], [1.0, 0.78], [1.0, 1.0], [0.0, 1.0]], "fill": "#0b241b", "opacity": 0.78, "z": 7 }
  ],
  "motifs": [
    { "type": "starfield", "count": 36, "region": [0.0, 0.02, 1.0, 0.38], "color": "#fff5cc", "size": 0.006, "opacity": 0.72, "seed": 12, "z": 8 },
    { "type": "grass", "count": 80, "region": [0.0, 0.78, 1.0, 1.0], "color": "#1a5c36", "size": 0.045, "opacity": 0.65, "seed": 21, "z": 9 }
  ],
  "textures": [
    { "type": "ripple", "count": 24, "region": [0.0, 0.56, 1.0, 0.78], "color": "#d8f3ff", "density": 0.6, "scale": 0.025, "opacity": 0.36, "blend": "screen", "seed": 31, "z": 9 },
    { "type": "hatching", "count": 90, "region": [0.0, 0.78, 1.0, 1.0], "color": "#2a7047", "density": 0.65, "scale": 0.035, "opacity": 0.42, "seed": 32, "z": 10 }
  ],
  "materials": [
    { "type": "water", "region": [0.0, 0.56, 1.0, 0.80], "colors": ["#8bdcff", "#0b3b71"], "intensity": 0.72, "scale": 0.035, "opacity": 0.58, "seed": 41, "z": 8 },
    { "type": "foliage", "region": [0.0, 0.78, 1.0, 1.0], "colors": ["#1e7a4a", "#071e18"], "intensity": 0.64, "scale": 0.045, "opacity": 0.52, "seed": 42, "z": 11 }
  ],
  "terrains": [
    { "type": "mountain", "points": [[0.02, 0.56], [0.22, 0.24], [0.42, 0.56], [0.64, 0.34], [0.96, 0.56]], "base": 0.78, "fill": "#405070", "shade": "#182030", "highlight": "#7890b0", "opacity": 0.76, "facets": true, "z": 4 }
  ],
  "reflections": [
    { "type": "vertical", "source": [0.0, 0.16, 1.0, 0.56], "target": [0.0, 0.56, 1.0, 0.80], "opacity": 0.36, "blur": 0.025, "fade": 0.62, "tint": "#2d88d8", "blend": "screen", "z": 8 }
  ],
  "warps": [
    { "type": "wave", "region": [0.0, 0.56, 1.0, 0.82], "direction": "horizontal", "amplitude": 0.018, "wavelength": 0.38, "phase": 0.20, "seed": 43, "z": 10 }
  ],
  "atmosphere": { "type": "horizon_fog", "color": "#d8e8f0", "horizon": 0.56, "height": 0.22, "strength": 0.32 },
  "veils": [
    { "type": "mist", "region": [0.04, 0.48, 0.96, 0.68], "color": "#d8e8f0", "opacity": 0.22, "blur": 0.026, "blend": "screen", "falloff": 0.18, "direction": "vertical", "z": 8 }
  ],
  "lights": [
    { "type": "radial", "x": 0.25, "y": 0.24, "radius": 0.35, "color": "#ffcf8a", "intensity": 0.48, "z": 10 },
    { "type": "shadow", "x": 0.50, "y": 0.95, "radius": 0.55, "intensity": 0.25, "z": 11 }
  ],
  "beams": [
    { "type": "sunbeam", "x": 0.25, "y": 0.24, "angle": 70.0, "length": 0.70, "spread": 24.0, "color": "#ffcf8a", "opacity": 0.22, "blur": 0.035, "blend": "screen", "count": 2, "seed": 44, "z": 9, "occlusions": [{ "type": "polygon", "points": [[0.08, 0.55], [0.46, 0.55], [0.46, 0.63], [0.08, 0.63]] }] }
  ],
  "clouds": [
    { "type": "cumulus", "region": [0.05, 0.08, 0.95, 0.34], "color": "#fff5dd", "shadow": "#8aa0b8", "opacity": 0.38, "blur": 0.025, "count": 3, "lobes": 5, "scale": 0.13, "blend": "screen", "seed": 45, "z": 4 }
  ],
  "shadows": [
    { "type": "ellipse", "x": 0.50, "y": 0.78, "width": 0.52, "height": 0.10, "color": "#101820", "opacity": 0.34, "blur": 0.035, "blend": "multiply", "z": 9 }
  ],
  "focus": { "type": "depth", "region": [0.04, 0.10, 0.84, 0.82], "blur": 0.018, "falloff": 0.12, "mode": "outside" },
  "style": { "grain": 0.08, "vignette": 0.12, "saturation": 0.32, "contrast": 0.22, "warmth": 0.18, "bloom": 0.22, "antialias": 1.0 }
}
```

Then run:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --scene-plan claude-imagegen-output/demo/scene-plan.json \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480 \
  --similarity-backend local
```

Outputs:

- `image.png`: generated RGB image.
- `metadata.json`: prompt, score, dimensions, seed, detected objects/colors, extracted `reference_palette` / `initial_palette`, threshold result, `score_details.cosine_score`, `initial_similarity_details` for image-to-image continuity, `image_caption`, `caption_similarity_score`, caption missing/unexpected evidence, candidate artifact paths, recommended candidate fields, local refinement actions, `refinement_delta` for refine runs, and `revision_hints` when Claude should revise a weak scene plan.
- `quality-report.json`: first-stop readiness report for Claude Code, combining prompt alignment, caption alignment, continuity, reference alignment, candidate recommendation, refine-run `refinement_delta`, and concrete `next_actions`.
- `critique-request.json`: Claude-vision judge request for the next iteration, including image/metadata/report paths, the prompt, quality status, allowed edit actions, and the JSON response schema to fill before running `refine --critique`.
- `progress.csv`: score per iteration.
- `candidates.json`: optional ranked candidate index when `--save-candidates N` is passed, including score details, captions, caption missing/unexpected evidence, and `selection_score` / `selection_reasons` for each candidate.
- `candidates/`: optional top-N candidate PNGs plus `contact-sheet.png` for visual comparison.
- `pixels.csv`: optional explicit `x,y,r,g,b` table when `--pixel-csv` is passed.

## Reference and Initial Images

```bash
claude-imagegen generate \
  --prompt "abstract botanical poster" \
  --reference-image /absolute/path/reference.png \
  --output-dir claude-imagegen-output/botanical \
  --max-iterations 48
```

Use `--initial-image /absolute/path/image.png` when you want the generated result to blend from an existing output.

## Iterative Refinement

Use `refine` when you want to continue from a previous `claude-imagegen` output directory. It uses the parent `image.png` as the initial image, reuses `scene-plan.json` from the parent directory when present, preserves parent dimensions by default, and writes lineage plus continuity metadata.

```bash
claude-imagegen refine \
  --from-dir claude-imagegen-output/demo \
  --prompt "cinematic red sun over a blue ocean with misty mountains, clouds, richer foreground grass, and stronger water reflections" \
  --output-dir claude-imagegen-output/demo-refined \
  --candidate-rank auto \
  --max-iterations 8 \
  --threshold 0.62
```

The refined `metadata.json` includes `refined_from`, `parent_image`, `parent_metadata`, `refinement_lineage_depth`, `initial_similarity_score`, `initial_similarity_details`, `parent_total_score`, `parent_quality_score`, `parent_caption`, `parent_caption_similarity_score`, `refinement_delta`, and `parent_candidate_*` fields when `--candidate-rank` is used. Use `--candidate-rank auto` to start from the saved candidate with the strongest combined total score, caption similarity, reference score, and caption-diagnostic penalties. Start each iteration by reading `quality-report.json` and `critique-request.json`; fill the request's `expected_response` into a `critique.json` after viewing `image.png`, then pass it to `refine --critique critique.json`. Use `refinement_delta.total_score_delta`, `refinement_delta.quality_score_delta`, and `refinement_delta.caption_similarity_delta` to compare the refined output against its parent, and use `initial_similarity_details.continuity_score`, `image_cosine_score`, `luminance_ssim_score`, `edge_cosine_score`, and `color_histogram_score` to confirm continuity with the previous image while `score_details.cosine_score`, `caption_similarity_score`, and `reference_score` track prompt/reference alignment.

Use `--save-candidates N` on either `generate` or `refine` when Claude Code should inspect alternatives instead of trusting only the best-scored final image. The generator writes `candidates.json`, `candidates/candidate-*.png`, and `candidates/contact-sheet.png`; each index entry records rank, iteration, image path, total/text/reference scores, score details, threshold status, candidate caption, caption similarity, candidate-level missing/unexpected prompt evidence, and `selection_score` / `selection_reasons`. The run metadata also records `recommended_candidate_rank`, `recommended_candidate_image`, `recommended_candidate_score`, and `recommended_candidate_reasons`. If a lower-ranked candidate is visually stronger, run `refine --from-dir <parent> --candidate-rank <rank>`; otherwise use `refine --from-dir <parent> --candidate-rank auto` to let the saved selection score choose the next initial image.

## Similarity Backends

The default scorer is `--similarity-backend local`, a deterministic shared feature-vector cosine scorer over prompt colors, objects, mood/style terms, image color presence, region/object proxies, edge density, and contrast. This keeps tests offline and fast.

For stronger text-image scoring, use an optional Transformers backend when `torch`, `transformers`, and the selected model are available. CLIP computes the familiar embedding cosine baseline:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --scene-plan claude-imagegen-output/demo/scene-plan.json \
  --output-dir claude-imagegen-output/demo-clip \
  --width 1024 \
  --height 768 \
  --similarity-backend transformers-clip \
  --similarity-model openai/clip-vit-base-patch32 \
  --similarity-device auto
```

`--similarity-device auto` uses CUDA through PyTorch when available, otherwise CPU. This optional backend scores prompt-image cosine similarity; it does not turn the renderer into a diffusion model.

SigLIP is also available and is a better fit for single prompt/image scoring loops because it uses an independent sigmoid score per image-text pair:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --scene-plan claude-imagegen-output/demo/scene-plan.json \
  --output-dir claude-imagegen-output/demo-siglip \
  --width 1024 \
  --height 768 \
  --similarity-backend transformers-siglip \
  --similarity-model google/siglip-base-patch16-224 \
  --similarity-device auto
```

Continuity scoring can be controlled separately from text-image scoring. By default, `--continuity-backend` inherits `--similarity-backend` when an `--initial-image` or refine parent is present. For stronger image-to-image continuity, use DINOv2:

```bash
claude-imagegen refine \
  --from-dir claude-imagegen-output/demo-siglip \
  --prompt "make the observatory larger while preserving the harbor composition" \
  --output-dir claude-imagegen-output/demo-dinov2-refine \
  --similarity-backend transformers-siglip \
  --similarity-model google/siglip-base-patch16-224 \
  --similarity-device auto \
  --continuity-backend transformers-dinov2 \
  --continuity-model facebook/dinov2-base \
  --continuity-device auto
```

When refining from an initial image with `--similarity-backend transformers-clip`, `initial_similarity_details` also includes `clip_image_cosine_score`; with `--similarity-backend transformers-siglip`, it includes `siglip_image_cosine_score`; with `--continuity-backend transformers-dinov2`, it includes `dinov2_image_cosine_score`. These are image-embedding cosines between the new image and the selected parent image. The final `initial_similarity_score` blends local continuity with the optional model-backed image cosine, giving Claude Code a stronger continuity signal when cached local model weights are available.

## Caption Backchecking

Every run defaults to `--caption-backend local`, a deterministic offline caption proxy that describes visible scene evidence such as sun, ocean, clouds, mountains, forest, or city structure. The generated `metadata.json` records `image_caption`, `caption_similarity_score`, `caption_backend`, `caption_model`, `caption_device`, `effective_caption_device`, `caption_similarity_backend`, `lexical_caption_similarity_score`, `semantic_caption_similarity_score`, `caption_missing_objects`, `caption_missing_colors`, `caption_unexpected_objects`, and `caption_unexpected_colors`.

For stronger image-to-text checking, use the optional Transformers BLIP backend when `torch`, `transformers`, and the selected model are available:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --scene-plan claude-imagegen-output/demo/scene-plan.json \
  --output-dir claude-imagegen-output/demo-blip \
  --width 1024 \
  --height 768 \
  --caption-backend transformers-blip \
  --caption-model Salesforce/blip-image-captioning-base \
  --caption-device auto
```

`--caption-device auto` uses CUDA through PyTorch when available, otherwise CPU. Caption backchecking is a second signal: it helps Claude Code spot when the image appears to contain different objects than the prompt asked for, even if the local cosine score is high. When caption similarity is low, missing requested objects or colors are also promoted into `revision_hints` so the next scene plan can make those elements visually explicit.

For stronger NLP comparison between the prompt and the generated caption, add sentence-embedding caption similarity:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --output-dir claude-imagegen-output/demo-caption-sbert \
  --width 1024 \
  --height 768 \
  --caption-backend transformers-blip \
  --caption-model Salesforce/blip-image-captioning-base \
  --caption-device auto \
  --caption-similarity-backend transformers-sentence \
  --caption-similarity-model sentence-transformers/all-MiniLM-L6-v2 \
  --caption-similarity-device auto
```

This keeps the explicit object/color gap diagnostics while letting `caption_similarity_score` use semantic prompt/caption closeness instead of only token overlap.

## Claude Code Plugin Layout

This repository is structured as a Claude Code plugin:

- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`
- `skills/generate-image/SKILL.md`
- `bin/claude-imagegen`
- `src/claude_imagegen/`

The skill tells Claude Code to write `scene-plan.json`, then call `claude-imagegen generate --scene-plan ...`. The executable prepends the plugin's bundled `src/` directory to `PYTHONPATH`. On a fresh machine, it creates a plugin-owned virtual environment on first run and installs this package there when `numpy` or `Pillow` are not already available.

## Install As A Claude Plugin

Install from GitHub on another machine:

```bash
claude plugin marketplace add rexkoh425/ClaudeImageGen
claude plugin install claude-imagegen@claude-imagegen
```

For project-only installation, use:

```bash
claude plugin install claude-imagegen@claude-imagegen --scope project
```

The repository is currently hosted at `https://github.com/rexkoh425/ClaudeImageGen`. If it remains private, authenticate GitHub access on the target machine before adding the marketplace.

To download or work on the package directly:

```bash
git clone https://github.com/rexkoh425/ClaudeImageGen.git
cd ClaudeImageGen
python -m pip install -e .
```

Validate the plugin and its marketplace manifest from the repo root:

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

For local development from a cloned checkout, add the checkout as a Claude Code marketplace:

```bash
claude plugin marketplace add ./
claude plugin install claude-imagegen@claude-imagegen
```

Restart Claude Code after install so the `generate-image` skill and `claude-imagegen` executable are loaded in the next session.

Scene plans have two composition levels:

- `objects`: semantic primitives such as sun, moon, mountain, ocean, cloud, forest, city, and foreground.
- `background`: top/bottom colors plus optional multi-stop gradients. `stops` let Claude specify sky bands, horizon glow, and directional color transitions with normalized positions instead of relying on a flat two-color blend.
- `elements`: Claude-authored detail primitives such as polylines, paths, polygons, ellipses, rectangles, glows, and arcs. `path` elements support `M`, `L`, `Q`, `C`, and `Z` commands for smoother waves, silhouettes, clouds, and shorelines. Filled rectangles, ellipses, polygons, and closed paths can use `gradient` with linear or radial color stops for shaded water, luminous haze, and material depth. Elements can also use `blur` plus `blend` values `normal`, `screen`, `multiply`, `overlay`, or `soft-light` for soft atmosphere, haze, shadow, material color, and reflections.
- `motifs`: compact repeated-detail instructions such as starfields, grass blades, rain streaks, window lights, and dots. These let Claude specify texture intent without listing every mark.
- `textures`: region-bound surface instructions such as hatching, crosshatch, ripple, paper, grain, noise, mist, and speckles. These let Claude add painterly surface breakup, water bands, atmospheric grain, and material detail without enumerating pixels.
- `materials`: compact semantic surface regions such as water, foliage, metal, or generic surfaces. These let Claude specify material intent once through region, colors, intensity, scale, opacity, seed, and z-order; the renderer expands that into bounded gradients and deterministic surface marks.
- `terrains`: Claude-authored ridge and landform primitives. Claude supplies normalized ridge points, a base line, colors, opacity, and optional facets; the renderer fills a deterministic silhouette and shaded planes.
- `reflections`: mirrored source-to-target region instructions, usually for water. Claude chooses the source and target regions plus opacity, blur, fade, tint, blend, and z-order; the renderer mirrors already-rendered pixels deterministically.
- `warps`: bounded sine-wave displacement instructions. Claude chooses the region, direction, amplitude, wavelength, phase, seed, and z-order; the renderer bends already-rendered pixels for water shimmer, heat haze, or softened reflections.
- `atmosphere`: horizon haze and aerial perspective controls. Claude specifies color, horizon, height, and strength; the renderer applies a smooth fog band for depth.
- `veils`: localized atmospheric layers for mist, fog sheets, smoke, rain wash, and soft glow. Claude specifies a bounded region, color, opacity, blur, blend mode, directional falloff, and z-order; the renderer expands that into a deterministic translucent overlay without applying haze to the whole image.
- `lights`: local radial lights, tints, rim lights, and shadows. These let Claude describe mood and depth explicitly instead of relying on flat procedural colors.
- `beams`: directional light shaft primitives. Claude specifies origin, angle, length, spread, color, opacity, blur, blend, count, seed, z-order, and optional occlusion masks; the renderer expands those into translucent wedge overlays and clips out Claude-specified rectangles, ellipses, or polygons.
- `clouds`: compact cloud-bank primitives. Claude specifies region, color, shadow, opacity, blur, count, lobes, scale, blend, seed, and z-order; the renderer expands those into deterministic puffy sky masses.
- `shadows`: compact cast-shadow and contact-shadow primitives. Claude specifies ellipse, rectangle, or polygon shape, normalized placement, color, opacity, blur, blend, and z-order; the renderer expands those into soft multiply overlays that ground objects and terrain.
- `focus`: Claude-authored depth-of-field control. Claude specifies a normalized focal region, blur strength, transition falloff, and whether to blur outside or inside the region; the renderer applies deterministic masked blur after composition.
- `style`: final image finishing controls: `grain`, `vignette`, `saturation`, `contrast`, `warmth`, `bloom`, and `antialias`. These let Claude specify the final grade after the composition is rendered; `antialias` renders the scene plan internally at higher resolution and downsamples to the capped output size for smoother hard geometry.

## Pixel Export and Token Cost

A 2048x2048 image has 4,194,304 pixels. A literal RGB table has 4,194,304 rows before metadata, and asking an LLM to emit that table directly would usually mean millions of tokens. This prototype therefore renders pixels programmatically and only exports `pixels.csv` on request.

When `quality-report.json` reports `status: "revise"`, its `next_actions` list is the primary checklist for the next scene-plan edit. `critique-request.json` is the paired Claude-vision checklist: open `image.png`, compare it against the prompt and report, fill the request's response schema as `critique.json`, then run `refine --critique critique.json` so structured visual edits are recorded and applied. `metadata.json` still includes the underlying `revision_hints` with concrete scene-plan changes for Claude Code to make next, such as adding missing objects, strengthening requested colors, increasing contrast, or moving closer to a reference image. Refine reports also include `refinement_delta` in both `metadata.json` and `quality-report.json` so Claude Code can tell whether the child improved or regressed in total score, quality score, caption similarity, and continuity compared with the parent. If a scene plan is provided, the generator also performs bounded local auto-refinement between iterations: it can add missing prompt objects such as clouds, add a matching cloud layer, and increase contrast or saturation when the local scorer shows weak evidence. If a reference or initial image is provided, `reference_palette` and `initial_palette` expose extracted hex colors that Claude can fold into the next scene plan. `image_caption`, `caption_similarity_score`, `lexical_caption_similarity_score`, `semantic_caption_similarity_score`, and `caption_missing_objects` / `caption_missing_colors` add a direct "what does this look like" check before revising the next scene plan. The intent is to keep the expensive semantic iteration in Claude Code while local work remains deterministic rendering, scoring, caption backchecking, quality reporting, critique request generation, and targeted refinement.

## Validation

```bash
python -m pytest
python -m claude_imagegen.cli generate --prompt "red sun over blue ocean" --output-dir claude-imagegen-output/smoke --width 160 --height 100 --max-iterations 12 --threshold 0.1
claude-imagegen verify --output-dir claude-imagegen-output/verification --size 320x192 --size 768x432 --size 1024x640
```

`claude-imagegen verify` writes `verification-report.json`, generates each requested size, saves candidate artifacts, runs one complex planned scene case, runs one `refine --candidate-rank auto` case, and records every output's `metadata.json`, `quality-report.json`, `critique-request.json`, and refine-case `refinement_delta`. The complex planned scene case writes a rich `scene-plan.json` with gradients, motifs, textures, materials, terrain, reflections, warps, atmosphere, veils, lights, beams, clouds, shadows, and focus; the report includes scene-plan feature counts and fails the case if that complexity is not present. Add `--strong-model --strong-model-device auto` to include one model-backed similarity plus BLIP verification case when local `torch`, `transformers`, and model weights are available. Use `--strong-similarity-backend transformers-siglip --similarity-model google/siglip-base-patch16-224` to run the strong case with SigLIP instead of CLIP. Add `--strong-continuity-backend transformers-dinov2 --continuity-model facebook/dinov2-base` to include an extra strong refine case with DINOv2 parent-image continuity. Add `--caption-similarity-backend transformers-sentence --caption-similarity-model sentence-transformers/all-MiniLM-L6-v2` to score BLIP captions against prompts with sentence embeddings.

If the local Claude Code build supports plugin validation:

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

## Current Limits

- The default scorer and captioner are lightweight local surrogates; optional `transformers-clip`, `transformers-siglip`, and `transformers-blip` can use stronger local models when dependencies and weights are available.
- Image quality is closer to procedural concept art than diffusion output.
- Keyword-only prompt understanding is dictionary-based; planned generation shifts composition and prompt interpretation to Claude Code through `scene-plan.json`.
- No GPU diffusion, external image API, or large model weights are used by default. GPU is only used by optional model-backed similarity and caption backchecking.

See `docs/research.md` for the research basis and why later versions should expand from scoring toward learned priors, upscaling, edit preservation, and stronger model-guided refinement.
