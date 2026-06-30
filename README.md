# Claude ImageGen

CPU-only Claude Code plugin prototype for generating a local PNG from a text prompt and optional reference or initial image.

The prototype does not download diffusion weights or require a GPU. It follows the same broad loop used by text-guided image optimization systems:

1. Parse the prompt into compact visual targets.
2. Render a candidate image from geometric and semantic primitives.
3. Score the result against text and optional reference-image features.
4. Mutate the candidate and repeat until the score reaches a threshold or the iteration budget is exhausted.

The maximum output size is capped at 720x480.

The higher-quality path is `--scene-plan`: Claude Code first writes a structured scene JSON with palette, background, object placement, depth, and style hints. Python then renders that explicit composition. This keeps the heavy semantic reasoning inside Claude Code while local CPU work stays limited to deterministic rendering and scoring.

## Quick Start

```bash
python -m pip install -e .
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480 \
  --max-iterations 32
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
  --height 480
```

Outputs:

- `image.png`: generated RGB image.
- `metadata.json`: prompt, score, dimensions, seed, detected objects/colors, extracted `reference_palette` / `initial_palette`, threshold result, and `revision_hints` when Claude should revise a weak scene plan.
- `progress.csv`: score per iteration.
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

A 720x480 image has 345,600 pixels. A literal RGB table has 345,600 rows before metadata, and asking an LLM to emit that table directly would usually mean millions of tokens. This prototype therefore renders pixels programmatically and only exports `pixels.csv` on request.

When `met_threshold` is false, `metadata.json` includes `revision_hints` with concrete scene-plan changes for Claude Code to make next, such as adding missing objects, strengthening requested colors, increasing contrast, or moving closer to a reference image. If a reference or initial image is provided, `reference_palette` and `initial_palette` expose extracted hex colors that Claude can fold into the next scene plan. The intent is to keep the expensive semantic iteration in Claude Code while local CPU work remains deterministic rendering and scoring.

## Validation

```bash
python -m pytest
python -m claude_imagegen.cli generate --prompt "red sun over blue ocean" --output-dir claude-imagegen-output/smoke --width 160 --height 100 --max-iterations 12 --threshold 0.1
```

If the local Claude Code build supports plugin validation:

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

## Current Limits

- The scorer is a lightweight CPU surrogate, not CLIP.
- Image quality is closer to procedural concept art than diffusion output.
- Keyword-only prompt understanding is dictionary-based; planned generation shifts composition and prompt interpretation to Claude Code through `scene-plan.json`.
- No GPU, external image API, or large model weights are used.

See `docs/research.md` for the research basis and why later versions should move from this surrogate scorer toward CLIP-like embeddings or lightweight latent models.
