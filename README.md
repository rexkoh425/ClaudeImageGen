# Claude ImageGen

Claude ImageGen is a Claude Code plugin for making local PNG images.

Default mode is lightweight and CPU-first. For higher-detail images, use the optional local GPU Diffusers backend. Claude can critique and refine prompts, but Claude does not become the image model; the actual photoreal image work needs a local image model such as SDXL Turbo.

## Install In Claude Code

Claude ImageGen is available through the Claude Code plugin marketplace from this GitHub repo.

On another machine, install Claude Code, sign in to GitHub if this repo is private, then run this inside Claude Code:

```text
/plugin marketplace add rexkoh425/ClaudeImageGen
/plugin install claude-imagegen@claude-imagegen
```

Restart Claude Code after installation so the `generate-image` skill and `claude-imagegen` command are loaded.

Check the base setup:

```bash
claude-imagegen setup
```

## Easier Local Setup

From a clone of this repo:

```bash
python -m pip install -e .
claude-imagegen setup
```

For the higher-detail local GPU path:

```bash
python -m pip install -e ".[diffusion]"
claude-imagegen setup --with-diffusion
```

## Fast CPU Image

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480 \
  --quality-target 0.9 \
  --save-candidates 4
```

For better CPU results, ask Claude Code to use the `generate-image` skill for a multi-refinement loop: it writes `scene-plan.json`, renders `image.png`, fills `critique.json` from `critique-request.json`, then runs `refine`.

## Higher-Detail GPU Image

After installing the diffusion extra:

```bash
claude-imagegen diffuse \
  --prompt "photorealistic cinematic glass greenhouse interior at deep night, crisp steel mullions, tropical plants with sharp leaf veins, warm tungsten hanging lamps, volumetric mist, wet black stone floor with mirror reflections, no people" \
  --output-dir claude-imagegen-output/greenhouse-gpu \
  --width 1024 \
  --height 768 \
  --seeds 101,202,303,404 \
  --device auto \
  --quality-target 0.9
```

`diffuse` writes multiple candidates, selects the strongest local candidate, and creates `candidates/contact-sheet.png`. Open `image.png` and `critique-request.json` with Claude vision before accepting a `0.9` target.

Keep diffusion prompts concise and put the most important details first; `metadata.json` records `prompt_length_warning` when SDXL-style text limits may truncate later details.

## Refinement

Continue from a previous CPU output:

```bash
claude-imagegen refine \
  --from-dir claude-imagegen-output/demo \
  --prompt "same coastal scene, stronger clouds, richer foreground grass, and clearer water reflections" \
  --output-dir claude-imagegen-output/demo-refined \
  --candidate-rank auto \
  --max-iterations 8
```

For visual feedback loops, fill `critique-request.json` as `critique.json`, then pass it back:

```bash
claude-imagegen refine \
  --from-dir claude-imagegen-output/demo \
  --prompt "same coastal scene with the critique applied" \
  --output-dir claude-imagegen-output/demo-refined \
  --critique claude-imagegen-output/demo/critique.json
```

## Outputs

Each run writes the main artifacts into the output directory:

- `image.png`: selected output image.
- `metadata.json`: prompt, settings, scores, selected seed or candidate, and refinement hints.
- `quality-report.json`: readiness report with concrete `next_actions`.
- `critique-request.json`: visual checklist for Claude Code to fill after inspecting `image.png`.
- `comparison-request.json`: refine-only parent/child comparison request.
- `candidates/`, `candidates.json`, and `candidates/contact-sheet.png`: alternatives for candidate-based generation.
- `verification-report.json`: created by `verify`, with nonblank image checks and CPU/GPU device evidence.

## Verify

Run tests:

```bash
python -m pytest
```

Run the plugin checks:

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

Run the built-in smoke suite:

```bash
claude-imagegen verify \
  --output-dir claude-imagegen-output/verification \
  --size 320x192 \
  --size 768x432 \
  --size 1024x640
```

For stronger local model checks:

```bash
claude-imagegen verify \
  --output-dir claude-imagegen-output/verification-strong \
  --size 320x192 \
  --strong-model \
  --strong-size 768x432 \
  --strong-similarity-backend transformers-siglip \
  --strong-continuity-backend transformers-dinov2 \
  --caption-similarity-backend transformers-sentence
```

Open `verification-report.json` and check:

- `image_summary`: confirms generated images are nonblank.
- `device_summary`: reports which CPU or GPU devices were used.
- per-case artifacts: include image, metadata, quality report, critique request, and comparison request when relevant.

## Quality Target

`--quality-target 0.9` is a gate, not a promise. A run should only be accepted when:

- local scores and `image_detail_score` are strong,
- Claude vision opens the PNG and gives a high `closeness_score`,
- `quality-report.json` has `target_quality_met: true`,
- GPT/Sora-level parity is not claimed unless an actual Claude visual judgement supports it.

In local testing on an RTX 5070 Ti, SDXL Turbo produced a much stronger greenhouse image than the CPU renderer, but Claude vision scored the best candidate `0.83`, not `0.9`. Treat that as improved quality, not solved parity.

## Current Limits

- Maximum output size is capped at 2048x2048 while preserving aspect ratio.
- CPU `generate` is a deterministic local renderer and verifier, not a diffusion model.
- GPU `diffuse` and strong model verification require optional local ML packages and model weights.
