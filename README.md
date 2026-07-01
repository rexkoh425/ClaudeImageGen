# Claude ImageGen

Claude ImageGen is a Claude Code plugin for making local PNG images from Claude-authored scene plans.

It is CPU-first by default. Optional PyTorch and Transformers checks can use CUDA when your local Python environment has GPU support, but the basic path does not download diffusion weights or require a GPU.

## Install In Claude Code

Claude ImageGen is available through the Claude Code plugin marketplace from this GitHub repo.

In Claude Code, run:

```text
/plugin marketplace add rexkoh425/ClaudeImageGen
/plugin install claude-imagegen@claude-imagegen
```

Restart Claude Code after installation so the `generate-image` skill and `claude-imagegen` command are loaded.

If the repo is private, sign in to GitHub on the target machine before adding the marketplace.

## Quick Start

Ask Claude Code to use the `generate-image` skill, or run the command directly:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480 \
  --max-iterations 32 \
  --save-candidates 4
```

For better quality, ask Claude Code to write a `scene-plan.json` first. Scene plans let Claude control palette, objects, background, lighting, atmosphere, textures, reflections, focus, and final style.

Then run:

```bash
claude-imagegen generate \
  --prompt "cinematic red sun over a blue ocean with misty mountains" \
  --scene-plan claude-imagegen-output/demo/scene-plan.json \
  --output-dir claude-imagegen-output/demo \
  --width 720 \
  --height 480
```

## Outputs

Each run writes the main artifacts into the output directory:

- `image.png`: the generated image.
- `metadata.json`: prompt, dimensions, score details, caption evidence, palette evidence, and refinement hints.
- `quality-report.json`: readiness report for Claude Code, including concrete `next_actions`.
- `critique-request.json`: visual checklist for Claude Code to fill after inspecting `image.png`.
- `comparison-request.json`: refine-only parent/child comparison request.
- `candidates/` and `candidates.json`: optional alternatives when `--save-candidates` is used.
- `verification-report.json`: created by `verify`, with nonblank image checks and CPU/GPU device evidence.

## Refinement

Use `refine` to continue from a previous output directory:

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

## CPU, GPU, And Validation

Run the local test suite:

```bash
python -m pytest
```

Generate a small smoke image:

```bash
python -m claude_imagegen.cli generate \
  --prompt "red sun over blue ocean" \
  --output-dir claude-imagegen-output/smoke \
  --width 160 \
  --height 100 \
  --max-iterations 12 \
  --threshold 0.1
```

Run the built-in verification command:

```bash
claude-imagegen verify \
  --output-dir claude-imagegen-output/verification \
  --size 320x192 \
  --size 768x432 \
  --size 1024x640
```

For stronger local model checks, use a Python environment with `torch`, `transformers`, and the model weights available:

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
- `device_summary`: reports which CPU or GPU devices were used by each backend.
- per-case artifacts: include each image, metadata, quality report, critique request, and refinement comparison request.

## Local Development

From this checkout:

```bash
python -m pip install -e .
```

Validate the Claude plugin and marketplace manifest:

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

The plugin files are:

- `.claude-plugin/plugin.json`
- `.claude-plugin/marketplace.json`
- `skills/generate-image/SKILL.md`
- `bin/claude-imagegen`
- `src/claude_imagegen/`

## Current Limits

- Maximum output size is capped at 2048x2048 while preserving aspect ratio.
- This is a deterministic local renderer and verifier, not a diffusion-model image generator.
- Strong model verification depends on locally installed optional ML packages and model availability.
