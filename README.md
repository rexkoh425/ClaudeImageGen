# Claude ImageGen

Claude ImageGen is a Claude Code plugin for making local PNG images.

The default path is lightweight and CPU-first. For stronger photoreal detail, install the optional local Diffusers/Torch backend and use your own CPU/GPU. Claude can plan, critique, compare, and refine, but Claude is not the image model.

## Install In Claude Code

Claude ImageGen is available through the Claude Code plugin marketplace from this GitHub repo.

On another machine, install Claude Code, sign in to GitHub if this repo is private, then run this inside Claude Code:

```text
/plugin marketplace add rexkoh425/ClaudeImageGen
/plugin install claude-imagegen@claude-imagegen
```

Restart Claude Code after installation so the `generate-image` skill and `claude-imagegen` command are loaded.

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

After installing the diffusion extra, use the photoreal profile for detailed night scenes:

```bash
claude-imagegen diffuse \
  --profile night-photoreal \
  --prompt "deep night glass greenhouse interior, tropical plants with sharp leaf veins, warm tungsten hanging lamps, volumetric mist, wet black stone floor with mirror reflections, no people" \
  --output-dir claude-imagegen-output/greenhouse-gpu \
  --width 1024 \
  --height 768 \
  --seeds 101,202,303,404 \
  --device auto \
  --quality-target 0.9
```

`diffuse` writes multiple candidates, selects the strongest prompt-aware local candidate, and creates `candidates/contact-sheet.png`. Open `image.png`, `candidates/contact-sheet.png`, and `critique-request.json` with Claude vision before accepting a `0.9` target. Keep prompts concise; `metadata.json` records `prompt_length_warning` when SDXL-style text limits may truncate later details.

## Pair Evaluation

Use this when comparing a raw image and an improved image. It does not generate anything; it writes the JSON request Claude should fill after opening both images.

```bash
claude-imagegen pair-eval \
  --prompt "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections" \
  --before claude-imagegen-output/base/image.png \
  --after claude-imagegen-output/refined/image.png \
  --pair-id greenhouse-v1 \
  --output-dir claude-imagegen-output/greenhouse-eval \
  --quality-target 0.9
```

Open `pair-evaluation-request.json` with Claude vision and fill its `expected_response`, then run `claude-imagegen eval-plan --evaluation claude-response.json --prompt "<same prompt>" --output-dir claude-imagegen-output/greenhouse-plan --quality-target 0.9`. Repeat `--evaluation` with multiple Claude responses to keep the gate conservative when scores disagree.

Do not claim GPT/Sora parity unless the after image scores at least `0.9` and the response marks the gate as met.

If Claude says the improved image is too bright or hazy for deep night, run a dark-preserving local postprocess:

```bash
claude-imagegen enhance-night --input-image claude-imagegen-output/refined/image.png \
  --prompt "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections" \
  --output-dir claude-imagegen-output/refined-night --quality-target 0.9
```

`enhance-night` writes a new `image.png`, `metadata.json`, and `pair-evaluation-request.json`; Claude must still score the before/after pair before acceptance.

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

For visual feedback loops, fill `critique-request.json` as `critique.json`, then pass it back with `refine --critique`.

## Outputs

Each run writes:

- `image.png`: selected output image.
- `metadata.json`: prompt, settings, scores, selected seed or candidate, and refinement hints.
- `quality-report.json`: readiness report with concrete `next_actions`.
- `critique-request.json`: visual checklist for Claude Code to fill after inspecting `image.png`.
- `comparison-request.json`: refine-only parent/child comparison request.
- `pair-evaluation-request.json`: before/after scoring request created by `pair-eval`.
- `candidates/`, `candidates.json`, and `candidates/contact-sheet.png`: alternatives.
- `verification-report.json`: created by `verify`, with image/device evidence.

## Verify

```bash
python -m pytest
```

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

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

Open `verification-report.json` and check `image_summary`, `device_summary`, nonblank artifacts, and failed case details.

## Quality Target

`--quality-target 0.9` is a gate, not a promise. A run should only be accepted when local scores and `image_detail_score` are strong, Claude vision gives a high `closeness_score`, `quality-report.json` has `target_quality_met: true`, and GPT/Sora-level parity is not claimed unless an actual Claude visual judgement supports it.

Current greenhouse testing on an RTX 5070 Ti improved through diffusion and postprocessing, but Claude pair-evaluation scored the best after image `0.84`, not `0.9`. Treat that as useful progress, not solved parity.

## Current Limits

- Maximum output size is capped at 2048x2048 while preserving aspect ratio.
- CPU `generate` is a deterministic local renderer and verifier, not a diffusion model.
- GPU `diffuse` and strong model verification require optional local ML packages and model weights.
