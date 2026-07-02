# Claude ImageGen

Claude ImageGen is a Claude Code plugin for making local PNG images from Claude-authored scene plans. Claude does the planning and critique, plus comparison and refinement. Your machine renders on CPU by default, with optional local GPU diffusion for stronger photoreal detail.

## Install In Claude Code

Claude ImageGen is available through the Claude Code plugin marketplace from this GitHub repo. You do not need to clone the repo for normal Claude Code use.

Fastest setup:

```text
/plugin marketplace add rexkoh425/ClaudeImageGen
/plugin install claude-imagegen@claude-imagegen
```

On another machine, install Claude Code first, sign in to GitHub if this repo is private, then run the same two commands in Claude Code.

Restart Claude Code after installation so the `generate-image` skill and `claude-imagegen` command are loaded. To update after a new release, run `/plugin install claude-imagegen@claude-imagegen` again and restart.

Then ask Claude Code:

```text
Use the generate-image skill to create <your prompt> with multi-refinement and a 0.9 gate.
```

The first use may set up more than expected because the plugin checks Python, creates or reuses a plugin-owned virtual environment, verifies numpy/Pillow, and optionally checks Torch/Diffusers/CUDA. That setup is local dependency work, not Claude doing image compute.

## Local Setup

Use this only when developing from a clone of this repo. Marketplace users normally do not need it.

CPU setup:

```bash
python -m pip install -e .
claude-imagegen setup
```

Optional GPU/photoreal setup:

```bash
python -m pip install -e ".[diffusion]"
claude-imagegen setup --with-diffusion
```

`setup` reports missing Python packages, optional Diffusers/Torch packages, CUDA visibility, and the next install command when something is missing. If basic CPU dependencies are missing, run `python -m pip install -e .`; if photoreal GPU dependencies are missing, run `python -m pip install -e ".[diffusion]"`.

## Best Result Loop

For simple images:

```bash
claude-imagegen generate --prompt "cinematic red sun over a blue ocean with misty mountains" --output-dir claude-imagegen-output/demo --width 720 --height 480 --quality-target 0.9 --save-candidates 4
```

For better CPU results, ask Claude Code to use the `generate-image` skill for a multi-refinement loop. Claude writes `scene-plan.json`, renders `image.png`, inspects `critique-request.json`, writes visual feedback, then reruns `refine` for 2-4 rounds when the image is close but not detailed enough.

The scene plan supports diagram and icon primitives: `text`, `arrow`, `rounded_rectangle`, `aperture`, `sparkle`, and explicit `stroke_width`. Use these for architecture diagrams, app icons, labels, connectors, and crisp vector-style shapes. For diagrams, keep labels inside boxes, keep badges inset, and give image tiles their own label space. `service tiles` and `image tiles` are treated as diagram/UI tiles; add explicit `floor`, `stone`, or `wet` only when you want a physical floor.

## Higher-Detail GPU Image

After installing the diffusion extra: `claude-imagegen diffuse --profile night-photoreal --prompt "deep night glass greenhouse, tropical plants, sharp leaves, warm tungsten lamps, mist beams, black wet mirror floor, no people" --output-dir claude-imagegen-output/greenhouse-gpu --width 1024 --height 768 --seeds 101,202,303,404 --device auto --quality-target 0.9`

For multi-refinement, rerun with `--initial-image <previous image.png> --strength 0.16`. The `night-photoreal` profile keeps lamp, mist-beam, floor, and leaf-detail terms compact so they do not fall out of CLIP's text window.

If Claude says the improved image is too bright or hazy for deep night: `claude-imagegen enhance-night --input-image claude-imagegen-output/refined/image.png --prompt "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections" --output-dir claude-imagegen-output/refined-night --quality-target 0.9 --shadow-lift 0.08 --foliage-clarity 0.35 --mist-beam-strength 0.45`

## Pair Evaluation

Use this when comparing a raw image and an improved image: `claude-imagegen pair-eval --prompt "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections" --before claude-imagegen-output/base/image.png --after claude-imagegen-output/refined/image.png --pair-id greenhouse-v1 --output-dir claude-imagegen-output/greenhouse-eval --quality-target 0.9`

Open `pair-evaluation-request.json` with Claude vision and fill its `expected_response`. For local metric evidence:

```bash
claude-imagegen audit-pair --before claude-imagegen-output/base/image.png --after claude-imagegen-output/refined/image.png --prompt "<same prompt>" --output-dir claude-imagegen-output/greenhouse-audit
```

Then plan the next step:

```bash
claude-imagegen eval-plan --evaluation claude-response.json --audit claude-imagegen-output/greenhouse-audit/pair-audit.json --prompt "<same prompt>" --output-dir claude-imagegen-output/greenhouse-plan --quality-target 0.9 --min-evaluations 2
```

Repeat `--evaluation` with multiple Claude responses when scores disagree. Do not claim GPT/Sora parity unless the after image scores at least `0.9` and the response marks the gate as met.

## Refinement

Continue from a previous CPU output: `claude-imagegen refine --from-dir claude-imagegen-output/demo --prompt "same coastal scene, stronger clouds, richer foreground grass, and clearer water reflections" --output-dir claude-imagegen-output/demo-refined --candidate-rank auto --max-iterations 8`

For visual feedback loops, fill `critique-request.json` as `critique.json`, then pass it back with `refine --critique`. For parent/child checks, fill `comparison-request.json`, then pass it with `refine --comparison`.

## Outputs

Each run writes the important artifacts beside the image:

- `image.png`: selected output image.
- `metadata.json`: prompt, settings, scores, selected seed or candidate, and refinement hints.
- `quality-report.json`: readiness report with concrete `next_actions`.
- `critique-request.json`: visual checklist for Claude Code to fill after inspecting `image.png`.
- `comparison-request.json`: refine-only parent/child comparison request.
- `pair-evaluation-request.json`: before/after scoring request created by `pair-eval`.
- `candidates/`, `candidates.json`, and `candidates/contact-sheet.png`: alternatives.
- `verification-report.json`: created by `verify`, with `device_summary`, `image_summary`, and nonblank checks.

## Verify

```bash
python -m pytest
```

```bash
claude plugin validate . --strict
claude plugin validate .claude-plugin/marketplace.json --strict
```

```bash
claude-imagegen verify --output-dir claude-imagegen-output/verification --size 320x192 --size 768x432 --size 1024x640
```

For stronger local model checks:

```bash
claude-imagegen verify --output-dir claude-imagegen-output/verification-strong --size 320x192 --strong-model --strong-size 768x432 --strong-similarity-backend transformers-siglip --strong-continuity-backend transformers-dinov2 --caption-similarity-backend transformers-sentence
```

Open `verification-report.json` and check `image_summary`, `device_summary`, nonblank artifacts, and failed case details.

## Quality Target

`--quality-target 0.9` is a gate, not a promise. A run should only be accepted when local scores and `image_detail_score` are strong, Claude vision gives a high `closeness_score`, `quality-report.json` has `target_quality_met: true`, and GPT/Sora-level parity is not claimed unless an actual Claude visual judgement supports it.

Current local testing shows diagrams and icon-style images can pass Claude visual acceptance, while strict photoreal greenhouse testing remains below the `0.9` target. Treat this as useful progress, not solved parity.

## Current Limits

- Maximum output size is capped at 2048x2048 while preserving aspect ratio.
- CPU `generate` is a deterministic local renderer and verifier, not a diffusion model.
- GPU `diffuse` and strong model verification require optional local ML packages and model weights.
