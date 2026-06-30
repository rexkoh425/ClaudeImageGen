# Claude ImageGen Prototype Design

## Goal

Build a Claude Code plugin prototype that accepts a required text prompt plus optional reference image and optional initial image, then generates a capped 720x480 PNG in the current project directory. The prototype must be CPU-only, testable without Claude Code, and packaged so Claude Code can invoke it as a skill plus executable.

## Key Assumptions

- "Cloud Code" is treated as Claude Code.
- The prototype should run on a Mac Pro-class CPU without requiring a local GPU.
- The first working version should not download large model weights or require external API keys.
- The generator should expose a real iteration loop and prompt/reference similarity score, but it does not need to match the quality of pretrained diffusion systems.
- Directly asking an LLM to emit every RGB triplet for 720x480 would require hundreds of thousands of values and likely millions of tokens, so the prototype renders compact image primitives and can optionally export the final RGB pixel table.

## Research-Informed Approach

Modern text-to-image systems avoid literal pixel-by-pixel language generation. DALL-E tokenizes text and image content into a shared autoregressive stream. CLIP provides a text-image similarity space that can be optimized against. CLIPDraw and VQGAN-CLIP show a practical "generate candidate, score against prompt, update candidate" loop. GLIDE, classifier-free guidance, and latent diffusion models show why large systems denoise or optimize in learned latent spaces rather than raw pixels. NVIDIA DLSS/neural rendering work is also relevant: generated or reconstructed pixels can be inferred from compact rendered signals instead of rendered independently from scratch.

This prototype uses the same high-level optimization pattern but replaces heavyweight neural models with a CPU-safe surrogate:

1. Prefer a Claude-authored scene plan JSON containing palette, background, object placement, depth, and style hints. Fall back to parsing the prompt into a target feature vector when no scene plan is provided.
2. Build a compact scene candidate: background gradient, horizon, geometric strokes, and semantic primitives such as sun, moon, mountains, water, trees, buildings, clouds, and abstract shapes.
3. Render the candidate to a 720x480 RGB image with Pillow/NumPy.
4. Score the image using cheap CPU features: color histogram, brightness, edge density, region color placement, scene-object proxies, and optional reference-image palette similarity.
5. Mutate the compact candidate and keep the best candidate until the score reaches a target threshold or max iterations.
6. Write `image.png`, `metadata.json`, `progress.csv`, and optional `pixels.csv`.

## Plugin Shape

The repository root is a Claude Code plugin:

- `.claude-plugin/plugin.json` declares plugin metadata.
- `skills/generate-image/SKILL.md` tells Claude Code when and how to run generation.
- `bin/claude-imagegen` is the executable Claude Code exposes on PATH in Bash tool calls.
- `src/claude_imagegen/` contains the testable Python implementation.
- `tests/` validates the generator without Claude Code.

## Python Components

- `prompt.py`: normalize prompt text and extract a `PromptSpec`.
- `palette.py`: map common color words to RGB and extract a reference palette.
- `scene.py`: define compact scene candidates and mutation logic.
- `scene_plan.py`: parse Claude-authored JSON scene plans.
- `render.py`: render a candidate or scene plan to RGB PIL images with a hard max resolution of 720x480.
- `score.py`: compute cosine-like scores for prompt and reference alignment.
- `generator.py`: run the iterative optimization loop and write outputs.
- `pixels.py`: export `x,y,r,g,b` CSV for users who need explicit pixel values.
- `cli.py`: provide `claude-imagegen generate`.

## Error Handling

- Missing prompt returns a CLI usage error.
- Unsupported or missing reference/initial image paths fail with a clear message.
- Requested dimensions larger than 720x480 are capped and recorded in metadata.
- A low final score is not fatal; metadata reports `met_threshold=false` so Claude Code can decide whether to rerun with more iterations or a different seed.

## Testing Strategy

- Unit tests cover prompt parsing, size capping, rendering, scoring, reference palette influence, and pixel CSV export.
- CLI tests run end-to-end in a temporary directory and verify image dimensions plus metadata.
- Plugin structure tests validate `.claude-plugin/plugin.json`, skill frontmatter, and executable presence.
- A sample generation command is run after tests to produce real output evidence, including the preferred `--scene-plan` path.

## Out of Scope for This Prototype

- GPU diffusion, Stable Diffusion, local CLIP weights, or API-backed image generation.
- Photorealism parity with commercial models.
- Marketplace publishing.
- Automatic Claude Code installation into the user profile.
