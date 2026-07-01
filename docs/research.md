# Research Notes

This prototype is grounded in text-guided image generation research, but intentionally chooses a CPU-first renderer that can run inside a Claude Code workflow without GPU model weights by default. Optional model-backed scoring can use local PyTorch/Transformers CLIP on CPU or CUDA when available.

## Relevant Papers and Systems

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020): provides the core idea of scoring image/text alignment in a shared embedding space. A future higher-quality version should replace this prototype's handcrafted scorer with a small CLIP-like embedding model if CPU cost is acceptable.
- [SigLIP: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343): replaces CLIP's batch softmax contrastive objective with independent sigmoid pair scoring, which is relevant when scoring one prompt/image pair at a time inside an iterative local loop.
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086): shows the value of combining captioning/filtering with image-text alignment. This project now includes caption-backchecking: a default local caption proxy plus an optional Transformers BLIP backend to caption the generated image and compare that caption back to the prompt before revising the scene plan.
- [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977): relevant because prompt-image similarity alone is not enough; later refinement should include preference or aesthetic reward signals when choosing among candidate images.
- [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092): DALL-E showed text-to-image generation through learned discrete image tokens, not direct natural-language RGB enumeration.
- [CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders](https://arxiv.org/abs/2106.14843): directly relevant to this prototype because it optimizes compact drawing primitives against a language-image score.
- [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741): shows text-guided diffusion for generation and editing, including guidance from text prompts.
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752): explains why modern systems operate in compressed latent spaces for efficiency instead of denoising every output pixel directly.
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598): a key guidance technique for steering diffusion models toward prompts without a separate classifier.
- [Deep Image Prior](https://arxiv.org/abs/1711.10925): relevant to image restoration and inpainting because a generator structure itself can act as a useful prior, even without a dataset-trained model.
- [NVIDIA DLSS](https://www.nvidia.com/en-us/geforce/technologies/dlss/): relevant as a production example of neural reconstruction/upscaling, where final pixels can be inferred from lower-resolution or partial rendered signals rather than fully rendered independently.

## Design Implications

Text-to-image systems do not usually ask a language model to list every pixel. They use one of three practical patterns:

1. Generate or denoise in a learned latent space, then decode to pixels.
2. Optimize a compact visual representation, such as strokes, vectors, patches, or scene parameters, against a text-image score.
3. Reconstruct or upsample pixels from cheaper signals using an image prior.

This plugin prototype uses pattern 2. It supports two compact representations:

- `SceneCandidate`: keyword-derived fallback candidate for rough drafts.
- `ScenePlan`: Claude-authored JSON that captures the prompt interpretation, palette, layout, foreground/background depth, and style hints before rendering.
- `background.stops`: Claude-authored multi-stop sky and backdrop gradients, where the model specifies normalized stop positions, colors, and direction to capture horizon glow or banded lighting without local semantic inference.
- `elements`: low-level Claude-authored detail primitives inside a scene plan, including lines, paths, polygons, ellipses, rectangles, arcs, and glows. Filled elements can include linear or radial gradients so Claude can specify local color transitions and depth without enumerating pixels. Elements can include blur and blend settings, including overlay and soft-light, so Claude can specify softness and compositing intent.
- `motifs`: repeated-detail instructions inside a scene plan, where Claude specifies count, region, color, size, seed, and z-order for details such as stars, rain, grass, window lights, or dots.
- `textures`: region-bound surface instructions inside a scene plan, where Claude specifies texture type, region, color, density, scale, opacity, blend mode, seed, and z-order for painterly hatching, water ripples, paper grain, mist, noise, or speckled material breakup.
- `materials`: compact semantic surface regions inside a scene plan, where Claude specifies a surface kind, region, color set, intensity, scale, opacity, seed, and z-order once, then the renderer expands that into bounded gradients and deterministic marks.
- `terrains`: compact ridge and landform geometry inside a scene plan, where Claude specifies normalized ridge points, base height, colors, opacity, facets, and z-order. The renderer fills deterministic silhouettes and shaded planes rather than choosing random mountain shapes.
- `reflections`: compact source-to-target mirror instructions, where Claude specifies which already-rendered region should be reflected into another region with opacity, blur, fade, tint, blend mode, and z-order. This captures broad water reflections without listing pixels.
- `warps`: compact bounded displacement instructions, where Claude specifies a region, direction, amplitude, wavelength, phase, seed, and z-order. The renderer remaps rows or columns deterministically to bend reflections and water surfaces without a learned model.
- `atmosphere`: Claude-authored horizon haze and aerial perspective controls, where the renderer applies a smooth deterministic fog band to improve depth without a learned model.
- `veils`: Claude-authored localized atmospheric overlays, where the model specifies a bounded region, color, opacity, blur, blend mode, directional falloff, and z-order for mist, smoke, rain wash, or glow without applying haze to the whole scene.
- `lights`: Claude-authored local illumination and shadow fields, letting the model specify focal light, rim tint, and shadow mood as compact compositing instructions.
- `beams`: Claude-authored directional light shafts, where the model specifies origin, screen angle, length, angular spread, opacity, blur, count, z-order, and optional occlusion masks. The renderer expands them into translucent wedges for sunrise, haze, window light, and forest shafts, then clips out Claude-specified mask shapes when rays should pass behind terrain or foreground layers.
- `clouds`: Claude-authored cloud-bank instructions, where the model specifies sky region, color, shadow, opacity, blur, count, lobe count, scale, blend mode, seed, and z-order. The renderer expands that compact cloud intent into deterministic puffy forms without requiring individual ellipse elements.
- `shadows`: Claude-authored cast-shadow and contact-shadow instructions, where the model specifies ellipse, rectangle, or polygon shadow shapes, placement, opacity, blur, blend mode, and z-order. The renderer expands those into soft multiply overlays that ground objects without adding local semantic decisions.
- `focus`: Claude-authored depth-of-field instructions, where the model specifies a normalized focal region, blur amount, falloff, and whether to blur inside or outside the region. The renderer applies deterministic masked blur after scene composition without deciding what the subject is locally.
- `style`: Claude-authored final grading controls for grain, vignette, saturation, contrast, warmth, bloom, and antialiasing. These keep final art direction explicit while the renderer performs only deterministic CPU post-processing; antialiasing uses a bounded higher-resolution intermediate raster and downsamples to the capped output size to smooth Claude-authored hard geometry.

The `ScenePlan` path is the preferred quality path because it makes Claude Code do the expensive semantic reasoning, detail placement, and lighting design, then leaves local work to deterministic rendering, deterministic motif expansion, light compositing, scoring, targeted auto-refinement, candidate ranking, quality reporting, and optional RGB export. The loop remains visible and testable, so Claude Code can inspect `quality-report.json`, use extracted `reference_palette` and `initial_palette` colors from user-provided images, inspect `candidates.json` and top-ranked candidate PNGs, and follow report `next_actions` or metadata `revision_hints` to revise missing objects, weak colors, low contrast, unclear mood, low caption evidence, weak continuity, or poor reference alignment before rerunning.

## Similarity Strategy

The current default scorer uses an explicit shared feature-vector cosine between prompt-derived text features and image-derived visual features. The text vector includes requested colors, objects, style words, and mood words. The image vector includes color presence, region/object proxies, brightness, edge density, contrast, warmth, and cloud/sky/water/foliage evidence. This is not CLIP quality, but it gives Claude Code a stable local optimization signal and makes every score auditable in `metadata.json`.

The optional `transformers-clip` scorer loads a CLIP model through Hugging Face Transformers and computes prompt-image embedding cosine directly. It can run on CPU or CUDA via `--similarity-device auto`. This is the first "strong model" extension point; it is intentionally optional so normal plugin tests remain offline and weight-free. When an initial or parent image is present, the same backend can also compute a CLIP image-image cosine that is blended with local continuity metrics.

The current default caption backcheck uses local visual heuristics to produce `image_caption` and `caption_similarity_score`. It is intentionally simple and auditable: it looks for visible evidence such as warm upper sun regions, blue lower water regions, bright cloud regions, green foliage, edges, contrast, and dark skyline structure. The generator compares the caption back to the prompt and records `caption_missing_objects`, `caption_missing_colors`, `caption_unexpected_objects`, and `caption_unexpected_colors`; low caption similarity promotes missing requested evidence into `revision_hints`. The optional `transformers-blip` backend loads BLIP through Hugging Face Transformers and captions the generated image directly. It can run on CPU or CUDA via `--caption-device auto`, giving Claude Code a stronger "what does this image appear to contain" signal when local model dependencies and weights are available. For the NLP comparison leg, `--caption-similarity-backend transformers-sentence` uses `sentence-transformers/all-MiniLM-L6-v2` embeddings to compare the generated caption back to the original prompt, while retaining explicit object/color gap diagnostics.

Good future similarity/refinement signals should be layered rather than singular:

1. CLIP/SigLIP-style prompt-image embedding cosine for broad semantic alignment.
2. Reference-image embedding cosine plus palette/layout comparisons for image-to-image refinement.
3. Structural continuity signals such as single-scale and multi-scale luminance SSIM, edge cosine, pixel cosine, color-histogram overlap, and a 3x3 region-similarity grid so edits can preserve layout instead of chasing only text alignment.
4. Caption-backchecking with local heuristics or a BLIP-style image captioner, then lexical or sentence-embedding prompt/caption comparison so Claude can compare what the generated image appears to contain against the requested prompt.
5. Preference/aesthetic reward such as ImageReward for choosing among multiple aligned candidates.
6. Hard local checks for resolution, aspect ratio, nonblankness, contrast, and requested object/color evidence.

The current `--save-candidates N` option supports a simple version of candidate ranking. It saves the top N scored images, a `candidates.json` index, and a visual `contact-sheet.png` so Claude Code can inspect alternatives when scores are close or when the numeric best candidate is not the best visual continuation. Each candidate index entry carries its own caption similarity, caption missing/unexpected evidence, `selection_score`, and `selection_reasons`, making selection less dependent on the final image's score alone. The `refine --candidate-rank auto` path closes this loop by letting Claude Code use the recommended candidate PNG as the next initial image while preserving parent candidate lineage in metadata; `refine --candidate-rank N` remains available when visual inspection should override the automatic recommendation.

For iterative editing, prompt alignment is not enough. Each refinement run should also measure continuity against the previous image. The current `refine` command records `initial_similarity_score` plus `initial_similarity_details` between the new output and the selected parent image. The local detail fields include image cosine, luminance SSIM, multi-scale luminance SSIM, edge cosine, color-histogram overlap, a 3x3 `region_similarity_scores` grid, `weakest_continuity_region`, local continuity, and final continuity. When `transformers-clip` is active, `clip_image_cosine_score` is added and blended into the final continuity score. Together with lineage metadata (`refined_from`, `parent_image`, `parent_metadata`, `parent_candidate_selection`, `refinement_lineage_depth`), this gives Claude Code two independent signals: whether the image still resembles the previous iteration, and whether it moved closer to the revised text/reference target.

`quality-report.json` aggregates those signals into an explicit `status`, `quality_score`, individual checks, and `next_actions`. It is intentionally not a hidden oracle: every check is backed by metadata fields such as prompt score, caption score, continuity score, reference score, and candidate recommendation score. This makes it useful as the first inspection point for Claude Code while preserving traceability for manual diagnosis.

## Why Not Raw Pixel Generation

At 2048x2048 there are 4,194,304 pixels. Even a compact textual RGB representation such as `x,y,r,g,b` is far too large for iterative LLM generation. It is also hard for an LLM to maintain global composition while emitting independent pixels. Programmatic rendering keeps global composition in a small candidate object and emits pixels only at the final step.

## Future Upgrade Path

- Add ONNX-backed CLIP/SigLIP/DINOv2 options for faster local scoring.
- Add more caption backends and object-grounding checks so Claude can revise based on localized missing or mistaken objects, not only global caption overlap.
- Add preference scoring, such as ImageReward-style candidate ranking, to choose better images when semantic cosine scores tie.
- Add a learned patch prior or tiny autoencoder if quality becomes more important than zero-weight portability.
- Expand the scene-plan schema with explicit camera, negative-space controls, and scene-level composition guides.
- Add richer path styling such as dashed strokes, variable stroke width, and path-level gradients.
- Add material presets that combine textures, motifs, and light responses for water, foliage, stone, fabric, skin, and metal.
- Add terrain variants for cliffs, rolling hills, dunes, snow caps, and foreground islands.
- Add more reflection controls such as horizon-aware masking and layered wave fields while keeping CPU rendering deterministic.
- Add weather presets such as storm clouds, fog banks, and low stratus layers using the cloud primitive.
- Add image-to-image edit modes: preserve composition from `--initial-image`, mutate palette/object overlays, and use a stricter reference similarity score.

## Similarity Methods Survey (2025)

This project generates images from code that Claude Code authors, with no hosted image API. The
verification side therefore matters as much as generation: we need trustworthy ways to measure
how close an image is to a text prompt and to a previous/target image. The current methods, and
how each maps to this code-only loop, are below.

### Image-to-text alignment (the "is this the right picture" question)

- **CLIPScore** ([CLIP](https://arxiv.org/abs/2103.00020)) is the standard baseline: cosine
  between CLIP text and image embeddings. It is cheap and scalable but behaves like a
  "bag of words", conflating compositional prompts (it can score "the horse eats the grass"
  and "the grass eats the horse" similarly). Good as a coarse signal, weak on attribute/relation
  binding. This project's optional `transformers-clip` backend is exactly CLIPScore.
- **SigLIP / SigLIP2** ([SigLIP](https://arxiv.org/abs/2303.15343)) replace CLIP's batch-softmax
  with an independent sigmoid per image-text pair. That is a better fit when scoring a single
  prompt/image pair in an iterative loop because there is no dependence on a batch of negatives,
  and SigLIP2 reports richer, more invertible features. This project implements the optional
  `transformers-siglip` backend with `google/siglip-base-patch16-224` as the default model-backed
  scorer, and also uses SigLIP image embeddings for continuity when refining from a parent image.
- **VQAScore** ([Lin et al., ECCV 2024](https://linzhiqiu.github.io/papers/vqascore/)) asks a
  visual-question-answering model "Does this figure show {text}?" and uses P(yes). It is
  state-of-the-art for image-text alignment correlation with humans across many benchmarks and
  handles compositional prompts far better than CLIPScore. **This is the single most important
  reference for this project**, because Claude Code can compute a VQAScore-style judgement
  natively: open the rendered PNG, answer how well each requested element is present, and return a
  structured 0-1 score. No external model is required.
- **TIT-Score / image-to-text-to-text consistency**
  ([2025](https://arxiv.org/pdf/2510.02987)) captions the image and compares the caption back to
  the prompt in text space, which is robust for long prompts. This is the existing caption
  backcheck (`caption_*` fields); the 2025 literature confirms it as a valid independent signal.
- **ImageReward** ([Xu et al., 2023](https://arxiv.org/abs/2304.05977)) fine-tunes BLIP on human
  preference data for an aesthetic/preference reward, useful for ranking among already-aligned
  candidates rather than for alignment itself.
- **LMM-as-evaluator** is the clearest 2025 trend: fine-tuned or prompted multimodal LLMs act as
  the direct judge of generated images (e.g. "Multi-Modal Language Models as Text-to-Image Model
  Evaluators", and grounded-reasoning diagnostics like ImageDoctor). This directly justifies the
  design choice in this repository: **Claude's own vision is the judge in the refinement loop.**

### Image-to-image similarity (the "did the edit keep what mattered" question)

- **SSIM / MS-SSIM / PSNR** are classic, model-free structural/error metrics. SSIM (luminance +
  contrast + structure) is good for continuity between refinement iterations but blind to semantics.
  Multi-scale SSIM ([Wang et al., 2003](https://ece.uwaterloo.ca/~z70wang/publications/msssim.html))
  checks structure after repeated low-pass/downsample steps, making it more useful when a refine run
  preserves broad layout while changing local detail. This project records both
  `luminance_ssim_score` and `multiscale_luminance_ssim_score`.
- **LPIPS** ([Zhang et al., 2018](https://arxiv.org/abs/1801.03924)) compares deep CNN features and
  tracks low-level human perception well, but is not semantic: two images can look similar and mean
  very different things.
- **CLIP and SigLIP image embeddings** provide semantic image vectors that can be blended with local
  continuity checks. The current optional backends add `clip_image_cosine_score` or
  `siglip_image_cosine_score` to `initial_similarity_details` during refine/initial-image runs.
- **DINOv2** ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193)) is self-supervised, 768-d, and
  **outperforms CLIP for pure image-to-image similarity**, especially at identifying the primary
  subject and fine-grained distinctions. This project implements it as the optional
  `transformers-dinov2` continuity backend with `facebook/dinov2-base`, separate from text-image
  scoring so a refine run can use SigLIP for prompt alignment and DINOv2 for parent-image
  continuity at the same time.
- **DreamSim** ([Fu et al., 2023](https://arxiv.org/abs/2306.09344)) fuses OpenCLIP + DINO features
  and is tuned on human similarity judgements, bridging low-level (LPIPS/SSIM) and high-level (CLIP)
  similarity. It is the best single learned image-image metric to target when weights are allowed.

### What this repository implements without any model weights or API

Because the default path must run with only `numpy` and `Pillow`, the project uses a **layered local
image embedding** (`embedding.py`) that approximates the structure-plus-semantics idea above without
a neural network:

1. Regional mean color on a grid, in a perceptual-ish space (luma + opponent color channels), to
   capture global layout and palette placement (a cheap stand-in for DINOv2's "primary elements").
2. Per-region hue/saturation/value histograms for color-distribution similarity (color-histogram
   intersection is a well-known robust image-similarity primitive).
3. Edge-orientation histograms per region (a HOG-like descriptor) for structure and silhouette
   continuity, complementing SSIM.
4. Global statistics: contrast, brightness, saturation, warmth, edge density.

These are concatenated and L2-normalized into one vector; image-to-image closeness is the cosine of
two such vectors, remapped to 0-1. This is intentionally not CLIP/DINOv2 quality, but it is
deterministic, auditable, fast, and — critically — always available, giving the refinement loop a
stable quantitative closeness number to optimize. The parent-child path also records
`region_similarity_scores`, `regional_continuity_score`, `weakest_continuity_region`, and
`weakest_continuity_region_score` so Claude Code can focus comparison on the local area that drifted
most. When `torch` + weights are present, the optional CLIP, SigLIP, or DINOv2 image-embedding
cosine is blended in for a stronger continuity signal.

### How the layered signals combine in the loop

The refinement loop is meant to use independent signals rather than a single oracle, mirroring the
survey above: VQAScore-style alignment (Claude vision judge), text-embedding cosine (local feature
vector or optional CLIP), caption backcheck (TIT-Score-style), structural continuity
(`luminance_ssim_score`, `multiscale_luminance_ssim_score`, edge cosine, color histograms, and
`weakest_continuity_region` from the 3x3 grid), and image-embedding cosine for continuity (local
layered embedding or optional CLIP/DINOv2). The
`quality-report.json` aggregates these into a status and `next_actions`, but every number remains
traceable to its source signal.

### Honest scope versus Nano Banana

[Gemini 2.5 Flash Image / "Nano Banana"](https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/)
is a learned diffusion/transformer model. Its headline strengths are photoreal synthesis,
conversational iterative editing, character consistency across edits, targeted natural-language
local edits, and multi-image fusion. A code-only generator that renders from a Claude-authored scene
representation can credibly pursue the *editing and verification* half of that list — iterative
targeted edits to the scene plan, continuity across edits measured by the image embedding, crisp
in-image text (vector/raster text is actually sharper than diffusion glyphs), and multi-element
composition. It cannot reach photoreal synthesis, because there is no learned image prior. The
project's value is the disciplined, fully auditable, model-in-the-loop refinement process, not
pixel-level photorealism.
