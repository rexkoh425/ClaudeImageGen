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

The `ScenePlan` path is the preferred quality path because it makes Claude Code do the expensive semantic reasoning, detail placement, and lighting design, then leaves local work to deterministic rendering, deterministic motif expansion, light compositing, scoring, targeted auto-refinement, candidate ranking, and optional RGB export. The loop remains visible and testable, so Claude Code can inspect `metadata.json`, use extracted `reference_palette` and `initial_palette` colors from user-provided images, inspect `candidates.json` and top-ranked candidate PNGs, and follow `revision_hints` to revise missing objects, weak colors, low contrast, unclear mood, low caption evidence, or poor reference alignment before rerunning.

## Similarity Strategy

The current default scorer uses an explicit shared feature-vector cosine between prompt-derived text features and image-derived visual features. The text vector includes requested colors, objects, style words, and mood words. The image vector includes color presence, region/object proxies, brightness, edge density, contrast, warmth, and cloud/sky/water/foliage evidence. This is not CLIP quality, but it gives Claude Code a stable local optimization signal and makes every score auditable in `metadata.json`.

The optional `transformers-clip` scorer loads a CLIP model through Hugging Face Transformers and computes prompt-image embedding cosine directly. It can run on CPU or CUDA via `--similarity-device auto`. This is the first "strong model" extension point; it is intentionally optional so normal plugin tests remain offline and weight-free.

The current default caption backcheck uses local visual heuristics to produce `image_caption` and `caption_similarity_score`. It is intentionally simple and auditable: it looks for visible evidence such as warm upper sun regions, blue lower water regions, bright cloud regions, green foliage, edges, contrast, and dark skyline structure. The generator compares the caption back to the prompt and records `caption_missing_objects`, `caption_missing_colors`, `caption_unexpected_objects`, and `caption_unexpected_colors`; low caption similarity promotes missing requested evidence into `revision_hints`. The optional `transformers-blip` backend loads BLIP through Hugging Face Transformers and captions the generated image directly. It can run on CPU or CUDA via `--caption-device auto`, giving Claude Code a stronger "what does this image appear to contain" signal when local model dependencies and weights are available.

Good future similarity/refinement signals should be layered rather than singular:

1. CLIP/SigLIP-style prompt-image embedding cosine for broad semantic alignment.
2. Reference-image embedding cosine plus palette/layout comparisons for image-to-image refinement.
3. Caption-backchecking with local heuristics or a BLIP-style image captioner so Claude can compare what the generated image appears to contain against the requested prompt.
4. Preference/aesthetic reward such as ImageReward for choosing among multiple aligned candidates.
5. Hard local checks for resolution, aspect ratio, nonblankness, contrast, and requested object/color evidence.

The current `--save-candidates N` option supports a simple version of candidate ranking. It saves the top N scored images and a `candidates.json` index so Claude Code can inspect alternatives when scores are close or when the numeric best candidate is not the best visual continuation.

For iterative editing, prompt alignment is not enough. Each refinement run should also measure continuity against the previous image. The current `refine` command records `initial_similarity_score` between the new output and the parent `image.png`, plus lineage metadata (`refined_from`, `parent_image`, `parent_metadata`, `refinement_lineage_depth`). This gives Claude Code two independent signals: whether the image still resembles the previous iteration, and whether it moved closer to the revised text/reference target.

## Why Not Raw Pixel Generation

At 2048x2048 there are 4,194,304 pixels. Even a compact textual RGB representation such as `x,y,r,g,b` is far too large for iterative LLM generation. It is also hard for an LLM to maintain global composition while emitting independent pixels. Programmatic rendering keeps global composition in a small candidate object and emits pixels only at the final step.

## Future Upgrade Path

- Add SigLIP and ONNX-backed CLIP/SigLIP options for faster local scoring.
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
