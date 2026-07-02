import json
from pathlib import Path

from PIL import Image, ImageDraw

from claude_imagegen.caption import CaptionResult
import claude_imagegen.caption as caption_module
import claude_imagegen.generator as generator_module
import claude_imagegen.score as score_module
from claude_imagegen.generator import GenerateOptions, _truncate_text_to_width, generate_image
from claude_imagegen.prompt import parse_prompt
from claude_imagegen.render import cap_dimensions, render_candidate
from claude_imagegen.scene import build_initial_candidate
from claude_imagegen.score import score_image


def test_cap_dimensions_allows_larger_outputs_with_aspect_preserving_cap():
    assert cap_dimensions(1536, 864) == (1536, 864)
    assert cap_dimensions(4096, 2048) == (2048, 1024)
    assert cap_dimensions(500, 300) == (500, 300)


def test_render_candidate_returns_rgb_image_with_requested_capped_size():
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=4)

    image = render_candidate(candidate, width=1024, height=640)

    assert image.mode == "RGB"
    assert image.size == (1024, 640)


def test_scoring_rewards_prompt_aligned_render_over_blank_canvas():
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=2)
    image = render_candidate(candidate, width=720, height=480)
    blank = Image.new("RGB", (720, 480), (240, 240, 240))

    rendered = score_image(image, spec)
    blank = score_image(blank, spec)

    assert "cosine_score" in rendered.details
    assert 0.0 <= rendered.details["cosine_score"] <= 1.0
    assert rendered.text_score > blank.text_score
    assert rendered.details["cosine_score"] > blank.details["cosine_score"]
    assert rendered.text_score >= 0.45


def test_reference_image_palette_influences_generated_output(tmp_path: Path):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), (25, 170, 75)).save(reference)

    result = generate_image(
        GenerateOptions(
            prompt="abstract botanical poster",
            output_dir=tmp_path / "out",
            reference_image=reference,
            width=160,
            height=100,
            max_iterations=8,
            seed=7,
            threshold=0.1,
        )
    )

    pixel = result.image.resize((1, 1)).getpixel((0, 0))
    assert pixel[1] > pixel[0]
    assert pixel[1] > pixel[2]
    assert result.metadata["reference_score"] > 0


def test_reference_and_initial_palettes_are_written_to_metadata(tmp_path: Path):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), (25, 170, 75)).save(reference)
    initial = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), (200, 40, 120)).save(initial)

    result = generate_image(
        GenerateOptions(
            prompt="abstract poster",
            output_dir=tmp_path / "out",
            reference_image=reference,
            initial_image=initial,
            width=80,
            height=50,
            max_iterations=2,
            seed=5,
            threshold=0.1,
        )
    )

    assert result.metadata["reference_palette"] == ["#19aa4b"]
    assert result.metadata["initial_palette"] == ["#c82878"]
    assert result.metadata["initial_similarity_score"] is not None
    assert 0.0 <= result.metadata["initial_similarity_score"] <= 1.0
    initial_details = result.metadata["initial_similarity_details"]
    assert initial_details["continuity_score"] == result.metadata["initial_similarity_score"]
    assert 0.0 <= initial_details["image_cosine_score"] <= 1.0
    assert 0.0 <= initial_details["luminance_ssim_score"] <= 1.0
    assert 0.0 <= initial_details["multiscale_luminance_ssim_score"] <= 1.0
    assert 0.0 <= initial_details["edge_cosine_score"] <= 1.0
    assert 0.0 <= initial_details["color_histogram_score"] <= 1.0
    region_scores = initial_details["region_similarity_scores"]
    assert set(region_scores) == {
        "top_left",
        "top_center",
        "top_right",
        "middle_left",
        "middle_center",
        "middle_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    }
    assert all(0.0 <= score <= 1.0 for score in region_scores.values())
    assert initial_details["weakest_continuity_region"] in region_scores
    assert initial_details["weakest_continuity_region_score"] == min(region_scores.values())
    report = json.loads((tmp_path / "out" / "quality-report.json").read_text(encoding="utf-8"))
    assert report["weakest_continuity_region"] == initial_details["weakest_continuity_region"]
    assert report["weakest_continuity_region_score"] == initial_details["weakest_continuity_region_score"]


def test_image_similarity_details_reward_identical_images_over_different_image(tmp_path: Path):
    assert hasattr(score_module, "image_similarity_details")

    reference = Image.new("RGB", (48, 32), (20, 40, 90))
    draw = ImageDraw.Draw(reference)
    draw.rectangle((4, 6, 26, 24), fill=(210, 50, 70))
    draw.ellipse((24, 4, 42, 22), fill=(240, 230, 170))
    reference_path = tmp_path / "reference.png"
    reference.save(reference_path)
    different = Image.new("RGB", (48, 32), (220, 220, 220))

    identical_details = score_module.image_similarity_details(reference, reference_path)
    different_details = score_module.image_similarity_details(different, reference_path)

    assert identical_details["continuity_score"] > different_details["continuity_score"]
    assert identical_details["image_cosine_score"] > different_details["image_cosine_score"]
    assert identical_details["luminance_ssim_score"] > different_details["luminance_ssim_score"]
    assert identical_details["multiscale_luminance_ssim_score"] > different_details["multiscale_luminance_ssim_score"]
    assert identical_details["multiscale_luminance_ssim_score"] >= 0.99
    assert identical_details["regional_continuity_score"] > different_details["regional_continuity_score"]
    assert identical_details["weakest_continuity_region_score"] > different_details["weakest_continuity_region_score"]
    assert identical_details["edge_cosine_score"] > different_details["edge_cosine_score"]
    assert identical_details["color_histogram_score"] > different_details["color_histogram_score"]
    assert identical_details["continuity_score"] >= 0.99


def test_initial_similarity_details_include_clip_image_cosine_when_clip_backend_is_active(tmp_path: Path, monkeypatch):
    initial = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), (200, 40, 120)).save(initial)

    monkeypatch.setattr(score_module, "_clip_text_image_score", lambda *args, **kwargs: 0.61)
    monkeypatch.setattr(score_module, "_clip_image_image_score", lambda *args, **kwargs: 0.83)

    result = generate_image(
        GenerateOptions(
            prompt="abstract poster",
            output_dir=tmp_path / "out",
            initial_image=initial,
            width=80,
            height=50,
            max_iterations=1,
            seed=5,
            threshold=0.1,
            similarity_backend="transformers-clip",
            similarity_device="cuda",
        )
    )

    initial_details = result.metadata["initial_similarity_details"]
    assert initial_details["clip_image_cosine_score"] == 0.83
    assert initial_details["continuity_score"] == result.metadata["initial_similarity_score"]
    assert result.metadata["similarity_backend"] == "transformers-clip"


def test_scoring_can_use_siglip_backend(monkeypatch):
    spec = parse_prompt("red sun over blue ocean")
    image = render_candidate(build_initial_candidate(spec, seed=3), width=96, height=64)

    monkeypatch.setattr(score_module, "_siglip_text_image_score", lambda *args, **kwargs: 0.74)

    score = score_image(
        image,
        spec,
        similarity_backend="transformers-siglip",
        similarity_model="google/siglip-base-patch16-224",
        similarity_device="cuda",
    )

    assert score.details["cosine_score"] == 0.74
    assert score.text_score > 0.0


def test_initial_similarity_details_include_siglip_image_cosine_when_siglip_backend_is_active(tmp_path: Path, monkeypatch):
    initial = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), (200, 40, 120)).save(initial)

    monkeypatch.setattr(score_module, "_siglip_text_image_score", lambda *args, **kwargs: 0.62)
    monkeypatch.setattr(score_module, "_siglip_image_image_score", lambda *args, **kwargs: 0.81)

    result = generate_image(
        GenerateOptions(
            prompt="abstract poster",
            output_dir=tmp_path / "out",
            initial_image=initial,
            width=80,
            height=50,
            max_iterations=1,
            seed=5,
            threshold=0.1,
            similarity_backend="transformers-siglip",
            similarity_model="google/siglip-base-patch16-224",
            similarity_device="cuda",
        )
    )

    initial_details = result.metadata["initial_similarity_details"]
    assert initial_details["siglip_image_cosine_score"] == 0.81
    assert initial_details["continuity_score"] == result.metadata["initial_similarity_score"]
    assert result.metadata["similarity_backend"] == "transformers-siglip"


def test_initial_similarity_can_use_dinov2_continuity_backend_independent_of_text_similarity(
    tmp_path: Path, monkeypatch
):
    initial = tmp_path / "initial.png"
    Image.new("RGB", (32, 32), (200, 40, 120)).save(initial)

    monkeypatch.setattr(score_module, "_dinov2_image_image_score", lambda *args, **kwargs: 0.87)

    result = generate_image(
        GenerateOptions(
            prompt="abstract poster",
            output_dir=tmp_path / "out",
            initial_image=initial,
            width=80,
            height=50,
            max_iterations=1,
            seed=5,
            threshold=0.1,
            similarity_backend="local",
            similarity_device="cpu",
            continuity_backend="transformers-dinov2",
            continuity_model="facebook/dinov2-base",
            continuity_device="cuda",
        )
    )

    initial_details = result.metadata["initial_similarity_details"]
    assert initial_details["dinov2_image_cosine_score"] == 0.87
    assert initial_details["continuity_score"] == result.metadata["initial_similarity_score"]
    assert result.metadata["similarity_backend"] == "local"
    assert result.metadata["continuity_backend"] == "transformers-dinov2"
    assert result.metadata["continuity_model"] == "facebook/dinov2-base"
    assert result.metadata["continuity_device"] == "cuda"
    assert result.metadata["effective_continuity_device"] == "cuda"


def test_scene_plan_generation_auto_refines_missing_prompt_objects(tmp_path: Path):
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "missing clouds on purpose",
                "palette": ["#102040", "#ff5533", "#286fc4", "#123d2a"],
                "background": {"top": "#102040", "bottom": "#205080"},
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.18, "color": "#ff5533"},
                    {"type": "mountain", "y": 0.55, "size": 0.28, "color": "#445570"},
                    {"type": "ocean", "y": 0.58, "color": "#286fc4"},
                    {"type": "foreground", "y": 0.80, "color": "#123d2a"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean with misty mountains and clouds",
            output_dir=tmp_path / "out",
            scene_plan=plan_path,
            width=240,
            height=150,
            max_iterations=2,
            threshold=0.99,
            auto_refine=True,
        )
    )

    assert result.metadata["refinement_rounds"] >= 1
    assert any("cloud" in action for action in result.metadata["refinement_actions"])
    assert "cloud" in result.metadata["scene_plan_objects"]
    assert result.metadata["scene_plan_cloud_count"] >= 1


def test_scene_plan_refinement_treats_diagram_primitives_as_prompt_evidence(tmp_path: Path):
    plan_path = tmp_path / "diagram-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "reference architecture diagram",
                "palette": ["#07122b", "#1f4fc4", "#f5b642", "#e6edf7"],
                "background": {"top": "#07122b", "bottom": "#102860"},
                "objects": [],
                "elements": [
                    {
                        "type": "rounded_rectangle",
                        "label": "cpu service tile",
                        "x": 0.24,
                        "y": 0.36,
                        "width": 0.25,
                        "height": 0.16,
                        "fill": "#123080",
                        "stroke": "#36d7ff",
                        "stroke_width": 0.012,
                    },
                    {
                        "type": "text",
                        "text": "CPU",
                        "x": 0.24,
                        "y": 0.36,
                        "size": 0.08,
                        "color": "#e6edf7",
                    },
                    {
                        "type": "rounded_rectangle",
                        "label": "gpu service tile",
                        "x": 0.66,
                        "y": 0.36,
                        "width": 0.25,
                        "height": 0.16,
                        "fill": "#123080",
                        "stroke": "#f5b642",
                        "stroke_width": 0.012,
                    },
                    {
                        "type": "text",
                        "text": "GPU",
                        "x": 0.66,
                        "y": 0.36,
                        "size": 0.08,
                        "color": "#e6edf7",
                    },
                    {
                        "type": "arrow",
                        "label": "compute route",
                        "points": [[0.38, 0.36], [0.52, 0.36]],
                        "stroke": "#f5b642",
                        "width": 0.018,
                    },
                ],
                "style": {"antialias": 1.0, "detail": 0.5, "sharpen": 0.45},
            }
        ),
        encoding="utf-8",
    )

    result = generate_image(
        GenerateOptions(
            prompt=(
                "premium reference architecture diagram with rounded-rectangle service tiles, "
                "gold and blue arrows, CPU and GPU badges, and a final image tile"
            ),
            output_dir=tmp_path / "out",
            scene_plan=plan_path,
            width=240,
            height=150,
            max_iterations=2,
            threshold=0.99,
            auto_refine=True,
            caption_backend="none",
        )
    )

    actions = " ".join(result.metadata["refinement_actions"])

    assert "floor" not in result.metadata["objects"]
    assert "floor" not in actions
    assert "abstract" not in actions
    assert "added missing object 'diagram'" not in actions


def test_scene_plan_refinement_adds_nonsemantic_detail_for_flat_diagrams(tmp_path: Path):
    plan_path = tmp_path / "flat-diagram-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "flat architecture diagram",
                "palette": ["#07122b", "#1f4fc4", "#f5b642", "#e6edf7"],
                "background": {"top": "#07122b", "bottom": "#102860"},
                "objects": [],
                "elements": [
                    {
                        "type": "rounded_rectangle",
                        "label": "source service tile",
                        "x": 0.28,
                        "y": 0.42,
                        "width": 0.30,
                        "height": 0.18,
                        "fill": "#123080",
                        "stroke": "#36d7ff",
                        "stroke_width": 0.01,
                    },
                    {
                        "type": "rounded_rectangle",
                        "label": "target service tile",
                        "x": 0.70,
                        "y": 0.42,
                        "width": 0.30,
                        "height": 0.18,
                        "fill": "#123080",
                        "stroke": "#f5b642",
                        "stroke_width": 0.01,
                    },
                    {
                        "type": "arrow",
                        "label": "diagram route",
                        "points": [[0.43, 0.42], [0.55, 0.42]],
                        "stroke": "#f5b642",
                        "width": 0.015,
                    },
                ],
                "style": {"antialias": 1.0, "detail": 0.05, "sharpen": 0.05},
            }
        ),
        encoding="utf-8",
    )

    result = generate_image(
        GenerateOptions(
            prompt="flat architecture diagram with rounded service tiles and gold arrows",
            output_dir=tmp_path / "out",
            scene_plan=plan_path,
            width=240,
            height=150,
            max_iterations=2,
            threshold=0.99,
            auto_refine=True,
            caption_backend="none",
        )
    )

    actions = " ".join(result.metadata["refinement_actions"])

    assert "added diagram detail texture" in actions
    assert "floor" not in actions
    assert "abstract" not in actions


def test_generate_metadata_records_similarity_backend(tmp_path: Path):
    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean",
            output_dir=tmp_path / "out",
            width=80,
            height=50,
            max_iterations=2,
            threshold=0.1,
            similarity_backend="local",
            similarity_device="cpu",
        )
    )

    assert result.metadata["similarity_backend"] == "local"
    assert result.metadata["similarity_device"] == "cpu"
    assert result.metadata["effective_similarity_device"] == "cpu"
    assert result.metadata["similarity_model"] is None
    assert "cosine_score" in result.metadata["score_details"]


def test_generate_metadata_records_caption_backcheck(tmp_path: Path):
    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean",
            output_dir=tmp_path / "out",
            width=80,
            height=50,
            max_iterations=2,
            threshold=0.1,
            caption_backend="local",
            caption_device="cpu",
        )
    )

    assert result.metadata["caption_backend"] == "local"
    assert result.metadata["caption_model"] is None
    assert result.metadata["caption_device"] == "cpu"
    assert result.metadata["effective_caption_device"] == "cpu"
    assert "sun" in result.metadata["image_caption"]
    assert "ocean" in result.metadata["image_caption"]
    assert 0.0 <= result.metadata["caption_similarity_score"] <= 1.0
    assert result.metadata["caption_similarity_score"] > 0.15


def test_generate_metadata_records_semantic_caption_similarity(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(caption_module, "_sentence_text_similarity_score", lambda *args, **kwargs: 0.73)

    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean",
            output_dir=tmp_path / "out",
            width=80,
            height=50,
            max_iterations=2,
            threshold=0.1,
            caption_backend="local",
            caption_device="cpu",
            caption_similarity_backend="transformers-sentence",
            caption_similarity_model="sentence-transformers/all-MiniLM-L6-v2",
            caption_similarity_device="cuda",
        )
    )

    assert result.metadata["caption_similarity_score"] == 0.73
    assert result.metadata["caption_similarity_backend"] == "transformers-sentence"
    assert result.metadata["caption_similarity_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.metadata["caption_similarity_device"] == "cuda"
    assert result.metadata["effective_caption_similarity_device"] == "cuda"
    assert result.metadata["semantic_caption_similarity_score"] == 0.73
    assert 0.0 <= result.metadata["lexical_caption_similarity_score"] <= 1.0


def test_generate_metadata_records_style_and_mood_words(tmp_path: Path):
    result = generate_image(
        GenerateOptions(
            prompt="cinematic dramatic red sun over blue ocean",
            output_dir=tmp_path / "out",
            width=64,
            height=40,
            max_iterations=1,
            threshold=0.1,
        )
    )

    assert result.metadata["style_words"] == ["cinematic"]
    assert result.metadata["mood_words"] == ["dramatic"]


def test_generate_revision_hints_include_caption_missing_evidence(tmp_path: Path, monkeypatch):
    def bad_caption(*args, **kwargs):
        return CaptionResult(
            caption="a blue bowl with white flowers on it",
            prompt_similarity_score=0.150769,
            backend="transformers-blip",
            model_name="Salesforce/blip-image-captioning-base",
            requested_device="auto",
            effective_device="cuda",
            tokens=("a", "blue", "bowl", "with", "white", "flowers", "on", "it"),
        )

    monkeypatch.setattr(generator_module, "caption_image", bad_caption)

    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean with clouds",
            output_dir=tmp_path / "out",
            width=80,
            height=50,
            max_iterations=2,
            threshold=0.99,
            caption_backend="transformers-blip",
        )
    )

    assert result.metadata["caption_missing_objects"] == ["cloud", "ocean", "sun"]
    assert result.metadata["caption_missing_colors"] == ["red"]
    assert result.metadata["caption_unexpected_objects"] == ["flower"]
    assert any("caption missed requested objects" in hint for hint in result.metadata["revision_hints"])


def test_generate_writes_quality_report_for_claude_iteration(tmp_path: Path):
    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean with clouds",
            output_dir=tmp_path / "out",
            width=96,
            height=64,
            max_iterations=2,
            threshold=0.99,
            save_candidates=2,
        )
    )

    quality_path = tmp_path / "out" / "quality-report.json"
    assert quality_path.exists()
    assert result.metadata["quality_report"] == str(quality_path)
    assert result.metadata["quality_status"] in {"pass", "review", "revise"}
    assert 0.0 <= result.metadata["quality_score"] <= 1.0

    report = json.loads(quality_path.read_text(encoding="utf-8"))
    assert report["status"] == result.metadata["quality_status"]
    assert report["quality_score"] == result.metadata["quality_score"]
    assert report["summary"]
    assert {check["name"] for check in report["checks"]} >= {
        "prompt_alignment",
        "caption_alignment",
        "candidate_recommendation",
    }
    assert report["next_actions"]


def test_generate_can_save_ranked_candidate_artifacts(tmp_path: Path):
    result = generate_image(
        GenerateOptions(
            prompt="red sun over blue ocean with clouds",
            output_dir=tmp_path / "out",
            width=96,
            height=64,
            max_iterations=4,
            threshold=0.99,
            save_candidates=2,
        )
    )

    candidates_path = tmp_path / "out" / "candidates.json"
    candidates_dir = tmp_path / "out" / "candidates"
    assert candidates_path.exists()
    assert candidates_dir.exists()
    assert result.metadata["candidate_count"] == 2
    assert result.metadata["candidate_index"] == str(candidates_path)
    assert result.metadata["candidate_contact_sheet"] == str(candidates_dir / "contact-sheet.png")
    assert len(result.metadata["candidate_images"]) == 2
    assert Path(result.metadata["candidate_contact_sheet"]).exists()

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert [candidate["rank"] for candidate in candidates] == [1, 2]
    assert candidates[0]["total_score"] >= candidates[1]["total_score"]
    assert result.metadata["recommended_candidate_rank"] in {1, 2}
    assert result.metadata["recommended_candidate_image"] in result.metadata["candidate_images"]
    assert 0.0 <= result.metadata["recommended_candidate_score"] <= 1.0
    assert 0.0 <= result.metadata["recommended_candidate_aesthetic_score"] <= 1.0
    for candidate in candidates:
        assert "caption" in candidate
        assert "caption_similarity_score" in candidate
        assert "caption_missing_objects" in candidate
        assert "caption_missing_colors" in candidate
        assert 0.0 <= candidate["aesthetic_score"] <= 1.0
        assert candidate["aesthetic_details"]["contrast_score"] >= 0.0
        assert 0.0 <= candidate["selection_score"] <= 1.0
        assert candidate["selection_reasons"]
        assert any("aesthetic_score" in reason for reason in candidate["selection_reasons"])
        image_path = Path(candidate["image"])
        assert image_path.exists()
        with Image.open(image_path) as image:
            assert image.size == (96, 64)
    with Image.open(result.metadata["candidate_contact_sheet"]) as contact_sheet:
        assert contact_sheet.width >= 96 * 2
        assert contact_sheet.height > 64


def test_contact_sheet_label_truncation_fits_tile_width():
    image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(image)
    text = "a very long generated caption that would otherwise overlap neighboring candidate tiles"

    truncated = _truncate_text_to_width(draw, text, max_width=90)

    assert len(truncated) < len(text)
    assert truncated.endswith("...")
    text_width = draw.textbbox((0, 0), truncated)[2]
    assert text_width <= 90
