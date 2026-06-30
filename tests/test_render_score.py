import json
from pathlib import Path

from PIL import Image

from claude_imagegen.caption import CaptionResult
import claude_imagegen.generator as generator_module
from claude_imagegen.generator import GenerateOptions, generate_image
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
    assert len(result.metadata["candidate_images"]) == 2

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert [candidate["rank"] for candidate in candidates] == [1, 2]
    assert candidates[0]["total_score"] >= candidates[1]["total_score"]
    for candidate in candidates:
        image_path = Path(candidate["image"])
        assert image_path.exists()
        with Image.open(image_path) as image:
            assert image.size == (96, 64)
