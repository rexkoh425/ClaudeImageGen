from pathlib import Path

from PIL import Image

from claude_imagegen.generator import GenerateOptions, generate_image
from claude_imagegen.prompt import parse_prompt
from claude_imagegen.render import cap_dimensions, render_candidate
from claude_imagegen.scene import build_initial_candidate
from claude_imagegen.score import score_image


def test_cap_dimensions_limits_output_to_720_by_480():
    assert cap_dimensions(4096, 2048) == (720, 480)
    assert cap_dimensions(500, 300) == (500, 300)


def test_render_candidate_returns_rgb_image_with_requested_capped_size():
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=4)

    image = render_candidate(candidate, width=900, height=900)

    assert image.mode == "RGB"
    assert image.size == (720, 480)


def test_scoring_rewards_prompt_aligned_render_over_blank_canvas():
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=2)
    image = render_candidate(candidate, width=720, height=480)
    blank = Image.new("RGB", (720, 480), (240, 240, 240))

    rendered_score = score_image(image, spec).text_score
    blank_score = score_image(blank, spec).text_score

    assert rendered_score > blank_score
    assert rendered_score >= 0.45


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
