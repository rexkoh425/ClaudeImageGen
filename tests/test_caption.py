from claude_imagegen.caption import caption_image, caption_prompt_diagnostics
from claude_imagegen.prompt import parse_prompt
from claude_imagegen.render import render_candidate
from claude_imagegen.scene import build_initial_candidate


def test_local_caption_backcheck_names_visible_prompt_elements():
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=3)
    image = render_candidate(candidate, width=160, height=100)

    result = caption_image(
        image,
        prompt=spec.original,
        backend="local",
        device="cpu",
    )

    assert result.backend == "local"
    assert result.model_name is None
    assert result.requested_device == "cpu"
    assert result.effective_device == "cpu"
    assert "sun" in result.caption
    assert "ocean" in result.caption
    assert 0.0 <= result.prompt_similarity_score <= 1.0
    assert result.prompt_similarity_score > 0.15


def test_caption_prompt_diagnostics_reports_missing_requested_evidence():
    diagnostics = caption_prompt_diagnostics(
        "red sun over blue ocean with clouds",
        "a blue bowl with white flowers on it",
    )

    assert diagnostics.missing_objects == ("cloud", "ocean", "sun")
    assert diagnostics.missing_colors == ("red",)
    assert diagnostics.unexpected_objects == ("flower",)


def test_local_caption_backcheck_names_simple_robot_portrait():
    spec = parse_prompt("red robot portrait over blue ocean with clouds")
    candidate = build_initial_candidate(spec, seed=0)
    image = render_candidate(candidate, width=160, height=100)

    result = caption_image(
        image,
        prompt=spec.original,
        backend="local",
        device="cpu",
    )

    assert "robot" in result.caption
    assert "portrait" in result.caption
