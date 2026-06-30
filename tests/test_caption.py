from claude_imagegen.caption import caption_image
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
