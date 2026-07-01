import json
from pathlib import Path

import claude_imagegen.caption as caption_module
from claude_imagegen.caption import caption_image, caption_prompt_diagnostics, caption_prompt_similarity
from claude_imagegen.prompt import parse_prompt
from claude_imagegen.render import render_candidate, render_scene_plan
from claude_imagegen.scene import build_initial_candidate
from claude_imagegen.scene_plan import parse_scene_plan


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


def test_caption_prompt_similarity_can_use_sentence_embedding_backend(monkeypatch):
    monkeypatch.setattr(caption_module, "_sentence_text_similarity_score", lambda *args, **kwargs: 0.84)

    score = caption_prompt_similarity(
        "a glass observatory above a neon harbor at sunset",
        "an illuminated building over water at dusk",
        backend="transformers-sentence",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cuda",
    )

    assert score == 0.84


def test_caption_image_records_semantic_caption_similarity_backend(monkeypatch):
    monkeypatch.setattr(caption_module, "_sentence_text_similarity_score", lambda *args, **kwargs: 0.77)
    spec = parse_prompt("red sun over blue ocean")
    candidate = build_initial_candidate(spec, seed=3)
    image = render_candidate(candidate, width=160, height=100)

    result = caption_image(
        image,
        prompt=spec.original,
        backend="local",
        device="cpu",
        similarity_backend="transformers-sentence",
        similarity_model="sentence-transformers/all-MiniLM-L6-v2",
        similarity_device="cuda",
    )

    assert result.prompt_similarity_score == 0.77
    assert result.lexical_prompt_similarity_score != result.prompt_similarity_score
    assert result.semantic_prompt_similarity_score == 0.77
    assert result.similarity_backend == "transformers-sentence"
    assert result.similarity_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.similarity_device == "cuda"
    assert result.effective_similarity_device == "cuda"


def test_local_caption_backcheck_names_greenhouse_scene_primitives(tmp_path: Path):
    prompt = (
        "cinematic glass greenhouse at night with layered tropical plants, mist, "
        "reflective wet stone floor, and warm hanging lights"
    )
    plan_path = tmp_path / "greenhouse-scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "captionable greenhouse scene",
                "palette": ["#0c2030", "#6f858a", "#1f7a4d", "#f8c86f", "#4d5558"],
                "background": {"top": "#0c2030", "bottom": "#09100d"},
                "objects": [
                    {"type": "greenhouse", "x": 0.5, "y": 0.16, "size": 0.9, "color": "#6f858a"},
                    {"type": "lamp", "x": 0.5, "y": 0.22, "size": 0.08, "color": "#f8c86f", "count": 3},
                    {"type": "plant", "x": 0.5, "y": 0.70, "size": 0.30, "color": "#1f7a4d", "count": 18},
                    {"type": "floor", "y": 0.78, "size": 0.22, "color": "#4d5558"},
                ],
                "style": {"antialias": 1.0, "detail": 0.7, "sharpen": 0.5},
            }
        ),
        encoding="utf-8",
    )
    image = render_scene_plan(parse_scene_plan(plan_path), width=160, height=100, seed=12)

    result = caption_image(image, prompt=prompt, backend="local", device="cpu")
    diagnostics = caption_prompt_diagnostics(prompt, result.caption)

    assert "greenhouse" in result.caption
    assert "plants" in result.caption
    assert "floor" in result.caption
    assert "lamps" in result.caption
    assert "greenhouse" not in diagnostics.missing_objects
    assert "plant" not in diagnostics.missing_objects
    assert "floor" not in diagnostics.missing_objects
    assert "lamp" not in diagnostics.missing_objects
    assert result.prompt_similarity_score > 0.45
