from claude_imagegen.prompt import parse_prompt


def test_parse_prompt_extracts_colors_objects_and_style():
    spec = parse_prompt("A cinematic red sun over a blue ocean with misty mountains")

    assert "red" in spec.color_words
    assert "blue" in spec.color_words
    assert "sun" in spec.objects
    assert "ocean" in spec.objects
    assert "mountain" in spec.objects
    assert "cinematic" in spec.style_words


def test_parse_prompt_provides_default_scene_when_prompt_is_sparse():
    spec = parse_prompt("quiet dream")

    assert spec.normalized == "quiet dream"
    assert spec.objects == ("abstract",)
    assert spec.color_words
