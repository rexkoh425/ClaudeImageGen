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


def test_parse_prompt_recognizes_greenhouse_scene_objects():
    spec = parse_prompt(
        "cinematic glass greenhouse at night with layered tropical plants, mist, "
        "reflective wet stone floor, and warm hanging lights"
    )

    assert "greenhouse" in spec.objects
    assert "plant" in spec.objects
    assert "floor" in spec.objects
    assert "lamp" in spec.objects
    assert "moon" not in spec.objects
    assert "cloud" not in spec.objects


def test_parse_prompt_does_not_treat_diagram_image_tile_as_floor():
    spec = parse_prompt(
        "clean architecture diagram for a local pipeline with rounded boxes, "
        "readable labels, arrows, CPU and GPU badges, and a final image tile"
    )

    assert "abstract" in spec.objects
    assert "floor" not in spec.objects
