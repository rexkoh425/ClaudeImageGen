from __future__ import annotations

import json

import pytest

from claude_imagegen.critique import (
    apply_critique_to_plan_dict,
    apply_comparison_to_plan_dict,
    build_comparison_request,
    comparison_signal,
    critique_signal,
    known_edit_actions,
    parse_comparison,
    parse_critique,
    write_comparison_request,
    write_critique_request,
)


def test_parse_clamps_score_and_infers_verdict() -> None:
    high = parse_critique({"closeness_score": 1.4})
    assert high.closeness_score == 1.0
    assert high.verdict == "accept"

    low = parse_critique({"closeness_score": -0.3})
    assert low.closeness_score == 0.0
    assert low.verdict == "revise"


def test_parse_honors_explicit_verdict_and_lists() -> None:
    critique = parse_critique(
        {
            "closeness_score": 0.5,
            "verdict": "ACCEPT",
            "summary": "looks close",
            "present": ["sun", "ocean"],
            "missing": ["clouds"],
            "wrong": ["mountains too bright"],
            "extra": ["red blob"],
            "element_checks": [
                {"kind": "object", "item": "sun", "present": True, "confidence": 0.8, "notes": "visible"}
            ],
            "notes": "n",
        }
    )
    assert critique.verdict == "accept"
    assert critique.present == ("sun", "ocean")
    assert critique.missing == ("clouds",)
    assert critique.wrong == ("mountains too bright",)
    assert critique.extra == ("red blob",)
    assert critique.element_checks == (
        {"kind": "object", "item": "sun", "present": True, "confidence": 0.8, "notes": "visible"},
    )


def test_parse_from_json_string() -> None:
    critique = parse_critique(json.dumps({"closeness_score": 0.6, "verdict": "revise"}))
    assert critique.closeness_score == 0.6
    assert critique.verdict == "revise"


def test_parse_and_apply_comparison_follow_up_edits() -> None:
    comparison = parse_comparison(
        {
            "alignment_score": 0.52,
            "continuity_score": 0.44,
            "improved": False,
            "preserved_identity": False,
            "better_image": "parent",
            "verdict": "revise",
            "summary": "Child lost the original sun scale.",
            "regressions": ["sun became too small", "palette drifted"],
            "follow_up_edits": [
                {"action": "resize_object", "type": "sun", "size": 0.24},
                {"action": "adjust_style", "field": "contrast", "delta": 0.1},
            ],
            "notes": "Use parent as continuity anchor.",
        }
    )

    assert comparison.verdict == "revise"
    assert comparison.better_image == "parent"
    assert comparison.regressions == ("sun became too small", "palette drifted")

    revised, actions = apply_comparison_to_plan_dict(
        {"objects": [{"type": "sun", "size": 0.1}], "style": {"contrast": 0.2}},
        comparison,
    )

    assert revised["objects"][0]["size"] == 0.24
    assert revised["style"]["contrast"] == 0.3
    assert any("resized 1 'sun'" in action for action in actions)

    signal = comparison_signal(comparison, applied_edits=actions)
    assert signal["judge"] == "claude-vision-refinement-comparison"
    assert signal["regressions"] == ["sun became too small", "palette drifted"]
    assert signal["applied_edits"] == actions


def test_apply_add_and_remove_object() -> None:
    plan = {"objects": [{"type": "robot", "x": 0.5, "y": 0.5}]}
    critique = parse_critique(
        {
            "edits": [
                {"action": "add_object", "type": "cloud", "x": 0.3, "y": 0.2, "color": "#ffffff"},
                {"action": "remove_object", "type": "robot"},
            ]
        }
    )
    revised, actions = apply_critique_to_plan_dict(plan, critique)
    kinds = {obj["type"] for obj in revised["objects"]}
    assert "cloud" in kinds
    assert "robot" not in kinds
    assert any("added object 'cloud'" in a for a in actions)
    assert any("removed 1 'robot'" in a for a in actions)
    # original plan is not mutated (deep copy)
    assert plan["objects"][0]["type"] == "robot"


def test_apply_recolor_move_resize_and_style() -> None:
    plan = {"objects": [{"type": "ocean", "x": 0.5, "y": 0.6, "size": 0.3, "color": "#000000"}], "style": {"contrast": 0.2}}
    critique = parse_critique(
        {
            "edits": [
                {"action": "recolor_object", "type": "ocean", "color": "#1d5fa8"},
                {"action": "move_object", "type": "ocean", "x": 0.4, "y": 0.7},
                {"action": "resize_object", "type": "ocean", "size": 0.5},
                {"action": "adjust_style", "field": "contrast", "delta": 0.15},
                {"action": "set_style", "field": "saturation", "value": 0.6},
            ]
        }
    )
    revised, _ = apply_critique_to_plan_dict(plan, critique)
    ocean = revised["objects"][0]
    assert ocean["color"] == "#1d5fa8"
    assert ocean["x"] == 0.4 and ocean["y"] == 0.7
    assert ocean["size"] == 0.5
    assert abs(revised["style"]["contrast"] - 0.35) < 1e-9
    assert abs(revised["style"]["saturation"] - 0.6) < 1e-9


def test_apply_set_palette_and_add_cloud() -> None:
    critique = parse_critique(
        {
            "edits": [
                {"action": "set_palette", "colors": ["#102040", "#ff5533"]},
                {"action": "add_cloud", "color": "#fff1dd"},
            ]
        }
    )
    revised, actions = apply_critique_to_plan_dict({}, critique)
    assert revised["palette"] == ["#102040", "#ff5533"]
    assert len(revised["clouds"]) == 1
    assert any("palette" in a for a in actions)


def test_apply_missing_object_element_check_adds_visible_cloud() -> None:
    critique = parse_critique(
        {
            "element_checks": [
                {
                    "kind": "object",
                    "item": "cloud",
                    "present": False,
                    "confidence": 0.2,
                    "notes": "not visible enough",
                }
            ]
        }
    )

    revised, actions = apply_critique_to_plan_dict({"objects": []}, critique)

    assert len(revised["clouds"]) == 1
    assert any("element_check: added default cloud bank" in action for action in actions)


def test_apply_low_confidence_color_element_check_strengthens_style() -> None:
    critique = parse_critique(
        {
            "element_checks": [
                {
                    "kind": "color",
                    "item": "red",
                    "present": True,
                    "confidence": 0.35,
                    "notes": "sun reads closer to pink",
                }
            ]
        }
    )

    revised, actions = apply_critique_to_plan_dict({"style": {"saturation": 0.2, "contrast": 0.3}}, critique)

    assert revised["style"]["saturation"] > 0.2
    assert revised["style"]["contrast"] > 0.3
    assert "#dc403a" in revised["palette"]
    assert any("element_check: strengthened checked color 'red'" in action for action in actions)


def test_apply_low_confidence_style_and_mood_checks_adjust_style() -> None:
    critique = parse_critique(
        {
            "element_checks": [
                {"kind": "style", "item": "cinematic", "present": True, "confidence": 0.4},
                {"kind": "mood", "item": "dramatic", "present": False, "confidence": 0.2},
            ]
        }
    )

    revised, actions = apply_critique_to_plan_dict(
        {"style": {"contrast": 0.2, "bloom": 0.1, "vignette": 0.0}},
        critique,
    )

    assert revised["style"]["contrast"] > 0.2
    assert revised["style"]["bloom"] > 0.1
    assert revised["style"]["vignette"] > 0.0
    assert any("element_check: strengthened checked style 'cinematic'" in action for action in actions)
    assert any("element_check: strengthened checked mood 'dramatic'" in action for action in actions)


def test_unknown_action_is_skipped_not_fatal() -> None:
    critique = parse_critique({"edits": [{"action": "teleport_object", "type": "sun"}]})
    revised, actions = apply_critique_to_plan_dict({"objects": []}, critique)
    assert revised["objects"] == []
    assert any("skipped unknown edit action 'teleport_object'" in a for a in actions)


def test_critique_signal_shape() -> None:
    critique = parse_critique(
        {
            "closeness_score": 0.7,
            "missing": ["clouds"],
            "element_checks": [{"kind": "object", "item": "cloud", "present": False}],
        }
    )
    signal = critique_signal(critique, applied_edits=["added object 'cloud'"])
    assert signal["judge"] == "claude-vision"
    assert signal["closeness_score"] == 0.7
    assert signal["missing"] == ["clouds"]
    assert signal["element_checks"] == [{"kind": "object", "item": "cloud", "present": False}]
    assert signal["applied_edits"] == ["added object 'cloud'"]


def test_write_critique_request_records_expected_judge_payload(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    metadata_path = tmp_path / "metadata.json"
    image_path.write_bytes(b"png")
    metadata = {
        "prompt": "red sun over blue ocean with clouds",
        "normalized_prompt": "red sun over blue ocean with clouds",
        "width": 320,
        "height": 192,
        "total_score": 0.58,
        "quality_report": str(tmp_path / "quality-report.json"),
        "quality_status": "revise",
        "quality_score": 0.44,
        "objects": ["sun", "ocean", "cloud"],
        "color_words": ["red", "blue"],
        "style_words": ["cinematic"],
        "mood_words": ["dramatic"],
        "caption_missing_objects": ["cloud"],
        "caption_missing_colors": ["red"],
        "revision_hints": ["Add missing clouds."],
    }

    request_path = write_critique_request(
        tmp_path,
        image_path=image_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request_path == tmp_path / "critique-request.json"
    assert metadata["critique_request"] == str(request_path)
    assert request["judge"] == "claude-vision"
    assert request["image"] == str(image_path)
    assert request["metadata"] == str(metadata_path)
    assert request["quality_report"] == str(tmp_path / "quality-report.json")
    assert request["prompt"] == "red sun over blue ocean with clouds"
    assert request["quality_status"] == "revise"
    assert request["quality_score"] == 0.44
    assert request["revision_hints"] == ["Add missing clouds."]
    assert request["visual_checklist"] == [
        {
            "kind": "object",
            "item": "cloud",
            "question": "Does the image clearly show the requested object: cloud?",
            "caption_backcheck": "missing",
            "priority": "high",
        },
        {
            "kind": "object",
            "item": "ocean",
            "question": "Does the image clearly show the requested object: ocean?",
            "caption_backcheck": "unknown",
            "priority": "normal",
        },
        {
            "kind": "object",
            "item": "sun",
            "question": "Does the image clearly show the requested object: sun?",
            "caption_backcheck": "unknown",
            "priority": "normal",
        },
        {
            "kind": "color",
            "item": "blue",
            "question": "Is the requested color visually present and attached to the right subject: blue?",
            "caption_backcheck": "unknown",
            "priority": "normal",
        },
        {
            "kind": "color",
            "item": "red",
            "question": "Is the requested color visually present and attached to the right subject: red?",
            "caption_backcheck": "missing",
            "priority": "high",
        },
        {
            "kind": "style",
            "item": "cinematic",
            "question": "Does the image clearly convey the requested visual style: cinematic?",
            "caption_backcheck": "unknown",
            "priority": "normal",
        },
        {
            "kind": "mood",
            "item": "dramatic",
            "question": "Does the image clearly convey the requested mood: dramatic?",
            "caption_backcheck": "unknown",
            "priority": "normal",
        },
    ]
    assert request["expected_response"]["closeness_score"] is None
    assert request["expected_response"]["verdict"] == "revise"
    assert request["expected_response"]["element_checks"][0] == {
        "kind": "object",
        "item": "cloud",
        "present": None,
        "confidence": None,
        "notes": "",
    }
    assert request["expected_response"]["present"] == []
    assert request["expected_response"]["edits"] == []
    assert "add_cloud" in request["allowed_edit_actions"]
    assert "resize_object" in request["allowed_edit_actions"]
    assert known_edit_actions() == sorted(request["allowed_edit_actions"])


def test_write_comparison_request_records_parent_child_judge_payload(tmp_path) -> None:
    parent_image = tmp_path / "parent.png"
    child_image = tmp_path / "image.png"
    metadata_path = tmp_path / "metadata.json"
    parent_metadata_path = tmp_path / "parent-metadata.json"
    parent_image.write_bytes(b"parent")
    child_image.write_bytes(b"child")
    metadata = {
        "prompt": "red sun over blue ocean with brighter clouds",
        "parent_prompt": "red sun over blue ocean",
        "quality_report": str(tmp_path / "quality-report.json"),
        "critique_request": str(tmp_path / "critique-request.json"),
        "total_score": 0.62,
        "quality_score": 0.68,
        "initial_similarity_score": 0.91,
        "initial_similarity_details": {
            "continuity_score": 0.91,
            "weakest_continuity_region": "top_left",
            "weakest_continuity_region_score": 0.72,
            "region_similarity_scores": {"top_left": 0.72, "middle_center": 0.94},
        },
        "refinement_delta": {
            "parent_total_score": 0.58,
            "current_total_score": 0.62,
            "total_score_delta": 0.04,
            "continuity_score": 0.91,
        },
    }

    request = build_comparison_request(
        parent_image=parent_image,
        child_image=child_image,
        metadata_path=metadata_path,
        parent_metadata_path=parent_metadata_path,
        metadata=metadata,
    )

    assert request["judge"] == "claude-vision-refinement-comparison"
    assert request["parent_image"] == str(parent_image)
    assert request["child_image"] == str(child_image)
    assert request["metadata"] == str(metadata_path)
    assert request["parent_metadata"] == str(parent_metadata_path)
    assert request["quality_report"] == str(tmp_path / "quality-report.json")
    assert request["critique_request"] == str(tmp_path / "critique-request.json")
    assert request["prompt"] == "red sun over blue ocean with brighter clouds"
    assert request["parent_prompt"] == "red sun over blue ocean"
    assert request["refinement_delta"]["total_score_delta"] == 0.04
    assert request["initial_similarity_score"] == 0.91
    assert request["initial_similarity_details"]["weakest_continuity_region"] == "top_left"
    assert request["initial_similarity_details"]["region_similarity_scores"]["middle_center"] == 0.94
    assert request["expected_response"]["alignment_score"] is None
    assert request["expected_response"]["continuity_score"] is None
    assert request["expected_response"]["better_image"] == "child"
    assert "follow_up_edits" in request["expected_response"]
    assert "add_cloud" in request["allowed_edit_actions"]

    request_path = write_comparison_request(
        tmp_path,
        parent_image=parent_image,
        child_image=child_image,
        metadata_path=metadata_path,
        parent_metadata_path=parent_metadata_path,
        metadata=metadata,
    )
    assert request_path == tmp_path / "comparison-request.json"
    assert metadata["comparison_request"] == str(request_path)
    persisted = json.loads(request_path.read_text(encoding="utf-8"))
    assert persisted == build_comparison_request(
        parent_image=parent_image,
        child_image=child_image,
        metadata_path=metadata_path,
        parent_metadata_path=parent_metadata_path,
        metadata=metadata,
    )


def test_missing_critique_file_raises() -> None:
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        parse_critique(Path("does-not-exist-critique.json"))
