from __future__ import annotations

import json

import pytest

from claude_imagegen.critique import (
    apply_critique_to_plan_dict,
    critique_signal,
    parse_critique,
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
            "notes": "n",
        }
    )
    assert critique.verdict == "accept"
    assert critique.present == ("sun", "ocean")
    assert critique.missing == ("clouds",)
    assert critique.wrong == ("mountains too bright",)
    assert critique.extra == ("red blob",)


def test_parse_from_json_string() -> None:
    critique = parse_critique(json.dumps({"closeness_score": 0.6, "verdict": "revise"}))
    assert critique.closeness_score == 0.6
    assert critique.verdict == "revise"


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


def test_unknown_action_is_skipped_not_fatal() -> None:
    critique = parse_critique({"edits": [{"action": "teleport_object", "type": "sun"}]})
    revised, actions = apply_critique_to_plan_dict({"objects": []}, critique)
    assert revised["objects"] == []
    assert any("skipped unknown edit action 'teleport_object'" in a for a in actions)


def test_critique_signal_shape() -> None:
    critique = parse_critique({"closeness_score": 0.7, "missing": ["clouds"]})
    signal = critique_signal(critique, applied_edits=["added object 'cloud'"])
    assert signal["judge"] == "claude-vision"
    assert signal["closeness_score"] == 0.7
    assert signal["missing"] == ["clouds"]
    assert signal["applied_edits"] == ["added object 'cloud'"]


def test_missing_critique_file_raises() -> None:
    from pathlib import Path

    with pytest.raises(FileNotFoundError):
        parse_critique(Path("does-not-exist-critique.json"))
