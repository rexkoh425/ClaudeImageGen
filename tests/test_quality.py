from claude_imagegen.quality import build_quality_report


def test_quality_report_surfaces_failed_visual_element_checks_as_next_actions():
    report = build_quality_report(
        {
            "total_score": 0.86,
            "threshold": 0.2,
            "caption_similarity_score": 0.88,
            "width": 96,
            "height": 64,
            "visual_critique": {
                "closeness_score": 0.62,
                "element_checks": [
                    {
                        "kind": "object",
                        "item": "cloud",
                        "present": False,
                        "confidence": 0.2,
                        "notes": "not visible",
                    },
                    {
                        "kind": "color",
                        "item": "red",
                        "present": True,
                        "confidence": 0.35,
                        "notes": "only weakly visible",
                    },
                    {
                        "kind": "object",
                        "item": "ocean",
                        "present": True,
                        "confidence": 0.9,
                    },
                    {
                        "kind": "style",
                        "item": "cinematic",
                        "present": True,
                        "confidence": 0.4,
                    },
                    {
                        "kind": "mood",
                        "item": "dramatic",
                        "present": False,
                        "confidence": 0.2,
                    },
                ],
            },
        }
    )

    visual_check = next(check for check in report["checks"] if check["name"] == "visual_judgement")
    assert report["status"] == "revise"
    assert visual_check["status"] == "revise"
    assert "Judge: make missing checked objects explicit: cloud." in report["next_actions"]
    assert "Judge: strengthen low-confidence checked colors: red." in report["next_actions"]
    assert "Judge: strengthen checked styles: cinematic." in report["next_actions"]
    assert "Judge: make checked moods more visually explicit: dramatic." in report["next_actions"]
    assert visual_check["element_checks"][0] == {
        "kind": "object",
        "item": "cloud",
        "present": False,
        "confidence": 0.2,
        "notes": "not visible",
    }
    assert visual_check["failed_element_checks"] == [
        {
            "kind": "object",
            "item": "cloud",
            "present": False,
            "confidence": 0.2,
            "notes": "not visible",
        },
        {
            "kind": "color",
            "item": "red",
            "present": True,
            "confidence": 0.35,
            "notes": "only weakly visible",
        },
        {
            "kind": "style",
            "item": "cinematic",
            "present": True,
            "confidence": 0.4,
        },
        {
            "kind": "mood",
            "item": "dramatic",
            "present": False,
            "confidence": 0.2,
        },
    ]


def test_quality_report_builds_refinement_guidance_for_parent_regressions():
    report = build_quality_report(
        {
            "total_score": 0.58,
            "threshold": 0.4,
            "caption_similarity_score": 0.51,
            "width": 96,
            "height": 64,
            "parent_total_score": 0.70,
            "parent_quality_score": 0.78,
            "parent_caption_similarity_score": 0.72,
            "initial_similarity_score": 0.62,
            "initial_similarity_details": {
                "weakest_continuity_region": "upper_left",
                "weakest_continuity_region_score": 0.41,
            },
        }
    )

    guidance = report["refinement_guidance"]
    assert guidance["decision"] == "revise"
    axes = {axis["axis"]: axis for axis in guidance["priority_axes"]}

    assert axes["prompt_alignment"]["delta"] == -0.12
    assert axes["prompt_alignment"]["severity"] == "revise"
    assert axes["quality"]["delta"] < -0.03
    assert axes["caption_alignment"]["delta"] == -0.21
    assert axes["continuity"]["score"] == 0.62
    assert axes["continuity"]["weakest_region"] == "upper_left"
    assert axes["continuity"]["weakest_region_score"] == 0.41

    assert "Refinement: restore prompt alignment that dropped versus the parent." in report["next_actions"]
    assert "Refinement: inspect failed quality checks before continuing; overall quality dropped versus the parent." in report["next_actions"]
    assert "Refinement: restore caption evidence for requested objects, colors, and relationships." in report["next_actions"]
    assert "Refinement: preserve parent layout near the upper left region; it has the weakest continuity score (0.410)." in report["next_actions"]
