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
                ],
            },
        }
    )

    visual_check = next(check for check in report["checks"] if check["name"] == "visual_judgement")
    assert report["status"] == "revise"
    assert visual_check["status"] == "revise"
    assert "Judge: make missing checked objects explicit: cloud." in report["next_actions"]
    assert "Judge: strengthen low-confidence checked colors: red." in report["next_actions"]
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
    ]
