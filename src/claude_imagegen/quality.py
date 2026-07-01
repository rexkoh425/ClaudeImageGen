from __future__ import annotations

import json
from pathlib import Path


def apply_quality_report(output_dir: Path, metadata: dict[str, object]) -> Path:
    report = build_quality_report(metadata)
    report_path = output_dir / "quality-report.json"
    report["report_path"] = str(report_path)
    metadata["quality_report"] = str(report_path)
    metadata["quality_status"] = report["status"]
    metadata["quality_score"] = report["quality_score"]
    if report.get("refinement_delta") is not None:
        metadata["refinement_delta"] = report["refinement_delta"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def build_quality_report(metadata: dict[str, object]) -> dict[str, object]:
    checks = _quality_checks(metadata)
    quality_score = _weighted_quality_score(checks)
    status = _quality_status(checks, quality_score)
    next_actions = _next_actions(metadata, checks, status)
    continuity_score = _float_or_none(metadata.get("initial_similarity_score"))
    continuity_details = metadata.get("initial_similarity_details")
    weakest_region = _weakest_continuity_region(continuity_details)
    weakest_region_score = _weakest_continuity_region_score(continuity_details)
    refinement_delta = _refinement_delta(metadata, quality_score=quality_score)

    return {
        "status": status,
        "quality_score": quality_score,
        "summary": _summary(status, quality_score, checks),
        "checks": checks,
        "next_actions": next_actions,
        "continuity_score": continuity_score if continuity_score is not None else None,
        "weakest_continuity_region": weakest_region,
        "weakest_continuity_region_score": weakest_region_score,
        "refinement_delta": refinement_delta,
        "recommended_candidate_rank": metadata.get("recommended_candidate_rank"),
        "recommended_candidate_score": metadata.get("recommended_candidate_score"),
        "recommended_candidate_aesthetic_score": metadata.get("recommended_candidate_aesthetic_score"),
        "revision_hints": _string_list(metadata.get("revision_hints")),
    }


def _quality_checks(metadata: dict[str, object]) -> list[dict[str, object]]:
    checks = [
        _check(
            name="prompt_alignment",
            score=_float(metadata.get("total_score"), 0.0),
            pass_threshold=_float(metadata.get("threshold"), 0.58),
            review_threshold=max(0.0, _float(metadata.get("threshold"), 0.58) * 0.82),
            weight=0.38,
            detail="Final prompt-image alignment score versus requested threshold.",
        ),
        _check(
            name="caption_alignment",
            score=_float(metadata.get("caption_similarity_score"), 0.0),
            pass_threshold=0.56,
            review_threshold=0.35,
            weight=0.22,
            detail=_caption_alignment_detail(metadata),
        ),
        _check(
            name="size",
            score=_size_score(metadata),
            pass_threshold=1.0,
            review_threshold=0.9,
            weight=0.08,
            detail="Output dimensions are valid and within the renderer cap.",
        ),
    ]

    critique = metadata.get("visual_critique")
    if isinstance(critique, dict) and "closeness_score" in critique:
        checks.append(
            _check(
                name="visual_judgement",
                score=_float(critique.get("closeness_score"), 0.0),
                pass_threshold=0.78,
                review_threshold=0.55,
                weight=0.30,
                detail="Claude-vision judge closeness (LMM-as-evaluator / VQAScore-style) after viewing the image.",
            )
        )

    initial_similarity = _float_or_none(metadata.get("initial_similarity_score"))
    if initial_similarity is not None:
        checks.append(
            _check(
                name="continuity",
                score=initial_similarity,
                pass_threshold=0.78,
                review_threshold=0.58,
                weight=0.20,
                detail="Image-to-image continuity against the selected parent or initial image.",
            )
        )

    if metadata.get("reference_image"):
        checks.append(
            _check(
                name="reference_alignment",
                score=_float(metadata.get("reference_score"), 0.0),
                pass_threshold=0.45,
                review_threshold=0.28,
                weight=0.12,
                detail="Palette/layout similarity to the supplied reference image.",
            )
        )

    if _float(metadata.get("candidate_count"), 0.0) > 0:
        checks.append(
            _check(
                name="candidate_recommendation",
                score=_float(metadata.get("recommended_candidate_score"), 0.0),
                pass_threshold=0.55,
                review_threshold=0.35,
                weight=0.10,
                detail="Best saved candidate score after combining visual score, caption evidence, aesthetic preference, and penalties.",
            )
        )

    caption_missing_objects = _string_list(metadata.get("caption_missing_objects"))
    caption_missing_colors = _string_list(metadata.get("caption_missing_colors"))
    if caption_missing_objects or caption_missing_colors:
        checks.append(
            {
                "name": "caption_gaps",
                "status": "revise",
                "score": 0.0,
                "pass_threshold": 1.0,
                "review_threshold": 1.0,
                "weight": 0.12,
                "detail": "Caption backcheck missed requested prompt evidence.",
                "missing_objects": caption_missing_objects,
                "missing_colors": caption_missing_colors,
            }
        )

    return checks


def _check(
    *,
    name: str,
    score: float,
    pass_threshold: float,
    review_threshold: float,
    weight: float,
    detail: str,
) -> dict[str, object]:
    rounded_score = round(score, 6)
    if score >= pass_threshold:
        status = "pass"
    elif score >= review_threshold:
        status = "review"
    else:
        status = "revise"
    return {
        "name": name,
        "status": status,
        "score": rounded_score,
        "pass_threshold": round(pass_threshold, 6),
        "review_threshold": round(review_threshold, 6),
        "weight": weight,
        "detail": detail,
    }


def _quality_status(checks: list[dict[str, object]], quality_score: float) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "revise" in statuses or quality_score < 0.45:
        return "revise"
    if "review" in statuses or quality_score < 0.72:
        return "review"
    return "pass"


def _weighted_quality_score(checks: list[dict[str, object]]) -> float:
    total_weight = sum(_float(check.get("weight"), 0.0) for check in checks)
    if total_weight <= 0:
        return 0.0
    score = sum(_float(check.get("score"), 0.0) * _float(check.get("weight"), 0.0) for check in checks) / total_weight
    return round(max(0.0, min(1.0, score)), 6)


def _next_actions(metadata: dict[str, object], checks: list[dict[str, object]], status: str) -> list[str]:
    actions = list(dict.fromkeys(_string_list(metadata.get("revision_hints"))))

    missing_objects = _string_list(metadata.get("caption_missing_objects"))
    if missing_objects:
        actions.append(f"Make requested objects visually explicit: {', '.join(missing_objects)}.")

    missing_colors = _string_list(metadata.get("caption_missing_colors"))
    if missing_colors:
        actions.append(f"Strengthen requested colors: {', '.join(missing_colors)}.")

    critique = metadata.get("visual_critique")
    if isinstance(critique, dict):
        critique_missing = _string_list(critique.get("missing"))
        critique_wrong = _string_list(critique.get("wrong"))
        critique_extra = _string_list(critique.get("extra"))
        if critique_missing:
            actions.append(f"Judge: add missing elements: {', '.join(critique_missing)}.")
        if critique_wrong:
            actions.append(f"Judge: fix incorrect elements: {', '.join(critique_wrong)}.")
        if critique_extra:
            actions.append(f"Judge: remove or downplay unrequested elements: {', '.join(critique_extra)}.")

    failed = [str(check.get("name")) for check in checks if check.get("status") == "revise"]
    if "prompt_alignment" in failed:
        actions.append("Revise the scene plan before rerunning; prompt alignment is below the requested threshold.")
    if "continuity" in failed:
        actions.append("Preserve more parent layout, palette, and silhouettes before applying new prompt changes.")
        weakest_region = _weakest_continuity_region(metadata.get("initial_similarity_details"))
        weakest_score = _weakest_continuity_region_score(metadata.get("initial_similarity_details"))
        if weakest_region and weakest_score is not None:
            actions.append(f"Inspect the {weakest_region.replace('_', ' ')} region; it has the weakest parent-child continuity score ({weakest_score:.3f}).")
    if "candidate_recommendation" in failed and metadata.get("candidate_index"):
        actions.append("Inspect candidates/contact-sheet.png before choosing the next refinement parent.")

    total_delta = _delta(metadata.get("total_score"), metadata.get("parent_total_score"))
    caption_delta = _delta(metadata.get("caption_similarity_score"), metadata.get("parent_caption_similarity_score"))
    if total_delta is not None and total_delta < -0.03:
        actions.append("Refinement lowered prompt alignment versus the parent; compare against the parent before continuing.")
    if caption_delta is not None and caption_delta < -0.05:
        actions.append("Refinement lowered caption alignment versus the parent; inspect whether requested visual evidence disappeared.")

    if not actions:
        if status == "pass":
            actions.append("No automatic revision required; inspect image.png for final visual acceptance.")
        else:
            actions.append("Review image.png, metadata.json, and candidates/contact-sheet.png before deciding whether to refine.")
    return actions[:8]


def _summary(status: str, quality_score: float, checks: list[dict[str, object]]) -> str:
    revise_checks = [str(check.get("name")) for check in checks if check.get("status") == "revise"]
    if revise_checks:
        return f"{status}: quality_score={quality_score:.3f}; revise {', '.join(revise_checks)}."
    review_checks = [str(check.get("name")) for check in checks if check.get("status") == "review"]
    if review_checks:
        return f"{status}: quality_score={quality_score:.3f}; review {', '.join(review_checks)}."
    return f"{status}: quality_score={quality_score:.3f}; all automatic checks passed."


def _caption_alignment_detail(metadata: dict[str, object]) -> str:
    backend = str(metadata.get("caption_similarity_backend") or "local")
    if backend in {"sentence", "transformers-sentence"}:
        return "Backchecked caption against the prompt with sentence-embedding semantic similarity plus object/color diagnostics."
    return "Backchecked caption overlap with requested prompt objects, colors, and tokens."


def _refinement_delta(metadata: dict[str, object], *, quality_score: float) -> dict[str, object] | None:
    parent_total = _float_or_none(metadata.get("parent_total_score"))
    parent_quality = _float_or_none(metadata.get("parent_quality_score"))
    parent_caption = _float_or_none(metadata.get("parent_caption_similarity_score"))
    if parent_total is None and parent_quality is None and parent_caption is None:
        return None

    current_total = _float_or_none(metadata.get("total_score"))
    current_caption = _float_or_none(metadata.get("caption_similarity_score"))
    return {
        "parent_total_score": _rounded_or_none(parent_total),
        "current_total_score": _rounded_or_none(current_total),
        "total_score_delta": _delta(current_total, parent_total),
        "parent_quality_score": _rounded_or_none(parent_quality),
        "current_quality_score": round(quality_score, 6),
        "quality_score_delta": _delta(quality_score, parent_quality),
        "parent_caption_similarity_score": _rounded_or_none(parent_caption),
        "current_caption_similarity_score": _rounded_or_none(current_caption),
        "caption_similarity_delta": _delta(current_caption, parent_caption),
        "continuity_score": _rounded_or_none(_float_or_none(metadata.get("initial_similarity_score"))),
    }


def _delta(current: object, parent: object) -> float | None:
    current_value = _float_or_none(current)
    parent_value = _float_or_none(parent)
    if current_value is None or parent_value is None:
        return None
    return round(current_value - parent_value, 6)


def _rounded_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _weakest_continuity_region(details: object) -> str | None:
    if not isinstance(details, dict):
        return None
    raw_region = details.get("weakest_continuity_region")
    if isinstance(raw_region, str) and raw_region:
        return raw_region
    region_scores = details.get("region_similarity_scores")
    if not isinstance(region_scores, dict) or not region_scores:
        return None
    parsed_scores = {
        str(region): score
        for region, score in ((key, _float_or_none(value)) for key, value in region_scores.items())
        if score is not None
    }
    if not parsed_scores:
        return None
    return min(parsed_scores.items(), key=lambda item: item[1])[0]


def _weakest_continuity_region_score(details: object) -> float | None:
    if not isinstance(details, dict):
        return None
    raw_score = _float_or_none(details.get("weakest_continuity_region_score"))
    if raw_score is not None:
        return round(raw_score, 6)
    region_scores = details.get("region_similarity_scores")
    if not isinstance(region_scores, dict) or not region_scores:
        return None
    parsed_scores = [_float_or_none(value) for value in region_scores.values()]
    parsed_scores = [score for score in parsed_scores if score is not None]
    if not parsed_scores:
        return None
    return round(min(parsed_scores), 6)


def _size_score(metadata: dict[str, object]) -> float:
    width = _float(metadata.get("width"), 0.0)
    height = _float(metadata.get("height"), 0.0)
    if width <= 0 or height <= 0:
        return 0.0
    return 1.0 if width <= 2048 and height <= 2048 else 0.0


def _float(value: object, default: float = 0.0) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else default


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []
