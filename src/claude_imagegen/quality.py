from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_detail_metrics(image: Image.Image) -> dict[str, float]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return {
            "detail_score": 0.0,
            "edge_density": 0.0,
            "edge_strength": 0.0,
            "luminance_std": 0.0,
        }

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_strength = float((dx.mean() + dy.mean()) / 2.0) if dx.size and dy.size else 0.0
    edge_density = float(((dx > 18.0).mean() + (dy > 18.0).mean()) / 2.0) if dx.size and dy.size else 0.0
    luminance_std = float(gray.std())

    contrast_score = min(1.0, luminance_std / 64.0)
    edge_strength_score = min(1.0, edge_strength / 22.0)
    edge_density_score = min(1.0, edge_density / 0.16)
    detail_score = max(0.0, min(1.0, 0.38 * contrast_score + 0.34 * edge_strength_score + 0.28 * edge_density_score))

    return {
        "detail_score": round(detail_score, 6),
        "edge_density": round(edge_density, 6),
        "edge_strength": round(edge_strength, 6),
        "luminance_std": round(luminance_std, 6),
    }

def apply_quality_report(output_dir: Path, metadata: dict[str, object]) -> Path:
    report = build_quality_report(metadata)
    report_path = output_dir / "quality-report.json"
    report["report_path"] = str(report_path)
    metadata["quality_report"] = str(report_path)
    metadata["quality_status"] = report["status"]
    metadata["quality_score"] = report["quality_score"]
    metadata["target_quality_met"] = report["target_quality_met"]
    if report.get("refinement_delta") is not None:
        metadata["refinement_delta"] = report["refinement_delta"]
    if report.get("refinement_guidance") is not None:
        metadata["refinement_guidance"] = report["refinement_guidance"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def build_quality_report(metadata: dict[str, object]) -> dict[str, object]:
    checks = _quality_checks(metadata)
    quality_score = _weighted_quality_score(checks)
    status = _quality_status(checks, quality_score)
    target_quality_met = _target_quality_met(metadata, checks, quality_score)
    continuity_score = _float_or_none(metadata.get("initial_similarity_score"))
    continuity_details = metadata.get("initial_similarity_details")
    weakest_region = _weakest_continuity_region(continuity_details)
    weakest_region_score = _weakest_continuity_region_score(continuity_details)
    refinement_delta = _refinement_delta(metadata, quality_score=quality_score)
    refinement_guidance = _refinement_guidance(metadata, refinement_delta=refinement_delta)
    next_actions = _next_actions(metadata, checks, status, refinement_guidance=refinement_guidance)

    return {
        "status": status,
        "quality_score": quality_score,
        "quality_target": _float_or_none(metadata.get("quality_target")),
        "target_quality_met": target_quality_met,
        "summary": _summary(status, quality_score, checks),
        "checks": checks,
        "next_actions": next_actions,
        "continuity_score": continuity_score if continuity_score is not None else None,
        "weakest_continuity_region": weakest_region,
        "weakest_continuity_region_score": weakest_region_score,
        "refinement_delta": refinement_delta,
        "refinement_guidance": refinement_guidance,
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

    detail_score = _float_or_none(metadata.get("image_detail_score"))
    if detail_score is not None:
        checks.append(
            _check(
                name="image_detail",
                score=detail_score,
                pass_threshold=0.72,
                review_threshold=0.45,
                weight=0.12,
                detail="CPU detail score from luminance variation, edge density, and edge strength; flat or low-detail images cannot pass high quality targets.",
            )
        )

    critique = metadata.get("visual_critique")
    if isinstance(critique, dict) and "closeness_score" in critique:
        element_checks = _element_checks(critique.get("element_checks"))
        failed_element_checks = _failed_element_checks(element_checks)
        visual_check = _check(
            name="visual_judgement",
            score=_float(critique.get("closeness_score"), 0.0),
            pass_threshold=0.78,
            review_threshold=0.55,
            weight=0.30,
            detail="Claude-vision judge closeness (LMM-as-evaluator / VQAScore-style) after viewing the image.",
        )
        if element_checks:
            visual_check["element_checks"] = element_checks
        if failed_element_checks:
            visual_check["failed_element_checks"] = failed_element_checks
            if any(check.get("present") is False for check in failed_element_checks):
                visual_check["status"] = "revise"
            elif visual_check["status"] == "pass":
                visual_check["status"] = "review"
        checks.append(
            visual_check
        )

    comparison = metadata.get("visual_comparison")
    if isinstance(comparison, dict):
        alignment_score = _float(comparison.get("alignment_score"), 0.0)
        continuity_score = _float(comparison.get("continuity_score"), 0.0)
        comparison_check = _check(
            name="visual_comparison",
            score=(alignment_score + continuity_score) / 2,
            pass_threshold=0.72,
            review_threshold=0.50,
            weight=0.16,
            detail="Claude-vision parent/child comparison of refinement alignment and continuity.",
        )
        comparison_check["alignment_score"] = round(alignment_score, 6)
        comparison_check["continuity_score"] = round(continuity_score, 6)
        comparison_check["better_image"] = str(comparison.get("better_image") or "")
        comparison_check["improved"] = comparison.get("improved")
        comparison_check["preserved_identity"] = comparison.get("preserved_identity")
        regressions = _string_list(comparison.get("regressions"))
        if regressions:
            comparison_check["regressions"] = regressions
        if (
            str(comparison.get("verdict") or "").lower() == "revise"
            or str(comparison.get("better_image") or "").lower() == "parent"
            or comparison.get("improved") is False
            or comparison.get("preserved_identity") is False
        ):
            comparison_check["status"] = "revise"
        checks.append(comparison_check)

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

    independent_gate = _independent_quality_gate(metadata)
    if independent_gate is not None:
        checks.append(independent_gate)

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


def _independent_quality_gate(metadata: dict[str, object]) -> dict[str, object] | None:
    target = _float_or_none(metadata.get("quality_target"))
    if target is None or target < 0.9:
        return None

    local_score = _float(metadata.get("total_score"), 0.0)
    detail_score = _float(metadata.get("image_detail_score"), 0.0)
    visual_score = _visual_closeness_score(metadata)
    visual_for_score = visual_score if visual_score is not None else 0.0
    score = max(0.0, min(1.0, 0.35 * local_score + 0.45 * visual_for_score + 0.20 * detail_score))
    check = _check(
        name="independent_quality_gate",
        score=score,
        pass_threshold=target,
        review_threshold=max(0.0, target - 0.12),
        weight=0.18,
        detail=(
            "High quality target gate that requires local prompt score, independent Claude visual judgement, "
            "and CPU detail evidence so the renderer cannot pass by optimizing only its own scorer."
        ),
    )
    local_floor = max(0.85, target - 0.05)
    detail_floor = 0.78
    if local_score < local_floor or detail_score < detail_floor or visual_score is None or visual_score < target:
        check["status"] = "revise"
    check["target"] = round(target, 6)
    check["local_score"] = round(local_score, 6)
    check["local_floor"] = round(local_floor, 6)
    check["detail_score"] = round(detail_score, 6)
    check["detail_floor"] = round(detail_floor, 6)
    check["visual_closeness_score"] = round(visual_score, 6) if visual_score is not None else None
    check["visual_floor"] = round(target, 6)
    return check


def _visual_closeness_score(metadata: dict[str, object]) -> float | None:
    critique = metadata.get("visual_critique")
    if not isinstance(critique, dict):
        return None
    return _float_or_none(critique.get("closeness_score"))


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


def _target_quality_met(metadata: dict[str, object], checks: list[dict[str, object]], quality_score: float) -> bool:
    target = _float_or_none(metadata.get("quality_target"))
    if target is None:
        return False
    independent_gate = next((check for check in checks if check.get("name") == "independent_quality_gate"), None)
    if independent_gate is not None:
        return independent_gate.get("status") == "pass"
    return quality_score >= target


def _weighted_quality_score(checks: list[dict[str, object]]) -> float:
    total_weight = sum(_float(check.get("weight"), 0.0) for check in checks)
    if total_weight <= 0:
        return 0.0
    score = sum(_float(check.get("score"), 0.0) * _float(check.get("weight"), 0.0) for check in checks) / total_weight
    return round(max(0.0, min(1.0, score)), 6)


def _next_actions(
    metadata: dict[str, object],
    checks: list[dict[str, object]],
    status: str,
    *,
    refinement_guidance: dict[str, object] | None = None,
) -> list[str]:
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
        actions.extend(_element_check_actions(_element_checks(critique.get("element_checks"))))

    comparison = metadata.get("visual_comparison")
    if isinstance(comparison, dict):
        regressions = _string_list(comparison.get("regressions"))
        if regressions:
            actions.append(f"Comparison: address regressions: {', '.join(regressions)}.")
        if str(comparison.get("better_image") or "").lower() == "parent":
            actions.append("Comparison: parent looked better; preserve more parent identity, layout, and palette.")

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
    if "independent_quality_gate" in failed:
        target = _float(metadata.get("quality_target"), 0.9)
        if _visual_closeness_score(metadata) is None:
            actions.append(f"Claude visual critique is required before accepting a {target:.3f} quality target.")
        if _float(metadata.get("image_detail_score"), 0.0) < 0.78:
            actions.append("Add more visible local detail: textures, materials, foreground marks, lighting edges, or sharpen/detail style controls.")
        if _float(metadata.get("total_score"), 0.0) < max(0.85, target - 0.05):
            actions.append("Raise local prompt alignment before accepting the high quality target.")

    if refinement_guidance is not None:
        for action in _refinement_guidance_actions(refinement_guidance):
            actions.append(action)
    else:
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


def _refinement_guidance_actions(guidance: dict[str, object]) -> list[str]:
    axes = guidance.get("priority_axes")
    if not isinstance(axes, list):
        return []
    actions: list[str] = []
    for axis in axes:
        if not isinstance(axis, dict):
            continue
        action = str(axis.get("action") or "").strip()
        if action:
            actions.append(action)
    return actions


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


def _element_checks(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    checks: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        label = str(item.get("item") or "").strip()
        if not kind or not label:
            continue
        check: dict[str, object] = {
            "kind": kind,
            "item": label,
        }
        if "present" in item:
            present = _bool_or_none(item.get("present"))
            if present is not None:
                check["present"] = present
        confidence = _float_or_none(item.get("confidence"))
        if confidence is not None:
            check["confidence"] = round(max(0.0, min(1.0, confidence)), 6)
        notes = str(item.get("notes") or "").strip()
        if notes:
            check["notes"] = notes
        checks.append(check)
    return checks


def _failed_element_checks(checks: list[dict[str, object]]) -> list[dict[str, object]]:
    return [check for check in checks if _element_check_failed(check)]


def _element_check_failed(check: dict[str, object]) -> bool:
    if check.get("present") is False:
        return True
    confidence = _float_or_none(check.get("confidence"))
    return confidence is not None and confidence < 0.5


def _element_check_actions(checks: list[dict[str, object]]) -> list[str]:
    missing_objects: list[str] = []
    missing_colors: list[str] = []
    low_confidence_objects: list[str] = []
    low_confidence_colors: list[str] = []
    weak_styles: list[str] = []
    weak_moods: list[str] = []

    for check in _failed_element_checks(checks):
        kind = str(check.get("kind") or "")
        item = str(check.get("item") or "")
        if not item:
            continue
        is_missing = check.get("present") is False
        if kind == "object":
            (missing_objects if is_missing else low_confidence_objects).append(item)
        elif kind == "color":
            (missing_colors if is_missing else low_confidence_colors).append(item)
        elif kind == "style":
            weak_styles.append(item)
        elif kind == "mood":
            weak_moods.append(item)

    actions: list[str] = []
    if missing_objects:
        actions.append(f"Judge: make missing checked objects explicit: {', '.join(dict.fromkeys(missing_objects))}.")
    if missing_colors:
        actions.append(f"Judge: make missing checked colors explicit: {', '.join(dict.fromkeys(missing_colors))}.")
    if low_confidence_objects:
        actions.append(f"Judge: clarify low-confidence checked objects: {', '.join(dict.fromkeys(low_confidence_objects))}.")
    if low_confidence_colors:
        actions.append(f"Judge: strengthen low-confidence checked colors: {', '.join(dict.fromkeys(low_confidence_colors))}.")
    if weak_styles:
        actions.append(f"Judge: strengthen checked styles: {', '.join(dict.fromkeys(weak_styles))}.")
    if weak_moods:
        actions.append(f"Judge: make checked moods more visually explicit: {', '.join(dict.fromkeys(weak_moods))}.")
    return actions


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "present"}:
            return True
        if normalized in {"false", "no", "missing", "absent"}:
            return False
    return None


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


def _refinement_guidance(metadata: dict[str, object], *, refinement_delta: dict[str, object] | None) -> dict[str, object] | None:
    if refinement_delta is None:
        return None

    axes: list[dict[str, object]] = []
    total_delta = _float_or_none(refinement_delta.get("total_score_delta"))
    if total_delta is not None and total_delta < -0.03:
        axes.append(
            {
                "axis": "prompt_alignment",
                "delta": round(total_delta, 6),
                "severity": "revise" if total_delta <= -0.08 else "review",
                "action": "Refinement: restore prompt alignment that dropped versus the parent.",
            }
        )

    quality_delta = _float_or_none(refinement_delta.get("quality_score_delta"))
    if quality_delta is not None and quality_delta < -0.03:
        axes.append(
            {
                "axis": "quality",
                "delta": round(quality_delta, 6),
                "severity": "revise" if quality_delta <= -0.08 else "review",
                "action": "Refinement: inspect failed quality checks before continuing; overall quality dropped versus the parent.",
            }
        )

    caption_delta = _float_or_none(refinement_delta.get("caption_similarity_delta"))
    if caption_delta is not None and caption_delta < -0.05:
        axes.append(
            {
                "axis": "caption_alignment",
                "delta": round(caption_delta, 6),
                "severity": "revise" if caption_delta <= -0.12 else "review",
                "action": "Refinement: restore caption evidence for requested objects, colors, and relationships.",
            }
        )

    continuity_score = _float_or_none(refinement_delta.get("continuity_score"))
    if continuity_score is not None and continuity_score < 0.78:
        weakest_region = _weakest_continuity_region(metadata.get("initial_similarity_details"))
        weakest_score = _weakest_continuity_region_score(metadata.get("initial_similarity_details"))
        continuity_axis: dict[str, object] = {
            "axis": "continuity",
            "score": round(continuity_score, 6),
            "severity": "revise" if continuity_score < 0.58 else "review",
            "action": "Refinement: preserve parent layout, palette, and silhouettes; continuity is below the pass threshold.",
        }
        if weakest_region:
            continuity_axis["weakest_region"] = weakest_region
            if weakest_score is not None:
                continuity_axis["weakest_region_score"] = round(weakest_score, 6)
                continuity_axis["action"] = (
                    f"Refinement: preserve parent layout near the {weakest_region.replace('_', ' ')} region; "
                    f"it has the weakest continuity score ({weakest_score:.3f})."
                )
        axes.append(continuity_axis)

    if any(str(axis.get("severity")) == "revise" for axis in axes):
        decision = "revise"
    elif axes:
        decision = "review"
    else:
        decision = "accept"

    axis_names = [str(axis.get("axis")) for axis in axes]
    return {
        "decision": decision,
        "priority_axes": axes,
        "summary": _refinement_guidance_summary(decision, axis_names),
    }


def _refinement_guidance_summary(decision: str, axis_names: list[str]) -> str:
    if not axis_names:
        return "accept: refinement did not regress tracked parent-child score axes."
    return f"{decision}: inspect {', '.join(axis_names)} before the next refinement."


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
