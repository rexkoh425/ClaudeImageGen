from __future__ import annotations

from collections.abc import Iterable, Mapping


def annotate_candidate_selection(candidate: dict[str, object]) -> dict[str, object]:
    score, reasons = compute_candidate_selection(candidate)
    candidate["selection_score"] = score
    candidate["selection_reasons"] = reasons
    return candidate


def select_recommended_candidate(candidates: Iterable[object]) -> dict[str, object]:
    best: dict[str, object] | None = None
    best_score = -1.0
    best_rank = 0

    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict):
            continue
        candidate = dict(raw_candidate)
        score = _float_or_none(candidate.get("selection_score"))
        if score is None:
            annotate_candidate_selection(candidate)
            score = _float(candidate.get("selection_score"))
        else:
            score = _clamp(score)
            candidate["selection_score"] = round(score, 6)
            candidate["selection_reasons"] = _string_list(candidate.get("selection_reasons")) or [
                f"precomputed selection_score={score:.3f}"
            ]

        rank = _rank(candidate)
        if best is None or score > best_score or (score == best_score and rank and rank < best_rank):
            best = candidate
            best_score = score
            best_rank = rank

    if best is None:
        raise ValueError("Candidate index does not contain any selectable candidates.")
    return best


def compute_candidate_selection(candidate: Mapping[str, object]) -> tuple[float, list[str]]:
    total_score = _float(candidate.get("total_score"))
    caption_similarity_score = _float(candidate.get("caption_similarity_score"))
    reference_score = _float(candidate.get("reference_score"))
    missing_objects = _string_list(candidate.get("caption_missing_objects"))
    missing_colors = _string_list(candidate.get("caption_missing_colors"))
    unexpected_objects = _string_list(candidate.get("caption_unexpected_objects"))
    unexpected_colors = _string_list(candidate.get("caption_unexpected_colors"))

    missing_object_penalty = 0.06 * len(missing_objects)
    missing_color_penalty = 0.04 * len(missing_colors)
    unexpected_object_penalty = 0.02 * len(unexpected_objects)
    unexpected_color_penalty = 0.01 * len(unexpected_colors)
    raw_score = (
        (0.64 * total_score)
        + (0.26 * caption_similarity_score)
        + (0.10 * reference_score)
        - missing_object_penalty
        - missing_color_penalty
        - unexpected_object_penalty
        - unexpected_color_penalty
    )
    score = round(_clamp(raw_score), 6)

    reasons = [
        f"total_score={total_score:.3f} weight=0.64",
        f"caption_similarity_score={caption_similarity_score:.3f} weight=0.26",
        f"reference_score={reference_score:.3f} weight=0.10",
    ]
    if missing_objects:
        reasons.append(f"missing_objects_penalty={missing_object_penalty:.3f}")
    if missing_colors:
        reasons.append(f"missing_colors_penalty={missing_color_penalty:.3f}")
    if unexpected_objects:
        reasons.append(f"unexpected_objects_penalty={unexpected_object_penalty:.3f}")
    if unexpected_colors:
        reasons.append(f"unexpected_colors_penalty={unexpected_color_penalty:.3f}")
    return score, reasons


def _float(value: object) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else 0.0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _rank(candidate: Mapping[str, object]) -> int:
    try:
        return int(candidate.get("rank", 0))
    except (TypeError, ValueError):
        return 0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
