from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any


@dataclass(frozen=True)
class EvalPlanOptions:
    prompt: str
    output_dir: Path
    evaluation: Path | None = None
    evaluations: tuple[Path, ...] = ()
    audits: tuple[Path, ...] = ()
    quality_target: float = 0.9
    min_evaluations: int = 2


@dataclass(frozen=True)
class EvalPlanResult:
    plan: dict[str, object]
    plan_path: Path


def build_eval_plan(options: EvalPlanOptions) -> EvalPlanResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    evaluation_paths = _evaluation_paths(options)
    samples = [_evaluation_sample(path, quality_target=options.quality_target) for path in evaluation_paths]
    conservative = min(samples, key=lambda sample: float(sample["after_score"]))
    after_scores = [float(sample["after_score"]) for sample in samples]
    detail_scores = [float(sample["detail_score"]) for sample in samples]
    best_after_score = min(after_scores)
    detail_score = min(detail_scores)
    after_score_median = round(float(statistics.median(after_scores)), 6)
    after_score_max = round(max(after_scores), 6)
    minimum_evaluations_met = len(samples) >= max(1, int(options.min_evaluations))
    acceptance_consensus_met = minimum_evaluations_met and all(bool(sample["gate_met"]) for sample in samples)
    parity = all(bool(sample["parity"]) for sample in samples)
    claude_quality_gate_met = acceptance_consensus_met and parity
    failure_modes: list[str] = []
    recommendations: list[str] = []
    for sample in samples:
        failure_modes.extend(_string_list(sample.get("failure_modes")))
        recommendations.extend(_string_list(sample.get("recommendations")))
    audit_samples = [_audit_sample(path) for path in options.audits]
    for audit in audit_samples:
        failure_modes.extend(_audit_failure_modes(audit))
        recommendations.extend(_string_list(audit.get("recommendations")))

    suggested_parameters = _suggest_enhance_parameters(failure_modes=failure_modes, recommendations=recommendations)
    suggested_parameters = _merge_audit_parameters(suggested_parameters, audit_samples)
    local_audit_gate_met = all(bool(audit.get("gate_met")) for audit in audit_samples) if audit_samples else True
    target_quality_met = claude_quality_gate_met and local_audit_gate_met
    next_action = "accept" if target_quality_met else "enhance-night"
    best_after_image = str(conservative.get("best_after_image") or "")
    command = _enhance_command(
        input_image=best_after_image,
        prompt=options.prompt,
        output_dir=options.output_dir / "enhance-night",
        parameters=suggested_parameters,
        quality_target=options.quality_target,
    )
    score_gap = round(max(0.0, options.quality_target - best_after_score), 6)
    plan: dict[str, object] = {
        "engine": "pair-evaluation-improvement-plan-v1",
        "evaluation": str(evaluation_paths[0]),
        "evaluations": [str(path) for path in evaluation_paths],
        "evaluation_count": len(samples),
        "audits": [str(path) for path in options.audits],
        "audit_count": len(audit_samples),
        "prompt": options.prompt,
        "quality_target": options.quality_target,
        "minimum_evaluations_required": max(1, int(options.min_evaluations)),
        "minimum_evaluations_met": minimum_evaluations_met,
        "local_audit_gate_met": local_audit_gate_met,
        "target_quality_met": target_quality_met,
        "acceptance_consensus_met": acceptance_consensus_met,
        "gpt_sora_parity_boolean": parity,
        "gpt_sora_parity_score": min(
            (
                score
                for score in (_float_or_none(sample.get("gpt_sora_parity_score")) for sample in samples)
                if score is not None
            ),
            default=None,
        ),
        "best_pair_id": str(conservative.get("best_pair_id") or conservative.get("pair_id") or ""),
        "best_after_image": best_after_image,
        "best_after_score": round(best_after_score, 6),
        "after_score_median": after_score_median,
        "after_score_max": after_score_max,
        "best_detail_score": round(detail_score, 6),
        "score_gap": score_gap,
        "next_action": next_action,
        "suggested_parameters": suggested_parameters,
        "recommended_command": "" if next_action == "accept" else command,
        "acceptance_reason": _acceptance_reason(
            target_quality_met=target_quality_met,
            parity=parity,
            acceptance_consensus_met=acceptance_consensus_met,
            local_audit_gate_met=local_audit_gate_met,
            minimum_evaluations_met=minimum_evaluations_met,
            minimum_evaluations_required=max(1, int(options.min_evaluations)),
            evaluation_count=len(samples),
            score_gap=score_gap,
            best_after_score=best_after_score,
            quality_target=options.quality_target,
        ),
        "failure_modes": list(dict.fromkeys(failure_modes)),
        "code_improvement_recommendations": list(dict.fromkeys(recommendations)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = options.output_dir / "improvement-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return EvalPlanResult(plan=plan, plan_path=plan_path)


def _evaluation_paths(options: EvalPlanOptions) -> tuple[Path, ...]:
    paths: list[Path] = []
    if options.evaluation is not None:
        paths.append(options.evaluation)
    paths.extend(options.evaluations)
    if not paths:
        raise ValueError("at least one evaluation file is required")
    return tuple(paths)


def _evaluation_sample(path: Path, *, quality_target: float) -> dict[str, object]:
    data = _load_json_object(path)
    pair_scores = _pair_scores(data)
    best_pair = _best_pair(data, pair_scores)
    after_score = _float(best_pair.get("after_score"), 0.0)
    detail_score = _float(best_pair.get("detail_score"), 0.0)
    parity = bool(data.get("gpt_sora_parity_boolean")) and bool(best_pair.get("parity_boolean"))
    gate_met = bool(data.get("acceptance_gate_met")) and after_score >= quality_target and parity
    return {
        "path": str(path),
        "pair_id": str(best_pair.get("id") or ""),
        "best_pair_id": str(data.get("best_pair_id") or ""),
        "best_after_image": str(data.get("best_after_image") or ""),
        "after_score": after_score,
        "detail_score": detail_score,
        "gpt_sora_parity_score": _float_or_none(data.get("gpt_sora_parity_score")),
        "parity": parity,
        "gate_met": gate_met,
        "failure_modes": _string_list(data.get("overall_failure_modes")) + _string_list(best_pair.get("failure_modes")),
        "recommendations": _string_list(data.get("code_improvement_recommendations"))
        + _string_list(best_pair.get("recommended_code_changes")),
    }


def _audit_sample(path: Path) -> dict[str, object]:
    data = _load_json_object(path)
    flags = data.get("flags")
    suggested = data.get("suggested_parameters")
    gate_met = True
    if isinstance(flags, dict):
        gate_met = bool(flags.get("night_mood_preserved", True)) and not any(
            bool(flags.get(name))
            for name in ("overbright_after", "detail_softening_risk", "highlight_clipping_risk", "haze_risk")
        )
    return {
        "path": str(path),
        "flags": flags if isinstance(flags, dict) else {},
        "suggested_parameters": suggested if isinstance(suggested, dict) else {},
        "recommendations": _string_list(data.get("recommendations")),
        "gate_met": gate_met,
    }


def _audit_failure_modes(audit: dict[str, object]) -> list[str]:
    flags = audit.get("flags")
    if not isinstance(flags, dict):
        return []
    failures: list[str] = []
    if bool(flags.get("overbright_after")):
        failures.append("local audit detected over-bright after image")
    if bool(flags.get("haze_risk")):
        failures.append("local audit detected haze or bloom washout")
    if bool(flags.get("highlight_clipping_risk")):
        failures.append("local audit detected highlight clipping")
    if bool(flags.get("detail_softening_risk")):
        failures.append("local audit detected detail softening")
    return failures


def _merge_audit_parameters(
    parameters: dict[str, float],
    audit_samples: list[dict[str, object]],
) -> dict[str, float]:
    merged = dict(parameters)
    for audit in audit_samples:
        suggested = audit.get("suggested_parameters")
        if not isinstance(suggested, dict):
            continue
        for key in ("night_luma_ceiling", "mist_cap", "highlight_rolloff"):
            value = _float_or_none(suggested.get(key))
            if value is not None:
                merged[key] = min(float(merged.get(key, value)), value)
        value = _float_or_none(suggested.get("local_contrast"))
        if value is not None:
            merged["local_contrast"] = max(float(merged.get("local_contrast", value)), value)
    return merged


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("evaluation must be a JSON object")
    return data


def _pair_scores(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("pair_scores")
    if not isinstance(value, list) or not value:
        raise ValueError("evaluation must include non-empty pair_scores")
    return [item for item in value if isinstance(item, dict)]


def _best_pair(data: dict[str, Any], pair_scores: list[dict[str, Any]]) -> dict[str, Any]:
    best_id = str(data.get("best_pair_id") or "")
    for pair in pair_scores:
        if str(pair.get("id") or "") == best_id:
            return pair
    return max(pair_scores, key=lambda pair: _float(pair.get("after_score"), 0.0))


def _suggest_enhance_parameters(*, failure_modes: list[str], recommendations: list[str]) -> dict[str, float]:
    text = " ".join(failure_modes + recommendations).lower()
    night_luma_ceiling = 0.32
    mist_cap = 0.2
    highlight_rolloff = 0.35
    local_contrast = 0.9
    if any(term in text for term in ("over-bright", "bright", "dusk", "twilight", "night-mood drift")):
        night_luma_ceiling = 0.3
    if any(term in text for term in ("haze", "mist", "bloom", "veil")):
        mist_cap = 0.16
    if any(term in text for term in ("clip", "clipping", "highlight", "lamp")):
        highlight_rolloff = 0.25
    if any(term in text for term in ("contrast", "floor", "mid-tone", "leaf", "detail")):
        local_contrast = 1.05
    return {
        "night_luma_ceiling": night_luma_ceiling,
        "mist_cap": mist_cap,
        "highlight_rolloff": highlight_rolloff,
        "local_contrast": local_contrast,
    }


def _enhance_command(
    *,
    input_image: str,
    prompt: str,
    output_dir: Path,
    parameters: dict[str, float],
    quality_target: float,
) -> str:
    return (
        "claude-imagegen enhance-night "
        f"--input-image \"{input_image}\" "
        f"--prompt \"{prompt}\" "
        f"--output-dir \"{output_dir}\" "
        f"--quality-target {quality_target:g} "
        f"--night-luma-ceiling {parameters['night_luma_ceiling']:g} "
        f"--mist-cap {parameters['mist_cap']:g} "
        f"--highlight-rolloff {parameters['highlight_rolloff']:g} "
        f"--local-contrast {parameters['local_contrast']:g}"
    )


def _acceptance_reason(
    *,
    target_quality_met: bool,
    parity: bool,
    acceptance_consensus_met: bool,
    local_audit_gate_met: bool,
    minimum_evaluations_met: bool,
    minimum_evaluations_required: int,
    evaluation_count: int,
    score_gap: float,
    best_after_score: float,
    quality_target: float,
) -> str:
    if target_quality_met and parity:
        return "Accept only after verifying the Claude evaluation response and artifact paths."
    if not minimum_evaluations_met:
        return (
            f"Do not accept: at least {minimum_evaluations_required} Claude evaluations are required "
            f"for this quality gate; only {evaluation_count} provided."
        )
    if evaluation_count > 1 and not acceptance_consensus_met:
        return (
            "Do not accept: multiple Claude evaluations disagree or at least one response failed the gate; "
            f"conservative after_score is {best_after_score:.2f}."
        )
    if not local_audit_gate_met:
        return "Do not accept: local pair audit found brightness, haze, clipping, or detail risks."
    return (
        f"Do not accept: after_score {best_after_score:.2f} is below target {quality_target:.2f} "
        f"by {score_gap:.2f}, or GPT/Sora parity is false."
    )


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
