from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairAuditOptions:
    before_image: Path
    after_image: Path
    prompt: str
    output_dir: Path
    night_luma_ceiling: float = 0.34
    quality_target: float = 0.9


@dataclass(frozen=True)
class PairAuditResult:
    audit: dict[str, object]
    audit_path: Path


def audit_pair(options: PairAuditOptions) -> PairAuditResult:
    if not options.prompt.strip():
        raise ValueError("prompt must not be empty")
    if not options.before_image.exists():
        raise FileNotFoundError(f"before image does not exist: {options.before_image}")
    if not options.after_image.exists():
        raise FileNotFoundError(f"after image does not exist: {options.after_image}")

    before_metrics = _image_metrics(options.before_image)
    after_metrics = _image_metrics(options.after_image)
    deltas = _deltas(before_metrics, after_metrics)
    flags = _flags(after_metrics=after_metrics, deltas=deltas, night_luma_ceiling=options.night_luma_ceiling)
    suggested_parameters = _suggested_parameters(flags=flags)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "pair-audit.json"
    recommended_command = _enhance_command(
        input_image=str(options.after_image),
        prompt=options.prompt,
        output_dir=output_dir / "enhance-night",
        quality_target=options.quality_target,
        parameters=suggested_parameters,
    )
    audit: dict[str, object] = {
        "engine": "pair-local-audit-v1",
        "before_image": str(options.before_image),
        "after_image": str(options.after_image),
        "prompt": options.prompt,
        "quality_target": options.quality_target,
        "night_luma_ceiling": options.night_luma_ceiling,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "deltas": deltas,
        "flags": flags,
        "suggested_parameters": suggested_parameters,
        "recommended_command": recommended_command,
        "recommendations": _recommendations(flags),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return PairAuditResult(audit=audit, audit_path=audit_path)


def _image_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = _luma(arr)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    lower = luma[luma.shape[0] // 2 :, :]
    edge_density = _edge_density(luma)
    return {
        "mean_luma": round(float(np.mean(luma)), 6),
        "median_luma": round(float(np.median(luma)), 6),
        "max_luma": round(float(np.max(luma)), 6),
        "highlight_clip_ratio": round(float(np.mean(luma > 0.94)), 6),
        "haze_ratio": round(float(np.mean((chroma < 0.12) & (luma > 0.32) & (luma < 0.82))), 6),
        "edge_density": round(edge_density, 6),
        "lower_luma_std": round(float(np.std(lower)), 6),
    }


def _deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = (
        "mean_luma",
        "median_luma",
        "max_luma",
        "highlight_clip_ratio",
        "haze_ratio",
        "edge_density",
        "lower_luma_std",
    )
    return {f"{key}_delta": round(after[key] - before[key], 6) for key in keys}


def _flags(*, after_metrics: dict[str, float], deltas: dict[str, float], night_luma_ceiling: float) -> dict[str, bool]:
    overbright = after_metrics["mean_luma"] > night_luma_ceiling or deltas["mean_luma_delta"] > 0.12
    return {
        "night_mood_preserved": not overbright,
        "overbright_after": overbright,
        "detail_softening_risk": deltas["edge_density_delta"] < -0.02 or deltas["lower_luma_std_delta"] < -0.01,
        "highlight_clipping_risk": after_metrics["highlight_clip_ratio"] > 0.002
        or deltas["highlight_clip_ratio_delta"] > 0.001,
        "haze_risk": after_metrics["haze_ratio"] > 0.12 or deltas["haze_ratio_delta"] > 0.08,
    }


def _suggested_parameters(*, flags: dict[str, bool]) -> dict[str, float]:
    return {
        "night_luma_ceiling": 0.3 if flags["overbright_after"] else 0.34,
        "mist_cap": 0.16 if flags["haze_risk"] else 0.22,
        "highlight_rolloff": 0.25 if flags["highlight_clipping_risk"] else 0.35,
        "local_contrast": 1.05 if flags["detail_softening_risk"] else 0.9,
    }


def _recommendations(flags: dict[str, bool]) -> list[str]:
    recommendations: list[str] = []
    if flags["overbright_after"]:
        recommendations.append("Apply a stricter night luminance ceiling before accepting deep-night prompts.")
    if flags["haze_risk"]:
        recommendations.append("Cap mist and bloom overlays so local detail does not wash out.")
    if flags["highlight_clipping_risk"]:
        recommendations.append("Use highlight rolloff around lamp cores to prevent clipping.")
    if flags["detail_softening_risk"]:
        recommendations.append("Recover lower-half and leaf detail with local contrast, not global exposure.")
    return recommendations


def _enhance_command(
    *,
    input_image: str,
    prompt: str,
    output_dir: Path,
    quality_target: float,
    parameters: dict[str, float],
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


def _edge_density(luma: np.ndarray) -> float:
    horizontal = np.abs(np.diff(luma, axis=1))
    vertical = np.abs(np.diff(luma, axis=0))
    return float((np.mean(horizontal > 0.045) + np.mean(vertical > 0.045)) / 2.0)


def _luma(arr: np.ndarray) -> np.ndarray:
    return (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
