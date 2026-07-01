"""Claude-vision critique: the model-in-the-loop judgement signal.

After Claude Code renders an image, it opens ``image.png`` with its own vision and
writes a structured critique (a small JSON file). This is the project's primary
verification signal, mirroring the 2025 "LMM-as-evaluator" / VQAScore findings that a
capable multimodal model judges prompt alignment better than CLIPScore alone — but with
no external API: the judge is Claude Code itself.

A critique captures:

* ``closeness_score`` (0-1): holistic judgement of how well the image matches the prompt.
* ``verdict``: ``accept`` or ``revise``.
* ``present`` / ``missing`` / ``wrong`` / ``extra``: requested elements and problems.
* ``edits``: optional concrete scene-plan edits to apply automatically.
* ``summary`` / ``notes``: natural-language rationale.

The module parses/validates the critique, produces a normalized signal for metadata and
the quality report, and can apply the structured edits to a scene-plan JSON dict. Edits
operate on the raw JSON dict (scene plans are authored as JSON), which keeps the apply
path simple and robust; unknown edit actions are skipped, never fatal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .palette import COLOR_RGB

ACCEPT = "accept"
REVISE = "revise"

_KNOWN_ACTIONS = {
    "add_object",
    "remove_object",
    "recolor_object",
    "move_object",
    "resize_object",
    "set_opacity",
    "set_style",
    "adjust_style",
    "set_palette",
    "add_element",
    "add_cloud",
}


@dataclass(frozen=True)
class VisualCritique:
    closeness_score: float
    verdict: str
    summary: str
    present: tuple[str, ...]
    missing: tuple[str, ...]
    wrong: tuple[str, ...]
    extra: tuple[str, ...]
    element_checks: tuple[dict[str, Any], ...]
    edits: tuple[dict[str, Any], ...]
    notes: str


def known_edit_actions() -> list[str]:
    """Return the supported structured edit actions for Claude-vision critiques."""
    return sorted(_KNOWN_ACTIONS)


def write_critique_request(
    output_dir: Path,
    *,
    image_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> Path:
    """Write the JSON request Claude Code should fill after visually inspecting image.png."""
    request_path = output_dir / "critique-request.json"
    request = build_critique_request(
        image_path=image_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    metadata["critique_request"] = str(request_path)
    return request_path


def write_comparison_request(
    output_dir: Path,
    *,
    parent_image: Path,
    child_image: Path,
    metadata_path: Path,
    parent_metadata_path: Path | None,
    metadata: dict[str, Any],
) -> Path:
    """Write the JSON request Claude Code should fill after comparing parent and child images."""
    request_path = output_dir / "comparison-request.json"
    request = build_comparison_request(
        parent_image=parent_image,
        child_image=child_image,
        metadata_path=metadata_path,
        parent_metadata_path=parent_metadata_path,
        metadata=metadata,
    )
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    metadata["comparison_request"] = str(request_path)
    return request_path


def build_critique_request(
    *,
    image_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable Claude-vision judge request from run metadata."""
    quality_status = str(metadata.get("quality_status") or "")
    suggested_verdict = ACCEPT if quality_status == "pass" else REVISE
    visual_checklist = _visual_checklist(metadata)
    return {
        "judge": "claude-vision",
        "instructions": (
            "Open image.png, answer each visual_checklist item like a VQAScore-style yes/no "
            "question, compare the image against the prompt and metadata, then write only JSON "
            "matching expected_response. Use edits only from allowed_edit_actions so refine "
            "--critique can apply them automatically."
        ),
        "image": str(image_path),
        "metadata": str(metadata_path),
        "quality_report": _str_or_none(metadata.get("quality_report")),
        "output_dir": str(metadata_path.parent),
        "prompt": str(metadata.get("prompt") or ""),
        "normalized_prompt": str(metadata.get("normalized_prompt") or ""),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "total_score": metadata.get("total_score"),
        "quality_status": metadata.get("quality_status"),
        "quality_score": metadata.get("quality_score"),
        "caption": metadata.get("image_caption"),
        "caption_similarity_score": metadata.get("caption_similarity_score"),
        "initial_similarity_score": metadata.get("initial_similarity_score"),
        "revision_hints": _str_list(metadata.get("revision_hints")),
        "visual_checklist": visual_checklist,
        "allowed_edit_actions": known_edit_actions(),
        "expected_response": {
            "closeness_score": None,
            "verdict": suggested_verdict,
            "element_checks": _expected_element_checks(visual_checklist),
            "summary": "",
            "present": [],
            "missing": [],
            "wrong": [],
            "extra": [],
            "edits": [],
            "notes": "",
        },
    }


def build_comparison_request(
    *,
    parent_image: Path,
    child_image: Path,
    metadata_path: Path,
    parent_metadata_path: Path | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a stable Claude-vision request for judging a refinement against its parent."""
    return {
        "judge": "claude-vision-refinement-comparison",
        "instructions": (
            "Open parent_image and child_image side by side. Decide whether the child is a better "
            "answer to the current prompt while preserving the important subject, layout, palette, "
            "and identity from the parent. Write only JSON matching expected_response. Use "
            "follow_up_edits only from allowed_edit_actions so refine --critique can apply them."
        ),
        "parent_image": str(parent_image),
        "child_image": str(child_image),
        "metadata": str(metadata_path),
        "parent_metadata": str(parent_metadata_path) if parent_metadata_path else None,
        "quality_report": _str_or_none(metadata.get("quality_report")),
        "critique_request": _str_or_none(metadata.get("critique_request")),
        "output_dir": str(metadata_path.parent),
        "prompt": str(metadata.get("prompt") or ""),
        "parent_prompt": str(metadata.get("parent_prompt") or ""),
        "total_score": metadata.get("total_score"),
        "quality_score": metadata.get("quality_score"),
        "initial_similarity_score": metadata.get("initial_similarity_score"),
        "initial_similarity_details": metadata.get("initial_similarity_details"),
        "refinement_delta": metadata.get("refinement_delta"),
        "caption": metadata.get("image_caption"),
        "parent_caption": metadata.get("parent_caption"),
        "caption_similarity_score": metadata.get("caption_similarity_score"),
        "parent_caption_similarity_score": metadata.get("parent_caption_similarity_score"),
        "allowed_edit_actions": known_edit_actions(),
        "expected_response": {
            "alignment_score": None,
            "continuity_score": None,
            "improved": None,
            "preserved_identity": None,
            "better_image": "child",
            "verdict": "accept",
            "summary": "",
            "regressions": [],
            "follow_up_edits": [],
            "notes": "",
        },
    }


def parse_critique(source: Path | str | dict[str, Any]) -> VisualCritique:
    """Parse a Claude-authored critique from a path, JSON string, or dict."""
    data = _load(source)
    if not isinstance(data, dict):
        raise ValueError("critique must be a JSON object")

    closeness = _clamp01(_to_float(data.get("closeness_score", data.get("score", 0.0)), 0.0))
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in {ACCEPT, REVISE}:
        verdict = ACCEPT if closeness >= 0.8 else REVISE

    return VisualCritique(
        closeness_score=closeness,
        verdict=verdict,
        summary=str(data.get("summary", "")).strip(),
        present=_str_tuple(data.get("present")),
        missing=_str_tuple(data.get("missing")),
        wrong=_str_tuple(data.get("wrong")),
        extra=_str_tuple(data.get("extra")),
        element_checks=_element_check_tuple(data.get("element_checks")),
        edits=_edit_tuple(data.get("edits")),
        notes=str(data.get("notes", "")).strip(),
    )


def critique_signal(critique: VisualCritique, applied_edits: list[str] | None = None) -> dict[str, Any]:
    """Normalized critique record for metadata and the quality report."""
    return {
        "judge": "claude-vision",
        "closeness_score": round(critique.closeness_score, 6),
        "verdict": critique.verdict,
        "summary": critique.summary,
        "present": list(critique.present),
        "missing": list(critique.missing),
        "wrong": list(critique.wrong),
        "extra": list(critique.extra),
        "element_checks": [dict(check) for check in critique.element_checks],
        "requested_edits": [dict(edit) for edit in critique.edits],
        "applied_edits": list(applied_edits or []),
        "notes": critique.notes,
    }


def apply_critique_to_plan_dict(
    plan: dict[str, Any], critique: VisualCritique
) -> tuple[dict[str, Any], list[str]]:
    """Apply the critique's structured edits to a scene-plan dict (mutates a copy)."""
    revised: dict[str, Any] = json.loads(json.dumps(plan))  # deep copy
    actions: list[str] = []
    actions.extend(_apply_element_check_edits(revised, critique.element_checks))
    for edit in critique.edits:
        action = str(edit.get("action", "")).strip().lower()
        if action not in _KNOWN_ACTIONS:
            actions.append(f"skipped unknown edit action '{action or '(missing)'}'")
            continue
        handler = _EDIT_HANDLERS[action]
        message = handler(revised, edit)
        if message:
            actions.append(message)
    return revised, actions


def apply_critique_to_plan_file(
    in_path: Path, out_path: Path, critique: VisualCritique
) -> list[str]:
    """Read a scene-plan file, apply critique edits, write the revised plan."""
    plan = json.loads(Path(in_path).read_text(encoding="utf-8-sig"))
    if not isinstance(plan, dict):
        raise ValueError("scene plan must be a JSON object")
    revised, actions = apply_critique_to_plan_dict(plan, critique)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(revised, indent=2), encoding="utf-8")
    return actions


# --- edit handlers -------------------------------------------------------------------


def _object_type(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("type", item.get("kind", ""))).strip().lower()


def _list_field(plan: dict[str, Any], key: str) -> list[Any]:
    value = plan.get(key)
    if isinstance(value, list):
        return value
    plan[key] = []
    return plan[key]


def _edit_add_object(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    kind = str(edit.get("type", edit.get("kind", "shape"))).strip().lower() or "shape"
    objects = _list_field(plan, "objects")
    new_object: dict[str, Any] = {
        "type": kind,
        "label": str(edit.get("label", f"critique-added {kind}")),
        "x": _clamp01(_to_float(edit.get("x", 0.5), 0.5)),
        "y": _clamp01(_to_float(edit.get("y", 0.5), 0.5)),
        "size": _clamp01(_to_float(edit.get("size", 0.18), 0.18)),
        "opacity": _clamp01(_to_float(edit.get("opacity", 1.0), 1.0)),
    }
    if edit.get("color") is not None:
        new_object["color"] = str(edit["color"])
    objects.append(new_object)
    return f"added object '{kind}'"


def _edit_remove_object(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    target = str(edit.get("type", edit.get("kind", ""))).strip().lower()
    if not target:
        return "skipped remove_object without type"
    objects = _list_field(plan, "objects")
    kept = [item for item in objects if _object_type(item) != target]
    removed = len(objects) - len(kept)
    plan["objects"] = kept
    return f"removed {removed} '{target}' object(s)" if removed else f"no '{target}' object to remove"


def _edit_recolor_object(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    target = str(edit.get("type", edit.get("kind", ""))).strip().lower()
    color = edit.get("color")
    if not target or color is None:
        return "skipped recolor_object without type/color"
    count = 0
    for item in _list_field(plan, "objects"):
        if _object_type(item) == target:
            item["color"] = str(color)
            count += 1
    return f"recolored {count} '{target}' object(s) to {color}"


def _edit_move_object(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    target = str(edit.get("type", edit.get("kind", ""))).strip().lower()
    if not target:
        return "skipped move_object without type"
    count = 0
    for item in _list_field(plan, "objects"):
        if _object_type(item) == target:
            if edit.get("x") is not None:
                item["x"] = _clamp01(_to_float(edit.get("x"), 0.5))
            if edit.get("y") is not None:
                item["y"] = _clamp01(_to_float(edit.get("y"), 0.5))
            count += 1
    return f"moved {count} '{target}' object(s)"


def _edit_resize_object(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    target = str(edit.get("type", edit.get("kind", ""))).strip().lower()
    if not target or edit.get("size") is None:
        return "skipped resize_object without type/size"
    count = 0
    for item in _list_field(plan, "objects"):
        if _object_type(item) == target:
            item["size"] = _clamp01(_to_float(edit.get("size"), 0.18))
            count += 1
    return f"resized {count} '{target}' object(s)"


def _edit_set_opacity(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    target = str(edit.get("type", edit.get("kind", ""))).strip().lower()
    if not target or edit.get("opacity") is None:
        return "skipped set_opacity without type/opacity"
    count = 0
    for item in _list_field(plan, "objects"):
        if _object_type(item) == target:
            item["opacity"] = _clamp01(_to_float(edit.get("opacity"), 1.0))
            count += 1
    return f"set opacity on {count} '{target}' object(s)"


def _edit_set_style(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    field = str(edit.get("field", "")).strip().lower()
    if not field or edit.get("value") is None:
        return "skipped set_style without field/value"
    style = plan.get("style")
    if not isinstance(style, dict):
        style = {}
        plan["style"] = style
    style[field] = _clamp01(_to_float(edit.get("value"), 0.0))
    return f"set style {field}={style[field]:.3f}"


def _edit_adjust_style(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    field = str(edit.get("field", "")).strip().lower()
    if not field or edit.get("delta") is None:
        return "skipped adjust_style without field/delta"
    style = plan.get("style")
    if not isinstance(style, dict):
        style = {}
        plan["style"] = style
    current = _to_float(style.get(field, 0.0), 0.0)
    style[field] = _clamp01(current + _to_float(edit.get("delta"), 0.0))
    return f"adjusted style {field} to {style[field]:.3f}"


def _edit_set_palette(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    colors = edit.get("colors")
    if not isinstance(colors, list) or not colors:
        return "skipped set_palette without colors"
    plan["palette"] = [str(color) for color in colors]
    return f"set palette to {len(plan['palette'])} colors"


def _edit_add_element(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    element = edit.get("element")
    if not isinstance(element, dict):
        return "skipped add_element without element object"
    _list_field(plan, "elements").append(dict(element))
    return f"added element '{str(element.get('type', 'shape'))}'"


def _edit_add_cloud(plan: dict[str, Any], edit: dict[str, Any]) -> str:
    cloud = edit.get("cloud")
    if isinstance(cloud, dict):
        _list_field(plan, "clouds").append(dict(cloud))
        return "added cloud bank"
    _list_field(plan, "clouds").append(
        {
            "type": "cumulus",
            "label": "critique-added cloud bank",
            "region": [0.08, 0.08, 0.94, 0.34],
            "color": str(edit.get("color", "#fff1dd")),
            "shadow": "#788ca8",
            "opacity": _clamp01(_to_float(edit.get("opacity", 0.4), 0.4)),
            "blur": 0.026,
            "count": 4,
            "lobes": 5,
            "scale": 0.12,
            "blend": "screen",
            "seed": 113,
            "z": 6,
        }
    )
    return "added default cloud bank"


def _apply_element_check_edits(plan: dict[str, Any], checks: tuple[dict[str, Any], ...]) -> list[str]:
    actions: list[str] = []
    for check in checks:
        kind = str(check.get("kind") or "").strip().lower()
        item = str(check.get("item") or "").strip().lower()
        if not kind or not item or not _element_check_failed(check):
            continue
        if kind == "object":
            message = _apply_element_object_gap(plan, item)
            if message:
                actions.append(f"element_check: {message}")
        elif kind == "color":
            message = _apply_element_color_gap(plan, item)
            if message:
                actions.append(f"element_check: {message}")
    return actions


def _apply_element_object_gap(plan: dict[str, Any], item: str) -> str:
    if item == "cloud":
        return _edit_add_cloud(plan, {})
    return _edit_add_object(
        plan,
        {
            "type": item,
            "label": f"critique-check-added {item}",
            "x": 0.5,
            "y": 0.45,
            "size": 0.22,
            "opacity": 1.0,
            "color": _color_hex(item),
        },
    )


def _apply_element_color_gap(plan: dict[str, Any], item: str) -> str:
    style = plan.get("style")
    if not isinstance(style, dict):
        style = {}
        plan["style"] = style
    style["saturation"] = _clamp01(_to_float(style.get("saturation", 0.35), 0.35) + 0.16)
    style["contrast"] = _clamp01(_to_float(style.get("contrast", 0.35), 0.35) + 0.08)

    palette = _list_field(plan, "palette")
    color = _color_hex(item)
    if color not in palette:
        palette.append(color)
    return f"strengthened checked color '{item}'"


def _element_check_failed(check: dict[str, Any]) -> bool:
    present = _bool_or_none(check.get("present"))
    if present is False:
        return True
    confidence = _to_float(check.get("confidence"), 1.0)
    return confidence < 0.5


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


def _color_hex(name: str) -> str:
    rgb = COLOR_RGB.get(name)
    if rgb is None:
        return "#f2f2f2"
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


_EDIT_HANDLERS = {
    "add_object": _edit_add_object,
    "remove_object": _edit_remove_object,
    "recolor_object": _edit_recolor_object,
    "move_object": _edit_move_object,
    "resize_object": _edit_resize_object,
    "set_opacity": _edit_set_opacity,
    "set_style": _edit_set_style,
    "adjust_style": _edit_adjust_style,
    "set_palette": _edit_set_palette,
    "add_element": _edit_add_element,
    "add_cloud": _edit_add_cloud,
}


# --- parsing helpers -----------------------------------------------------------------


def _load(source: Path | str | dict[str, Any]) -> Any:
    if isinstance(source, dict):
        return source
    text = source
    if isinstance(source, Path) or (isinstance(source, str) and _looks_like_path(source)):
        path = Path(source)
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")
        elif isinstance(source, Path):
            raise FileNotFoundError(f"Critique file not found: {source}")
    return json.loads(text)


def _looks_like_path(value: str) -> bool:
    stripped = value.strip()
    return not stripped.startswith("{") and (stripped.endswith(".json") or "/" in stripped or "\\" in stripped)


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _str_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _visual_checklist(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    missing_objects = set(_str_list(metadata.get("caption_missing_objects")))
    missing_colors = set(_str_list(metadata.get("caption_missing_colors")))
    checklist: list[dict[str, Any]] = []

    for item in sorted(set(_str_list(metadata.get("objects")))):
        checklist.append(
            {
                "kind": "object",
                "item": item,
                "question": f"Does the image clearly show the requested object: {item}?",
                "caption_backcheck": "missing" if item in missing_objects else "unknown",
                "priority": "high" if item in missing_objects else "normal",
            }
        )

    for item in sorted(set(_str_list(metadata.get("color_words")))):
        checklist.append(
            {
                "kind": "color",
                "item": item,
                "question": f"Is the requested color visually present and attached to the right subject: {item}?",
                "caption_backcheck": "missing" if item in missing_colors else "unknown",
                "priority": "high" if item in missing_colors else "normal",
            }
        )

    return sorted(
        checklist,
        key=lambda check: (
            0 if check["kind"] == "object" else 1,
            str(check["item"]),
        ),
    )


def _expected_element_checks(checklist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(item.get("kind", "")),
            "item": str(item.get("item", "")),
            "present": None,
            "confidence": None,
            "notes": "",
        }
        for item in checklist
    ]


def _element_check_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    checks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        check = dict(item)
        if "confidence" in check and check["confidence"] is not None:
            check["confidence"] = _clamp01(_to_float(check["confidence"], 0.0))
        checks.append(check)
    return tuple(checks)


def _edit_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
