import json
import subprocess
import sys
from pathlib import Path


def _write_pair_response(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_scores": [
                    {
                        "id": "greenhouse-existing-best",
                        "before_score": 0.71,
                        "after_score": 0.86,
                        "detail_score": 0.87,
                        "winner": "after",
                        "parity_boolean": False,
                        "failure_modes": [
                            "night-mood drift from over-brightening",
                            "haze/bloom softening reduces mid-tone and floor contrast",
                            "near-clipping lamp highlights",
                        ],
                        "recommended_code_changes": [
                            "add a night-luminance ceiling",
                            "parameterize and cap bloom + mist-veil strength",
                        ],
                    }
                ],
                "best_pair_id": "greenhouse-existing-best",
                "best_after_image": "claude-imagegen-output\\postprocess-rays-2\\rays-balanced-final.png",
                "gpt_sora_parity_score": 0.83,
                "gpt_sora_parity_boolean": False,
                "acceptance_gate_met": False,
                "overall_failure_modes": [
                    "night-mood drift from over-brightening",
                    "haze/bloom softening reduces mid-tone and floor contrast",
                    "near-clipping lamp highlights",
                ],
                "code_improvement_recommendations": [
                    "add a night-luminance ceiling",
                    "parameterize and cap bloom + mist-veil strength",
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_accepting_pair_response(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "pair_scores": [
                    {
                        "id": "greenhouse-existing-best",
                        "before_score": 0.8,
                        "after_score": 0.9,
                        "detail_score": 0.89,
                        "winner": "after",
                        "parity_boolean": True,
                        "failure_modes": [
                            "slight loss of the 'deep night' mood due to global brightening",
                            "mild lamp-halo bloom that softens some fine leaf edges near the light sources",
                        ],
                    }
                ],
                "best_pair_id": "greenhouse-existing-best",
                "best_after_image": "claude-imagegen-output\\postprocess-rays-2\\rays-balanced-final.png",
                "gpt_sora_parity_score": 0.9,
                "gpt_sora_parity_boolean": True,
                "acceptance_gate_met": True,
                "overall_failure_modes": [
                    "global brightening trades away some deep-night mood",
                    "local lamp bloom slightly softens nearby high-frequency detail",
                ],
                "code_improvement_recommendations": [
                    "add a night-preservation exposure clamp",
                    "use masked local tone-mapping for emitters",
                ],
            }
        ),
        encoding="utf-8",
    )


def test_pair_evaluation_plan_recommends_dark_preserving_next_step(tmp_path: Path):
    from claude_imagegen.eval_plan import EvalPlanOptions, build_eval_plan

    response_path = tmp_path / "pair-response.json"
    _write_pair_response(response_path)

    result = build_eval_plan(
        EvalPlanOptions(
            evaluation=response_path,
            prompt="deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "plan",
            quality_target=0.9,
        )
    )

    assert result.plan_path.exists()
    assert result.plan["target_quality_met"] is False
    assert result.plan["best_after_score"] == 0.86
    assert result.plan["score_gap"] == 0.04
    assert result.plan["next_action"] == "enhance-night"
    assert result.plan["suggested_parameters"] == {
        "night_luma_ceiling": 0.3,
        "mist_cap": 0.16,
        "highlight_rolloff": 0.25,
        "local_contrast": 1.05,
    }
    assert result.plan["best_after_image"] == "claude-imagegen-output\\postprocess-rays-2\\rays-balanced-final.png"
    assert "claude-imagegen enhance-night" in result.plan["recommended_command"]
    assert "--night-luma-ceiling 0.3" in result.plan["recommended_command"]
    assert "Do not accept" in result.plan["acceptance_reason"]


def test_pair_evaluation_plan_requires_consensus_across_multiple_claude_judges(tmp_path: Path):
    from claude_imagegen.eval_plan import EvalPlanOptions, build_eval_plan

    strict_response = tmp_path / "strict-response.json"
    accepting_response = tmp_path / "accepting-response.json"
    _write_pair_response(strict_response)
    _write_accepting_pair_response(accepting_response)

    result = build_eval_plan(
        EvalPlanOptions(
            evaluations=(strict_response, accepting_response),
            prompt="deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "plan",
            quality_target=0.9,
        )
    )

    assert result.plan["evaluation_count"] == 2
    assert result.plan["target_quality_met"] is False
    assert result.plan["acceptance_consensus_met"] is False
    assert result.plan["best_after_score"] == 0.86
    assert result.plan["after_score_median"] == 0.88
    assert result.plan["after_score_max"] == 0.9
    assert result.plan["gpt_sora_parity_boolean"] is False
    assert result.plan["next_action"] == "enhance-night"
    assert "multiple Claude evaluations disagree" in result.plan["acceptance_reason"]


def test_pair_evaluation_plan_requires_minimum_judge_count_for_acceptance(tmp_path: Path):
    from claude_imagegen.eval_plan import EvalPlanOptions, build_eval_plan

    accepting_response = tmp_path / "accepting-response.json"
    _write_accepting_pair_response(accepting_response)

    result = build_eval_plan(
        EvalPlanOptions(
            evaluation=accepting_response,
            prompt="deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "plan",
            quality_target=0.9,
        )
    )

    assert result.plan["evaluation_count"] == 1
    assert result.plan["minimum_evaluations_required"] == 2
    assert result.plan["target_quality_met"] is False
    assert result.plan["acceptance_consensus_met"] is False
    assert result.plan["best_after_score"] == 0.9
    assert result.plan["next_action"] == "enhance-night"
    assert "at least 2 Claude evaluations" in result.plan["acceptance_reason"]


def test_cli_eval_plan_writes_improvement_plan_without_images(tmp_path: Path):
    response_path = tmp_path / "pair-response.json"
    _write_pair_response(response_path)
    output_dir = tmp_path / "plan"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "eval-plan",
            "--evaluation",
            str(response_path),
            "--prompt",
            "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections",
            "--output-dir",
            str(output_dir),
            "--quality-target",
            "0.9",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan_path = output_dir / "improvement-plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["next_action"] == "enhance-night"
    assert plan["target_quality_met"] is False
    assert "Improvement plan" in completed.stdout


def test_cli_eval_plan_accepts_multiple_evaluation_files_conservatively(tmp_path: Path):
    strict_response = tmp_path / "strict-response.json"
    accepting_response = tmp_path / "accepting-response.json"
    _write_pair_response(strict_response)
    _write_accepting_pair_response(accepting_response)
    output_dir = tmp_path / "plan"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "eval-plan",
            "--evaluation",
            str(strict_response),
            "--evaluation",
            str(accepting_response),
            "--prompt",
            "deep night glass greenhouse interior with lamps, mist, leaf detail, and wet floor reflections",
            "--output-dir",
            str(output_dir),
            "--quality-target",
            "0.9",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output_dir / "improvement-plan.json").read_text(encoding="utf-8"))
    assert plan["evaluation_count"] == 2
    assert plan["acceptance_consensus_met"] is False
    assert plan["best_after_score"] == 0.86
