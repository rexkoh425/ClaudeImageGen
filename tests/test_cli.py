import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from claude_imagegen import verify as verify_module
from claude_imagegen.cli import build_parser


def test_cli_accepts_siglip_similarity_backend_options():
    parser = build_parser()

    generate_args = parser.parse_args(
        [
            "generate",
            "--prompt",
            "red sun over blue ocean",
            "--output-dir",
            "out",
            "--similarity-backend",
            "transformers-siglip",
            "--similarity-model",
            "google/siglip-base-patch16-224",
            "--continuity-backend",
            "transformers-dinov2",
            "--continuity-model",
            "facebook/dinov2-base",
            "--continuity-device",
            "cuda",
            "--caption-similarity-backend",
            "transformers-sentence",
            "--caption-similarity-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--caption-similarity-device",
            "cuda",
        ]
    )
    assert generate_args.similarity_backend == "transformers-siglip"
    assert generate_args.similarity_model == "google/siglip-base-patch16-224"
    assert generate_args.continuity_backend == "transformers-dinov2"
    assert generate_args.continuity_model == "facebook/dinov2-base"
    assert generate_args.continuity_device == "cuda"
    assert generate_args.caption_similarity_backend == "transformers-sentence"
    assert generate_args.caption_similarity_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert generate_args.caption_similarity_device == "cuda"

    refine_args = parser.parse_args(
        [
            "refine",
            "--from-dir",
            "base",
            "--prompt",
            "add clouds",
            "--output-dir",
            "refined",
            "--similarity-backend",
            "transformers-siglip",
            "--continuity-backend",
            "transformers-dinov2",
            "--caption-similarity-backend",
            "transformers-sentence",
        ]
    )
    assert refine_args.similarity_backend == "transformers-siglip"
    assert refine_args.continuity_backend == "transformers-dinov2"
    assert refine_args.caption_similarity_backend == "transformers-sentence"

    verify_args = parser.parse_args(
        [
            "verify",
            "--strong-model",
            "--strong-similarity-backend",
            "transformers-siglip",
            "--similarity-model",
            "google/siglip-base-patch16-224",
            "--strong-continuity-backend",
            "transformers-dinov2",
            "--continuity-model",
            "facebook/dinov2-base",
            "--caption-similarity-backend",
            "transformers-sentence",
            "--caption-similarity-model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--strong-size",
            "96x64",
            "--strong-size",
            "144x96",
        ]
    )
    assert verify_args.strong_similarity_backend == "transformers-siglip"
    assert verify_args.similarity_model == "google/siglip-base-patch16-224"
    assert verify_args.strong_continuity_backend == "transformers-dinov2"
    assert verify_args.continuity_model == "facebook/dinov2-base"
    assert verify_args.caption_similarity_backend == "transformers-sentence"
    assert verify_args.caption_similarity_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert verify_args.strong_sizes == [(96, 64), (144, 96)]


def test_cli_generate_writes_image_metadata_progress_and_optional_pixels(tmp_path: Path):
    output_dir = tmp_path / "generated"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red sun over blue ocean",
            "--output-dir",
            str(output_dir),
            "--width",
            "80",
            "--height",
            "48",
            "--max-iterations",
            "6",
            "--threshold",
            "0.99",
            "--seed",
            "11",
            "--caption-backend",
            "local",
            "--caption-device",
            "cpu",
            "--save-candidates",
            "2",
            "--pixel-csv",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"
    quality_path = output_dir / "quality-report.json"
    progress_path = output_dir / "progress.csv"
    pixels_path = output_dir / "pixels.csv"
    candidates_path = output_dir / "candidates.json"
    contact_sheet_path = output_dir / "candidates" / "contact-sheet.png"
    critique_request_path = output_dir / "critique-request.json"

    assert image_path.exists()
    assert metadata_path.exists()
    assert quality_path.exists()
    assert progress_path.exists()
    assert pixels_path.exists()
    assert candidates_path.exists()
    assert contact_sheet_path.exists()
    assert critique_request_path.exists()
    assert "image.png" in completed.stdout
    assert "Caption" in completed.stdout
    assert "Quality" in completed.stdout
    assert "Critique request" in completed.stdout

    with Image.open(image_path) as image:
        assert image.mode == "RGB"
        assert image.size == (80, 48)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt"] == "red sun over blue ocean"
    assert metadata["width"] == 80
    assert metadata["height"] == 48
    assert metadata["iterations"] >= 1
    assert metadata["caption_backend"] == "local"
    assert metadata["effective_caption_device"] == "cpu"
    assert "sun" in metadata["image_caption"]
    assert "ocean" in metadata["image_caption"]
    assert metadata["caption_similarity_score"] > 0.15
    assert metadata["candidate_count"] == 2
    assert metadata["candidate_index"] == str(candidates_path)
    assert metadata["candidate_contact_sheet"] == str(contact_sheet_path)
    assert metadata["quality_report"] == str(quality_path)
    assert metadata["critique_request"] == str(critique_request_path)
    assert metadata["quality_status"] in {"pass", "review", "revise"}
    assert 0.0 <= metadata["quality_score"] <= 1.0

    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality_report["status"] == metadata["quality_status"]
    assert quality_report["quality_score"] == metadata["quality_score"]

    critique_request = json.loads(critique_request_path.read_text(encoding="utf-8"))
    assert critique_request["image"] == str(image_path)
    assert critique_request["metadata"] == str(metadata_path)
    assert critique_request["quality_report"] == str(quality_path)
    assert critique_request["expected_response"]["verdict"] == "revise"
    assert "add_cloud" in critique_request["allowed_edit_actions"]

    with progress_path.open(newline="", encoding="utf-8") as handle:
        progress_rows = list(csv.DictReader(handle))
    assert progress_rows

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert len(candidates) == 2
    assert all(candidate["caption"] for candidate in candidates)
    assert all("caption_similarity_score" in candidate for candidate in candidates)
    assert all((output_dir / candidate["image"]).exists() or Path(candidate["image"]).exists() for candidate in candidates)

    with pixels_path.open(newline="", encoding="utf-8") as handle:
        pixel_rows = list(csv.reader(handle))
    assert pixel_rows[0] == ["x", "y", "r", "g", "b"]
    assert len(pixel_rows) == (80 * 48) + 1


def test_cli_generate_accepts_scene_plan_file(tmp_path: Path):
    output_dir = tmp_path / "planned"
    plan_path = tmp_path / "scene-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "title": "CLI planned scene",
                "palette": ["#102040", "#ff5533"],
                "background": {"top": "#102040", "bottom": "#205080"},
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.18, "color": "#ff5533"},
                    {"type": "foreground", "y": 0.75, "color": "#123d2a"},
                ],
            }
        ),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red sun over detailed green foreground",
            "--scene-plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
            "--width",
            "120",
            "--height",
            "80",
            "--max-iterations",
            "3",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["scene_plan_used"] is True
    assert metadata["scene_plan_title"] == "CLI planned scene"


def test_cli_refine_uses_previous_output_as_initial_image(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    base_dir = tmp_path / "base"
    base_plan = base_dir / "scene-plan.json"
    base_dir.mkdir()
    base_plan.write_text(
        json.dumps(
            {
                "title": "Base refinable scene",
                "palette": ["#102040", "#ff5533", "#286fc4"],
                "background": {"top": "#102040", "bottom": "#205080"},
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.18, "color": "#ff5533"},
                    {"type": "ocean", "y": 0.58, "color": "#286fc4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red sun over blue ocean",
            "--scene-plan",
            str(base_plan),
            "--output-dir",
            str(base_dir),
            "--width",
            "120",
            "--height",
            "80",
            "--max-iterations",
            "2",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    refined_dir = tmp_path / "refined"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "refine",
            "--from-dir",
            str(base_dir),
            "--prompt",
            "red sun over blue ocean with clouds",
            "--output-dir",
            str(refined_dir),
            "--max-iterations",
            "2",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "Refined" in completed.stdout
    assert "Critique request" in completed.stdout
    assert "Refinement delta" in completed.stdout
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    quality_report = json.loads((refined_dir / "quality-report.json").read_text(encoding="utf-8"))
    assert metadata["refined_from"] == str(base_dir)
    assert metadata["parent_image"] == str(base_dir / "image.png")
    assert metadata["parent_metadata"] == str(base_dir / "metadata.json")
    assert metadata["initial_image"] == str(base_dir / "image.png")
    assert metadata["initial_similarity_score"] is not None
    assert metadata["parent_quality_score"] is not None
    assert metadata["refinement_lineage_depth"] == 1
    assert metadata["scene_plan_refine_actions"]
    assert metadata["quality_report"] == str(refined_dir / "quality-report.json")
    assert metadata["critique_request"] == str(refined_dir / "critique-request.json")
    critique_request = json.loads((refined_dir / "critique-request.json").read_text(encoding="utf-8"))
    assert critique_request["image"] == str(refined_dir / "image.png")
    assert critique_request["metadata"] == str(refined_dir / "metadata.json")
    assert critique_request["quality_report"] == str(refined_dir / "quality-report.json")
    assert "continuity" in {check["name"] for check in quality_report["checks"]}
    assert quality_report["continuity_score"] == metadata["initial_similarity_score"]
    assert quality_report["refinement_delta"] == metadata["refinement_delta"]
    assert metadata["refinement_delta"]["parent_total_score"] == metadata["parent_total_score"]
    assert metadata["refinement_delta"]["current_total_score"] == metadata["total_score"]
    assert metadata["refinement_delta"]["total_score_delta"] == round(
        metadata["total_score"] - metadata["parent_total_score"], 6
    )
    assert metadata["refinement_delta"]["parent_quality_score"] == metadata["parent_quality_score"]
    assert metadata["refinement_delta"]["current_quality_score"] == metadata["quality_score"]
    assert metadata["refinement_delta"]["quality_score_delta"] == round(
        metadata["quality_score"] - metadata["parent_quality_score"], 6
    )
    assert metadata["refinement_delta"]["continuity_score"] == metadata["initial_similarity_score"]


def test_cli_refine_records_and_applies_visual_critique(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    base_dir = tmp_path / "base-critique"
    base_plan = base_dir / "scene-plan.json"
    base_dir.mkdir()
    base_plan.write_text(
        json.dumps(
            {
                "title": "Critique refinable scene",
                "palette": ["#102040", "#ff5533", "#286fc4"],
                "background": {"top": "#102040", "bottom": "#205080"},
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.12, "color": "#ff5533"},
                    {"type": "ocean", "y": 0.58, "color": "#286fc4"},
                ],
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red sun over blue ocean",
            "--scene-plan",
            str(base_plan),
            "--output-dir",
            str(base_dir),
            "--width",
            "120",
            "--height",
            "80",
            "--max-iterations",
            "2",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    critique_path = tmp_path / "critique.json"
    critique_path.write_text(
        json.dumps(
            {
                "closeness_score": 0.41,
                "verdict": "revise",
                "summary": "The sun and ocean are present, but clouds are missing.",
                "present": ["sun", "ocean"],
                "missing": ["clouds"],
                "edits": [
                    {"action": "add_cloud", "color": "#fff1dd"},
                    {"action": "resize_object", "type": "sun", "size": 0.2},
                ],
            }
        ),
        encoding="utf-8",
    )

    refined_dir = tmp_path / "refined-critique"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "refine",
            "--from-dir",
            str(base_dir),
            "--prompt",
            "red sun over blue ocean",
            "--critique",
            str(critique_path),
            "--output-dir",
            str(refined_dir),
            "--max-iterations",
            "2",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "Critique revise closeness 0.41" in completed.stdout
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    refined_plan = json.loads((refined_dir / "scene-plan.json").read_text(encoding="utf-8"))
    quality_report = json.loads((refined_dir / "quality-report.json").read_text(encoding="utf-8"))

    critique = metadata["visual_critique"]
    assert critique["judge"] == "claude-vision"
    assert critique["closeness_score"] == 0.41
    assert critique["verdict"] == "revise"
    assert critique["missing"] == ["clouds"]
    assert any("added default cloud bank" in action for action in critique["applied_edits"])
    assert any("critique: added default cloud bank" in action for action in metadata["scene_plan_refine_actions"])
    assert len(refined_plan["clouds"]) == 1
    assert refined_plan["objects"][0]["size"] == 0.2
    assert "visual_judgement" in {check["name"] for check in quality_report["checks"]}
    assert any("Judge: add missing elements: clouds." == action for action in quality_report["next_actions"])


def test_cli_refine_can_start_from_saved_candidate_rank(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    base_dir = tmp_path / "base-candidates"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red robot portrait over blue ocean with clouds",
            "--output-dir",
            str(base_dir),
            "--width",
            "120",
            "--height",
            "80",
            "--max-iterations",
            "4",
            "--threshold",
            "0.99",
            "--save-candidates",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    candidates = json.loads((base_dir / "candidates.json").read_text(encoding="utf-8"))
    selected = candidates[1]
    selected_image = Path(selected["image"])

    refined_dir = tmp_path / "refined-from-candidate"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "refine",
            "--from-dir",
            str(base_dir),
            "--candidate-rank",
            "2",
            "--prompt",
            "red robot portrait over blue ocean with brighter clouds",
            "--output-dir",
            str(refined_dir),
            "--max-iterations",
            "1",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "Candidate rank 2" in completed.stdout
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parent_candidate_rank"] == 2
    assert metadata["parent_candidate_image"] == str(selected_image)
    assert metadata["parent_candidate_iteration"] == selected["iteration"]
    assert metadata["parent_candidate_total_score"] == selected["total_score"]
    assert metadata["parent_candidate_caption"] == selected["caption"]
    assert metadata["parent_image"] == str(selected_image)
    assert metadata["initial_image"] == str(selected_image)


def test_cli_refine_can_auto_select_saved_candidate_rank(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    base_dir = tmp_path / "base-auto-candidates"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "generate",
            "--prompt",
            "red robot portrait over blue ocean with clouds",
            "--output-dir",
            str(base_dir),
            "--width",
            "120",
            "--height",
            "80",
            "--max-iterations",
            "4",
            "--threshold",
            "0.99",
            "--save-candidates",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    candidates_path = base_dir / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates[0]["selection_score"] = 0.10
    candidates[0]["selection_reasons"] = ["forced lower score for test"]
    candidates[1]["selection_score"] = 0.98
    candidates[1]["selection_reasons"] = ["forced higher score for test"]
    candidates_path.write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    selected = candidates[1]
    selected_image = Path(selected["image"])

    refined_dir = tmp_path / "refined-auto-candidate"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "refine",
            "--from-dir",
            str(base_dir),
            "--candidate-rank",
            "auto",
            "--prompt",
            "red robot portrait over blue ocean with brighter clouds",
            "--output-dir",
            str(refined_dir),
            "--max-iterations",
            "1",
            "--threshold",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    assert "Candidate rank 2" in completed.stdout
    assert "Candidate selection auto" in completed.stdout
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["parent_candidate_selection"] == "auto"
    assert metadata["parent_candidate_rank"] == 2
    assert metadata["parent_candidate_image"] == str(selected_image)
    assert metadata["parent_candidate_selection_score"] == 0.98
    assert metadata["parent_candidate_selection_reasons"] == ["forced higher score for test"]
    assert metadata["parent_image"] == str(selected_image)
    assert metadata["initial_image"] == str(selected_image)


def test_cli_verify_runs_size_and_refine_smoke_suite(tmp_path: Path):
    output_dir = tmp_path / "verify-suite"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "verify",
            "--output-dir",
            str(output_dir),
            "--size",
            "80x48",
            "--size",
            "128x72",
            "--max-iterations",
            "2",
            "--threshold",
            "0.99",
            "--save-candidates",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )

    report_path = output_dir / "verification-report.json"
    assert "Verification" in completed.stdout
    assert str(report_path) in completed.stdout
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert [case["size"] for case in report["cases"] if case["type"] == "generate"] == ["80x48", "128x72"]
    complex_case = next(case for case in report["cases"] if case["type"] == "complex-plan")
    assert complex_case["scene_plan_used"] is True
    assert complex_case["scene_plan_feature_count"] >= 12
    assert complex_case["scene_plan_material_count"] >= 1
    assert complex_case["scene_plan_terrain_count"] >= 1
    assert complex_case["scene_plan_reflection_count"] >= 1
    assert complex_case["scene_plan_warp_count"] >= 1
    assert complex_case["scene_plan_beam_count"] >= 1
    assert complex_case["scene_plan_cloud_count"] >= 1
    assert complex_case["scene_plan_shadow_count"] >= 1
    assert complex_case["scene_plan_focus_used"] is True
    assert any(case["type"] == "refine" for case in report["cases"])
    assert report["strong_model"] == "not-requested"

    for case in report["cases"]:
        case_dir = Path(case["output_dir"])
        assert (case_dir / "image.png").exists()
        assert (case_dir / "metadata.json").exists()
        assert (case_dir / "quality-report.json").exists()
        assert (case_dir / "critique-request.json").exists()
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        assert case["critique_request"] == str(case_dir / "critique-request.json")
        assert case["caption_backend"] == metadata["caption_backend"]
        assert case["caption_model"] == metadata["caption_model"]
        if case["type"] == "generate":
            assert f"{metadata['width']}x{metadata['height']}" == case["size"]
            assert (case_dir / "candidates.json").exists()
            assert (case_dir / "candidates" / "contact-sheet.png").exists()
        if case["type"] == "complex-plan":
            assert (case_dir / "scene-plan.json").exists()
            assert metadata["scene_plan_used"] is True
            assert metadata["scene_plan_material_count"] >= 1
            assert metadata["scene_plan_focus_used"] is True
        if case["type"] == "refine":
            assert metadata["parent_candidate_selection"] == "auto"
            assert metadata["initial_similarity_score"] is not None
            assert case["refinement_delta"] == metadata["refinement_delta"]


def test_verification_runs_strong_cases_for_explicit_strong_sizes(tmp_path: Path, monkeypatch):
    def fake_result(output_dir: Path, *, width: int, height: int, options: object, refined: bool = False):
        output_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (width, height), (32, 64, 128))
        image_path = output_dir / "image.png"
        metadata_path = output_dir / "metadata.json"
        progress_path = output_dir / "progress.csv"
        quality_path = output_dir / "quality-report.json"
        critique_path = output_dir / "critique-request.json"
        image.save(image_path)
        progress_path.write_text("iteration,total_score\n1,0.5\n", encoding="utf-8")
        quality_path.write_text("{}", encoding="utf-8")
        critique_path.write_text("{}", encoding="utf-8")

        similarity_backend = getattr(options, "similarity_backend", "local")
        continuity_backend = getattr(options, "continuity_backend", None) or similarity_backend
        caption_backend = getattr(options, "caption_backend", "local")
        caption_similarity_backend = getattr(options, "caption_similarity_backend", "local")
        effective_device = "cuda" if similarity_backend != "local" else "cpu"
        metadata = {
            "width": width,
            "height": height,
            "quality_report": str(quality_path),
            "quality_status": "review",
            "quality_score": 0.5,
            "total_score": 0.5,
            "caption_backend": caption_backend,
            "caption_model": getattr(options, "caption_model", None),
            "caption_similarity_score": 0.7,
            "caption_similarity_backend": caption_similarity_backend,
            "caption_similarity_model": getattr(options, "caption_similarity_model", None),
            "effective_caption_similarity_device": effective_device,
            "initial_similarity_score": 0.91 if refined else None,
            "refinement_delta": {"total_score_delta": 0.02} if refined else None,
            "similarity_backend": similarity_backend,
            "similarity_model": getattr(options, "similarity_model", None),
            "continuity_backend": continuity_backend,
            "continuity_model": getattr(options, "continuity_model", None),
            "effective_similarity_device": effective_device,
            "effective_continuity_device": effective_device,
            "effective_caption_device": effective_device if caption_backend != "local" else "cpu",
            "parent_candidate_selection": "auto" if getattr(options, "candidate_rank", None) == "auto" else None,
            "candidate_count": 0,
            "scene_plan_used": output_dir.name.startswith("complex-plan"),
            "scene_plan_background_stop_count": 1,
            "scene_plan_element_count": 1,
            "scene_plan_gradient_count": 1,
            "scene_plan_motif_count": 1,
            "scene_plan_texture_count": 1,
            "scene_plan_material_count": 1,
            "scene_plan_terrain_count": 1,
            "scene_plan_reflection_count": 1,
            "scene_plan_warp_count": 1,
            "scene_plan_veil_count": 1,
            "scene_plan_light_count": 1,
            "scene_plan_beam_count": 1,
            "scene_plan_cloud_count": 1,
            "scene_plan_shadow_count": 1,
            "scene_plan_atmosphere_used": True,
            "scene_plan_focus_used": True,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return SimpleNamespace(
            image=image,
            metadata=metadata,
            image_path=image_path,
            metadata_path=metadata_path,
            progress_path=progress_path,
            pixels_path=None,
            candidates_path=None,
        )

    def fake_generate(options):
        return fake_result(options.output_dir, width=options.width, height=options.height, options=options)

    def fake_refine(options):
        parent_metadata = json.loads((options.from_dir / "metadata.json").read_text(encoding="utf-8"))
        width = options.width or int(parent_metadata["width"])
        height = options.height or int(parent_metadata["height"])
        return fake_result(options.output_dir, width=width, height=height, options=options, refined=True)

    monkeypatch.setattr(verify_module, "generate_image", fake_generate)
    monkeypatch.setattr(verify_module, "refine_image", fake_refine)

    report = verify_module.run_verification(
        verify_module.VerifyOptions(
            output_dir=tmp_path / "verify-strong-sizes",
            sizes=((80, 48),),
            max_iterations=1,
            threshold=0.1,
            save_candidates=1,
            strong_model=True,
            strong_similarity_backend="transformers-siglip",
            strong_continuity_backend="transformers-dinov2",
            caption_similarity_backend="transformers-sentence",
            strong_sizes=((96, 64), (144, 96)),
        )
    )

    assert report["status"] == "pass"
    assert report["strong_model"] == "pass"
    assert report["strong_sizes"] == ["96x64", "144x96"]
    assert [case["size"] for case in report["cases"] if case["type"] == "strong-model"] == ["96x64", "144x96"]
    assert [case["size"] for case in report["cases"] if case["type"] == "strong-continuity"] == ["96x64", "144x96"]
    strong_cases = [case for case in report["cases"] if case["type"].startswith("strong")]
    assert all(case["similarity_backend"] == "transformers-siglip" for case in strong_cases)
    assert all(case["effective_similarity_device"] == "cuda" for case in strong_cases)
