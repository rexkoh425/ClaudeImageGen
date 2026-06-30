import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


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
    progress_path = output_dir / "progress.csv"
    pixels_path = output_dir / "pixels.csv"
    candidates_path = output_dir / "candidates.json"

    assert image_path.exists()
    assert metadata_path.exists()
    assert progress_path.exists()
    assert pixels_path.exists()
    assert candidates_path.exists()
    assert "image.png" in completed.stdout
    assert "Caption" in completed.stdout

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

    with progress_path.open(newline="", encoding="utf-8") as handle:
        progress_rows = list(csv.DictReader(handle))
    assert progress_rows

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert len(candidates) == 2
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
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["refined_from"] == str(base_dir)
    assert metadata["parent_image"] == str(base_dir / "image.png")
    assert metadata["parent_metadata"] == str(base_dir / "metadata.json")
    assert metadata["initial_image"] == str(base_dir / "image.png")
    assert metadata["initial_similarity_score"] is not None
    assert metadata["refinement_lineage_depth"] == 1
    assert metadata["scene_plan_refine_actions"]
