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
            "--quality-target",
            "0.9",
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
    assert generate_args.quality_target == 0.9
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
            "--quality-target",
            "0.9",
            "--similarity-backend",
            "transformers-siglip",
            "--continuity-backend",
            "transformers-dinov2",
            "--caption-similarity-backend",
            "transformers-sentence",
        ]
    )
    assert refine_args.similarity_backend == "transformers-siglip"
    assert refine_args.quality_target == 0.9
    assert refine_args.continuity_backend == "transformers-dinov2"
    assert refine_args.caption_similarity_backend == "transformers-sentence"

    comparison_refine_args = parser.parse_args(
        [
            "refine",
            "--from-dir",
            "base",
            "--prompt",
            "add clouds",
            "--output-dir",
            "refined",
            "--comparison",
            "comparison.json",
        ]
    )
    assert str(comparison_refine_args.comparison) == "comparison.json"

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

    setup_args = parser.parse_args(["setup"])
    assert setup_args.command == "setup"

    setup_diffusion_args = parser.parse_args(["setup", "--with-diffusion"])
    assert setup_diffusion_args.command == "setup"
    assert setup_diffusion_args.with_diffusion is True

    diffuse_args = parser.parse_args(
        [
            "diffuse",
            "--prompt",
            "photoreal glass greenhouse interior at night",
            "--output-dir",
            "out",
            "--initial-image",
            "input.png",
            "--width",
            "1024",
            "--height",
            "768",
            "--seeds",
            "101,202",
            "--steps",
            "4",
            "--strength",
            "0.28",
            "--device",
            "cuda",
            "--quality-target",
            "0.9",
        ]
    )
    assert diffuse_args.command == "diffuse"
    assert diffuse_args.initial_image == Path("input.png")
    assert diffuse_args.seeds == (101, 202)
    assert diffuse_args.strength == 0.28
    assert diffuse_args.device == "cuda"
    assert diffuse_args.quality_target == 0.9


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
            "--quality-target",
            "0.9",
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
    assert 0.0 <= metadata["recommended_candidate_aesthetic_score"] <= 1.0
    assert metadata["quality_report"] == str(quality_path)
    assert metadata["critique_request"] == str(critique_request_path)
    assert metadata["quality_status"] in {"pass", "review", "revise"}
    assert 0.0 <= metadata["quality_score"] <= 1.0
    assert metadata["quality_target"] == 0.9
    assert metadata["target_quality_met"] is False
    assert 0.0 <= metadata["image_detail_score"] <= 1.0
    assert metadata["image_detail_metrics"]["detail_score"] == metadata["image_detail_score"]

    quality_report = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality_report["status"] == metadata["quality_status"]
    assert quality_report["quality_score"] == metadata["quality_score"]
    assert quality_report["target_quality_met"] == metadata["target_quality_met"]
    assert quality_report["recommended_candidate_aesthetic_score"] == metadata["recommended_candidate_aesthetic_score"]

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
    assert all(0.0 <= candidate["aesthetic_score"] <= 1.0 for candidate in candidates)
    assert all(candidate["aesthetic_details"]["brightness_score"] >= 0.0 for candidate in candidates)
    assert all(any("aesthetic_score" in reason for reason in candidate["selection_reasons"]) for candidate in candidates)
    assert all((output_dir / candidate["image"]).exists() or Path(candidate["image"]).exists() for candidate in candidates)

    with pixels_path.open(newline="", encoding="utf-8") as handle:
        pixel_rows = list(csv.reader(handle))
    assert pixel_rows[0] == ["x", "y", "r", "g", "b"]
    assert len(pixel_rows) == (80 * 48) + 1


def test_diffusion_generation_writes_multi_seed_artifacts(tmp_path: Path, monkeypatch):
    from claude_imagegen import diffusion as diffusion_module
    from claude_imagegen.diffusion import DiffusionOptions, generate_diffusion_image

    class FakePipeline:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs):
            call_index = len(self.calls)
            self.calls.append(kwargs)
            seed = (101, 202)[call_index]
            width = int(kwargs["width"])
            height = int(kwargs["height"])
            if seed == 202:
                image = Image.new("RGB", (width, height), (9, 18, 20))
                for x in range(8, width - 8, 16):
                    for y in range(12, height - 8):
                        image.putpixel((x, y), (20, 140, 42))
                for x in range(width // 3, width // 3 + 8):
                    for y in range(height // 5, height // 5 + 8):
                        image.putpixel((x, y), (245, 168, 70))
                for x in range(0, width):
                    y = int((height * 0.62) + ((x - width / 2) * 0.12))
                    if 0 <= y < height:
                        image.putpixel((x, y), (215, 185, 125))
            else:
                image = Image.new("RGB", (width, height), (128, 128, 128))
                for x in range(0, width, 12):
                    for y in range(height):
                        image.putpixel((x, y), (180, 180, 180))
            return SimpleNamespace(images=[image])

    fake_pipeline = FakePipeline()

    def fake_load_pipeline(*, model: str, device: str):
        assert model == "stabilityai/sdxl-turbo"
        assert device == "cuda"
        return fake_pipeline, "cuda"

    monkeypatch.setattr(diffusion_module, "_load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(diffusion_module, "_torch_generator", lambda *, seed, device: None)

    output_dir = tmp_path / "diffusion"
    result = generate_diffusion_image(
        DiffusionOptions(
            prompt="photoreal glass greenhouse interior at night with tropical plants and tungsten lamps",
            output_dir=output_dir,
            width=97,
            height=65,
            seeds=(101, 202),
            device="cuda",
            quality_target=0.9,
        )
    )

    assert result.image_path == output_dir / "image.png"
    assert result.image_path.exists()
    assert result.metadata_path.exists()
    assert result.quality_report_path.exists()
    assert result.critique_request_path.exists()
    assert result.candidates_path.exists()
    assert result.contact_sheet_path.exists()
    assert len(fake_pipeline.calls) == 2
    assert all(
        call["prompt"] == "photoreal glass greenhouse interior at night with tropical plants and tungsten lamps"
        for call in fake_pipeline.calls
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["engine"] == "diffusers-text-to-image-v1"
    assert metadata["backend"] == "diffusers"
    assert metadata["model"] == "stabilityai/sdxl-turbo"
    assert metadata["effective_device"] == "cuda"
    assert metadata["width"] == 96
    assert metadata["height"] == 64
    assert metadata["candidate_count"] == 2
    assert metadata["selected_seed"] == 202
    assert metadata["quality_target"] == 0.9
    assert metadata["target_quality_met"] is False
    assert metadata["visual_critique_required"] is True
    assert metadata["selection_strategy"] == "prompt-aware-detail-aesthetic-v1"
    assert set(metadata["prompt_focus_terms"]) >= {"night", "plant", "lamp"}
    assert metadata["prompt_signal_score"] == metadata["recommended_candidate_prompt_signal_score"]

    candidates = json.loads(result.candidates_path.read_text(encoding="utf-8"))
    assert {candidate["seed"] for candidate in candidates} == {101, 202}
    assert all(Path(candidate["image"]).exists() for candidate in candidates)
    assert all(0.0 <= candidate["selection_score"] <= 1.0 for candidate in candidates)
    assert all(0.0 <= candidate["prompt_signal_score"] <= 1.0 for candidate in candidates)
    assert all(0.0 <= candidate["prompt_critical_score"] <= 1.0 for candidate in candidates)
    assert all("prompt_signal_details" in candidate for candidate in candidates)
    assert metadata["recommended_candidate_prompt_critical_score"] == candidates[0]["prompt_critical_score"]
    assert all(any("prompt_signal_score" in reason for reason in candidate["selection_reasons"]) for candidate in candidates)

    critique_request = json.loads(result.critique_request_path.read_text(encoding="utf-8"))
    assert critique_request["image"] == str(result.image_path)
    assert critique_request["prompt"] == "photoreal glass greenhouse interior at night with tropical plants and tungsten lamps"


def test_diffusion_profile_applies_photoreal_defaults(tmp_path: Path, monkeypatch):
    from claude_imagegen import diffusion as diffusion_module
    from claude_imagegen.diffusion import DiffusionOptions, generate_diffusion_image

    captured: dict[str, object] = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (18, 40, 32))])

    def fake_load_pipeline(*, model: str, device: str):
        captured["model"] = model
        captured["device"] = device
        return FakePipeline(), "cuda"

    monkeypatch.setattr(diffusion_module, "_load_pipeline", fake_load_pipeline)
    monkeypatch.setattr(diffusion_module, "_torch_generator", lambda *, seed, device: None)

    result = generate_diffusion_image(
        DiffusionOptions(
            prompt="deep night photoreal greenhouse with tungsten lamps",
            output_dir=tmp_path / "diffusion-profile",
            profile="night-photoreal",
            width=96,
            height=64,
            seeds=(7,),
            device="auto",
        )
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert captured["model"] == "SG161222/RealVisXL_V5.0"
    assert captured["device"] == "auto"
    assert metadata["diffusion_profile"] == "night-photoreal"
    assert metadata["model"] == "SG161222/RealVisXL_V5.0"
    assert metadata["steps"] == 28
    assert metadata["guidance_scale"] == 7.0
    assert "furniture" in metadata["negative_prompt"]
    assert metadata["normalized_prompt"].startswith("photorealistic high-detail DSLR")


def test_night_diffusion_profile_promotes_strict_quality_features(tmp_path: Path, monkeypatch):
    from claude_imagegen import diffusion as diffusion_module
    from claude_imagegen.diffusion import DiffusionOptions, generate_diffusion_image

    captured: dict[str, object] = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            captured["prompt"] = kwargs["prompt"]
            return SimpleNamespace(images=[Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (18, 40, 32))])

        def to(self, device: str):
            captured["device"] = device
            return self

    monkeypatch.setattr(diffusion_module, "_load_pipeline", lambda *, model, device: (FakePipeline(), "cuda"))
    monkeypatch.setattr(diffusion_module, "_torch_generator", lambda *, seed, device: None)

    result = generate_diffusion_image(
        DiffusionOptions(
            prompt=(
                "deep night glass greenhouse interior with tropical plants, sharp leaf veins, "
                "warm tungsten hanging lamps, visible interior volumetric mist in the lamp beams, "
                "wet black stone floor with coherent mirror reflections, no people"
            ),
            output_dir=tmp_path / "diffusion-strict-features",
            profile="night-photoreal",
            width=96,
            height=64,
            seeds=(7,),
            device="auto",
        )
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    normalized_prompt = metadata["normalized_prompt"]
    assert metadata["prompt_token_estimate"] <= 77
    assert metadata["prompt_length_warning"] is None
    assert "greenhouse interior" in normalized_prompt.lower()
    assert "tropical plants" in normalized_prompt.lower()
    first_clause = normalized_prompt.split(",", maxsplit=8)[:8]
    early_prompt = ",".join(first_clause).lower()
    assert "interior volumetric mist" in early_prompt
    assert "mirror-wet floor reflections" in early_prompt
    assert "crisp leaf-vein microdetail" in early_prompt
    assert "warm tungsten hanging lamps" in early_prompt
    assert "volumetric light beams from warm tungsten lamps" in early_prompt
    assert captured["prompt"] == normalized_prompt


def test_diffusion_candidate_selection_penalizes_missing_required_prompt_terms(tmp_path: Path):
    from PIL import ImageDraw

    from claude_imagegen.diffusion import _candidate_entry

    image = Image.new("RGB", (128, 96), (52, 72, 68))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 58, 128, 96), fill=(70, 85, 82))
    draw.rectangle((8, 66, 120, 86), fill=(160, 170, 158))

    entry = _candidate_entry(
        image,
        image_path=tmp_path / "no-lamp.png",
        seed=1,
        prompt_focus_terms=("night", "lamp", "mist", "reflection"),
    )

    assert entry["prompt_signal_details"]["term_scores"]["lamp"] == 0.0
    assert entry["prompt_critical_score"] == 0.0
    assert entry["selection_score"] < 0.35
    assert any("prompt_critical_score=0.000" in reason for reason in entry["selection_reasons"])


def test_diffusion_image_to_image_refinement_uses_initial_image(tmp_path: Path, monkeypatch):
    from claude_imagegen import diffusion as diffusion_module
    from claude_imagegen.diffusion import DiffusionOptions, generate_diffusion_image

    initial_image = tmp_path / "initial.png"
    Image.new("RGB", (80, 60), (12, 28, 24)).save(initial_image)
    captured: dict[str, object] = {}

    class FakePipeline:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (20, 42, 34))])

        def to(self, device: str):
            captured["device"] = device
            return self

    monkeypatch.setattr(diffusion_module, "_load_image_to_image_pipeline", lambda *, model, device: (FakePipeline(), "cuda"))
    monkeypatch.setattr(diffusion_module, "_torch_generator", lambda *, seed, device: None)

    result = generate_diffusion_image(
        DiffusionOptions(
            prompt="deep night glass greenhouse with warm lamps, mist, plants, and wet floor reflections",
            output_dir=tmp_path / "img2img",
            profile="night-photoreal",
            initial_image=initial_image,
            strength=0.28,
            width=96,
            height=64,
            seeds=(7,),
            device="auto",
            quality_target=0.9,
        )
    )

    assert captured["image"].size == (96, 64)
    assert captured["strength"] == 0.28
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["diffusion_mode"] == "image-to-image"
    assert metadata["initial_image"] == str(initial_image)
    assert metadata["strength"] == 0.28


def test_diffusion_generation_records_prompt_length_warning(tmp_path: Path, monkeypatch):
    from claude_imagegen import diffusion as diffusion_module
    from claude_imagegen.diffusion import DiffusionOptions, generate_diffusion_image

    class FakePipeline:
        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (int(kwargs["width"]), int(kwargs["height"])), (24, 80, 120))])

    monkeypatch.setattr(diffusion_module, "_load_pipeline", lambda *, model, device: (FakePipeline(), "cuda"))
    monkeypatch.setattr(diffusion_module, "_torch_generator", lambda *, seed, device: None)

    long_prompt = " ".join(f"detail{i}" for i in range(90))
    result = generate_diffusion_image(
        DiffusionOptions(
            prompt=long_prompt,
            output_dir=tmp_path / "diffusion-long-prompt",
            width=96,
            height=64,
            seeds=(101,),
            device="cuda",
        )
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt_token_estimate"] == 90
    assert "77-token" in metadata["prompt_length_warning"]


def test_cli_pair_eval_writes_claude_request_without_generating(tmp_path: Path):
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (16, 16), (80, 80, 80)).save(before)
    Image.new("RGB", (16, 16), (20, 40, 30)).save(after)
    output_dir = tmp_path / "pair-eval"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "pair-eval",
            "--prompt",
            "photoreal greenhouse with night mist and lamps",
            "--before",
            str(before),
            "--after",
            str(after),
            "--pair-id",
            "greenhouse-v1",
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
    request_path = output_dir / "pair-evaluation-request.json"
    assert request_path.exists()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["judge"] == "claude-vision-pair-evaluation"
    assert request["prompt"] == "photoreal greenhouse with night mist and lamps"
    assert request["quality_target"] == 0.9
    assert request["pairs"] == [
        {"id": "greenhouse-v1", "before_image": str(before), "after_image": str(after)}
    ]
    assert request["expected_response"]["acceptance_gate_met"] is False
    assert "after_score" in request["expected_response"]["pair_scores"][0]


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
    assert "Comparison request" in completed.stdout
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
    assert metadata["comparison_request"] == str(refined_dir / "comparison-request.json")
    critique_request = json.loads((refined_dir / "critique-request.json").read_text(encoding="utf-8"))
    assert critique_request["image"] == str(refined_dir / "image.png")
    assert critique_request["metadata"] == str(refined_dir / "metadata.json")
    assert critique_request["quality_report"] == str(refined_dir / "quality-report.json")
    comparison_request = json.loads((refined_dir / "comparison-request.json").read_text(encoding="utf-8"))
    assert comparison_request["parent_image"] == str(base_dir / "image.png")
    assert comparison_request["child_image"] == str(refined_dir / "image.png")
    assert comparison_request["metadata"] == str(refined_dir / "metadata.json")
    assert comparison_request["parent_metadata"] == str(base_dir / "metadata.json")
    assert comparison_request["refinement_delta"] == metadata["refinement_delta"]
    assert comparison_request["expected_response"]["better_image"] == "child"
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


def test_cli_refine_records_and_applies_visual_comparison(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")

    base_dir = tmp_path / "base-comparison"
    base_plan = base_dir / "scene-plan.json"
    base_dir.mkdir()
    base_plan.write_text(
        json.dumps(
            {
                "title": "Comparison refinable scene",
                "palette": ["#102040", "#ff5533", "#286fc4"],
                "background": {"top": "#102040", "bottom": "#205080"},
                "objects": [
                    {"type": "sun", "x": 0.25, "y": 0.25, "size": 0.1, "color": "#ff5533"},
                    {"type": "ocean", "y": 0.58, "color": "#286fc4"},
                ],
                "style": {"contrast": 0.2},
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

    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "alignment_score": 0.52,
                "continuity_score": 0.43,
                "improved": False,
                "preserved_identity": False,
                "better_image": "parent",
                "verdict": "revise",
                "summary": "Child regressed visually.",
                "regressions": ["sun became too small", "lost parent palette"],
                "follow_up_edits": [
                    {"action": "resize_object", "type": "sun", "size": 0.24},
                    {"action": "adjust_style", "field": "contrast", "delta": 0.1},
                ],
            }
        ),
        encoding="utf-8",
    )

    refined_dir = tmp_path / "refined-comparison"
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
            "--comparison",
            str(comparison_path),
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

    assert "Comparison revise alignment 0.52 continuity 0.43" in completed.stdout
    metadata = json.loads((refined_dir / "metadata.json").read_text(encoding="utf-8"))
    refined_plan = json.loads((refined_dir / "scene-plan.json").read_text(encoding="utf-8"))
    quality_report = json.loads((refined_dir / "quality-report.json").read_text(encoding="utf-8"))

    comparison = metadata["visual_comparison"]
    assert comparison["judge"] == "claude-vision-refinement-comparison"
    assert comparison["better_image"] == "parent"
    assert comparison["regressions"] == ["sun became too small", "lost parent palette"]
    assert any("resized 1 'sun'" in action for action in comparison["applied_edits"])
    assert any("comparison: resized 1 'sun'" in action for action in metadata["scene_plan_refine_actions"])
    assert refined_plan["objects"][0]["size"] == 0.24
    assert refined_plan["style"]["contrast"] == 0.3
    assert any("Comparison: address regressions: sun became too small, lost parent palette." == action for action in quality_report["next_actions"])


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
    assert metadata["parent_candidate_aesthetic_score"] == selected["aesthetic_score"]
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
    assert metadata["parent_candidate_aesthetic_score"] == selected["aesthetic_score"]
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
    assert "Devices" in completed.stdout
    assert "Images nonblank" in completed.stdout
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
    assert report["device_summary"]["devices"] == ["cpu"]
    assert report["device_summary"]["cpu_case_count"] == len(report["cases"])
    assert report["device_summary"]["cuda_case_count"] == 0
    assert report["image_summary"]["case_count"] == len(report["cases"])
    assert report["image_summary"]["nonblank_cases"] == len(report["cases"])
    assert report["image_summary"]["blank_cases"] == 0

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
        assert case["image_nonblank"] is True
        assert case["image_variance_sum"] > 0
        assert case["image_stats"]["nonblank"] is True
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
            assert (case_dir / "comparison-request.json").exists()
            assert case["comparison_request"] == str(case_dir / "comparison-request.json")
            assert metadata["comparison_request"] == str(case_dir / "comparison-request.json")
            assert metadata["parent_candidate_selection"] == "auto"
            assert metadata["initial_similarity_score"] is not None
            assert case["refinement_delta"] == metadata["refinement_delta"]


def test_verification_runs_strong_cases_for_explicit_strong_sizes(tmp_path: Path, monkeypatch):
    def fake_result(output_dir: Path, *, width: int, height: int, options: object, refined: bool = False):
        output_dir.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (width, height), (32, 64, 128))
        image.putpixel((0, 0), (96, 128, 196))
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
    assert report["device_summary"]["devices"] == ["cpu", "cuda"]
    assert report["device_summary"]["cuda_case_count"] == len(strong_cases)
    assert report["device_summary"]["cpu_case_count"] == 3
    assert report["device_summary"]["role_devices"]["similarity"]["cuda"] == len(strong_cases)
    assert "transformers-siglip" in report["device_summary"]["similarity_backends"]
    assert "transformers-dinov2" in report["device_summary"]["continuity_backends"]
    assert report["image_summary"]["nonblank_cases"] == len(report["cases"])
    assert report["image_summary"]["blank_cases"] == 0


def test_verification_case_report_fails_blank_images(tmp_path: Path):
    output_dir = tmp_path / "blank-case"
    output_dir.mkdir()
    image_path = output_dir / "image.png"
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.csv"
    quality_path = output_dir / "quality-report.json"
    critique_path = output_dir / "critique-request.json"

    Image.new("RGB", (16, 12), (32, 64, 128)).save(image_path)
    progress_path.write_text("iteration,total_score\n1,0.5\n", encoding="utf-8")
    quality_path.write_text("{}", encoding="utf-8")
    critique_path.write_text("{}", encoding="utf-8")
    metadata = {
        "width": 16,
        "height": 12,
        "quality_report": str(quality_path),
        "candidate_count": 0,
        "caption_backend": "local",
        "caption_model": None,
        "similarity_backend": "local",
        "continuity_backend": "local",
        "caption_similarity_backend": "local",
        "effective_similarity_device": "cpu",
        "effective_continuity_device": "cpu",
        "effective_caption_device": "cpu",
        "effective_caption_similarity_device": "cpu",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    case = verify_module._case_report(
        "generate",
        SimpleNamespace(
            metadata=metadata,
            image_path=image_path,
            metadata_path=metadata_path,
            progress_path=progress_path,
        ),
        requested_size=(16, 12),
    )

    assert case["status"] == "fail"
    assert case["image_nonblank"] is False
    assert case["image_variance_sum"] == 0
    assert case["image_stats"]["nonblank"] is False
