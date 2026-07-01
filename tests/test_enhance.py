import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _hazy_night_fixture(path: Path) -> Image.Image:
    width, height = 96, 64
    arr = np.full((height, width, 3), 0.52, dtype=np.float32)
    for y in range(height):
        arr[y, :, :] += (y / max(1, height - 1)) * 0.04
    for x in range(width):
        stripe = 0.025 if (x // 6) % 2 == 0 else -0.01
        arr[height // 2 :, x, :] += stripe
    arr[:, :, 1] += 0.03
    arr[8:18, 42:54, :] = (1.0, 0.93, 0.62)
    arr[18:24, 46:50, :] = (1.0, 0.86, 0.40)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    image.save(path)
    return image


def _luma_stats(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    lower = luma[luma.shape[0] // 2 :, :]
    return {
        "mean_luma": float(np.mean(luma)),
        "max_luma": float(np.max(luma)),
        "lower_luma_std": float(np.std(lower)),
    }


def test_enhance_night_preserves_darkness_and_writes_pair_request(tmp_path: Path):
    from claude_imagegen.enhance import EnhanceNightOptions, enhance_night_image

    input_path = tmp_path / "hazy-input.png"
    before = _hazy_night_fixture(input_path)
    before_stats = _luma_stats(before)

    result = enhance_night_image(
        EnhanceNightOptions(
            input_image=input_path,
            prompt="deep night greenhouse with warm lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "enhanced",
            quality_target=0.9,
            night_luma_ceiling=0.34,
            mist_cap=0.22,
            highlight_rolloff=0.35,
            local_contrast=0.9,
        )
    )

    assert result.image_path.exists()
    assert result.metadata_path.exists()
    assert result.pair_evaluation_request_path.exists()
    after_stats = _luma_stats(result.image)
    assert after_stats["mean_luma"] <= 0.35
    assert after_stats["max_luma"] < before_stats["max_luma"]
    assert after_stats["lower_luma_std"] > before_stats["lower_luma_std"]

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["engine"] == "night-preserving-postprocess-v1"
    assert metadata["input_image"] == str(input_path)
    assert metadata["quality_target"] == 0.9
    assert metadata["night_luma_ceiling"] == 0.34
    assert metadata["after_mean_luma"] <= 0.35
    assert metadata["after_lower_luma_std"] > metadata["before_lower_luma_std"]
    assert metadata["acceptance_requires_pair_evaluation"] is True

    request = json.loads(result.pair_evaluation_request_path.read_text(encoding="utf-8"))
    assert request["judge"] == "claude-vision-pair-evaluation"
    assert request["pairs"] == [
        {"id": "enhance-night", "before_image": str(input_path), "after_image": str(result.image_path)}
    ]
    assert request["quality_target"] == 0.9


def test_cli_enhance_night_writes_artifacts_without_diffusion(tmp_path: Path):
    input_path = tmp_path / "hazy-input.png"
    _hazy_night_fixture(input_path)
    output_dir = tmp_path / "cli-enhanced"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "enhance-night",
            "--input-image",
            str(input_path),
            "--prompt",
            "deep night greenhouse with mist and wet reflections",
            "--output-dir",
            str(output_dir),
            "--night-luma-ceiling",
            "0.34",
            "--mist-cap",
            "0.22",
            "--highlight-rolloff",
            "0.35",
            "--local-contrast",
            "0.9",
            "--quality-target",
            "0.9",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "image.png").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "pair-evaluation-request.json").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "local-postprocess"
    assert metadata["acceptance_requires_pair_evaluation"] is True
    assert "Pair evaluation request" in completed.stdout
