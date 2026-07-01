import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


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


def _crushed_shadow_fixture(path: Path) -> Image.Image:
    width, height = 96, 64
    arr = np.full((height, width, 3), 0.11, dtype=np.float32)
    arr[: height // 2, :, :] = 0.16
    for x in range(width):
        stripe = 0.035 if (x // 5) % 2 == 0 else -0.025
        arr[height // 2 :, x, :] += stripe
    arr[8:18, 42:54, :] = (0.85, 0.66, 0.36)
    arr[38:58, 18:82, 1] += 0.035
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    image.save(path)
    return image


def _crushed_chroma_noise_fixture(path: Path) -> Image.Image:
    width, height = 96, 64
    arr = np.full((height, width, 3), 0.055, dtype=np.float32)
    arr[: height // 2, :, :] = 0.14
    lower = arr[height // 2 :, :, :]
    lower[::2, ::3, 2] += 0.045
    lower[::3, ::2, 0] += 0.035
    arr[10:18, 42:54, :] = (0.78, 0.58, 0.30)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    image.save(path)
    return image


def _midtone_foliage_fixture(path: Path) -> Image.Image:
    width, height = 96, 64
    arr = np.full((height, width, 3), 0.075, dtype=np.float32)
    arr[: height // 2, :, :] = 0.14
    arr[height // 2 :, :, :] = (0.16, 0.22, 0.15)
    arr[8:18, 42:54, :] = (0.78, 0.58, 0.30)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB")
    image.save(path)
    return image


def _soft_foliage_fixture(path: Path) -> Image.Image:
    width, height = 96, 64
    arr = np.full((height, width, 3), 0.08, dtype=np.float32)
    arr[:, : width // 2, :] = (0.10, 0.22, 0.09)
    arr[:, width // 2 :, :] = (0.08, 0.08, 0.09)
    for x in range(6, width // 2 - 4, 8):
        arr[8:56, x : x + 2, :] = (0.18, 0.40, 0.16)
    arr[28:36, width // 2 + 8 : width - 8, :] = (0.72, 0.55, 0.30)
    image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), "RGB").filter(
        ImageFilter.GaussianBlur(radius=1.4)
    )
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
        "lower_luma_p10": float(np.percentile(lower, 10)),
    }


def _dark_chroma_p95(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    chroma = arr.max(axis=2) - arr.min(axis=2)
    dark_chroma = chroma[luma < 0.2]
    return float(np.percentile(dark_chroma, 95))


def _lower_half_median_luma(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    lower = luma[luma.shape[0] // 2 :, :]
    return float(np.median(lower))


def _region_edge_density(image: Image.Image, region: tuple[int, int, int, int]) -> float:
    arr = np.asarray(image.convert("RGB").crop(region), dtype=np.float32) / 255.0
    luma = (0.2126 * arr[:, :, 0]) + (0.7152 * arr[:, :, 1]) + (0.0722 * arr[:, :, 2])
    horizontal = np.abs(np.diff(luma, axis=1))
    vertical = np.abs(np.diff(luma, axis=0))
    return float((np.mean(horizontal > 0.018) + np.mean(vertical > 0.018)) / 2.0)


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


def test_enhance_night_can_lift_crushed_shadows_without_breaking_night(tmp_path: Path):
    from claude_imagegen.enhance import EnhanceNightOptions, enhance_night_image

    input_path = tmp_path / "crushed-input.png"
    before = _crushed_shadow_fixture(input_path)
    before_stats = _luma_stats(before)

    result = enhance_night_image(
        EnhanceNightOptions(
            input_image=input_path,
            prompt="deep night greenhouse with warm lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "enhanced",
            quality_target=0.9,
            night_luma_ceiling=0.32,
            mist_cap=0.16,
            highlight_rolloff=0.25,
            local_contrast=1.05,
            shadow_lift=0.12,
        )
    )

    after_stats = _luma_stats(result.image)
    assert after_stats["mean_luma"] <= 0.33
    assert after_stats["lower_luma_p10"] > before_stats["lower_luma_p10"] + 0.025
    assert after_stats["lower_luma_std"] >= before_stats["lower_luma_std"] * 0.85

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["shadow_lift"] == 0.12
    assert metadata["after_lower_luma_p10"] > metadata["before_lower_luma_p10"]


def test_enhance_night_shadow_lift_does_not_amplify_chroma_speckles(tmp_path: Path):
    from claude_imagegen.enhance import EnhanceNightOptions, enhance_night_image

    input_path = tmp_path / "chroma-noise-input.png"
    before = _crushed_chroma_noise_fixture(input_path)
    before_chroma_p95 = _dark_chroma_p95(before)

    result = enhance_night_image(
        EnhanceNightOptions(
            input_image=input_path,
            prompt="deep night greenhouse with warm lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "enhanced",
            quality_target=0.9,
            night_luma_ceiling=0.32,
            mist_cap=0.16,
            highlight_rolloff=0.25,
            local_contrast=1.05,
            shadow_lift=0.12,
        )
    )

    after_chroma_p95 = _dark_chroma_p95(result.image)
    assert after_chroma_p95 <= before_chroma_p95 + 0.02


def test_enhance_night_shadow_lift_preserves_readable_midtones(tmp_path: Path):
    from claude_imagegen.enhance import EnhanceNightOptions, enhance_night_image

    input_path = tmp_path / "midtone-input.png"
    before = _midtone_foliage_fixture(input_path)
    before_median = _lower_half_median_luma(before)

    result = enhance_night_image(
        EnhanceNightOptions(
            input_image=input_path,
            prompt="deep night greenhouse with warm lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "enhanced",
            quality_target=0.9,
            night_luma_ceiling=0.32,
            mist_cap=0.16,
            highlight_rolloff=0.25,
            local_contrast=1.05,
            shadow_lift=0.12,
        )
    )

    after_median = _lower_half_median_luma(result.image)
    assert after_median <= before_median + 0.015


def test_enhance_night_foliage_clarity_targets_green_texture(tmp_path: Path):
    from claude_imagegen.enhance import EnhanceNightOptions, enhance_night_image

    input_path = tmp_path / "soft-foliage-input.png"
    before = _soft_foliage_fixture(input_path)
    foliage_region = (0, 0, 48, 64)
    floor_region = (56, 0, 96, 64)
    before_foliage_edges = _region_edge_density(before, foliage_region)
    before_floor_edges = _region_edge_density(before, floor_region)

    result = enhance_night_image(
        EnhanceNightOptions(
            input_image=input_path,
            prompt="deep night greenhouse with warm lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "enhanced",
            quality_target=0.9,
            night_luma_ceiling=0.32,
            mist_cap=0.16,
            highlight_rolloff=0.25,
            local_contrast=0.2,
            shadow_lift=0.0,
            foliage_clarity=0.8,
        )
    )

    after_foliage_edges = _region_edge_density(result.image, foliage_region)
    after_floor_edges = _region_edge_density(result.image, floor_region)
    assert after_foliage_edges > before_foliage_edges + 0.02
    assert after_floor_edges <= before_floor_edges + 0.01

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["foliage_clarity"] == 0.8


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
            "--shadow-lift",
            "0.12",
            "--foliage-clarity",
            "0.4",
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
    assert metadata["shadow_lift"] == 0.12
    assert metadata["foliage_clarity"] == 0.4
    assert metadata["acceptance_requires_pair_evaluation"] is True
    assert "Pair evaluation request" in completed.stdout
