"""Dependency-free perceptual image embedding for image-to-image similarity.

This module produces a single L2-normalized feature vector per image using only
``numpy`` and ``Pillow``. The goal is not CLIP/DINOv2 quality; it is a deterministic,
auditable, always-available image-to-image closeness signal for the refinement loop,
approximating the structure-plus-color ideas behind learned metrics:

* Regional mean color on a grid, in a luma + opponent-color space (global layout/palette).
* Per-region hue histograms weighted by saturation and value (color distribution).
* Per-region edge-orientation histograms, HOG-like (structure and silhouette).
* Global statistics: contrast, brightness, saturation, warmth, edge density.

Each block is individually L2-normalized and weighted so no single block dominates,
then concatenated and L2-normalized again. ``embedding_cosine`` returns the clamped
cosine of two such vectors in ``[0, 1]`` (identical images -> ~1.0).

When ``torch`` + weights are available, callers may blend in a CLIP/DINOv2 image
cosine; this module deliberately has no model dependency so it always runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Public knobs. Kept module-level so tests and callers can reference them.
EMBED_SIZE = (128, 128)
GRID = 4
HUE_BINS = 8
VALUE_BINS = 4
ORIENT_BINS = 9

# Relative weight of each feature block before the final normalization.
_BLOCK_WEIGHTS = {
    "color": 1.0,
    "hue": 1.0,
    "value": 0.6,
    "orientation": 1.0,
    "global": 0.5,
}


def image_embedding(image: Image.Image) -> np.ndarray:
    """Return an L2-normalized float32 embedding vector for ``image``."""
    array = _normalized_rgb_array(image)
    blocks = [
        (_BLOCK_WEIGHTS["color"], _regional_color_block(array)),
        (_BLOCK_WEIGHTS["hue"], _regional_hue_block(array)),
        (_BLOCK_WEIGHTS["value"], _regional_value_block(array)),
        (_BLOCK_WEIGHTS["orientation"], _regional_orientation_block(array)),
        (_BLOCK_WEIGHTS["global"], _global_stats_block(array)),
    ]
    parts = [weight * _l2_normalize(block) for weight, block in blocks]
    vector = np.concatenate(parts).astype(np.float32)
    return _l2_normalize(vector)


def embedding_cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine of two embeddings, clamped to ``[0, 1]``."""
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return _clamp01(float(np.dot(left, right)) / denominator)


def image_embedding_similarity(image: Image.Image, comparison: Image.Image | Path | str) -> float:
    """Convenience: embed two images (or an image and a path) and return cosine."""
    comparison_image = _coerce_image(comparison)
    return embedding_cosine(image_embedding(image), image_embedding(comparison_image))


def _coerce_image(value: Image.Image | Path | str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    with Image.open(Path(value)) as handle:
        return handle.convert("RGB")


def _normalized_rgb_array(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").resize(EMBED_SIZE, Image.Resampling.BICUBIC)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _grid_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    row_edges = np.linspace(0, height, GRID + 1, dtype=int)
    col_edges = np.linspace(0, width, GRID + 1, dtype=int)
    slices: list[tuple[slice, slice]] = []
    for r in range(GRID):
        for c in range(GRID):
            slices.append((slice(row_edges[r], row_edges[r + 1]), slice(col_edges[c], col_edges[c + 1])))
    return slices


def _regional_color_block(array: np.ndarray) -> np.ndarray:
    """Mean luma + opponent color per grid cell."""
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    opp_rg = (red - green + 1.0) / 2.0  # red-green opponent, shifted to [0, 1]
    opp_yb = ((red + green) / 2.0 - blue + 1.0) / 2.0  # yellow-blue opponent
    height, width = luma.shape
    values: list[float] = []
    for row_slice, col_slice in _grid_slices(height, width):
        values.append(float(luma[row_slice, col_slice].mean()))
        values.append(float(opp_rg[row_slice, col_slice].mean()))
        values.append(float(opp_yb[row_slice, col_slice].mean()))
    return np.asarray(values, dtype=np.float32)


def _rgb_to_hsv(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    delta = maximum - minimum
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]

    hue = np.zeros_like(maximum)
    safe = delta > 1e-6
    # Red is max.
    mask = safe & (maximum == red)
    hue[mask] = (((green - blue)[mask] / delta[mask]) % 6.0)
    # Green is max.
    mask = safe & (maximum == green)
    hue[mask] = ((blue - red)[mask] / delta[mask]) + 2.0
    # Blue is max.
    mask = safe & (maximum == blue)
    hue[mask] = ((red - green)[mask] / delta[mask]) + 4.0
    hue = (hue / 6.0) % 1.0  # normalize to [0, 1)

    saturation = np.where(maximum > 1e-6, delta / np.maximum(maximum, 1e-6), 0.0)
    value = maximum
    return hue, saturation, value


def _regional_hue_block(array: np.ndarray) -> np.ndarray:
    hue, saturation, value = _rgb_to_hsv(array)
    weight = saturation * value  # ignore hue of dark/gray pixels
    bins = np.clip((hue * HUE_BINS).astype(int), 0, HUE_BINS - 1)
    height, width = hue.shape
    values: list[float] = []
    for row_slice, col_slice in _grid_slices(height, width):
        cell_bins = bins[row_slice, col_slice].reshape(-1)
        cell_weight = weight[row_slice, col_slice].reshape(-1)
        histogram = np.bincount(cell_bins, weights=cell_weight, minlength=HUE_BINS)
        total = float(histogram.sum())
        if total > 0:
            histogram = histogram / total
        values.extend(histogram.tolist())
    return np.asarray(values, dtype=np.float32)


def _regional_value_block(array: np.ndarray) -> np.ndarray:
    _, _, value = _rgb_to_hsv(array)
    bins = np.clip((value * VALUE_BINS).astype(int), 0, VALUE_BINS - 1)
    height, width = value.shape
    values: list[float] = []
    for row_slice, col_slice in _grid_slices(height, width):
        cell_bins = bins[row_slice, col_slice].reshape(-1)
        histogram = np.bincount(cell_bins, minlength=VALUE_BINS).astype(np.float32)
        total = float(histogram.sum())
        if total > 0:
            histogram = histogram / total
        values.extend(histogram.tolist())
    return np.asarray(values, dtype=np.float32)


def _regional_orientation_block(array: np.ndarray) -> np.ndarray:
    luma = array.mean(axis=2)
    gradient_y, gradient_x = np.gradient(luma)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = (np.arctan2(gradient_y, gradient_x) % np.pi) / np.pi  # unsigned [0, 1)
    bins = np.clip((orientation * ORIENT_BINS).astype(int), 0, ORIENT_BINS - 1)
    height, width = luma.shape
    values: list[float] = []
    for row_slice, col_slice in _grid_slices(height, width):
        cell_bins = bins[row_slice, col_slice].reshape(-1)
        cell_magnitude = magnitude[row_slice, col_slice].reshape(-1)
        histogram = np.bincount(cell_bins, weights=cell_magnitude, minlength=ORIENT_BINS)
        total = float(histogram.sum())
        if total > 0:
            histogram = histogram / total
        values.extend(histogram.tolist())
    return np.asarray(values, dtype=np.float32)


def _global_stats_block(array: np.ndarray) -> np.ndarray:
    luma = array.mean(axis=2)
    _, saturation, _ = _rgb_to_hsv(array)
    red, blue = array[:, :, 0], array[:, :, 2]
    gradient_y, gradient_x = np.gradient(luma)
    edge_density = float(np.sqrt(gradient_x**2 + gradient_y**2).mean())
    return np.asarray(
        [
            float(luma.mean()),
            float(luma.std()),
            float(saturation.mean()),
            float((red - blue).mean() + 0.5),  # warmth, shifted positive
            _clamp01(edge_density * 6.0),
        ],
        dtype=np.float32,
    )


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
