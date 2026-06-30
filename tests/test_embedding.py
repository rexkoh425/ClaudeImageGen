from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from claude_imagegen.embedding import (
    embedding_cosine,
    image_embedding,
    image_embedding_similarity,
)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (200, 200)) -> Image.Image:
    return Image.new("RGB", size, color)


def _gradient(
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    size: tuple[int, int] = (200, 200),
) -> Image.Image:
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / (height - 1)
        array[y, :, :] = [int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)]
    return Image.fromarray(array)


def test_embedding_is_unit_norm_and_stable_dimension() -> None:
    vector = image_embedding(_gradient((20, 40, 90), (200, 150, 90)))
    assert vector.ndim == 1
    assert vector.shape[0] > 64
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_identical_images_score_one() -> None:
    image = _gradient((20, 40, 90), (200, 150, 90))
    assert image_embedding_similarity(image, image) > 0.999


def test_blurred_image_stays_close() -> None:
    image = _gradient((20, 40, 90), (200, 150, 90))
    blurred = image.filter(ImageFilter.GaussianBlur(3))
    assert image_embedding_similarity(image, blurred) > 0.9


def test_resize_is_approximately_invariant() -> None:
    image = _gradient((10, 60, 120), (230, 180, 70))
    resized = image.resize((400, 320))
    assert image_embedding_similarity(image, resized) > 0.98


def test_different_images_score_lower_than_identical() -> None:
    red = _solid((220, 30, 30))
    blue = _solid((30, 40, 220))
    identical = image_embedding_similarity(red, red)
    different = image_embedding_similarity(red, blue)
    assert different < identical
    assert different < 0.7


def test_cosine_is_clamped_to_unit_interval() -> None:
    a = image_embedding(_solid((10, 200, 10)))
    b = image_embedding(_solid((200, 10, 200)))
    score = embedding_cosine(a, b)
    assert 0.0 <= score <= 1.0


def test_same_composition_beats_unrelated_image() -> None:
    base = _gradient((15, 35, 80), (210, 150, 80))
    similar = base.filter(ImageFilter.GaussianBlur(1))
    unrelated = _solid((20, 30, 210))
    assert image_embedding_similarity(base, similar) > image_embedding_similarity(base, unrelated)
