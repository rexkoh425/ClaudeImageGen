from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def export_pixel_csv(image: Image.Image, path: Path) -> None:
    rgb = image.convert("RGB")
    width, height = rgb.size
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "r", "g", "b"])
        pixels = rgb.load()
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                writer.writerow([x, y, r, g, b])
