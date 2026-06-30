from __future__ import annotations

from pathlib import Path

from PIL import Image

RGB = tuple[int, int, int]

COLOR_RGB: dict[str, RGB] = {
    "black": (16, 18, 22),
    "blue": (39, 110, 210),
    "brown": (126, 86, 52),
    "coral": (255, 127, 80),
    "cyan": (41, 190, 210),
    "gold": (235, 178, 62),
    "gray": (128, 132, 140),
    "green": (44, 158, 88),
    "grey": (128, 132, 140),
    "indigo": (72, 72, 166),
    "magenta": (208, 64, 166),
    "orange": (232, 126, 54),
    "pink": (236, 116, 158),
    "purple": (130, 80, 178),
    "red": (220, 64, 58),
    "teal": (32, 150, 150),
    "violet": (142, 92, 205),
    "white": (240, 238, 230),
    "yellow": (240, 212, 78),
}


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def blend(a: RGB, b: RGB, amount: float) -> RGB:
    amount = max(0.0, min(1.0, amount))
    return (
        clamp_channel(a[0] * (1 - amount) + b[0] * amount),
        clamp_channel(a[1] * (1 - amount) + b[1] * amount),
        clamp_channel(a[2] * (1 - amount) + b[2] * amount),
    )


def palette_from_words(color_words: tuple[str, ...]) -> tuple[RGB, ...]:
    colors = tuple(COLOR_RGB[word] for word in color_words if word in COLOR_RGB)
    return colors or (COLOR_RGB["blue"], COLOR_RGB["gold"])


def extract_reference_palette(path: Path, limit: int = 5) -> tuple[RGB, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found: {path}")

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((32, 32))
        quantized = rgb.quantize(colors=max(1, limit)).convert("RGB")
        colors = quantized.getcolors(maxcolors=32 * 32)

    if not colors:
        return ()

    ordered = sorted(colors, key=lambda item: item[0], reverse=True)
    return tuple(color for _, color in ordered[:limit])


def average_color(image: Image.Image) -> RGB:
    rgb = image.convert("RGB").resize((1, 1))
    return rgb.getpixel((0, 0))
