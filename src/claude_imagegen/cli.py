from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .generator import GenerateOptions, generate_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="claude-imagegen")
    subcommands = parser.add_subparsers(dest="command", required=True)

    generate = subcommands.add_parser("generate", help="Generate a CPU-rendered image from a text prompt.")
    generate.add_argument("--prompt", required=True, help="Text prompt to render.")
    generate.add_argument("--reference-image", type=Path, help="Optional image whose palette should influence output.")
    generate.add_argument("--initial-image", type=Path, help="Optional existing image to blend into the search.")
    generate.add_argument("--scene-plan", type=Path, help="Optional Claude-authored JSON scene plan for explicit composition.")
    generate.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output"))
    generate.add_argument("--width", type=int, default=720)
    generate.add_argument("--height", type=int, default=480)
    generate.add_argument("--max-iterations", type=int, default=32)
    generate.add_argument("--threshold", type=float, default=0.58)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--pixel-csv", action="store_true", help="Also write pixels.csv with x,y,r,g,b rows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        result = generate_image(
            GenerateOptions(
                prompt=args.prompt,
                output_dir=args.output_dir,
                reference_image=args.reference_image,
                initial_image=args.initial_image,
                scene_plan=args.scene_plan,
                width=args.width,
                height=args.height,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                seed=args.seed,
                pixel_csv=args.pixel_csv,
            )
        )
        print(f"Generated {result.image_path}")
        print(f"Metadata {result.metadata_path}")
        print(f"Score {result.metadata['total_score']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
