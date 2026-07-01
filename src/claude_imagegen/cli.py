from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .generator import GenerateOptions, generate_image
from .refine import RefineOptions, refine_image
from .verify import DEFAULT_VERIFY_SIZES, VerifyOptions, parse_size, run_verification

SIMILARITY_BACKENDS = ("local", "transformers-clip", "transformers-siglip")
STRONG_SIMILARITY_BACKENDS = ("transformers-clip", "transformers-siglip")
CONTINUITY_BACKENDS = ("local", "transformers-clip", "transformers-siglip", "transformers-dinov2")
STRONG_CONTINUITY_BACKENDS = ("local", "transformers-clip", "transformers-siglip", "transformers-dinov2")
CAPTION_SIMILARITY_BACKENDS = ("local", "transformers-sentence")


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
    generate.add_argument(
        "--save-candidates",
        type=int,
        default=0,
        help="Save the top N ranked candidate images plus candidates.json for inspection.",
    )
    generate.add_argument(
        "--similarity-backend",
        choices=SIMILARITY_BACKENDS,
        default="local",
        help="Similarity scorer for prompt/image alignment.",
    )
    generate.add_argument(
        "--similarity-model",
        help="Optional model id/path for model-backed similarity backends.",
    )
    generate.add_argument(
        "--similarity-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional model-backed similarity scoring.",
    )
    generate.add_argument(
        "--continuity-backend",
        choices=CONTINUITY_BACKENDS,
        help="Image-to-image continuity scorer for --initial-image; defaults to --similarity-backend.",
    )
    generate.add_argument(
        "--continuity-model",
        help="Optional model id/path for model-backed continuity scoring.",
    )
    generate.add_argument(
        "--continuity-device",
        choices=("auto", "cpu", "cuda"),
        help="Device for optional model-backed continuity scoring; defaults to --similarity-device.",
    )
    generate.add_argument(
        "--caption-backend",
        choices=("none", "local", "transformers-blip"),
        default="local",
        help="Caption backcheck backend for what the image appears to contain.",
    )
    generate.add_argument(
        "--caption-model",
        help="Optional model id/path for --caption-backend transformers-blip.",
    )
    generate.add_argument(
        "--caption-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional model-backed caption backchecking.",
    )
    generate.add_argument(
        "--caption-similarity-backend",
        choices=CAPTION_SIMILARITY_BACKENDS,
        default="local",
        help="Prompt/caption text similarity scorer.",
    )
    generate.add_argument(
        "--caption-similarity-model",
        help="Optional model id/path for --caption-similarity-backend transformers-sentence.",
    )
    generate.add_argument(
        "--caption-similarity-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional semantic prompt/caption similarity scoring.",
    )
    generate.add_argument(
        "--no-auto-refine",
        dest="auto_refine",
        action="store_false",
        help="Disable local scene-plan refinement between iterations.",
    )
    generate.set_defaults(auto_refine=True)

    refine = subcommands.add_parser("refine", help="Refine from a previous claude-imagegen output directory.")
    refine.add_argument("--from-dir", type=Path, required=True, help="Previous output directory containing image.png.")
    refine.add_argument("--prompt", required=True, help="Revised text prompt to render.")
    refine.add_argument("--output-dir", type=Path, required=True)
    refine.add_argument("--reference-image", type=Path, help="Optional target/reference image for similarity scoring.")
    refine.add_argument("--scene-plan", type=Path, help="Optional scene plan override; defaults to the parent output scene-plan.json when present.")
    refine.add_argument(
        "--critique",
        type=Path,
        help="Optional Claude-authored visual critique JSON (closeness_score, verdict, missing/wrong/extra, edits) recorded after viewing the parent image.",
    )
    refine.add_argument("--width", type=int, help="Output width; defaults to parent metadata width.")
    refine.add_argument("--height", type=int, help="Output height; defaults to parent metadata height.")
    refine.add_argument("--candidate-rank", help="Use a ranked candidate number, or 'auto', from the parent output candidates.json as the initial image.")
    refine.add_argument("--max-iterations", type=int, default=32)
    refine.add_argument("--threshold", type=float, default=0.58)
    refine.add_argument("--seed", type=int, default=0)
    refine.add_argument("--pixel-csv", action="store_true", help="Also write pixels.csv with x,y,r,g,b rows.")
    refine.add_argument(
        "--save-candidates",
        type=int,
        default=0,
        help="Save the top N ranked candidate images plus candidates.json for inspection.",
    )
    refine.add_argument(
        "--similarity-backend",
        choices=SIMILARITY_BACKENDS,
        default="local",
        help="Similarity scorer for prompt/image alignment.",
    )
    refine.add_argument(
        "--similarity-model",
        help="Optional model id/path for model-backed similarity backends.",
    )
    refine.add_argument(
        "--similarity-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional model-backed similarity scoring.",
    )
    refine.add_argument(
        "--continuity-backend",
        choices=CONTINUITY_BACKENDS,
        help="Image-to-image continuity scorer for the parent/initial image; defaults to --similarity-backend.",
    )
    refine.add_argument(
        "--continuity-model",
        help="Optional model id/path for model-backed continuity scoring.",
    )
    refine.add_argument(
        "--continuity-device",
        choices=("auto", "cpu", "cuda"),
        help="Device for optional model-backed continuity scoring; defaults to --similarity-device.",
    )
    refine.add_argument(
        "--caption-backend",
        choices=("none", "local", "transformers-blip"),
        default="local",
        help="Caption backcheck backend for what the image appears to contain.",
    )
    refine.add_argument(
        "--caption-model",
        help="Optional model id/path for --caption-backend transformers-blip.",
    )
    refine.add_argument(
        "--caption-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional model-backed caption backchecking.",
    )
    refine.add_argument(
        "--caption-similarity-backend",
        choices=CAPTION_SIMILARITY_BACKENDS,
        default="local",
        help="Prompt/caption text similarity scorer.",
    )
    refine.add_argument(
        "--caption-similarity-model",
        help="Optional model id/path for --caption-similarity-backend transformers-sentence.",
    )
    refine.add_argument(
        "--caption-similarity-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional semantic prompt/caption similarity scoring.",
    )
    refine.add_argument(
        "--no-auto-refine",
        dest="auto_refine",
        action="store_false",
        help="Disable local scene-plan refinement between iterations.",
    )
    refine.set_defaults(auto_refine=True)

    verify = subcommands.add_parser("verify", help="Run a local generation/refinement verification suite.")
    verify.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output/verification"))
    verify.add_argument(
        "--size",
        action="append",
        type=_parse_cli_size,
        dest="sizes",
        help="Verification output size as WIDTHxHEIGHT. Repeat for multiple sizes.",
    )
    verify.add_argument(
        "--prompt",
        default="cinematic red robot portrait over blue ocean with clouds, reflections, and atmospheric light",
        help="Prompt used for generated verification cases.",
    )
    verify.add_argument(
        "--refine-prompt",
        default="cinematic red robot portrait over blue ocean with brighter clouds and stronger water reflections",
        help="Prompt used for the auto-candidate refinement verification case.",
    )
    verify.add_argument("--max-iterations", type=int, default=3)
    verify.add_argument("--threshold", type=float, default=0.99)
    verify.add_argument("--save-candidates", type=int, default=2)
    verify.add_argument("--strong-model", action="store_true", help="Also run one model-backed similarity plus BLIP verification case.")
    verify.add_argument(
        "--strong-similarity-backend",
        choices=STRONG_SIMILARITY_BACKENDS,
        default="transformers-clip",
        help="Model-backed similarity scorer to use for --strong-model.",
    )
    verify.add_argument(
        "--strong-model-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for optional model-backed verification.",
    )
    verify.add_argument(
        "--strong-size",
        type=_parse_cli_size,
        action="append",
        dest="strong_sizes",
        help="Output size for model-backed verification cases as WIDTHxHEIGHT. Repeat for multiple strong sizes.",
    )
    verify.add_argument("--similarity-model", help="Optional similarity model id/path for --strong-model.")
    verify.add_argument(
        "--strong-continuity-backend",
        choices=STRONG_CONTINUITY_BACKENDS,
        default="local",
        help="Optional image-to-image continuity scorer for an extra strong-model refine case.",
    )
    verify.add_argument("--continuity-model", help="Optional continuity model id/path for --strong-continuity-backend.")
    verify.add_argument("--caption-model", help="Optional BLIP model id/path for --strong-model.")
    verify.add_argument(
        "--caption-similarity-backend",
        choices=CAPTION_SIMILARITY_BACKENDS,
        default="local",
        help="Prompt/caption text similarity scorer for strong-model cases.",
    )
    verify.add_argument("--caption-similarity-model", help="Optional semantic prompt/caption similarity model id/path.")
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
                save_candidates=args.save_candidates,
                auto_refine=args.auto_refine,
                similarity_backend=args.similarity_backend,
                similarity_model=args.similarity_model,
                similarity_device=args.similarity_device,
                continuity_backend=args.continuity_backend,
                continuity_model=args.continuity_model,
                continuity_device=args.continuity_device,
                caption_backend=args.caption_backend,
                caption_model=args.caption_model,
                caption_device=args.caption_device,
                caption_similarity_backend=args.caption_similarity_backend,
                caption_similarity_model=args.caption_similarity_model,
                caption_similarity_device=args.caption_similarity_device,
            )
        )
        print(f"Generated {result.image_path}")
        print(f"Metadata {result.metadata_path}")
        print(f"Score {result.metadata['total_score']}")
        print(f"Quality {result.metadata['quality_status']} {result.metadata['quality_score']} ({result.metadata['quality_report']})")
        print(f"Caption {result.metadata['image_caption']}")
        if result.metadata.get("critique_request"):
            print(f"Critique request {result.metadata['critique_request']}")
        if result.candidates_path:
            print(f"Candidates {result.candidates_path}")
            print(f"Contact sheet {result.metadata['candidate_contact_sheet']}")
        return 0

    if args.command == "refine":
        result = refine_image(
            RefineOptions(
                from_dir=args.from_dir,
                prompt=args.prompt,
                output_dir=args.output_dir,
                reference_image=args.reference_image,
                scene_plan=args.scene_plan,
                critique=args.critique,
                width=args.width,
                height=args.height,
                candidate_rank=args.candidate_rank,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                seed=args.seed,
                pixel_csv=args.pixel_csv,
                save_candidates=args.save_candidates,
                auto_refine=args.auto_refine,
                similarity_backend=args.similarity_backend,
                similarity_model=args.similarity_model,
                similarity_device=args.similarity_device,
                continuity_backend=args.continuity_backend,
                continuity_model=args.continuity_model,
                continuity_device=args.continuity_device,
                caption_backend=args.caption_backend,
                caption_model=args.caption_model,
                caption_device=args.caption_device,
                caption_similarity_backend=args.caption_similarity_backend,
                caption_similarity_model=args.caption_similarity_model,
                caption_similarity_device=args.caption_similarity_device,
            )
        )
        print(f"Refined {result.image_path}")
        print(f"Metadata {result.metadata_path}")
        print(f"Score {result.metadata['total_score']}")
        print(f"Quality {result.metadata['quality_status']} {result.metadata['quality_score']} ({result.metadata['quality_report']})")
        print(f"Caption {result.metadata['image_caption']}")
        if result.metadata.get("critique_request"):
            print(f"Critique request {result.metadata['critique_request']}")
        if result.metadata.get("comparison_request"):
            print(f"Comparison request {result.metadata['comparison_request']}")
        if result.metadata.get("parent_candidate_rank"):
            print(f"Candidate rank {result.metadata['parent_candidate_rank']}")
        if result.metadata.get("parent_candidate_selection"):
            print(f"Candidate selection {result.metadata['parent_candidate_selection']}")
        if result.candidates_path:
            print(f"Candidates {result.candidates_path}")
            print(f"Contact sheet {result.metadata['candidate_contact_sheet']}")
        print(f"Initial similarity {result.metadata['initial_similarity_score']}")
        refinement_delta = result.metadata.get("refinement_delta")
        if isinstance(refinement_delta, dict):
            print(
                "Refinement delta "
                f"total {refinement_delta.get('total_score_delta')} "
                f"quality {refinement_delta.get('quality_score_delta')} "
                f"caption {refinement_delta.get('caption_similarity_delta')}"
            )
        critique_signal_data = result.metadata.get("visual_critique")
        if isinstance(critique_signal_data, dict):
            print(
                f"Critique {critique_signal_data.get('verdict')} "
                f"closeness {critique_signal_data.get('closeness_score')} "
                f"(applied {len(critique_signal_data.get('applied_edits', []))} edits)"
            )
        return 0

    if args.command == "verify":
        report = run_verification(
            VerifyOptions(
                output_dir=args.output_dir,
                sizes=tuple(args.sizes) if args.sizes else DEFAULT_VERIFY_SIZES,
                prompt=args.prompt,
                refine_prompt=args.refine_prompt,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                save_candidates=args.save_candidates,
                strong_model=args.strong_model,
                strong_similarity_backend=args.strong_similarity_backend,
                strong_model_device=args.strong_model_device,
                strong_sizes=tuple(args.strong_sizes) if args.strong_sizes else None,
                similarity_model=args.similarity_model,
                strong_continuity_backend=args.strong_continuity_backend,
                continuity_model=args.continuity_model,
                caption_model=args.caption_model,
                caption_similarity_backend=args.caption_similarity_backend,
                caption_similarity_model=args.caption_similarity_model,
            )
        )
        print(f"Verification {report['status']} ({report['report_path']})")
        print(f"Cases {len(report['cases'])}")
        print(f"Strong model {report['strong_model']}")
        for case in report["cases"]:
            print(f"Case {case['type']} {case['status']} {case['size']} {case.get('quality_status', 'none')} {case['output_dir']}")
        return 0 if report["status"] == "pass" else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _parse_cli_size(value: str) -> tuple[int, int]:
    try:
        return parse_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
