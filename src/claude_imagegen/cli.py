from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import sys

from .critique import write_pair_evaluation_request
from .diffusion import DIFFUSION_PROFILE_NAMES, DiffusionOptions, generate_diffusion_image
from .enhance import EnhanceNightOptions, enhance_night_image
from .eval_plan import EvalPlanOptions, build_eval_plan
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
    generate.add_argument(
        "--quality-target",
        type=float,
        help="Optional high-quality acceptance target; 0.9 requires independent Claude visual critique plus local detail evidence.",
    )
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

    diffuse = subcommands.add_parser(
        "diffuse",
        help="Generate higher-detail images with optional local Diffusers/Torch backends.",
    )
    diffuse.add_argument("--prompt", required=True, help="Text prompt to generate.")
    diffuse.add_argument("--negative-prompt", help="Negative prompt for the diffusion model.")
    diffuse.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output/diffusion"))
    diffuse.add_argument("--model", help="Diffusers text-to-image model id/path; defaults come from --profile.")
    diffuse.add_argument(
        "--profile",
        choices=DIFFUSION_PROFILE_NAMES,
        default="turbo",
        help="Local generation profile. Use night-photoreal for detailed deep-night photoreal attempts.",
    )
    diffuse.add_argument("--width", type=int, default=1024)
    diffuse.add_argument("--height", type=int, default=768)
    diffuse.add_argument("--steps", type=int, help="Override the profile's diffusion step count.")
    diffuse.add_argument("--guidance-scale", type=float, help="Override the profile's classifier-free guidance scale.")
    diffuse.add_argument(
        "--seeds",
        type=_parse_seed_list,
        default=(101, 202, 303, 404),
        help="Comma-separated seed list for multi-candidate generation.",
    )
    diffuse.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for local diffusion generation.",
    )
    diffuse.add_argument(
        "--quality-target",
        type=float,
        help="Optional acceptance target; 0.9 still requires Claude visual critique of image.png.",
    )
    diffuse.add_argument(
        "--prompt-focus",
        type=_parse_csv_list,
        default=("auto",),
        help="Comma-separated prompt-critical terms for candidate ranking; default auto derives terms from prompt.",
    )

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
    refine.add_argument(
        "--comparison",
        type=Path,
        help="Optional Claude-authored parent/child comparison JSON with follow_up_edits to apply before rendering.",
    )
    refine.add_argument("--width", type=int, help="Output width; defaults to parent metadata width.")
    refine.add_argument("--height", type=int, help="Output height; defaults to parent metadata height.")
    refine.add_argument("--candidate-rank", help="Use a ranked candidate number, or 'auto', from the parent output candidates.json as the initial image.")
    refine.add_argument("--max-iterations", type=int, default=32)
    refine.add_argument("--threshold", type=float, default=0.58)
    refine.add_argument(
        "--quality-target",
        type=float,
        help="Optional high-quality acceptance target; 0.9 requires independent Claude visual critique plus local detail evidence.",
    )
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

    pair_eval = subcommands.add_parser(
        "pair-eval",
        help="Write a Claude vision request for scoring existing before/after image pairs without generating images.",
    )
    pair_eval.add_argument("--prompt", required=True, help="Prompt the images should satisfy.")
    pair_eval.add_argument("--before", type=Path, action="append", required=True, help="Existing before image. Repeat with --after.")
    pair_eval.add_argument("--after", type=Path, action="append", required=True, help="Existing after image. Repeat with --before.")
    pair_eval.add_argument("--pair-id", action="append", help="Optional id for each before/after pair.")
    pair_eval.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output/pair-eval"))
    pair_eval.add_argument("--quality-target", type=float, default=0.9)
    pair_eval.add_argument("--notes", default="", help="Optional context for Claude's evaluator request.")

    enhance_night = subcommands.add_parser(
        "enhance-night",
        help="Postprocess an existing local image while preserving deep-night exposure.",
    )
    enhance_night.add_argument("--input-image", type=Path, required=True, help="Existing image to enhance.")
    enhance_night.add_argument("--prompt", required=True, help="Prompt the enhanced image should satisfy.")
    enhance_night.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output/enhance-night"))
    enhance_night.add_argument("--quality-target", type=float, default=0.9)
    enhance_night.add_argument("--night-luma-ceiling", type=float, default=0.34)
    enhance_night.add_argument("--mist-cap", type=float, default=0.22)
    enhance_night.add_argument("--highlight-rolloff", type=float, default=0.35)
    enhance_night.add_argument("--local-contrast", type=float, default=0.9)

    eval_plan = subcommands.add_parser(
        "eval-plan",
        help="Turn a Claude pair-evaluation response into a concrete next-step improvement plan.",
    )
    eval_plan.add_argument(
        "--evaluation",
        type=Path,
        action="append",
        required=True,
        help="Claude-filled pair evaluation JSON. Repeat to aggregate multiple judge passes conservatively.",
    )
    eval_plan.add_argument("--prompt", required=True, help="Prompt the image pair should satisfy.")
    eval_plan.add_argument("--output-dir", type=Path, default=Path("claude-imagegen-output/eval-plan"))
    eval_plan.add_argument("--quality-target", type=float, default=0.9)
    eval_plan.add_argument(
        "--min-evaluations",
        type=int,
        default=2,
        help="Minimum Claude judge responses required before accepting the quality gate.",
    )

    setup = subcommands.add_parser("setup", help="Check first-run dependencies and print setup status.")
    setup.add_argument(
        "--with-diffusion",
        action="store_true",
        help="Also report optional local Diffusers/Torch dependencies for the higher-detail backend.",
    )
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
                quality_target=args.quality_target,
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

    if args.command == "diffuse":
        result = generate_diffusion_image(
            DiffusionOptions(
                prompt=args.prompt,
                output_dir=args.output_dir,
                negative_prompt=args.negative_prompt,
                model=args.model,
                profile=args.profile,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                seeds=args.seeds,
                device=args.device,
                quality_target=args.quality_target,
                prompt_focus=args.prompt_focus,
            )
        )
        print(f"Generated {result.image_path}")
        print(f"Metadata {result.metadata_path}")
        print(f"Quality {result.metadata['quality_status']} {result.metadata['quality_score']} ({result.metadata['quality_report']})")
        print(f"Profile {result.metadata['diffusion_profile']} model {result.metadata['model']}")
        print(f"Selected seed {result.metadata['selected_seed']} on {result.metadata['effective_device']}")
        print(f"Prompt signal {result.metadata['prompt_signal_score']} terms {', '.join(result.metadata['prompt_focus_terms'])}")
        if result.metadata.get("prompt_length_warning"):
            print(f"Warning {result.metadata['prompt_length_warning']}")
        print(f"Candidates {result.candidates_path}")
        print(f"Contact sheet {result.contact_sheet_path}")
        print(f"Critique request {result.critique_request_path}")
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
                comparison=args.comparison,
                width=args.width,
                height=args.height,
                candidate_rank=args.candidate_rank,
                max_iterations=args.max_iterations,
                threshold=args.threshold,
                quality_target=args.quality_target,
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
        comparison_signal_data = result.metadata.get("visual_comparison")
        if isinstance(comparison_signal_data, dict):
            print(
                f"Comparison {comparison_signal_data.get('verdict')} "
                f"alignment {comparison_signal_data.get('alignment_score')} "
                f"continuity {comparison_signal_data.get('continuity_score')} "
                f"(applied {len(comparison_signal_data.get('applied_edits', []))} edits)"
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
        device_summary = report.get("device_summary")
        if isinstance(device_summary, dict):
            devices = ", ".join(str(device) for device in device_summary.get("devices", [])) or "none"
            print(
                f"Devices {devices} "
                f"(cpu cases {device_summary.get('cpu_case_count', 0)}, "
                f"cuda cases {device_summary.get('cuda_case_count', 0)})"
            )
        image_summary = report.get("image_summary")
        if isinstance(image_summary, dict):
            print(
                f"Images nonblank {image_summary.get('nonblank_cases', 0)}/"
                f"{image_summary.get('case_count', 0)} "
                f"(blank {image_summary.get('blank_cases', 0)}, "
                f"min variance {image_summary.get('min_variance_sum')})"
            )
        for case in report["cases"]:
            print(f"Case {case['type']} {case['status']} {case['size']} {case.get('quality_status', 'none')} {case['output_dir']}")
        return 0 if report["status"] == "pass" else 1

    if args.command == "pair-eval":
        if len(args.before) != len(args.after):
            parser.error("--before and --after must be supplied the same number of times")
        pair_ids = args.pair_id or []
        if pair_ids and len(pair_ids) != len(args.before):
            parser.error("--pair-id must be supplied once per --before/--after pair")
        pairs = [
            {
                "id": pair_ids[index] if pair_ids else f"pair-{index + 1}",
                "before_image": str(before),
                "after_image": str(after),
            }
            for index, (before, after) in enumerate(zip(args.before, args.after))
        ]
        request_path = write_pair_evaluation_request(
            args.output_dir,
            prompt=args.prompt,
            pairs=pairs,
            quality_target=args.quality_target,
            notes=args.notes,
        )
        print(f"Pair evaluation request {request_path}")
        print("Open this JSON with Claude vision and fill expected_response before claiming 0.9 quality.")
        return 0

    if args.command == "enhance-night":
        result = enhance_night_image(
            EnhanceNightOptions(
                input_image=args.input_image,
                prompt=args.prompt,
                output_dir=args.output_dir,
                quality_target=args.quality_target,
                night_luma_ceiling=args.night_luma_ceiling,
                mist_cap=args.mist_cap,
                highlight_rolloff=args.highlight_rolloff,
                local_contrast=args.local_contrast,
            )
        )
        print(f"Enhanced {result.image_path}")
        print(f"Metadata {result.metadata_path}")
        print(f"Mean luma {result.metadata['before_mean_luma']} -> {result.metadata['after_mean_luma']}")
        print(f"Lower contrast {result.metadata['before_lower_luma_std']} -> {result.metadata['after_lower_luma_std']}")
        print(f"Pair evaluation request {result.pair_evaluation_request_path}")
        return 0

    if args.command == "eval-plan":
        result = build_eval_plan(
            EvalPlanOptions(
                evaluations=tuple(args.evaluation),
                prompt=args.prompt,
                output_dir=args.output_dir,
                quality_target=args.quality_target,
                min_evaluations=args.min_evaluations,
            )
        )
        print(f"Improvement plan {result.plan_path}")
        print(f"Next action {result.plan['next_action']}")
        print(f"Score gap {result.plan['score_gap']}")
        if result.plan.get("recommended_command"):
            print(f"Recommended command {result.plan['recommended_command']}")
        return 0

    if args.command == "setup":
        status = _setup_status(include_diffusion=args.with_diffusion)
        print("claude-imagegen setup ok" if status["ready"] else "claude-imagegen setup incomplete")
        print(f"Python {status['python_version']} ({status['python_executable']})")
        for dependency in status["dependencies"]:
            state = "ok" if dependency["available"] else "missing"
            print(f"{dependency['name']} {state}")
        print("First run bootstrap creates a plugin-owned virtual environment when numpy or Pillow are missing.")
        if args.with_diffusion:
            print("Diffusion optional ready" if status["diffusion_ready"] else "Diffusion optional incomplete")
            for dependency in status["diffusion_dependencies"]:
                state = "ok" if dependency["available"] else "missing"
                print(f"{dependency['name']} {state}")
            print('Install diffusion extras with: python -m pip install -e ".[diffusion]"')
        return 0 if status["ready"] else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _parse_cli_size(value: str) -> tuple[int, int]:
    try:
        return parse_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(seed.strip()) for seed in value.split(",") if seed.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_csv_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one item")
    return items


def _setup_status(*, include_diffusion: bool = False) -> dict[str, object]:
    dependencies = [
        {"name": "numpy", "module": "numpy"},
        {"name": "Pillow", "module": "PIL"},
    ]
    statuses = [
        {
            "name": dependency["name"],
            "available": importlib.util.find_spec(str(dependency["module"])) is not None,
        }
        for dependency in dependencies
    ]
    status: dict[str, object] = {
        "ready": all(bool(dependency["available"]) for dependency in statuses),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "dependencies": statuses,
    }
    if include_diffusion:
        diffusion_dependencies = [
            {"name": "torch", "module": "torch"},
            {"name": "diffusers", "module": "diffusers"},
            {"name": "accelerate", "module": "accelerate"},
            {"name": "transformers", "module": "transformers"},
        ]
        diffusion_statuses = [
            {
                "name": dependency["name"],
                "available": importlib.util.find_spec(str(dependency["module"])) is not None,
            }
            for dependency in diffusion_dependencies
        ]
        status["diffusion_dependencies"] = diffusion_statuses
        status["diffusion_ready"] = all(bool(dependency["available"]) for dependency in diffusion_statuses)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
