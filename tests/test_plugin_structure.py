import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.1.16"


def test_plugin_manifest_has_required_metadata():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "claude-imagegen"
    assert manifest["displayName"] == "Claude ImageGen"
    assert manifest["version"] == EXPECTED_VERSION
    assert "Image generation" in manifest["description"]
    assert manifest["license"] == "MIT"
    assert "image-generation" in manifest["keywords"]


def test_repo_contains_installable_claude_marketplace_manifest():
    marketplace = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))

    assert marketplace["name"] == "claude-imagegen"
    assert "GitHub" in marketplace["description"]
    assert marketplace["version"]
    assert marketplace["owner"]["name"] == "ClaudeImageGen"
    assert marketplace["plugins"] == [
        {
            "name": "claude-imagegen",
            "displayName": "Claude ImageGen",
            "description": "Generate local CPU-first images from Claude-authored scene plans with optional GPU diffusion.",
            "version": EXPECTED_VERSION,
            "source": "./",
            "category": "creative",
            "keywords": ["image-generation", "scene-plan", "cpu-renderer", "caption-backcheck", "diffusers"],
            "tags": ["image-generation", "creative", "cpu", "caption", "gpu"],
            "license": "MIT",
            "strict": True,
            "defaultEnabled": True,
        }
    ]


def test_claude_skill_and_executable_are_present():
    skill = ROOT / "skills" / "generate-image" / "SKILL.md"
    executable = ROOT / "bin" / "claude-imagegen"

    skill_text = skill.read_text(encoding="utf-8")

    assert skill.exists()
    assert "claude-imagegen generate" in skill_text
    assert "claude-imagegen diffuse" in skill_text
    assert "--initial-image" in skill_text
    assert "--strength 0.16" in skill_text
    assert "claude-imagegen pair-eval" in skill_text
    assert "claude-imagegen enhance-night" in skill_text
    assert "claude-imagegen eval-plan" in skill_text
    assert "claude-imagegen audit-pair" in skill_text
    assert "--min-evaluations" in skill_text
    assert "--audit" in skill_text
    assert "--shadow-lift" in skill_text
    assert "--foliage-clarity" in skill_text
    assert "pair-evaluation-request.json" in skill_text
    assert "--scene-plan" in skill_text
    assert "scene-plan.json" in skill_text
    assert '"stops"' in skill_text
    assert '"elements"' in skill_text
    assert "`text`" in skill_text
    assert "`arrow`" in skill_text
    assert "`rounded_rectangle`" in skill_text
    assert "`aperture`" in skill_text
    assert "`sparkle`" in skill_text
    assert "`stroke_width`" in skill_text
    assert '"path"' in skill_text
    assert '"blur"' in skill_text
    assert '"blend"' in skill_text
    assert "`overlay`" in skill_text
    assert "`soft-light`" in skill_text
    assert '"gradient"' in skill_text
    assert '"motifs"' in skill_text
    assert '"textures"' in skill_text
    assert '"materials"' in skill_text
    assert '"terrains"' in skill_text
    assert '"reflections"' in skill_text
    assert '"warps"' in skill_text
    assert '"atmosphere"' in skill_text
    assert '"veils"' in skill_text
    assert '"lights"' in skill_text
    assert '"beams"' in skill_text
    assert '"occlusions"' in skill_text
    assert '"clouds"' in skill_text
    assert '"shadows"' in skill_text
    assert '"focus"' in skill_text
    assert '"saturation"' in skill_text
    assert '"contrast"' in skill_text
    assert '"warmth"' in skill_text
    assert '"bloom"' in skill_text
    assert '"antialias"' in skill_text
    assert '"detail"' in skill_text
    assert '"sharpen"' in skill_text
    assert "revision_hints" in skill_text
    assert "critique-request.json" in skill_text
    assert "comparison-request.json" in skill_text
    assert "refine --critique" in skill_text
    assert "refine --comparison" in skill_text
    assert "`update_element`" in skill_text
    assert "visual_comparison" in skill_text
    assert "visual_checklist" in skill_text
    assert "element_checks" in skill_text
    assert "failed checklist items" in skill_text
    assert "checklist-derived edits" in skill_text
    assert "style/mood checks" in skill_text
    assert "refinement_delta" in skill_text
    assert "refinement_guidance" in skill_text
    assert "multiscale_luminance_ssim_score" in skill_text
    assert "weakest_continuity_region" in skill_text
    assert "--strong-size" in skill_text
    assert "complex planned scene" in skill_text
    assert "reference_palette" in skill_text
    assert "initial_palette" in skill_text
    assert "--caption-backend" in skill_text
    assert "caption_similarity_score" in skill_text
    assert "aesthetic_score" in skill_text
    assert "device_summary" in skill_text
    assert "image_summary" in skill_text
    assert "nonblank" in skill_text
    assert "--quality-target 0.9" in skill_text
    assert "multi-refinement" in skill_text
    assert "avoid label overlap" in skill_text
    assert "inset badges" in skill_text
    assert "separate image tiles and labels" in skill_text
    assert "Do not shorten the user's prompt" in skill_text
    assert "Do not drop `--quality-target 0.9`" in skill_text
    assert "GPT/Sora parity" in skill_text
    assert '"greenhouse"' in skill_text
    assert '"plant"' in skill_text
    assert '"lamp"' in skill_text
    assert '"floor"' in skill_text
    assert "claude-imagegen setup" in skill_text
    assert "claude-imagegen setup --with-diffusion" in skill_text
    assert executable.exists()
    executable_text = executable.read_text(encoding="utf-8")
    assert "claude_imagegen.cli" in executable_text
    assert "CLAUDE_PLUGIN_DATA" in executable_text
    assert "python3 -m venv" in executable_text
    assert 'pip" install' in executable_text
    assert "PIP_DISABLE_PIP_VERSION_CHECK=1" in executable_text
    assert "--quiet" in executable_text


def test_shell_entrypoint_is_forced_to_lf_on_checkout():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "bin/* text eol=lf" in attributes


def test_readme_documents_claude_plugin_install_flow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(readme.splitlines()) <= 170
    assert "Fastest setup" in readme
    assert "Best Result Loop" in readme
    assert "Claude does the planning and critique" in readme
    assert "Claude ImageGen is available through the Claude Code plugin marketplace" in readme
    assert "On another machine" in readme
    assert "sign in to GitHub if this repo is private" in readme
    assert "Restart Claude Code after installation" in readme
    assert "python -m pip install -e ." in readme
    assert 'python -m pip install -e ".[diffusion]"' in readme
    assert "claude-imagegen diffuse" in readme
    assert "--initial-image" in readme
    assert "--strength 0.16" in readme
    assert "claude-imagegen pair-eval" in readme
    assert "claude-imagegen enhance-night" in readme
    assert "claude-imagegen eval-plan" in readme
    assert "claude-imagegen audit-pair" in readme
    assert "--min-evaluations" in readme
    assert "--audit" in readme
    assert "--shadow-lift" in readme
    assert "--foliage-clarity" in readme
    assert "--mist-beam-strength" in readme
    assert "claude-imagegen setup --with-diffusion" in readme
    assert "--profile night-photoreal" in readme
    assert "`rounded_rectangle`" in readme
    assert "`aperture`" in readme
    assert "`sparkle`" in readme
    assert "`stroke_width`" in readme
    assert "scene-plan.json" in readme
    assert "image.png" in readme
    assert "metadata.json" in readme
    assert "quality-report.json" in readme
    assert "claude-imagegen setup" in readme
    assert "--quality-target 0.9" in readme
    assert "GPT/Sora-level parity is not claimed" in readme
    assert "multi-refinement" in readme
    assert "critique-request.json" in readme
    assert "comparison-request.json" in readme
    assert "pair-evaluation-request.json" in readme
    assert "verification-report.json" in readme
    assert "device_summary" in readme
    assert "image_summary" in readme
    assert "nonblank" in readme
    assert "--strong-size" in readme
    assert "## Install In Claude Code" in readme
    assert "/plugin marketplace add rexkoh425/ClaudeImageGen" in readme
    assert "/plugin install claude-imagegen@claude-imagegen" in readme
    assert "claude plugin marketplace add ./" not in readme
    assert "claude plugin validate . --strict" in readme
    assert "claude plugin validate .claude-plugin/marketplace.json --strict" in readme
