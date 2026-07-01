import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_plugin_manifest_has_required_metadata():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "claude-imagegen"
    assert manifest["displayName"] == "Claude ImageGen"
    assert manifest["version"]
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
            "description": "Generate local CPU-first images from Claude-authored scene plans with optional model-backed checks.",
            "version": "0.1.0",
            "source": "./",
            "category": "creative",
            "keywords": ["image-generation", "scene-plan", "cpu-renderer", "caption-backcheck"],
            "tags": ["image-generation", "creative", "cpu", "caption"],
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
    assert "--scene-plan" in skill_text
    assert "scene-plan.json" in skill_text
    assert '"stops"' in skill_text
    assert '"elements"' in skill_text
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
    assert "revision_hints" in skill_text
    assert "critique-request.json" in skill_text
    assert "comparison-request.json" in skill_text
    assert "refine --critique" in skill_text
    assert "refinement_delta" in skill_text
    assert "--strong-size" in skill_text
    assert "complex planned scene" in skill_text
    assert "reference_palette" in skill_text
    assert "initial_palette" in skill_text
    assert "--caption-backend" in skill_text
    assert "caption_similarity_score" in skill_text
    assert executable.exists()
    executable_text = executable.read_text(encoding="utf-8")
    assert "claude_imagegen.cli" in executable_text
    assert "CLAUDE_PLUGIN_DATA" in executable_text
    assert "python3 -m venv" in executable_text
    assert 'pip" install' in executable_text


def test_shell_entrypoint_is_forced_to_lf_on_checkout():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "bin/* text eol=lf" in attributes


def test_readme_documents_claude_plugin_install_flow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "critique-request.json" in readme
    assert "comparison-request.json" in readme
    assert "refinement_delta" in readme
    assert "--strong-size" in readme
    assert "complex planned scene" in readme
    assert "claude plugin marketplace add rexkoh425/ClaudeImageGen" in readme
    assert "claude plugin install claude-imagegen@claude-imagegen" in readme
    assert "claude plugin marketplace add ./" in readme
    assert "claude plugin validate . --strict" in readme
    assert "claude plugin validate .claude-plugin/marketplace.json --strict" in readme
