# Claude ImageGen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only Claude Code plugin prototype that generates a 720x480 capped PNG from a prompt plus optional reference/initial image.

**Architecture:** Package the repo as a Claude Code plugin and a small Python package. Claude Code invokes a skill, the skill runs `claude-imagegen`, and the Python generator iteratively mutates compact scene candidates until a surrogate text/reference similarity score reaches a threshold or iteration limit.

**Tech Stack:** Python 3.10+, Pillow, NumPy, pytest, Claude Code plugin manifest/skill/bin layout.

---

## File Structure

- `.claude-plugin/plugin.json`: Claude Code plugin manifest.
- `skills/generate-image/SKILL.md`: Claude Code workflow instructions.
- `bin/claude-imagegen`: shell executable for Claude Code.
- `pyproject.toml`: package metadata, dependencies, console script, pytest config.
- `README.md`: usage, limits, examples, research grounding.
- `docs/research.md`: concise paper notes and design implications.
- `src/claude_imagegen/*.py`: prompt parsing, scene planning, rendering, scoring, generation, CLI.
- `tests/*.py`: unit, CLI, and plugin-structure tests.

## Tasks

- [ ] **Task 1: Create failing tests for public behavior**

  Add tests for prompt parsing, dimension capping, scoring, CLI output files, reference palette influence, pixel CSV export, and plugin packaging. Run `python -m pytest` and confirm the tests fail because implementation files do not exist.

- [ ] **Task 2: Add package skeleton and prompt/palette parsing**

  Create `pyproject.toml`, package init, prompt parsing, palette helpers, and dataclasses. Run targeted tests and confirm parser/palette tests pass while downstream tests still fail.

- [ ] **Task 3: Add scene rendering and scoring**

  Implement compact scene candidate creation, mutation, Pillow rendering, RGB output capping, and surrogate cosine scoring. Run renderer/scorer tests.

- [ ] **Task 4: Add generator loop, CLI, and pixel export**

  Implement the iterative search loop, output metadata/progress files, optional pixel CSV, and command-line interface. Run CLI tests.

- [ ] **Task 5: Add Claude Code plugin files and docs**

  Add manifest, skill, executable shim, README, and research notes. Validate plugin JSON and run plugin structure tests. If the local Claude Code supports it, run `claude plugin validate . --strict`.

- [ ] **Task 6: End-to-end validation**

  Install dependencies in editable mode if needed, run the full test suite, generate a sample image, inspect metadata, and record the exact commands/results.

## Plan Self-Review

- Spec coverage: all design requirements map to the six tasks.
- Placeholder scan: no task depends on unspecified implementation details.
- Type consistency: the planned public command is consistently `claude-imagegen generate`.
