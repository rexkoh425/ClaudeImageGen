import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pair_images(tmp_path: Path) -> tuple[Path, Path]:
    width, height = 96, 64
    before = np.full((height, width, 3), 0.18, dtype=np.float32)
    after = np.full((height, width, 3), 0.48, dtype=np.float32)
    for x in range(0, width, 8):
        before[:, x : x + 2, :] = 0.42
        after[:, x : x + 2, :] = 0.5
    before[8:16, 40:52, :] = (0.95, 0.68, 0.28)
    after[8:18, 40:54, :] = (1.0, 0.96, 0.78)
    for x in range(width):
        before[height // 2 :, x, :] += 0.05 if (x // 6) % 2 == 0 else -0.02
        after[height // 2 :, x, :] += 0.01 if (x // 6) % 2 == 0 else -0.005
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.fromarray(np.uint8(np.clip(before, 0.0, 1.0) * 255), "RGB").save(before_path)
    Image.fromarray(np.uint8(np.clip(after, 0.0, 1.0) * 255), "RGB").save(after_path)
    return before_path, after_path


def test_audit_pair_detects_overbright_hazy_after_image(tmp_path: Path):
    from claude_imagegen.pair_audit import PairAuditOptions, audit_pair

    before_path, after_path = _write_pair_images(tmp_path)

    result = audit_pair(
        PairAuditOptions(
            before_image=before_path,
            after_image=after_path,
            prompt="deep night greenhouse with lamps, mist, leaf detail, and wet floor reflections",
            output_dir=tmp_path / "audit",
            night_luma_ceiling=0.34,
        )
    )

    assert result.audit_path.exists()
    audit = result.audit
    assert audit["engine"] == "pair-local-audit-v1"
    assert audit["before_image"] == str(before_path)
    assert audit["after_image"] == str(after_path)
    assert audit["flags"]["night_mood_preserved"] is False
    assert audit["flags"]["overbright_after"] is True
    assert audit["flags"]["detail_softening_risk"] is True
    assert audit["flags"]["highlight_clipping_risk"] is True
    assert audit["deltas"]["mean_luma_delta"] > 0.2
    assert audit["deltas"]["edge_density_delta"] < 0
    assert audit["suggested_parameters"]["night_luma_ceiling"] == 0.3
    assert audit["suggested_parameters"]["mist_cap"] == 0.16
    assert "claude-imagegen enhance-night" in audit["recommended_command"]


def test_cli_audit_pair_writes_json_without_generating_image(tmp_path: Path):
    before_path, after_path = _write_pair_images(tmp_path)
    output_dir = tmp_path / "audit"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "claude_imagegen.cli",
            "audit-pair",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--prompt",
            "deep night greenhouse with lamps, mist, leaf detail, and wet floor reflections",
            "--output-dir",
            str(output_dir),
            "--night-luma-ceiling",
            "0.34",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    audit_path = output_dir / "pair-audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["flags"]["overbright_after"] is True
    assert "Pair audit" in completed.stdout
