import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

# Skip the entire module if PyYAML is not available
if importlib.util.find_spec("yaml") is None:
    pytest.skip("PyYAML not installed", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "tools" / "validate_config.py"


def _write_valid_configs(directory: Path) -> None:
    (directory / "recursive.yaml").write_text(
        "mode: auto\npsi_threshold: 0.5\nmax_depth: 3\n", encoding="utf-8"
    )
    (directory / "vault.yaml").write_text(
        "memory_retention: 10\nsimilarity_threshold: 0.9\n", encoding="utf-8"
    )
    (directory / "matrix_response_paths.yaml").write_text(
        "fault_responses:\n  error: ignore\ndefault_action: hold\n",
        encoding="utf-8",
    )
    (directory / "braid.yaml").write_text(
        "patterns:\n  - a\nconfidence_threshold: 0.1\n", encoding="utf-8"
    )
    (directory / "logging.yaml").write_text(
        "level: INFO\nformat: '%(levelname)s'\noutput: stdout\n", encoding="utf-8"
    )


def _run_cli(cfg_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), "--config-dir", str(cfg_dir)],
        text=True,
        capture_output=True,
    )


def test_valid_configs_exit_zero():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_dir = Path(tmp)
        _write_valid_configs(cfg_dir)
        result = _run_cli(cfg_dir)
        assert result.returncode == 0
        assert "Overall Status: [OK] All OK" in result.stdout


def test_invalid_configs_exit_nonzero():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_dir = Path(tmp)
        _write_valid_configs(cfg_dir)
        (cfg_dir / "vault.yaml").write_text(
            "memory_retention: 'ten'\nsimilarity_threshold: 0.9\n",
            encoding="utf-8",
        )
        result = _run_cli(cfg_dir)
        assert result.returncode != 0
        assert "Overall Status: [FAIL] Issues Found" in result.stdout 