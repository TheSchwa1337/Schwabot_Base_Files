from pathlib import Path

# Shared constants across the Schwabot code-base
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_FRACTAL_PATH = CONFIG_DIR / "fractal_core.yaml" 