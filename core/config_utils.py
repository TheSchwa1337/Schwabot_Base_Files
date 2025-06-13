from pathlib import Path
import yaml
from config.schemas.quantization import QuantizationSchema
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_FRACTAL_PATH = CONFIG_DIR / "fractal_core.yaml"

def load_yaml_config(path: Path) -> dict:
    if not path.exists():
        logger.error(f"YAML config not found at: {path}")
        raise FileNotFoundError(f"YAML config not found at: {path}")
    with path.open("r") as f:
        logger.info(f"Loading YAML config from {path}")
        return yaml.safe_load(f)

def create_default_fractal_config(path: Path = DEFAULT_FRACTAL_PATH):
    default = {
        "profile": {
            "name": "default",
            "type": "quantization",
            "parameters": {
                "decay_power": 1.5,
                "terms": 12,
                "dimension": 8,
                "epsilon_q": 0.003,
                "precision": 0.001
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(default, f)

def get_profile_params_from_yaml(path: Path) -> dict:
    config = load_yaml_config(path)
    try:
        params = config["profile"]["parameters"]
        # Validate with Pydantic
        schema = QuantizationSchema(**params)
        return schema.dict()
    except (KeyError, TypeError):
        raise ValueError(f"Malformed YAML: missing 'profile.parameters' in {path}")
