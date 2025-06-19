from pathlib import Path
from core.config_utils import get_profile_params_from_yaml
from core.quantization_profile import QuantizationProfile


def test_load_fractal_profile():
    path = Path("tests/configs/sample_fractal.yaml")
    params = get_profile_params_from_yaml(path)
    profile = QuantizationProfile(**params)
    assert profile.dimension == 6
    assert 0 < profile.epsilon_q < 1