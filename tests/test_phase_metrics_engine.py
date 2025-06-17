import sys
import types
import numpy as np
import pytest
from unittest.mock import patch, mock_open

# Provide a minimal cupy stub so module imports succeed
cupy_stub = types.SimpleNamespace(array=np.array, asarray=np.asarray)
sys.modules.setdefault('cupy', cupy_stub)


@pytest.fixture
def engine():
    mock_config = {
        "metrics": {
            "window_sizes": {
                "short": 100,
                "medium": 500,
                "long": 1000
            },
            "update_interval_ms": 1000,
            "gpu_acceleration": False,
            "drift_detection": {
                "window_size": 1000,
                "threshold": 0.7
            },
            "entropy": {
                "bins": 50,
                "min_probability": 1e-10
            }
        }
    }
    
    with patch('builtins.open', mock_open(read_data='{"metrics": {"window_sizes": {"short": 100, "medium": 500, "long": 1000}, "update_interval_ms": 1000, "gpu_acceleration": false, "drift_detection": {"window_size": 1000, "threshold": 0.7}, "entropy": {"bins": 50, "min_probability": 1e-10}}}')):
        with patch('json.load', return_value=mock_config):
            from core.phase_engine.phase_metrics_engine import PhaseMetricsEngine
            eng = PhaseMetricsEngine()
            # Ensure GPU disabled for predictable CPU execution
            eng.use_gpu = False
            return eng


def test_compute_metrics_cpu_fallback(engine):
    prices = np.array([1.0, 1.1, 1.2, 1.3])
    volumes = np.array([10, 12, 11, 13])

    with patch('core.phase_engine.phase_metrics_engine.cp.asarray') as mock_asarray:
        metrics = engine.compute_metrics(prices, volumes)
        mock_asarray.assert_not_called()

    expected_keys = {
        'profit_trend',
        'stability',
        'memory_coherence',
        'paradox_pressure',
        'entropy_rate',
        'thermal_state',
        'bit_depth',
        'trust_score',
    }
    assert set(metrics.keys()) == expected_keys


def test_validate_metrics(engine):
    prices = np.array([1.0, 1.1, 1.2, 1.3])
    volumes = np.array([10, 12, 11, 13])
    metrics = engine.compute_metrics(prices, volumes)

    # Valid metrics should return no errors
    assert engine.validate_metrics(metrics) == []

    # Introduce invalid values
    invalid = dict(metrics)
    invalid['stability'] = 1.5  # outside [0,1]
    invalid['bit_depth'] = 300  # outside allowed range
    errors = engine.validate_metrics(invalid)
    assert errors and all(isinstance(e, str) for e in errors)


def test_initialize_gpu_error_handling(monkeypatch):
    mock_config = {
        "metrics": {
            "window_sizes": {
                "short": 100,
                "medium": 500,
                "long": 1000
            },
            "update_interval_ms": 1000,
            "gpu_acceleration": True,
            "drift_detection": {
                "window_size": 1000,
                "threshold": 0.7
            },
            "entropy": {
                "bins": 50,
                "min_probability": 1e-10
            }
        }
    }
    
    with patch('builtins.open', mock_open()):
        with patch('json.load', return_value=mock_config):
            from core.phase_engine.phase_metrics_engine import PhaseMetricsEngine
            eng = PhaseMetricsEngine()
            eng.use_gpu = True

            def raise_error(_):
                raise RuntimeError('GPU failure')

            monkeypatch.setattr('core.phase_engine.phase_metrics_engine.cp.array', raise_error)
            eng._initialize_gpu()
            assert eng.use_gpu is False 