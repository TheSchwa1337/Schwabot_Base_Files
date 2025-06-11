import pytest
from core.sfsss_strategy_bundler import SFSSSStrategyBundler

@pytest.fixture
def bundler():
    return SFSSSStrategyBundler()

@pytest.mark.parametrize("drift, echo, expected", [
    (0.8, 1.2, 'Tier3_HighProfit'),
    (0.5, 0.6, 'Tier2_MidProfit'),
    (0.3, 0.4, 'Tier1_LowProfit'),
    (0.1, 0.0, 'Tier0_Observe'),
])
def test_bundle_strategies_by_tier(drift, echo, expected, bundler):
    bundle = bundler.bundle_strategies_by_tier(drift, echo, 'Hint')
    assert bundle['strategy'].startswith(expected)

def test_bundle_parameters_present(bundler):
    bundle = bundler.bundle_strategies_by_tier(0.9, 1.5, None)
    assert 'params' in bundle
    assert isinstance(bundle['params'], dict) 