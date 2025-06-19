"""
Mathematical Infrastructure Integration Test
==========================================

Comprehensive test for the mathematical foundations of the recursive trading system.
Tests integration between mathlib, mathlib_v2, math_core, \
    and Klein bottle systems.
"""

from core.math_core import UnifiedMathematicalProcessor, MATHEMATICAL_CONSTANTS  # noqa: F401
from core.mathlib_v2 import CoreMathLibV2, SmartStop, klein_bottle_collapse  # noqa: F401
from core.mathlib import CoreMathLib, GradedProfitVector  # noqa: F401
import pytest  # noqa: F401
import numpy as np  # noqa: F401
import sys  # noqa: F401
import os  # noqa: F401
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestMathematicalIntegration:
    """Test suite for comprehensive mathematical integration"""

    @pytest.fixture
    def math_lib(self):
        """Create CoreMathLib instance"""
        return CoreMathLib(
            base_volume=1000.0,
            tick_freq=60.0,
            profit_coef=1.2,
            threshold=0.5
        )

    @pytest.fixture
    def math_lib_v2(self):
        """Create CoreMathLibV2 instance"""
        return CoreMathLibV2(
            base_volume=1000.0,
            tick_freq=60.0,
            profit_coef=1.2,
            threshold=0.5
        )

    @pytest.fixture
    def math_processor(self):
        """Create UnifiedMathematicalProcessor instance"""
        return UnifiedMathematicalProcessor()

    @pytest.fixture
    def sample_data(self):
        """Generate sample market data"""
        np.random.seed(42)
        n_points = 100

        # Generate realistic price data with trend and noise
        t = np.linspace(0, 1, n_points)
        trend = 100 + 20 * t + 5 * np.sin(2 * np.pi * t * 5)
        noise = np.random.normal(0, 1, n_points)
        prices = trend + noise

        # Generate correlated volume data
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volumes = price_changes * 1000 + np.random.normal(0, 100, n_points)
        volumes = np.abs(volumes)  # Ensure positive volumes

        return {
            'prices': prices,
            'volumes': volumes,
            'high': prices + np.abs(np.random.normal(0, 0.5, n_points)),
            'low': prices - np.abs(np.random.normal(0, 0.5, n_points))
        }

    def test_basic_math_operations(self, math_lib):
        """Test basic mathematical operations"""
        # Test vector operations
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        similarity = math_lib.cosine_similarity(a, b)
        assert 0 <= similarity <= 1

        distance = math_lib.euclidean_distance(a, b)
        assert distance > 0

        normalized = math_lib.normalize_vector(a)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_graded_profit_vector(self, math_lib):
        """Test GradedProfitVector functionality"""
        trade_data = {
            'profit': 0.05,
            'volume_allocated': 1000.0,
            'time_held': 3600.0,
            'signal_strength': 0.8,
            'smart_money_score': 0.7
        }

        graded_vector = math_lib.grading_vector(trade_data)
        assert isinstance(graded_vector, GradedProfitVector)
        assert graded_vector.profit == 0.05

        # Test vector operations
        vector_array = graded_vector.to_array()
        assert len(vector_array) == 5
        assert vector_array[0] == 0.05

    def test_advanced_strategies(self, math_lib, sample_data):
        """Test advanced trading strategies"""
        prices = sample_data['prices']
        volumes = sample_data['volumes']

        results = math_lib.apply_advanced_strategies(prices, volumes)

        # Verify all expected keys are present
        expected_keys = {
            'returns', 'bollinger_upper', 'bollinger_lower',
            'momentum', 'mean_reversion', 'z_scores',
            'vpt', 'profit_ratios', 'sharpe_ratio',
            'log_return', 'entropy', 'std'
        }

        assert all(key in results for key in expected_keys)

        # Verify reasonable values
        assert isinstance(results['sharpe_ratio'], float)
        assert isinstance(results['entropy'], float)
        assert results['entropy'] >= 0

    def test_mathlib_v2_extensions(self, math_lib_v2, sample_data):
        """Test CoreMathLibV2 extensions"""
        prices = sample_data['prices']
        volumes = sample_data['volumes']
        high = sample_data['high']
        low = sample_data['low']

        # Test VWAP calculation
        vwap = math_lib_v2.calculate_vwap(prices, volumes)
        assert len(vwap) == len(prices)
        assert np.all(vwap > 0)

        # Test ATR calculation
        atr = math_lib_v2.calculate_atr(high, low, prices)
        assert len(atr) == len(prices)
        assert np.all(atr >= 0)

        # Test RSI calculation
        rsi = math_lib_v2.calculate_rsi(prices)
        assert len(rsi) == len(prices)
        assert np.all((rsi >= 0) & (rsi <= 100))

        # Test Kelly fraction
        returns = np.diff(prices) / prices[:-1]
        kelly = math_lib_v2.calculate_kelly_fraction(returns)
        assert isinstance(kelly, float)

    def test_smart_stop_functionality(self):
        """Test SmartStop adaptive stop-loss system"""
        entry_price = 100.0
        stop = SmartStop(
            initial_stop=0.02,
            trailing_distance=0.01,
            max_loss=0.05,
            profit_lock_threshold=0.03
        )

        # Test initial state
        assert stop.current_stop == 0.02
        assert not stop.is_active

        # Test update with profitable move
        current_price = 105.0  # 5% profit
        _ = stop.update(current_price, entry_price)  # noqa: F841

        assert result['profit_pct'] == 0.05
        assert result['is_trailing']  # Should activate trailing
        assert not result['should_exit']

        # Test reset
        stop.reset()
        assert stop.current_stop == 0.02
        assert not stop.is_active

    def test_klein_bottle_integration(self):
        """Test Klein bottle mathematical functions"""
        # Test Klein bottle collapse field
        field = klein_bottle_collapse(dim=20)
        assert field.shape == (20, 20)
        assert np.all(np.isfinite(field))

        # Test mathematical constants
        assert MATHEMATICAL_CONSTANTS['KLEIN_BOTTLE_EULER'] == 0
        assert MATHEMATICAL_CONSTANTS['KLEIN_BOTTLE_GENUS'] == 1
        assert MATHEMATICAL_CONSTANTS['QUANTUM_COHERENCE_THRESHOLD'] == 0.85

    def test_unified_mathematical_processor(self, math_processor):
        """Test unified mathematical processor"""
        # Test initialization
        assert len(math_processor.analyzers) > 0
        assert hasattr(math_processor, 'results')

        # Test constants access
        constants = MATHEMATICAL_CONSTANTS
        assert 'KLEIN_BOTTLE_EULER' in constants
        assert 'QUANTUM_COHERENCE_THRESHOLD' in constants
        assert 'CYCLIC_COVERAGE' in constants

    def test_memory_kernel_operations(self, math_lib_v2):
        """Test memory kernel and time decay functions"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test memory kernel application
        weighted_values = math_lib_v2.apply_memory_kernel(values)
        assert len(weighted_values) > 0
        assert np.all(np.isfinite(weighted_values))

    def test_ornstein_uhlenbeck_simulation(self, math_lib_v2):
        """Test Ornstein-Uhlenbeck process simulation"""
        x0 = 100.0
        mu = 105.0
        n_steps = 50

        ou_process = math_lib_v2.simulate_ornstein_uhlenbeck(x0, mu, n_steps)

        assert len(ou_process) == n_steps
        assert ou_process[0] == x0
        assert np.all(np.isfinite(ou_process))

    def test_risk_parity_weights(self, math_lib_v2):
        """Test risk parity weight calculation"""
        volatilities = np.array([0.1, 0.2, 0.15, 0.25])

        weights = math_lib_v2.calculate_risk_parity_weights(volatilities)

        assert len(weights) == len(volatilities)
        assert abs(np.sum(weights) - 1.0) < 1e-6  # Should sum to 1
        assert np.all(weights > 0)  # All positive weights

        # Higher volatility should get lower weight
        assert weights[3] < weights[0]  # 0.25 vol < 0.1 vol

    def test_comprehensive_integration(
        self,
            math_lib,
            math_lib_v2,
            sample_data
    ):
        """Test comprehensive integration between all mathematical components"""
        prices = sample_data['prices']
        volumes = sample_data['volumes']
        high = sample_data['high']
        low = sample_data['low']

        # Run base strategies
        base_results = math_lib.apply_advanced_strategies(prices, volumes)

        # Run extended v2 strategies
        extended_results = math_lib_v2.apply_advanced_strategies_v2(
            prices,
            volumes,
            high,
            low
        )

        # Verify integration
        assert 'sharpe_ratio' in base_results
        assert 'vwap' in extended_results
        assert 'rsi' in extended_results
        assert 'kelly_fraction' in extended_results

        # Test that extended results include base results
        for key in base_results:
            if key in extended_results:
                # Allow for small numerical differences
                if isinstance(base_results[key], (int, float)):
                    assert abs(
                        base_results[key] -
                        extended_results[key]) < 1e-6
                else:
                    assert np.allclose(
                        base_results[key],
                        extended_results[key],
                        rtol=1e-6
                    )

    def test_mathematical_constants_integrity(self):
        """Test mathematical constants for consistency and validity"""
        constants = MATHEMATICAL_CONSTANTS

        # Test Klein bottle properties
        # Correct Euler characteristic
        assert constants['KLEIN_BOTTLE_EULER'] == 0
        assert constants['KLEIN_BOTTLE_GENUS'] == 1  # Correct genus

        # Test threshold values are in valid ranges
        assert 0 < constants['QUANTUM_COHERENCE_THRESHOLD'] < 1
        assert 0 < constants['WARP_STABILITY_THRESHOLD'] < 1
        assert constants['TPF_PARADOX_THRESHOLD'] > 0

        # Test dormant state count is positive
        assert constants['DORMANT_STATE_COUNT'] > 0
        assert constants['CYCLIC_COVERAGE'] > 0


if __name__ == "__main__":
    # Run the tests when executed directly
    pytest.main([__file__, "-v"])