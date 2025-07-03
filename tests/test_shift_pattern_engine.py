import numpy as np
        import time
from core.advanced_drift_shell_integration import (
from core.type_defs import Tensor, Entropy
from datetime import datetime
import pytest

"""
Comprehensive test suite for the Shift Pattern Engine.

Tests all differential states and their mathematical implementations.
"""


    ShiftPatternEngine,
    AdvancedDriftShellIntegration,
    GrayscaleDriftTensorCore,
    AdvancedTensorMemoryFeedback,
)


class TestShiftPatternEngine:
    """Test suite for Shift Pattern Engine core functionality."""

    @pytest.fixture
    def engine(self):
        """Create test engine instance."""
        return ShiftPatternEngine(
            shift_durations={
                "BTC": {"short": 16, "mid": 72, "long": 672},
                "XRP": {"short": 12, "mid": 48, "long": 480},
                "ETH": {"short": 14, "mid": 60, "long": 600},
            },
            decay_rate=0.1,
            coherence_threshold=0.05,
        )

    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.decay_rate == 0.1
        assert engine.coherence_threshold == 0.05
        assert "BTC" in engine.shift_durations
        assert "XRP" in engine.shift_durations
        assert "ETH" in engine.shift_durations
        assert engine.phase_history == []
        assert engine.tensor_decay_weights == []

    def test_ferris_wheel_phase(self, engine):
        """Test Ferris Wheel phase computation."""
        # Test basic phase computation
        phase = engine.compute_ferris_wheel_phase(tick_count=72, period=144)
        assert 0 <= phase <= 2 * np.pi
        assert np.isclose(phase, np.pi)

        # Test phase wrapping
        phase_wrapped = engine.compute_ferris_wheel_phase(tick_count=288, period=144)
        assert np.isclose(phase_wrapped, 0)

        # Test edge cases
        phase_zero = engine.compute_ferris_wheel_phase(tick_count=0, period=144)
        assert np.isclose(phase_zero, 0)

        phase_half = engine.compute_ferris_wheel_phase(tick_count=36, period=144)
        assert np.isclose(phase_half, np.pi / 2)

    def test_phase_shift_detection(self, engine):
        """Test phase shift detection logic."""
        # Test ascent to peak
        current_phase = np.pi / 4
        previous_phase = 0
        shift_type = engine.detect_phase_shift(current_phase, previous_phase)
        assert shift_type == "ascent_peak"

        # Test peak to descent
        current_phase = 3 * np.pi / 4
        previous_phase = np.pi / 2
        shift_type = engine.detect_phase_shift(current_phase, previous_phase)
        assert shift_type == "peak_descent"

        # Test descent to trough
        current_phase = 5 * np.pi / 4
        previous_phase = np.pi
        shift_type = engine.detect_phase_shift(current_phase, previous_phase)
        assert shift_type == "descent_trough"

        # Test trough to ascent
        current_phase = 7 * np.pi / 4
        previous_phase = 3 * np.pi / 2
        shift_type = engine.detect_phase_shift(current_phase, previous_phase)
        assert shift_type == "trough_ascent"

        # Test phase wrapping
        current_phase = 0
        previous_phase = 7 * np.pi / 4
        shift_type = engine.detect_phase_shift(current_phase, previous_phase)
        assert shift_type == "trough_ascent"

    def test_tensor_decay_weight(self, engine):
        """Test tensor decay weight computation."""
        # Test exponential decay
        weight_0 = engine.compute_tensor_decay_weight(time_index=0)
        weight_1 = engine.compute_tensor_decay_weight(time_index=1)
        weight_5 = engine.compute_tensor_decay_weight(time_index=5)

        assert np.isclose(weight_0, 1.0)
        assert np.isclose(weight_1, np.exp(-engine.decay_rate))
        assert np.isclose(weight_5, np.exp(-5 * engine.decay_rate))

        # Test decay properties
        assert weight_0 > weight_1 > weight_5
        assert all(0 < w <= 1 for w in [weight_0, weight_1, weight_5])

        # Test with different decay rates
        engine.decay_rate = 0.2
        weight_fast = engine.compute_tensor_decay_weight(time_index=1)
        assert weight_fast < weight_1  # Faster decay

    def test_thermal_pressure(self, engine):
        """Test thermal pressure computation."""
        # Test basic pressure computation
        pressure = engine.compute_thermal_pressure(volume_ema=1.2, volatility=0.15)
        assert 0 < pressure < 2  # Reasonable range

        # Test pressure with zero volatility
        pressure_zero_vol = engine.compute_thermal_pressure(
            volume_ema=1.0, volatility=0.0
        )
        assert pressure_zero_vol > 0

        # Test pressure with high volatility
        pressure_high_vol = engine.compute_thermal_pressure(
            volume_ema=1.0, volatility=1.0
        )
        assert pressure_high_vol > pressure_zero_vol

        # Test pressure with different volume ratios
        pressure_low_vol = engine.compute_thermal_pressure(
            volume_ema=0.5, volatility=0.15
        )
        pressure_high_vol_ratio = engine.compute_thermal_pressure(
            volume_ema=2.0, volatility=0.15
        )

        # Higher volume should give higher pressure
        assert pressure_high_vol_ratio > pressure_low_vol

    def test_entropy_coherence_shift(self, engine):
        """Test entropy-coherence shift detection."""
        # Test trigger condition
        current_coherence = 0.8
        previous_coherence = 0.9
        should_trigger = engine.compute_entropy_coherence_shift(
            current_coherence, previous_coherence
        )
        assert should_trigger is True

        # Test no trigger condition
        current_coherence = 0.9
        previous_coherence = 0.8
        should_trigger = engine.compute_entropy_coherence_shift(
            current_coherence, previous_coherence
        )
        assert should_trigger is False

        # Test threshold boundary
        current_coherence = 0.85
        previous_coherence = 0.9
        should_trigger = engine.compute_entropy_coherence_shift(
            current_coherence, previous_coherence
        )
        assert should_trigger is True

        # Test with different thresholds
        engine.coherence_threshold = 0.1
        should_trigger = engine.compute_entropy_coherence_shift(
            current_coherence, previous_coherence
        )
        assert should_trigger is False  # Delta is -0.05, threshold is 0.1

    def test_api_penalty_decay(self, engine):
        """Test API penalty decay computation."""
        # Test basic penalty computation
        confidence = 0.9
        error_count = 2
        penalized = engine.compute_api_penalty_decay(confidence, error_count)
        assert penalized < confidence
        assert penalized > 0

        # Test with zero errors
        penalized_zero = engine.compute_api_penalty_decay(confidence, 0)
        assert np.isclose(penalized_zero, confidence)

        # Test with high error count
        penalized_high = engine.compute_api_penalty_decay(confidence, 10)
        assert penalized_high < penalized

        # Test with different tau values
        penalized_tau5 = engine.compute_api_penalty_decay(
            confidence, error_count, tau=5.0
        )
        penalized_tau20 = engine.compute_api_penalty_decay(
            confidence, error_count, tau=20.0
        )
        assert penalized_tau5 < penalized_tau20  # Lower tau = faster decay

    def test_time_lock_phase_drift(self, engine):
        """Test time lock phase drift computation."""
        # Test basic drift computation
        short_phase = 0.5
        mid_phase = 1.2
        long_phase = 2.1

        drift_magnitude, drift_direction = engine.compute_time_lock_phase_drift(
            short_phase, mid_phase, long_phase
        )

        assert drift_magnitude >= 0
        assert drift_direction in [-1, 0, 1]

        # Test with aligned phases
        drift_magnitude_aligned, drift_direction_aligned = (
            engine.compute_time_lock_phase_drift(0.5, 0.5, 0.5)
        )
        assert np.isclose(drift_magnitude_aligned, 0)

        # Test with maximum misalignment
        drift_magnitude_max, drift_direction_max = engine.compute_time_lock_phase_drift(
            0, np.pi, 2 * np.pi
        )
        assert drift_magnitude_max > 0

        # Test drift direction logic
        # When short-mid difference > mid-long difference, direction should be positive
        drift_magnitude_test, drift_direction_test = (
            engine.compute_time_lock_phase_drift(0, 1.0, 0.5)
        )
        assert drift_direction_test == 1

    def test_shift_duration_retrieval(self, engine):
        """Test shift duration retrieval for different assets."""
        # Test BTC durations
        btc_short = engine.get_shift_duration("BTC", "short")
        btc_mid = engine.get_shift_duration("BTC", "mid")
        btc_long = engine.get_shift_duration("BTC", "long")

        assert btc_short == 16
        assert btc_mid == 72
        assert btc_long == 672

        # Test XRP durations
        xrp_short = engine.get_shift_duration("XRP", "short")
        xrp_mid = engine.get_shift_duration("XRP", "mid")
        xrp_long = engine.get_shift_duration("XRP", "long")

        assert xrp_short == 12
        assert xrp_mid == 48
        assert xrp_long == 480

        # Test unknown asset (should use default)
        unknown_short = engine.get_shift_duration("UNKNOWN", "short")
        assert unknown_short == 16  # Default value

        # Test unknown shift type (should use short)
        unknown_type = engine.get_shift_duration("BTC", "unknown")
        assert unknown_type == 16  # Short duration as fallback


class TestGrayscaleDriftTensorCore:
    """Test suite for Grayscale Drift Tensor Core."""

    @pytest.fixture
    def core(self):
        """Create test core instance."""
        return GrayscaleDriftTensorCore(psi_infinity=1.618033988749895)

    def test_initialization(self, core):
        """Test core initialization."""
        assert np.isclose(core.psi_infinity, 1.618033988749895)  # Golden ratio

    def test_compute_drift_field(self, core):
        """Test drift field computation."""
        # Test basic computation
        drift_value = core.compute_drift_field(x=1.0, y=2.0, z=0.5, time=1.0)
        assert isinstance(drift_value, float)
        assert not np.isnan(drift_value)

        # Test time decay
        drift_t0 = core.compute_drift_field(x=1.0, y=2.0, z=0.5, time=0.0)
        drift_t1 = core.compute_drift_field(x=1.0, y=2.0, z=0.5, time=1.0)
        assert drift_t0 > drift_t1  # Decay over time

        # Test spatial variation
        drift_x1 = core.compute_drift_field(x=1.0, y=2.0, z=0.5, time=1.0)
        drift_x2 = core.compute_drift_field(x=2.0, y=2.0, z=0.5, time=1.0)
        assert not np.isclose(drift_x1, drift_x2)  # Different x values

    def test_allocate_ring_drift(self, core):
        """Test ring drift allocation."""
        # Test basic allocation
        drift_value = core.allocate_ring_drift(layer_index=3, entropy_gradient=0.1)
        assert isinstance(drift_value, float)
        assert not np.isnan(drift_value)

        # Test layer dependency
        drift_layer1 = core.allocate_ring_drift(layer_index=1, entropy_gradient=0.1)
        drift_layer5 = core.allocate_ring_drift(layer_index=5, entropy_gradient=0.1)
        assert not np.isclose(drift_layer1, drift_layer5)

        # Test entropy gradient dependency
        drift_grad1 = core.allocate_ring_drift(layer_index=3, entropy_gradient=0.1)
        drift_grad2 = core.allocate_ring_drift(layer_index=3, entropy_gradient=0.2)
        assert not np.isclose(drift_grad1, drift_grad2)

    def test_gamma_node_coupling(self, core):
        """Test gamma node coupling."""
        # Test basic coupling
        coupling_value = core.gamma_node_coupling(node_depth=2, drift_signal=0.5)
        assert isinstance(coupling_value, float)
        assert not np.isnan(coupling_value)

        # Test depth dependency
        coupling_depth1 = core.gamma_node_coupling(node_depth=1, drift_signal=0.5)
        coupling_depth5 = core.gamma_node_coupling(node_depth=5, drift_signal=0.5)
        assert coupling_depth1 > coupling_depth5  # Deeper nodes have lower weight

        # Test signal dependency
        coupling_signal1 = core.gamma_node_coupling(node_depth=2, drift_signal=0.1)
        coupling_signal2 = core.gamma_node_coupling(node_depth=2, drift_signal=0.9)
        assert coupling_signal2 > coupling_signal1  # Higher signal = higher coupling


class TestAdvancedTensorMemoryFeedback:
    """Test suite for Advanced Tensor Memory Feedback."""

    @pytest.fixture
    def memory(self):
        """Create test memory instance."""
        return AdvancedTensorMemoryFeedback(max_history=10, decay_rate=0.1)

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return Tensor(np.random.rand(4, 4))

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.max_history == 10
        assert memory.decay_rate == 0.1
        assert memory.history_stack == []

    def test_record_tensor_history(self, memory, sample_tensor):
        """Test tensor history recording."""
        # Record first entry
        memory.record_tensor_history(
            tensor=sample_tensor, entropy_delta=0.1, metadata={"test": True}
        )

        assert len(memory.history_stack) == 1
        entry = memory.history_stack[0]
        assert entry["tensor"].shape == sample_tensor.shape
        assert entry["entropy_delta"] == Entropy(0.1)
        assert entry["metadata"]["test"] is True
        assert isinstance(entry["timestamp"], datetime)

        # Test with float entropy delta
        memory.record_tensor_history(tensor=sample_tensor, entropy_delta=0.2)
        assert len(memory.history_stack) == 2
        assert entry["entropy_delta"] == Entropy(0.1)  # First entry unchanged

    def test_max_history_limit(self, memory, sample_tensor):
        """Test maximum history limit enforcement."""
        # Add more entries than max_history
        for i in range(15):
            memory.record_tensor_history(tensor=sample_tensor, entropy_delta=float(i))

        assert len(memory.history_stack) == memory.max_history
        # Oldest entries should be removed
        assert memory.history_stack[0]["entropy_delta"] == Entropy(5.0)
        assert memory.history_stack[-1]["entropy_delta"] == Entropy(14.0)

    def test_compute_recursive_feedback(self, memory, sample_tensor):
        """Test recursive feedback computation."""
        # Add some history
        for i in range(5):
            memory.record_tensor_history(
                tensor=sample_tensor * (i + 1), entropy_delta=float(i + 1)
            )

        # Test feedback computation
        feedback_tensor = memory.compute_recursive_feedback(
            current_tensor=sample_tensor, recursion_depth=3
        )

        assert isinstance(feedback_tensor, Tensor)
        assert feedback_tensor.shape == sample_tensor.shape
        assert not np.array_equal(feedback_tensor, sample_tensor)  # Should be modified

    def test_compute_recursive_feedback_empty_history(self, memory, sample_tensor):
        """Test feedback computation with empty history."""
        feedback_tensor = memory.compute_recursive_feedback(
            current_tensor=sample_tensor, recursion_depth=3
        )

        assert isinstance(feedback_tensor, Tensor)
        assert np.array_equal(feedback_tensor, sample_tensor)  # Should be unchanged

    def test_get_memory_statistics(self, memory, sample_tensor):
        """Test memory statistics computation."""
        # Test empty memory
        stats_empty = memory.get_memory_statistics()
        assert stats_empty["entries"] == 0
        assert stats_empty["avg_entropy"] == 0.0
        assert stats_empty["oldest_entry"] is None
        assert stats_empty["newest_entry"] is None
        assert stats_empty["total_memory_mb"] == 0.0

        # Test with entries
        memory.record_tensor_history(sample_tensor, 0.1)
        memory.record_tensor_history(sample_tensor, 0.2)

        stats = memory.get_memory_statistics()
        assert stats["entries"] == 2
        assert stats["avg_entropy"] == 0.15
        assert stats["oldest_entry"] is not None
        assert stats["newest_entry"] is not None
        assert stats["total_memory_mb"] > 0.0

    def test_clear_old_entries(self, memory, sample_tensor):
        """Test clearing old entries."""
        # Add entries with different timestamps
        memory.record_tensor_history(sample_tensor, 0.1)
        memory.record_tensor_history(sample_tensor, 0.2)

        # Simulate time passing by modifying timestamps
        memory.history_stack[0]["timestamp"] = datetime.now()
        memory.history_stack[1]["timestamp"] = datetime.now()

        # Clear entries older than 1 hour
        removed_count = memory.clear_old_entries(max_age_hours=1.0)
        assert removed_count == 0  # No entries should be removed

        # Clear entries older than 0 hours (should remove all)
        removed_count = memory.clear_old_entries(max_age_hours=0.0)
        assert removed_count == 2
        assert len(memory.history_stack) == 0


class TestAdvancedDriftShellIntegration:
    """Test suite for Advanced Drift Shell Integration."""

    @pytest.fixture
    def integration(self):
        """Create test integration instance."""
        return AdvancedDriftShellIntegration(
            shell_radius=144.44, thermal_conductivity=0.024, energy_scale=1.0
        )

    @pytest.fixture
    def sample_tensor(self):
        """Create sample tensor for testing."""
        return Tensor(np.random.rand(8, 8))

    def test_initialization(self, integration):
        """Test integration initialization."""
        assert integration.grayscale_core is not None
        assert integration.tensor_memory is not None
        assert integration.shift_engine is not None

        # Components may be None if imports fail
        assert hasattr(integration, "drift_engine")
        assert hasattr(integration, "quantum_engine")
        assert hasattr(integration, "thermal_allocator")
        assert hasattr(integration, "phase_harmonizer")

    def test_analyze_shift_patterns(self, integration, sample_tensor):
        """Test shift pattern analysis."""
        metadata = {"tick_count": 100, "asset": "BTC"}

        results = integration.analyze_shift_patterns(sample_tensor, metadata)

        assert "current_phase" in results
        assert "tensor_decay_weight" in results
        assert "thermal_pressure" in results
        assert "drift_magnitude" in results
        assert "drift_direction" in results
        assert "btc_short_duration" in results
        assert "xrp_short_duration" in results

        # Check data types
        assert isinstance(results["current_phase"], float)
        assert isinstance(results["tensor_decay_weight"], float)
        assert isinstance(results["thermal_pressure"], float)
        assert isinstance(results["drift_magnitude"], float)
        assert isinstance(results["drift_direction"], (int, float))
        assert isinstance(results["btc_short_duration"], int)
        assert isinstance(results["xrp_short_duration"], int)

    def test_get_system_statistics(self, integration):
        """Test system statistics retrieval."""
        stats = integration.get_system_statistics()

        assert "components_available" in stats
        assert "memory" in stats

        # Check component availability
        components = stats["components_available"]
        assert isinstance(components["drift_engine"], bool)
        assert isinstance(components["quantum_engine"], bool)
        assert isinstance(components["thermal_allocator"], bool)
        assert isinstance(components["phase_harmonizer"], bool)

        # Check memory statistics
        memory_stats = stats["memory"]
        assert "entries" in memory_stats
        assert "avg_entropy" in memory_stats
        assert "total_memory_mb" in memory_stats

    def test_cleanup_old_data(self, integration):
        """Test old data cleanup."""
        # Add some data to memory
        sample_tensor = Tensor(np.random.rand(4, 4))
        integration.tensor_memory.record_tensor_history(sample_tensor, 0.1)
        integration.tensor_memory.record_tensor_history(sample_tensor, 0.2)

        # Clean up old data
        removed_count = integration.cleanup_old_data(max_age_hours=24.0)
        assert isinstance(removed_count, int)
        assert removed_count >= 0


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_full_integration_workflow(self):
        """Test complete integration workflow."""
        integration = AdvancedDriftShellIntegration()
        sample_tensor = Tensor(np.random.rand(8, 8))
        hash_patterns = ["a1b2c3d4", "e5f6g7h8"]
        metadata = {"tick_count": 100, "asset": "BTC"}

        # Test integration without quantum state
        results = integration.integrate_all_components(
            current_tensor=sample_tensor, hash_patterns=hash_patterns, metadata=metadata
        )

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check that shift pattern results are included
        assert "current_phase" in results
        assert "tensor_decay_weight" in results
        assert "thermal_pressure" in results

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        engine = ShiftPatternEngine()

        # Test with extreme values
        extreme_phase = engine.compute_ferris_wheel_phase(tick_count=1000000, period=1)
        assert 0 <= extreme_phase <= 2 * np.pi

        # Test with zero values
        zero_phase = engine.compute_ferris_wheel_phase(tick_count=0, period=1)
        assert np.isclose(zero_phase, 0)

        # Test with negative values (should handle gracefully)
        negative_phase = engine.compute_ferris_wheel_phase(tick_count=-10, period=10)
        assert 0 <= negative_phase <= 2 * np.pi

    def test_performance_characteristics(self):
        """Test performance characteristics."""
        engine = ShiftPatternEngine()

        # Test computational efficiency

        start_time = time.time()
        for i in range(1000):
            engine.compute_ferris_wheel_phase(i)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second for 1000 computations

        # Test memory usage
        memory = AdvancedTensorMemoryFeedback(max_history=1000)
        sample_tensor = Tensor(np.random.rand(10, 10))

        start_time = time.time()
        for i in range(100):
            memory.record_tensor_history(sample_tensor, float(i))
        end_time = time.time()

        assert end_time - start_time < 1.0  # Less than 1 second for 100 recordings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
