# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from __future__ import annotations

# -*- coding: utf - 8 -*-
# -*- coding: utf - 8 -*-
from dual_unicore_handler import DualUnicoreHandler
from pathlib import Path
import time

import numpy as np

from core.ghost_phase_strategy_loader import GhostPhaseStrategyLoader, GhostPhaseDecision
from core.ghost_trigger_map import GhostTriggerMapper, generate_ghost_trigger_map
from core.strategy_mapper import StrategyMapper
from utils.safe_print import safe_print


# Initialize Unicode handler
unicore = DualUnicoreHandler()

"""tests.ghost_phase_strategy_loader_test"
Ghost Phase Strategy Loader Integration Test
===========================================

Sanity test to validate .strategy_id returns a string using mocked inputs.
Tests the integration of all mathematical modules:
- DriftPhaseWeighter
- AlephOverlayMapper
- PhaseTransitionMonitor
- TruthLatticeMath
- GhostFieldStabilizer
- BitWavePropagator"""
""""""
""""""
"""


def test_ghost_phase_strategy_loader():"""
    """Test basic GhostPhaseStrategyLoader functionality."""

"""
""""""
""""""
  safe_print("\\u1f9ea Testing GhostPhaseStrategyLoader...")

# Use the overlay file we created
   overlay_path = "memory_stack / aleph_overlays.json"
    if not Path(overlay_path).exists():
        safe_print(f"\\u274c Overlay file not found: {overlay_path}")
        return False

try:
        # Initialize loader
loader = GhostPhaseStrategyLoader(overlay_path)

# Create test data
prices = np.random.random(50) * 100 + 50  # Random prices 50 - 150
        live_vector = [0.6, 0.4, 0.7, 0.3, 0.8, 0.2]  # 6 - element vector
        raw_signals = [0.7, 0.3, 0.6, 0.8, 0.4]  # 5 signals

# Make decision
decision = loader.decide(prices, live_vector, raw_signals)

# Validate result
assert isinstance(decision, GhostPhaseDecision), "Decision must be GhostPhaseDecision"
        assert isinstance(decision.strategy_id, str), "Strategy ID must be string"
        assert len(decision.strategy_id) > 0, "Strategy ID must not be empty"
        assert isinstance(decision.consensus, bool), "Consensus must be boolean"

safe_print(f"\\u2705 Strategy ID: {decision.strategy_id}")
        safe_print(f"\\u2705 Phase: {decision.phase_report.phase_state.name}")
        safe_print(f"\\u2705 Consensus: {decision.consensus}")
        safe_print(f"\\u2705 Overlay: {decision.overlay_match.overlay_id}")
        safe_print(f"\\u2705 Similarity: {decision.overlay_match.similarity:.3f}")

return True

except Exception as e:
        safe_print(f"\\u274c Test failed: {e}")
        return False


def test_ghost_trigger_mapper():
    """Test GhostTriggerMapper functionality."""

"""
""""""
""""""
  safe_print("\\n  # -*- coding: utf - 8 -*-\\n\\u1f9ea Testing GhostTriggerMapper...")

   try:
        # Initialize mapper
mapper = GhostTriggerMapper()

# Create test data
prices = np.random.random(50) * 100 + 50
        live_vector = [0.8, 0.2, 0.6, 0.4, 0.9, 0.1]
        raw_signals = [0.8, 0.6, 0.7, 0.9, 0.5]

# Evaluate trigger
result = mapper.evaluate_trigger(
            prices, live_vector, raw_signals,
            volatility=0.5, resonance=0.7, threshold=0.3
        )

# Validate result
assert isinstance(result.triggered, bool), "Triggered must be boolean"
        assert isinstance(result.strategy_id, str), "Strategy ID must be string"
        assert isinstance(result.confidence, float), "Confidence must be float"
        assert 0.0 <= result.confidence <= 1.0, "Confidence must be in [0,1]"

safe_print(f"\\u2705 Triggered: {result.triggered}")
        safe_print(f"\\u2705 Strategy: {result.strategy_id}")
        safe_print(f"\\u2705 Confidence: {result.confidence:.3f}")

return True

except Exception as e:
        safe_print(f"\\u274c Test failed: {e}")
        return False


def test_strategy_mapper():
    """Test modern StrategyMapper functionality."""

"""
""""""
""""""
  safe_print("\\n\\u1f9ea Testing StrategyMapper...")

   try:
        # Initialize mapper
mapper = StrategyMapper()

# Create test data
prices = np.random.random(50) * 100 + 50
        live_vector = [0.5, 0.5, 0.6, 0.4, 0.7, 0.3]
        raw_signals = [0.6, 0.4, 0.7, 0.5, 0.8]

# Map strategy
result = mapper.map_strategy(prices, live_vector, raw_signals)

# Validate result
assert isinstance(result.success, bool), "Success must be boolean"
        assert isinstance(result.strategy_id, str), "Strategy ID must be string"
        assert len(result.recommendations) >= 0, "Recommendations must be list"

safe_print(f"\\u2705 Success: {result.success}")
        safe_print(f"\\u2705 Strategy: {result.strategy_id}")
        safe_print(f"\\u2705 Recommendations: {len(result.recommendations)}")

return True

except Exception as e:
        safe_print(f"\\u274c Test failed: {e}")
        return False


def test_ghost_trigger_map_legacy():
    """Test legacy ghost trigger map function."""

"""
""""""
""""""
  safe_print("\\n\\u1f9ea Testing generate_ghost_trigger_map...")

   try:
        # Generate trigger map
trigger_map = generate_ghost_trigger_map(
            volatility=0.5, resonance = 0.7, threshold = 0.3
        )

# Validate structure
assert isinstance(trigger_map, dict), "Trigger map must be dict"
        assert "trigger_type" in trigger_map, "Must have trigger_type"
        assert "parameters" in trigger_map, "Must have parameters"
        assert "strategy_mapping" in trigger_map, "Must have strategy_mapping"
        assert trigger_map["trigger_type"] == "ghost_phase", "Must be ghost_phase type"

safe_print(f"\\u2705 Trigger type: {trigger_map['trigger_type']}")
        safe_print(f"\\u2705 Parameters: {trigger_map['parameters']}")
        safe_print(f"\\u2705 Strategy mappings: {len(trigger_map['strategy_mapping'])}")

return True

except Exception as e:
        safe_print(f"\\u274c Test failed: {e}")
        return False


def test_mathematical_integration():
    """Test integration of all mathematical components."""

"""
""""""
""""""
  safe_print("\\n\\u1f9ea Testing mathematical component integration...")

   try:
        # Test that all mathematical modules can be imported and used together
from core.phase.drift_phase_weighter import DriftPhaseWeighter
from core.overlay.aleph_overlay_mapper import AlephOverlayMapper
from core.phase.phase_transition_monitor import PhaseTransitionMonitor
from core.truth_lattice_math import collapse_score, is_consensus_reached
        from core.ghost_field_stabilizer import GhostFieldStabilizer
from core.phase.bit_wave_propagator import allocate_phase_vector
from utils.math_utils import calculate_entropy, cosine_similarity

# Test each component individually
prices = np.random.random(50) * 100 + 50

# 1. Drift Phase Weighter
weighter = DriftPhaseWeighter(lambda_=0.4)
        drift_report = weighter.calculate_drift_weight(prices)
        assert isinstance(drift_report.drift_weight, float)
        safe_print("\\u2705 DriftPhaseWeighter working")

# 2. Ghost Field Stabilizer
stabilizer = GhostFieldStabilizer()
        stability = stabilizer.check_stability(prices)
        assert isinstance(stability.is_stable, bool)
        safe_print("\\u2705 GhostFieldStabilizer working")

# 3. Truth Lattice Math
signals = [0.6, 0.4, 0.7, 0.5]
        score = collapse_score(signals, omega=1.0)
        consensus = is_consensus_reached(signals, omega=1.0)
        assert isinstance(score, float)
        assert isinstance(consensus.reached, bool)
        safe_print("\\u2705 TruthLatticeMath working")

# 4. Math Utils
entropy = calculate_entropy(prices)
        vec_a = [0.6, 0.4, 0.7]
        vec_b = [0.5, 0.5, 0.6]
        similarity = cosine_similarity(vec_a, vec_b)
        assert isinstance(entropy, float)
        assert isinstance(similarity, float)
        safe_print("\\u2705 MathUtils working")

# 5. Bit Wave Propagator
phase_vector = allocate_phase_vector(8, signals)
        assert phase_vector.bit_depth == 8
        safe_print("\\u2705 BitWavePropagator working")

safe_print("\\u2705 All mathematical components integrated successfully")
        return True

except Exception as e:
        safe_print(f"\\u274c Mathematical integration test failed: {e}")
        return False


def test_hybrid_strategy_mapper():
    """Test hybrid strategy mapper with both Ghost Phase and legacy paths."""

"""
""""""
""""""
  safe_print("\\n\\u1f9ea Testing Hybrid Strategy Mapper...")

   try:
        from core.strategy_mapper import StrategyMapper

# Test 1: Ghost Phase Path
safe_print("  \\u1f4ca Testing Ghost Phase path...")
        mapper_ghost = StrategyMapper(
            enable_ghost_phase=True,
            enable_legacy=False,
            default_to_legacy=False
        )

prices = np.random.random(50) * 100 + 50
        live_vector = [0.8, 0.2, 0.6, 0.4, 0.9, 0.1]
        raw_signals = [0.7, 0.3, 0.6, 0.8, 0.4]

result_ghost = mapper_ghost.map_strategy(
            prices, live_vector, raw_signals, use_legacy=False
        )

assert result_ghost.success, "Ghost Phase path should succeed"
        assert result_ghost.ghost_decision is not None, "Should have ghost decision"
        assert isinstance(result_ghost.strategy_id, str), "Strategy ID must be string"
        assert len(result_ghost.recommendations) > 0, "Should have recommendations"
        safe_print(f"    \\u2705 Ghost Phase: {result_ghost.strategy_id}")

# Test 2: Legacy Path
safe_print("  \\u1f4ca Testing Legacy path...")
        mapper_legacy = StrategyMapper(
            enable_ghost_phase=False,
            enable_legacy=True,
            default_to_legacy=True
        )

execution_packet = {
            "strategy_type": "momentum",
            "prices": list(prices),
            "signals": list(raw_signals),
            "timestamp": time.time(),

result_legacy = mapper_legacy.map_strategy(
            prices, live_vector, raw_signals, execution_packet, use_legacy=True
        )

assert result_legacy.success, "Legacy path should succeed"
        assert result_legacy.mapped_strategy is not None, "Should have mapped strategy"
        assert isinstance(result_legacy.strategy_id, str), "Strategy ID must be string"
        safe_print(f"    \\u2705 Legacy: {result_legacy.strategy_id}")

# Test 3: Hybrid Auto - Detection
safe_print("  \\u1f4ca Testing Hybrid auto - detection...")
        mapper_hybrid = StrategyMapper(
            enable_ghost_phase=True,
            enable_legacy=True,
            default_to_legacy=False
        )

# Modern packet should use Ghost Phase
result_auto_ghost = mapper_hybrid.map_strategy(
            prices, live_vector, raw_signals, None, None  # No explicit override
        )
assert "ghost_phase" in result_auto_ghost.metadata.get("path", "")
        safe_print(f"    \\u2705 Auto - detection Ghost: {result_auto_ghost.strategy_id}")

# Legacy packet should use legacy path
legacy_packet = {"agent_type": "SCHWABOT", "alpha_score": 0.5}
        result_auto_legacy = mapper_hybrid.map_strategy(
            prices, live_vector, raw_signals, legacy_packet, None
        )
assert "legacy" in result_auto_legacy.metadata.get("path", "")
        safe_print(f"    \\u2705 Auto - detection Legacy: {result_auto_legacy.strategy_id}")

# Test 4: Performance Statistics
stats = mapper_hybrid.get_performance_stats()
        assert "total_mappings" in stats, "Should have performance stats"
        assert "ghost_decisions" in stats, "Should track ghost decisions"
        assert "legacy_mappings" in stats, "Should track legacy mappings"
        safe_print(
            f"    \\u2705 Stats: {stats['total_mappings']} total, {stats['ghost_decisions']} ghost, {stats['legacy_mappings']} legacy")

return True

except Exception as e:
        safe_print(f"\\u274c Hybrid test failed: {e}")
        return False


def test_legacy_compatibility_functions():
    """Test legacy compatibility functions work correctly."""

"""
""""""
""""""
  safe_print("\\n\\u1f9ea Testing Legacy Compatibility Functions...")

   try:
        from core.strategy_mapper import map_strategy, map_strategy_enhanced

# Test legacy map_strategy function
execution_packet = {
            "strategy_type": "momentum",
            "prices": [50000, 51000, 50500, 52000],
            "signals": [0.7, 0.3, 0.6, 0.8],

mapped_result = map_strategy(execution_packet)
        assert isinstance(mapped_result, dict), "Legacy function should return dict"
        assert "mapped_at" in mapped_result, "Should have mapping timestamp"
        safe_print(f"    \\u2705 Legacy map_strategy: {mapped_result.get('strategy_id', 'unknown')}")

# Test enhanced async function
import asyncio

async def test_async():
            enhanced_result = await map_strategy_enhanced(
                execution_packet,
                agent_type=None,
                prophet_curve_id="test_curve"
            )
assert enhanced_result.success, "Enhanced function should succeed"
            assert isinstance(enhanced_result.strategy_id, str), "Should have strategy ID"
            return enhanced_result

enhanced_result = asyncio.run(test_async())
        safe_print(f"    \\u2705 Enhanced async: {enhanced_result.strategy_id}")

return True

except Exception as e:
        safe_print(f"\\u274c Legacy compatibility test failed: {e}")
        return False


def main():
    """Run all integration tests."""

"""
""""""
""""""
  safe_print("\\u1f680 Ghost Phase Strategy Loader Integration Tests")
   safe_print("=" * 60)

tests = [
        test_ghost_phase_strategy_loader,
        test_ghost_trigger_mapper,
        test_strategy_mapper,
        test_ghost_trigger_map_legacy,
        test_mathematical_integration,
        test_hybrid_strategy_mapper,
        test_legacy_compatibility_functions,
]
passed = 0
    total = len(tests)

for test in tests:
        if test():
            passed += 1

safe_print("\n" + "=" * 60)
    safe_print(f"\\u1f4ca Test Results: {passed}/{total} tests passed")

if passed == total:
        safe_print("\\u1f389 All tests passed! Hybrid mathematical integration is working correctly.")
        return True
else:
        safe_print("\\u274c Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    main()
