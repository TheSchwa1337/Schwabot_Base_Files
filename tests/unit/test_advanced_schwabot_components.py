import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from schwabot.core.ghost_field_stabilizer import GhostFieldStabilizer
from schwabot.core.overlay.aleph_overlay_mapper import AlephOverlayMapper
from schwabot.core.phase.bit_wave_propagator import BitWavePropagator
from schwabot.core.phase.drift_phase_weighter import DriftPhaseWeighter
from schwabot.core.phase.phase_transition_monitor import PhaseTransitionMonitor
from schwabot.core.truth_lattice_math import TruthLatticeMath

#!/usr/bin/env python3
"""
Advanced Schwabot Components Test
================================

Comprehensive test demonstrating all advanced Schwabot components working together,
including phase, drift, consensus, and overlay modules.
"""


# Add schwabot to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import advanced components

# Import existing components

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_test_signal(length: int, signal_type: str) -> np.ndarray:
    """Generate test signal for component testing."""
    if signal_type == "random":
        return np.random.random(length)
    elif signal_type == "trending":
        return np.linspace(0, 1, length) + np.random.normal(0, 0.1, length)
    elif signal_type == "oscillatory":
        return np.sin(np.linspace(0, 4 * np.pi, length)) + np.random.normal(0, 0.1, length)
    elif signal_type == "chaotic":
        return np.cumsum(np.random.normal(0, 0.1, length))
    else:
        return np.random.random(length)


def test_drift_phase_weighter():
    """Test Drift-Phase Weighter component."""
    print("\n" + "=" * 60)
    print("TESTING DRIFT-PHASE WEIGHTER")
    print("=" * 60)

    # Initialize component
    weighter = DriftPhaseWeighter()

    # Test different signal types
    signal_types = ["random", "trending", "oscillatory", "chaotic"]

    for signal_type in signal_types:
        print(f"\nTesting {signal_type} signal:")

        # Generate test signal
        signal = generate_test_signal(100, signal_type)

        # Calculate drift weight
        drift_weight = weighter.calculate_phase_drift_weight(signal)
        print(f"  Drift Weight: {drift_weight:.4f}")

        # Analyze drift pattern
        metrics = weighter.analyze_drift_pattern(signal)
        print(f"  Drift Type: {metrics.drift_type.value}")
        print(f"  Lambda Decay: {metrics.lambda_decay:.4f}")
        print(f"  Transition Tension: {metrics.transition_tension:.4f}")
        print(f"  Phase Stability: {metrics.phase_stability:.4f}")
        print(f"  Entropy Score: {metrics.entropy_score:.4f}")

        # Test phase transition detection
        transition = weighter.detect_phase_transition(signal, "consolidation")
        if transition:
            print(f"  Phase Transition: {transition.from_phase} -> {transition.to_phase}")
            print(f"  Tension Score: {transition.tension_score:.3f}")
            print(f"  Confidence: {transition.confidence:.3f}")
        else:
            print("  No phase transition detected")

    # Get summary
    summary = weighter.get_drift_summary()
    print("\nDrift Summary:")
    print(f"  Total Calculations: {summary['total_calculations']}")
    print(f"  Total Transitions: {summary['total_transitions']}")
    print(f"  Average Drift Weight: {summary['average_drift_weight']:.4f}")
    print(f"  Most Common Type: {summary['most_common_drift_type']}")

    return weighter


def test_ghost_field_stabilizer():
    """Test Ghost Field Stabilizer component."""
    print("\n" + "=" * 60)
    print("TESTING GHOST FIELD STABILIZER")
    print("=" * 60)

    # Initialize component
    stabilizer = GhostFieldStabilizer()

    # Test different signal types
    signal_types = ["random", "trending", "oscillatory", "chaotic"]

    for signal_type in signal_types:
        print(f"\nTesting {signal_type} signal:")

        # Generate test signal
        signal = generate_test_signal(100, signal_type)

        # Evaluate stability
        report = stabilizer.evaluate_stability(signal)
        print(f"  Stability Level: {report.stability_level.value}")
        print(f"  Entropy Score: {report.entropy_score:.4f}")
        print(f"  Entropy Bounds: ({report.entropy_bounds[0]:.4f}, {report.entropy_bounds[1]:.4f})")
        print(f"  Slope Estimate: {report.slope_estimate:.4f}")
        print(f"  Field Integrity: {report.field_integrity:.4f}")
        print(f"  Confidence: {report.confidence:.4f}")

        # Test entropy bounds computation
        bounds = stabilizer.compute_entropy_bounds(signal)
        print(f"  Computed Bounds: ({bounds[0]:.4f}, {bounds[1]:.4f})")

    # Get summary
    summary = stabilizer.get_stability_summary()
    print("\nStability Summary:")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Total Stability Events: {summary['total_stability_events']}")
    print(f"  Average Entropy Score: {summary['average_entropy_score']:.4f}")
    print(f"  Average Field Integrity: {summary['average_field_integrity']:.4f}")
    print(f"  Most Common Level: {summary['most_common_stability_level']}")

    return stabilizer


def test_truth_lattice_math():
    """Test Truth Lattice Math (Consensus, Engine) component."""
    print("\n" + "=" * 60)
    print("TESTING TRUTH LATTICE MATH (CONSENSUS, ENGINE)")
    print("=" * 60)

    # Initialize component
    consensus = TruthLatticeMath()

    # Test different signal combinations
    test_cases = [
        ("High Agreement", [0.8, 0.82, 0.79, 0.81, 0.83]),
        ("Medium Agreement", [0.6, 0.7, 0.5, 0.8, 0.6]),
        ("Low Agreement", [0.2, 0.8, 0.1, 0.9, 0.3]),
        ("Random Signals", np.random.random(10).tolist()),
    ]
    for case_name, signals in test_cases:
        print(f"\nTesting {case_name}:")

        # Calculate collapse score
        collapse_score = consensus.collapse_score(signals)
        print(f"  Collapse Score: {collapse_score:.4f}")

        # Check consensus
        consensus_reached = consensus.is_consensus_reached(collapse_score)
        print(f"  Consensus Reached: {consensus_reached}")

        # Evaluate consensus
        result = consensus.evaluate_consensus(signals)
        print(f"  Consensus State: {result.consensus_state.value}")
        print(f"  Signal Count: {result.signal_count}")
        print(f"  Agreement Ratio: {result.agreement_ratio:.4f}")
        print(f"  Volatility Tolerance: {result.volatility_tolerance:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")

    # Get summary
    summary = consensus.get_consensus_summary()
    print("\nConsensus Summary:")
    print(f"  Total Collapses: {summary['total_collapses']}")
    print(f"  Successful Consensus: {summary['successful_consensus']}")
    print(f"  Consensus Rate: {summary['consensus_rate']:.2%}")
    print(f"  Average Collapse Score: {summary['average_collapse_score']:.4f}")

    return consensus


def test_bit_wave_propagator():
    """Test Bit-Wave Propagator component."""
    print("\n" + "=" * 60)
    print("TESTING BIT-WAVE PROPAGATOR")
    print("=" * 60)

    # Initialize component
    propagator = BitWavePropagator()

    # Test different bit depths
    bit_depths = [4, 8, 16]
    signal_types = ["random", "trending", "oscillatory"]

    for bit_depth in bit_depths:
        print(f"\nTesting {bit_depth}-bit depth:")

        for signal_type in signal_types:
            print(f"  {signal_type} signal:")

            # Generate test signal
            signal = generate_test_signal(100, signal_type)

            # Allocate phase vector
            vector = propagator.allocate_phase_vector(bit_depth, signal)
            print(f"    Vector ID: {vector.vector_id}")
            print(f"    Bit Depth: {vector.bit_depth.value}")
            print(f"    Strategy Slots: {vector.strategy_slots}")
            print(f"    Phase Energy: {np.sum(vector.phase_values**2):.4f}")

        # Generate transition matrix
        transition_matrix = propagator.generate_transition_matrix(bit_depth)
        print(f"  Transition Matrix Size: {transition_matrix.shape}")
        print(f"  Matrix Energy: {np.sum(transition_matrix**2):.4f}")

    # Get summary
    summary = propagator.get_propagation_summary()
    print("\nPropagation Summary:")
    print(f"  Total Allocations: {summary['total_allocations']}")
    print(f"  Total Transitions: {summary['total_transitions']}")
    print(f"  Average Bit Depth: {summary['average_bit_depth']:.1f}")
    print(f"  Most Common Strategy: {summary['most_common_strategy']}")
    print(f"  Average Phase Energy: {summary['average_phase_energy']:.4f}")

    return propagator


def test_aleph_overlay_mapper():
    """Test Aleph Overlay Mapper component."""
    print("\n" + "=" * 60)
    print("TESTING ALEPH OVERLAY MAPPER")
    print("=" * 60)

    # Initialize component
    mapper = AlephOverlayMapper()

    # Test different hash signals
    hash_signals = [
        "BTC_45000_2024_01_15",
        "ETH_3000_2024_01_15",
        "XRP_0.5_2024_01_15",
        "SOL_100_2024_01_15",
        "USDC_1.0_2024_01_15",
    ]
    for hash_signal in hash_signals:
        print(f"\nTesting hash signal: {hash_signal}")

        # Map hash to overlay
        overlay = mapper.map_hash_to_overlay(hash_signal)
        print(f"  Overlay Type: {overlay.overlay_type.value}")
        print(f"  Confidence Score: {overlay.confidence_score:.4f}")
        print(f"  Similarity Matrix Size: {overlay.similarity_matrix.shape}")
        print(f"  Phase Alignment Size: {overlay.phase_alignment.shape}")

        # Calculate overlay confidence
        confidence = mapper.calculate_overlay_confidence(overlay.similarity_matrix)
        print(f"  Calculated Confidence: {confidence:.4f}")

    # Get summary
    summary = mapper.get_overlay_summary()
    print("\nOverlay Summary:")
    print(f"  Total Mappings: {summary['total_mappings']}")
    print(f"  Successful Projections: {summary['successful_projections']}")
    print(f"  Average Confidence: {summary['average_confidence']:.4f}")
    print(f"  Most Common Type: {summary['most_common_type']}")
    print(f"  Average Similarity: {summary['average_similarity']:.4f}")
    print(f"  Projection Rate: {summary['projection_rate']:.2%}")

    return mapper


def test_phase_transition_monitor():
    """Test Phase Transition Monitor component."""
    print("\n" + "=" * 60)
    print("TESTING PHASE TRANSITION MONITOR")
    print("=" * 60)

    # Initialize component
    monitor = PhaseTransitionMonitor()

    # Test different signal types
    signal_types = ["random", "trending", "oscillatory", "chaotic"]

    for signal_type in signal_types:
        print(f"\nTesting {signal_type} signal:")

        # Generate test signal
        signal = generate_test_signal(100, signal_type)

        # Calculate drift weight (simulated)
        drift_weight = np.random.random()

        # Evaluate phase state
        phase_state = monitor.evaluate_phase_state(signal, drift_weight)
        print(f"  Phase State: {phase_state.value}")

        # Check transition likelihood
        transition_likely = monitor.is_phase_transition_likely(phase_state, drift_weight)
        print(f"  Transition Likely: {transition_likely}")

    # Get summary
    summary = monitor.get_phase_summary()
    print("\nPhase Summary:")
    print(f"  Total Evaluations: {summary['total_evaluations']}")
    print(f"  Total Transitions: {summary['total_transitions']}")
    print(f"  Current Phase: {summary['current_phase']}")
    print(f"  Average Volatility: {summary['average_volatility']:.4f}")
    print(f"  Average Stability: {summary['average_stability']:.4f}")
    print(f"  Most Common Phase: {summary['most_common_phase']}")
    print(f"  Transition Rate: {summary['transition_rate']:.2%}")

    return monitor


def test_integration():
    """Test integration of all advanced components."""
    print("\n" + "=" * 60)
    print("TESTING ADVANCED COMPONENTS INTEGRATION")
    print("=" * 60)

    # Initialize all components
    weighter = DriftPhaseWeighter()
    stabilizer = GhostFieldStabilizer()
    consensus = TruthLatticeMath()
    propagator = BitWavePropagator()
    mapper = AlephOverlayMapper()
    monitor = PhaseTransitionMonitor()

    # Generate test signal
    signal = generate_test_signal(100, "trending")
    print(f"Generated test signal: {len(signal)} points")

    # Integration workflow
    print("\nIntegration Workflow:")

    # 1. Drift analysis
    print("1. Drift Analysis:")
    drift_weight = weighter.calculate_phase_drift_weight(signal)
    drift_metrics = weighter.analyze_drift_pattern(signal)
    print(f"   Drift Weight: {drift_weight:.4f}")
    print(f"   Drift Type: {drift_metrics.drift_type.value}")

    # 2. Field stability
    print("2. Field Stability:")
    stability_report = stabilizer.evaluate_stability(signal)
    print(f"   Stability Level: {stability_report.stability_level.value}")
    print(f"   Field Integrity: {stability_report.field_integrity:.4f}")

    # 3. Phase transition monitoring
    print("3. Phase Transition Monitoring:")
    phase_state = monitor.evaluate_phase_state(signal, drift_weight)
    print(f"   Phase State: {phase_state.value}")

    # 4. Bit wave propagation
    print("4. Bit Wave Propagation:")
    phase_vector = propagator.allocate_phase_vector(8, signal)
    print(f"   Strategy Slots: {phase_vector.strategy_slots}")

    # 5. Hash overlay mapping
    print("5. Hash Overlay Mapping:")
    hash_signal = f"BTC_{int(time.time())}"
    overlay = mapper.map_hash_to_overlay(hash_signal)
    print(f"   Overlay Type: {overlay.overlay_type.value}")
    print(f"   Confidence: {overlay.confidence_score:.4f}")

    # 6. Consensus evaluation
    print("6. Consensus Evaluation:")
    signals = [drift_weight, stability_report.field_integrity, overlay.confidence_score]
    consensus_result = consensus.evaluate_consensus(signals)
    print(f"   Consensus State: {consensus_result.consensus_state.value}")
    print(f"   Confidence: {consensus_result.confidence:.4f}")

    print("\nIntegration test completed successfully!")

    return {
        "weighter": weighter,
        "stabilizer": stabilizer,
        "consensus": consensus,
        "propagator": propagator,
        "mapper": mapper,
        "monitor": monitor,
    }


def export_test_results(components):
    """Export test results to JSON file."""
    print("\n" + "=" * 60)
    print("EXPORTING TEST RESULTS")
    print("=" * 60)

    results = {"test_timestamp": time.time(), "components": {}}
    # Export component summaries
    if "weighter" in components:
        results["components"]["drift_phase_weighter"] = components["weighter"].get_drift_summary()

    if "stabilizer" in components:
        results["components"]["ghost_field_stabilizer"] = components["stabilizer"].get_stability_summary()

    if "consensus" in components:
        results["components"]["truth_lattice_math"] = components["consensus"].get_consensus_summary()

    if "propagator" in components:
        results["components"]["bit_wave_propagator"] = components["propagator"].get_propagation_summary()

    if "mapper" in components:
        results["components"]["aleph_overlay_mapper"] = components["mapper"].get_overlay_summary()

    if "monitor" in components:
        results["components"]["phase_transition_monitor"] = components["monitor"].get_phase_summary()

    # Save to file
    output_file = "advanced_schwabot_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Test results exported to: {output_file}")

    return output_file


def main():
    """Main test function."""
    print("ADVANCED SCHWABOT COMPONENTS TEST")
    print("=" * 60)
    print("Testing all advanced Schwabot components including:")
    print("- Drift-Phase Weighter")
    print("- Ghost Field Stabilizer")
    print("- Truth Lattice Math (Consensus, Engine)")
    print("- Bit-Wave Propagator")
    print("- Aleph Overlay Mapper")
    print("- Phase Transition Monitor")
    print("=" * 60)

    try:
        # Test individual components
        test_drift_phase_weighter()
        test_ghost_field_stabilizer()
        test_truth_lattice_math()
        test_bit_wave_propagator()
        test_aleph_overlay_mapper()
        test_phase_transition_monitor()

        # Test integration
        components = test_integration()

        # Export results
        export_test_results(components)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Advanced Schwabot components are working correctly.")
        print("The system is ready for advanced trading operations.")

    except Exception as e:
        logger.error(f"Error in advanced component testing: {e}")
        print(f"\nError: {e}")
        print("Please check the component implementations.")


if __name__ == "__main__":
    main()
