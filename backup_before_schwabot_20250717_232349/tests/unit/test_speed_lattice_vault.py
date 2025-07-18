import json
import os
import sys
import time

"""
Test Script for Speed Lattice Vault System - SP 1.27 AE
Comprehensive validation of temporal anchor equations, recursive drift catch loops,
and strategic containment zones with full delta mapping capabilities.
"""


# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

    SpeedLatticeVault,
    LatticeVaultExecutor,
    ChronoBiasLevel,
    AnchorPhase,
    DeltaMap,
    ShiftPattern,
)


def test_core_initialization():
    """Test core system initialization"""
    print("🧪 Testing Core System Initialization...")

    # Test basic initialization
    vault = SpeedLatticeVault(warp_speed=10000, cycles=64)

    assert vault.warp_speed == 10000, ()
        f"Expected warp speed 10000, got {vault.warp_speed}"
    )
    assert vault.cycles == 64, f"Expected cycles 64, got {vault.cycles}"
    assert vault.phase_lock == "Ω-Phase", ()
        f"Expected phase lock Ω-Phase, got {vault.phase_lock}"
    )
    assert vault.drift_matrix.shape == (64, 64), ()
        f"Expected drift matrix shape (64, 64), got {vault.drift_matrix.shape}"
    )

    print("✅ Core initialization test passed")
    return vault


def test_chrono_bias_calculation(vault: SpeedLatticeVault):
    """Test chronological bias calculation"""
    print("🧪 Testing Chronological Bias Calculation...")

    # Test bias calculation
    bias = vault._calculate_chrono_bias(vault.drift_matrix, 0.0)
    assert isinstance(bias, float), f"Expected float, got {type(bias)}"
    assert bias >= 0, f"Expected non-negative bias, got {bias}"

    # Test with different time states
    biases = []
    for t in [0.0, 1.0, 2.0, 3.0]:
        bias = vault._calculate_chrono_bias(vault.drift_matrix, t)
        biases.append(bias)
        assert isinstance(bias, float), f"Expected float for t={t}, got {type(bias)}"

    print()
        f"✅ Chrono bias calculation test passed - Bias range: {min(biases):.4f} to {max(biases):.4f}"
    )
    return biases


def test_stability_factor_calculation(vault: SpeedLatticeVault):
    """Test stability factor calculation"""
    print("🧪 Testing Stability Factor Calculation...")

    # Test stability factor calculation
    drift_correction = 0.1
    stability = vault._calculate_stability_factor(drift_correction)

    assert isinstance(stability, float), f"Expected float, got {type(stability)}"
    assert stability > 0, f"Expected positive stability, got {stability}"

    print(f"✅ Stability factor calculation test passed - Stability: {stability:.6f}")
    return stability


def test_surround_chronomancy(vault: SpeedLatticeVault):
    """Test surround chronomancy recursive drift catch loop"""
    print("🧪 Testing Surround Chronomancy...")

    # Test with different bias levels
    test_cases = []
        (0.5, "low_bias"),  # Should sustain recursion
        (0.15, "high_bias"),  # Should inject feedback layer
        (-0.1, "negative_bias"),  # Should activate fallback
    ]
    results = []
    for bias_level, case_name in test_cases:
        # Create test drift matrix with specific bias
        test_matrix = vault.drift_matrix.copy()
        test_matrix *= bias_level

        result = vault.surround_chronomancy(test_matrix, 0.0)

        assert "anchor_bias" in result, f"Missing anchor_bias in {case_name}"
        assert "action" in result, f"Missing action in {case_name}"
        assert "chrono_bias" in result, f"Missing chrono_bias in {case_name}"
        assert "stability_factor" in result, f"Missing stability_factor in {case_name}"

        results.append((case_name, result))
        print()
            f"  {case_name}: Bias={result['chrono_bias']:.4f}, Action={result['action']}"
        )

    print("✅ Surround chronomancy test passed")
    return results


def test_containment_zones(vault: SpeedLatticeVault):
    """Test strategic containment zones"""
    print("🧪 Testing Strategic Containment Zones...")

    test_biases = [0.2, 0.8, 0.18, 0.30]
    expected_zones = []
        ChronoBiasLevel.ECHO_CROWN,
        ChronoBiasLevel.AEON_RIM,
        ChronoBiasLevel.VORTEX_MARGIN,
        ChronoBiasLevel.COLLAPSE_ARC,
    ]
    for bias, expected_zone in zip(test_biases, expected_zones):
        zone = vault.get_containment_zone(bias)
        assert zone == expected_zone, ()
            f"Expected {expected_zone} for bias {bias}, got {zone}"
        )
        print(f"  Bias {bias:.2f} → Zone: {zone.name}")

    print("✅ Containment zones test passed")
    return test_biases


def test_anchor_points(vault: SpeedLatticeVault):
    """Test SP 1.27 AE anchor points calculation"""
    print("🧪 Testing SP 1.27 AE Anchor Points...")

    t_state = 1.0
    anchor_points = vault.calculate_anchor_points(t_state)

    # Verify all anchor phases are present
    expected_phases = []
        AnchorPhase.T1_ZERO_PHASE,
        AnchorPhase.T2_MID_CYCLE,
        AnchorPhase.T3_PHASE_FLIP,
        AnchorPhase.T4_PROFIT_COLLAPSE,
    ]
    for phase in expected_phases:
        assert phase in anchor_points, f"Missing anchor phase {phase}"
        assert isinstance(anchor_points[phase], float), ()
            f"Expected float for {phase}, got {type(anchor_points[phase])}"
        )
        print(f"  {phase.value}: {anchor_points[phase]:.4f}")

    print("✅ Anchor points test passed")
    return anchor_points


def test_shift_pattern_generation(vault: SpeedLatticeVault):
    """Test shift pattern generation"""
    print("🧪 Testing Shift Pattern Generation...")

    patterns = []
    for tick in range(10):
        pattern = vault.generate_shift_pattern(tick)

        assert isinstance(pattern, ShiftPattern), ()
            f"Expected ShiftPattern, got {type(pattern)}"
        )
        assert pattern.tick == tick, f"Expected tick {tick}, got {pattern.tick}"
        assert isinstance(pattern.delta_t, float), ()
            f"Expected float delta_t, got {type(pattern.delta_t)}"
        )
        assert isinstance(pattern.delta_psi, float), ()
            f"Expected float delta_psi, got {type(pattern.delta_psi)}"
        )
        assert isinstance(pattern.vault_sync, bool), ()
            f"Expected bool vault_sync, got {type(pattern.vault_sync)}"
        )
        assert isinstance(pattern.action_trigger, str), ()
            f"Expected str action_trigger, got {type(pattern.action_trigger)}"
        )
        assert isinstance(pattern.phase_lock, str), ()
            f"Expected str phase_lock, got {type(pattern.phase_lock)}"
        )

        patterns.append(pattern)
        print()
            f"  Tick {tick}: ΔT={pattern.delta_t:.6f}, Δψ={pattern.delta_psi:.6f}, Action={pattern.action_trigger}"
        )

    print("✅ Shift pattern generation test passed")
    return patterns


def test_delta_map_creation(vault: SpeedLatticeVault):
    """Test full delta map creation"""
    print("🧪 Testing Full Delta Map Creation...")

    # Generate some shift patterns first
    for tick in range(64):
        vault.generate_shift_pattern(tick)

    # Create delta map
    delta_map = vault.create_full_delta_map()

    assert isinstance(delta_map, DeltaMap), f"Expected DeltaMap, got {type(delta_map)}"
    assert isinstance(delta_map.delta_psi, list), ()
        f"Expected list delta_psi, got {type(delta_map.delta_psi)}"
    )
    assert isinstance(delta_map.delta_t, list), ()
        f"Expected list delta_t, got {type(delta_map.delta_t)}"
    )
    assert isinstance(delta_map.delta_xi, list), ()
        f"Expected list delta_xi, got {type(delta_map.delta_xi)}"
    )
    assert isinstance(delta_map.vault_sync, bool), ()
        f"Expected bool vault_sync, got {type(delta_map.vault_sync)}"
    )
    assert isinstance(delta_map.echo_map, str), ()
        f"Expected str echo_map, got {type(delta_map.echo_map)}"
    )
    assert isinstance(delta_map.timestamp, float), ()
        f"Expected float timestamp, got {type(delta_map.timestamp)}"
    )

    # Verify vector lengths
    assert len(delta_map.delta_psi) == len(delta_map.delta_t), ()
        "Delta vectors should have same length"
    )
    assert len(delta_map.delta_t) == len(delta_map.delta_xi), ()
        "Delta vectors should have same length"
    )

    print()
        f"✅ Delta map creation test passed - Vector length: {len(delta_map.delta_psi)}"
    )
    print(f"  Vault Sync: {delta_map.vault_sync}")
    print(f"  Echo Map: {delta_map.echo_map}")
    print(f"  Timestamp: {delta_map.timestamp}")

    return delta_map


def test_warp_cycle_execution(vault: SpeedLatticeVault):
    """Test full warp cycle execution"""
    print("🧪 Testing Full Warp Cycle Execution...")

    # Execute warp cycle with reduced cycles for testing
    vault.cycles = 32  # Reduced for testing
    result = vault.execute_warp_cycle()

    assert "cycle_results" in result, "Missing cycle_results in warp result"
    assert "final_delta_map" in result, "Missing final_delta_map in warp result"
    assert "total_cycles" in result, "Missing total_cycles in warp result"
    assert "warp_speed" in result, "Missing warp_speed in warp result"
    assert "execution_time" in result, "Missing execution_time in warp result"

    assert len(result["cycle_results"]) == 32, ()
        f"Expected 32 cycle results, got {len(result['cycle_results'])}"
    )
    assert result["total_cycles"] == 32, ()
        f"Expected 32 total cycles, got {result['total_cycles']}"
    )
    assert result["warp_speed"] == 10000, ()
        f"Expected warp speed 10000, got {result['warp_speed']}"
    )

    print()
        f"✅ Warp cycle execution test passed - {len(result['cycle_results'])} cycles completed"
    )

    # Show sample cycle results
    for i in [0, 8, 16, 24, 31]:
        cycle = result["cycle_results"][i]
        print()
            f"  Cycle {i}: Bias={cycle['chrono_result']['chrono_bias']:.4f}, "
            f"Zone={cycle['containment_zone'].name}"
        )

    return result


def test_system_status(vault: SpeedLatticeVault):
    """Test system status reporting"""
    print("🧪 Testing System Status Reporting...")

    status = vault.get_system_status()

    required_fields = []
        "warp_speed",
        "cycles_completed",
        "chrono_bias",
        "stability_factor",
        "vault_sync",
        "echo_pulse_active",
        "fractal_feedback_loop",
        "phase_lock",
        "containment_zone",
        "resonance_history_size",
        "feedback_state_size",
        "delta_maps_count",
    ]
    for field in required_fields:
        assert field in status, f"Missing field {field} in system status"

    print("✅ System status test passed")
    print(f"  Warp Speed: {status['warp_speed']}x")
    print(f"  Containment Zone: {status['containment_zone']}")
    print(f"  Vault Sync: {status['vault_sync']}")
    print(f"  Resonance History: {status['resonance_history_size']} entries")

    return status


def test_lattice_vault_executor():
    """Test LatticeVaultExecutor class"""
    print("🧪 Testing LatticeVaultExecutor...")

    executor = LatticeVaultExecutor(warp_speed=10000)

    # Test strategy cycle execution
    result = executor.execute_strategy_cycle()

    assert "timestamp" in result, "Missing timestamp in execution result"
    assert "warp_result" in result, "Missing warp_result in execution result"
    assert "system_status" in result, "Missing system_status in execution result"
    assert "market_data_integrated" in result, ()
        "Missing market_data_integrated in execution result"
    )

    # Test with market data
    market_data = {"price_change": 0.2, "volume": 1000000, "volatility": 0.15}
    result_with_market = executor.execute_strategy_cycle(market_data)
    assert result_with_market["market_data_integrated"], ()
        "Market data should be integrated"
    )

    # Test execution summary
    summary = executor.get_execution_summary()
    assert "total_executions" in summary, "Missing total_executions in summary"
    assert summary["total_executions"] == 2, ()
        f"Expected 2 executions, got {summary['total_executions']}"
    )

    print("✅ LatticeVaultExecutor test passed")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Average Chrono Bias: {summary['average_chrono_bias']:.4f}")

    return executor


def test_data_export(vault: SpeedLatticeVault, executor: LatticeVaultExecutor):
    """Test data export functionality"""
    print("🧪 Testing Data Export Functionality...")

    # Test delta map export
    delta_map_file = vault.export_delta_map_data()
    assert os.path.exists(delta_map_file), ()
        f"Delta map file not created: {delta_map_file}"
    )

    # Verify JSON structure
    with open(delta_map_file, "r") as f:
        delta_data = json.load(f)

    assert "system_status" in delta_data, "Missing system_status in exported data"
    assert "delta_maps" in delta_data, "Missing delta_maps in exported data"
    assert "shift_patterns" in delta_data, "Missing shift_patterns in exported data"

    # Test full report export
    report_file = executor.export_full_report()
    assert os.path.exists(report_file), f"Report file not created: {report_file}"

    # Verify JSON structure
    with open(report_file, "r") as f:
        report_data = json.load(f)

    assert "execution_summary" in report_data, "Missing execution_summary in report"
    assert "execution_history" in report_data, "Missing execution_history in report"
    assert "delta_map_file" in report_data, "Missing delta_map_file in report"

    print("✅ Data export test passed")
    print(f"  Delta Map File: {delta_map_file}")
    print(f"  Report File: {report_file}")

    return delta_map_file, report_file


def test_performance_benchmark():
    """Test performance benchmarks"""
    print("🧪 Testing Performance Benchmarks...")

    # Test different warp speeds
    warp_speeds = [1000, 5000, 10000]
    performance_results = {}

    for speed in warp_speeds:
        start_time = time.time()
        vault = SpeedLatticeVault(warp_speed=speed, cycles=64)
        vault.execute_warp_cycle()
        end_time = time.time()

        execution_time = end_time - start_time
        performance_results[speed] = execution_time

        print(f"  Warp Speed {speed}x: {execution_time:.3f}s")

    print("✅ Performance benchmark test passed")
    return performance_results


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Speed Lattice Vault System - SP 1.27 AE")
    print("Comprehensive Test Suite")
    print("=" * 60)

    test_results = {}

    try:
        # Core system tests
        vault = test_core_initialization()
        test_results["core_initialization"] = "PASSED"

        test_results["chrono_bias"] = test_chrono_bias_calculation(vault)
        test_results["stability_factor"] = test_stability_factor_calculation(vault)
        test_results["surround_chronomancy"] = test_surround_chronomancy(vault)
        test_results["containment_zones"] = test_containment_zones(vault)
        test_results["anchor_points"] = test_anchor_points(vault)
        test_results["shift_patterns"] = test_shift_pattern_generation(vault)
        test_results["delta_map"] = test_delta_map_creation(vault)
        test_results["warp_cycle"] = test_warp_cycle_execution(vault)
        test_results["system_status"] = test_system_status(vault)

        # Executor tests
        executor = test_lattice_vault_executor()
        test_results["executor"] = "PASSED"

        # Export tests
        delta_file, report_file = test_data_export(vault, executor)
        test_results["data_export"] = "PASSED"

        # Performance tests
        test_results["performance"] = test_performance_benchmark()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Speed Lattice Vault System Ready")
        print("=" * 60)

        # Generate test summary
        summary = {}
            "test_status": "ALL_PASSED",
            "timestamp": time.time(),
            "export_files": {"delta_map": delta_file, "report": report_file},
            "performance_results": test_results["performance"],
        }
        # Save test summary
        summary_file = f"speed_lattice_vault_test_summary_{int(time.time())}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📁 Test summary saved to: {summary_file}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
