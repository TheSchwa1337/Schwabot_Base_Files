import logging
import subprocess
import sys
import time

import numpy as np

from core.dynamic_handoff_orchestrator import DynamicHandoffOrchestrator
from core.internal_state.fileization_manager import FileizationManager
from core.internal_state.state_continuity_manager import StateType
from core.internal_state.visualizer_integration import VisualizerIntegration

#!/usr/bin/env python3
"""
State Continuity System Test
===========================

Comprehensive test of the internal state management system, including:
- StateContinuityManager functionality
- FileizationManager operations
- VisualizerIntegration connections
- JSON hang-up prevention
- Lint compliance validation
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_state_continuity_manager():
    """Test StateContinuityManager functionality."""
    print("\n🔧 Testing StateContinuityManager")
    print("=" * 50)

    try:
            StateContinuityManager,
            StateType,
        )

        # Create manager
        manager = StateContinuityManager()
        print("✅ StateContinuityManager created")

        # Test state updates
        test_data = {}
            "price": 50000,
            "volume": 1000,
            "timestamp": time.time(),
            "indicators": {"rsi": 65.5, "macd": 0.2},
        }
        state_key = manager.update_state()
            StateType.TRADING_STATE,
            test_data,
            agent="BTC",
            phase=32,
            metadata={"source": "test"},
        )
        print(f"✅ Created state: {state_key}")

        # Test state retrieval
        retrieved_state = manager.get_state(StateType.TRADING_STATE, "BTC", 32)
        if retrieved_state:
            print(f"✅ Retrieved state: {retrieved_state.state_type.value}")
        else:
            print("❌ Failed to retrieve state")

        # Test visualization data
        viz_data = manager.get_visualization_data(StateType.TRADING_STATE)
        print(f"✅ Visualization data: {len(viz_data.get('states', []))} states")

        # Test panel data
        panel_data = manager.get_panel_data("trading_panel")
        print(f"✅ Panel data: {panel_data.get('active_states', 0)} active states")

        # Test continuity report
        report = manager.get_continuity_report()
        print(f"✅ Continuity report: {report.get('active_states', 0)} active states")

        return True

    except Exception as e:
        print(f"❌ StateContinuityManager test failed: {e}")
        return False


def test_fileization_manager():
    """Test FileizationManager functionality."""
    print("\n📁 Testing FileizationManager")
    print("=" * 50)

    try:

        # Create manager
        manager = FileizationManager()
        print("✅ FileizationManager created")

        # Test numpy array save/load
        test_array = np.random.rand(10, 10)
        path = manager.save_state(test_array, tag="test", phase=32, agent="BTC")
        print(f"✅ Saved array to: {path}")

        loaded_array = manager.load_state("test", 32, agent="BTC")
        if loaded_array is not None:
            print(f"✅ Loaded array shape: {loaded_array.shape}")
        else:
            print("❌ Failed to load array")

        # Test validation
        valid = manager.validate_state()
            loaded_array, expected_shape=(10, 10), expected_type=np.ndarray
        )
        print(f"✅ Validation result: {valid}")

        # Test dict save/load
        test_dict = {"key": "value", "number": 42, "list": [1, 2, 3]}
        path = manager.save_state(test_dict, tag="test_dict", phase=16, agent="USDC")
        print(f"✅ Saved dict to: {path}")

        loaded_dict = manager.load_state("test_dict", 16, agent="USDC", ext="json")
        if loaded_dict is not None:
            print(f"✅ Loaded dict: {loaded_dict}")
        else:
            print("❌ Failed to load dict")

        # Clean up
        manager.clear_states()
        print("✅ Cleared all states")

        return True

    except Exception as e:
        print(f"❌ FileizationManager test failed: {e}")
        return False


def test_visualizer_integration():
    """Test VisualizerIntegration functionality."""
    print("\n📊 Testing VisualizerIntegration")
    print("=" * 50)

    try:

        # Create integration
        integration = VisualizerIntegration()
        print("✅ VisualizerIntegration created")

        # Test integration status
        status = integration.get_integration_status()
        print(f"✅ Integration status: {status}")

        # Test state updates
        test_data = {}
            "price": 50000,
            "volume": 1000,
            "timestamp": time.time(),
            "indicators": {"rsi": 65.5, "macd": 0.2},
        }
        state_key = integration.update_state()
            StateType.TRADING_STATE,
            test_data,
            agent="BTC",
            phase=32,
            metadata={"source": "test"},
        )
        print(f"✅ Created state: {state_key}")

        # Test visualization data
        viz_data = integration.get_visualization_data(StateType.TRADING_STATE)
        print(f"✅ Visualization data: {len(viz_data.get('states', []))} states")

        # Test panel data
        panel_data = integration.get_panel_data("trading_panel")
        print(f"✅ Panel data: {panel_data}")

        return True

    except Exception as e:
        print(f"❌ VisualizerIntegration test failed: {e}")
        return False


def test_orchestrator_integration():
    """Test DynamicHandoffOrchestrator integration."""
    print("\n🔄 Testing DynamicHandoffOrchestrator Integration")
    print("=" * 50)

    try:

        # Create orchestrator
        orchestrator = DynamicHandoffOrchestrator()
        print("✅ DynamicHandoffOrchestrator created")

        # Test routing with state continuity
        test_data = np.random.rand(100, 100)
        result = orchestrator.route(test_data, phase=32, agent="BTC", utilization=0.5)
        print(f"✅ Routed data: {type(result)}")

        # Test multi-phase handoff
        phases = [2, 4, 8, 16, 32, 42]
        results = orchestrator.multi_phase_handoff(test_data, phases, agent="USDC")
        print(f"✅ Multi-phase results: {len(results)} phases")

        # Test state continuity report
        report = orchestrator.get_state_continuity_report()
        print(f"✅ State continuity report: {report}")

        # Test visualization data
        viz_data = orchestrator.get_visualization_data("handoff")"
        print()
            f"✅ Handoff visualization data: {len(viz_data.get('states', []))} states"
        )

        return True

    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False


def test_json_hangup_prevention():
    """Test JSON hang-up prevention mechanisms."""
    print("\n🛡️ Testing JSON Hang-up Prevention")
    print("=" * 50)

    try:
            StateContinuityManager,
            StateType,
        )

        # Create manager with short timeout
        manager = StateContinuityManager(max_json_timeout=1.0)
        print("✅ Created manager with short timeout")

        # Test large data handling
        large_data = {}
            "large_array": np.random.rand(1000, 1000).tolist(),
            "nested_data": {"level1": {"level2": {"level3": [i for i in range(1000)]}}},
            "timestamp": time.time(),
        }
        # This should not hang due to timeout protection
        start_time = time.time()
        state_key = manager.update_state()
            StateType.MATHEMATICAL_STATE, large_data, agent="BTC", phase=32
        )
        elapsed = time.time() - start_time

        print(f"✅ Large data processed in {elapsed:.2f}s")
        print(f"✅ State key: {state_key}")

        # Test file save with timeout
        start_time = time.time()
        filename = manager.save_state_to_file(state_key)
        elapsed = time.time() - start_time

        print(f"✅ File saved in {elapsed:.2f}s: {filename}")

        return True

    except Exception as e:
        print(f"❌ JSON hang-up prevention test failed: {e}")
        return False


def test_lint_compliance():
    """Test lint compliance of all modules."""
    print("\n🔍 Testing Lint Compliance")
    print("=" * 50)

    try:

        # Test flake8 compliance
        modules = []
            "core/internal_state/state_continuity_manager.py",
            "core/internal_state/fileization_manager.py",
            "core/internal_state/visualizer_integration.py",
            "core/internal_state/__init__.py",
            "core/dynamic_handoff_orchestrator.py",
        ]
        all_passed = True
        for module in modules:
            try:
                result = subprocess.run()
                    ["flake8", module, "--max-line-length=120"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode == 0:
                    print(f"✅ {module}: Lint compliant")
                else:
                    print(f"❌ {module}: Lint errors")
                    print(f"   {result.stdout}")
                    all_passed = False

            except subprocess.TimeoutExpired:
                print(f"⚠️ {module}: Lint check timed out")
                all_passed = False
            except Exception as e:
                print(f"❌ {module}: Lint check failed - {e}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"❌ Lint compliance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 State Continuity System Test Suite")
    print("=" * 60)

    tests = []
        ("StateContinuityManager", test_state_continuity_manager),
        ("FileizationManager", test_fileization_manager),
        ("VisualizerIntegration", test_visualizer_integration),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("JSON Hang-up Prevention", test_json_hangup_prevention),
        ("Lint Compliance", test_lint_compliance),
    ]
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! State continuity system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
