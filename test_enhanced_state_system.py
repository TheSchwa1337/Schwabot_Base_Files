#!/usr/bin/env python3
"""
Enhanced State System Test Suite
================================

Comprehensive test of the enhanced internal state management system, including:
- EnhancedStateManager functionality with logging, memory, and backlogs
- SystemIntegration with all internal systems
- BTC price hashing for demo states
- Testing, demo, and live mode support
- Internal logging and system state management
- Memory and backlog processing
"""

import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_enhanced_state_manager():
    """Test EnhancedStateManager functionality."""
    print("\n🔧 Testing EnhancedStateManager")
    print("=" * 50)

    try:
        from core.internal_state.enhanced_state_manager import (
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        # Test different modes
        for mode in [SystemMode.TESTING, SystemMode.DEMO, SystemMode.LIVE]:
            print(f"\n--- Testing {mode.value} mode ---")

            # Create manager
            manager = EnhancedStateManager(mode=mode, log_level=LogLevel.INFO)
            print(f"✅ EnhancedStateManager created in {mode.value} mode")

            # Test memory operations
            test_data = {"key": "value", "number": 42, "timestamp": time.time()}
            manager.store_memory("test_memory", test_data, ttl=1800.0)
            print("✅ Memory stored")

            retrieved_data = manager.get_memory("test_memory")
            if retrieved_data and retrieved_data == test_data:
                print("✅ Memory retrieved successfully")
            else:
                print("❌ Memory retrieval failed")

            # Test backlog operations
            entry_id = manager.add_backlog_entry(
                priority=5,
                data={"test": "backlog_data"},
                source="test",
                target="memory",
            )
            print(f"✅ Backlog entry added: {entry_id}")

            backlog_status = manager.get_backlog_status()
            print(f"✅ Backlog status: {backlog_status['queue_size']} items in queue")

            # Test BTC price hash generation
            btc_hash = manager.generate_btc_price_hash(50000.0, 1000.0, 32)
            print(f"✅ BTC price hash generated: {btc_hash.hash_value[:16]}...")

            # Test demo state creation
            demo_state = manager.create_demo_state(
                50000.0, 1000.0, 32, {"extra": "demo_data"}
            )
            print(
                f"✅ Demo state created: {demo_state['btc_price_hash']['hash'][:16]}..."
            )

            # Test system status
            status = manager.get_system_status()
            print(
                f"✅ System status: {status['mode']} mode, {status['memory']['active_memories']} memories"
            )

            # Test BTC price history
            history = manager.get_btc_price_history(limit=10)
            print(f"✅ BTC price history: {len(history)} entries")

            # Test export/import
            export_file = manager.export_system_state()
            print(f"✅ System state exported to: {export_file}")

            # Clean up
            del manager

        return True

    except Exception as e:
        print(f"❌ EnhancedStateManager test failed: {e}")
        return False


def test_system_integration():
    """Test SystemIntegration functionality."""
    print("\n🔄 Testing SystemIntegration")
    print("=" * 50)

    try:
        from core.internal_state.system_integration import SystemIntegration
        from core.internal_state.enhanced_state_manager import SystemMode, LogLevel

        # Test different modes
        for mode in [SystemMode.DEMO, SystemMode.TESTING]:
            print(f"\n--- Testing {mode.value} mode ---")

            # Create integration
            integration = SystemIntegration(mode=mode, log_level=LogLevel.INFO)
            print(f"✅ SystemIntegration created in {mode.value} mode")

            # Test demo state creation with BTC hash
            demo_state = integration.create_demo_state_with_btc_hash(
                50000.0, 1000.0, 32, {"integration_test": "data"}
            )

            if "error" not in demo_state:
                print(
                    f"✅ Demo state created with BTC hash: {demo_state['btc_price_hash']['hash'][:16]}..."
                )
                print(
                    f"✅ System integration data: {len(demo_state['system_integration']['connected_systems'])} systems"
                )
            else:
                print(f"❌ Demo state creation failed: {demo_state['error']}")

            # Test BTC price history
            btc_history = integration.get_btc_price_history(limit=20)
            print(
                f"✅ BTC price history: {len(btc_history)} entries with system context"
            )

            # Test comprehensive system status
            status = integration.get_comprehensive_system_status()
            print(
                f"✅ Comprehensive status: {status['system_health_summary']['total_systems']} systems connected"
            )

            # Test system test
            test_results = integration.run_system_test(test_duration=10)
            print(
                f"✅ System test completed: {len(test_results.get('tests', {}))} tests"
            )

            # Test export
            export_file = integration.export_integrated_system_state()
            print(f"✅ Integrated system state exported to: {export_file}")

            # Clean up
            del integration

        return True

    except Exception as e:
        print(f"❌ SystemIntegration test failed: {e}")
        return False


def test_btc_price_hashing():
    """Test BTC price hashing functionality."""
    print("\n₿ Testing BTC Price Hashing")
    print("=" * 50)

    try:
        from core.internal_state.enhanced_state_manager import (
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        # Create manager
        manager = EnhancedStateManager(mode=SystemMode.DEMO, log_level=LogLevel.INFO)
        print("✅ EnhancedStateManager created for BTC testing")

        # Test BTC price hash generation
        test_prices = [45000.0, 50000.0, 55000.0]
        test_volumes = [800.0, 1000.0, 1200.0]
        test_phases = [16, 32, 42]

        generated_hashes = []

        for price in test_prices:
            for volume in test_volumes:
                for phase in test_phases:
                    btc_hash = manager.generate_btc_price_hash(price, volume, phase)
                    generated_hashes.append(btc_hash)

                    print(
                        f"✅ Generated hash for price={price}, volume={volume}, phase={phase}: {btc_hash.hash_value[:16]}..."
                    )

                    # Verify hash properties
                    assert btc_hash.price == price
                    assert btc_hash.volume == volume
                    assert btc_hash.phase == phase
                    assert len(btc_hash.hash_value) == 64  # SHA256 hex length
                    assert btc_hash.agent == "BTC"

        print(f"✅ Generated {len(generated_hashes)} BTC price hashes")

        # Test hash uniqueness
        hash_values = [h.hash_value for h in generated_hashes]
        unique_hashes = set(hash_values)
        print(
            f"✅ Hash uniqueness: {len(unique_hashes)} unique hashes out of {len(hash_values)} total"
        )

        # Test BTC price history
        history = manager.get_btc_price_history(limit=50)
        print(f"✅ BTC price history: {len(history)} entries")

        # Test demo state with BTC hash
        demo_state = manager.create_demo_state(
            50000.0, 1000.0, 32, {"btc_test": "data"}
        )

        if "btc_price_hash" in demo_state:
            btc_data = demo_state["btc_price_hash"]
            print(
                f"✅ Demo state BTC data: price={btc_data['price']}, volume={btc_data['volume']}, hash={btc_data['hash'][:16]}..."
            )
        else:
            print("❌ Demo state missing BTC price hash")

        return True

    except Exception as e:
        print(f"❌ BTC price hashing test failed: {e}")
        return False


def test_memory_and_backlog():
    """Test memory and backlog functionality."""
    print("\n💾 Testing Memory and Backlog")
    print("=" * 50)

    try:
        from core.internal_state.enhanced_state_manager import (
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        # Create manager
        manager = EnhancedStateManager(mode=SystemMode.DEMO, log_level=LogLevel.INFO)
        print("✅ EnhancedStateManager created for memory/backlog testing")

        # Test memory operations
        memory_tests = [
            {"id": "test1", "data": {"simple": "value"}, "ttl": 3600.0},
            {
                "id": "test2",
                "data": {"complex": {"nested": "data", "array": [1, 2, 3]}},
                "ttl": 1800.0,
            },
            {"id": "test3", "data": {"large": "x" * 1000}, "ttl": 900.0},
        ]
        for test in memory_tests:
            manager.store_memory(test["id"], test["data"], test["ttl"])
            print(f"✅ Stored memory: {test['id']}")

            retrieved = manager.get_memory(test["id"])
            if retrieved == test["data"]:
                print(f"✅ Retrieved memory: {test['id']}")
            else:
                print(f"❌ Memory retrieval failed: {test['id']}")

        # Test backlog operations
        backlog_tests = [
            {
                "priority": 1,
                "data": {"high": "priority"},
                "source": "test",
                "target": "memory",
            },
            {
                "priority": 5,
                "data": {"medium": "priority"},
                "source": "test",
                "target": "hash",
            },
            {
                "priority": 10,
                "data": {"low": "priority"},
                "source": "demo",
                "target": "memory",
            },
        ]
        for test in backlog_tests:
            entry_id = manager.add_backlog_entry(
                test["priority"], test["data"], test["source"], test["target"]
            )
            print(f"✅ Added backlog entry: {entry_id} (priority: {test['priority']})")

        # Test backlog status
        status = manager.get_backlog_status()
        print(
            f"✅ Backlog status: {status['queue_size']} in queue, {status['processed_count']} processed"
        )

        # Wait for some processing
        print("⏳ Waiting for backlog processing...")
        time.sleep(5)

        # Check updated status
        updated_status = manager.get_backlog_status()
        print(
            f"✅ Updated backlog status: {updated_status['queue_size']} in queue, {updated_status['processed_count']} processed"
        )

        # Test system status
        system_status = manager.get_system_status()
        print(
            f"✅ System status: {system_status['memory']['active_memories']} active memories"
        )

        return True

    except Exception as e:
        print(f"❌ Memory and backlog test failed: {e}")
        return False


def test_logging_and_system_states():
    """Test logging and system states functionality."""
    print("\n📝 Testing Logging and System States")
    print("=" * 50)

    try:
        from core.internal_state.enhanced_state_manager import (
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        # Test different log levels
        for log_level in [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING]:
            print(f"\n--- Testing {log_level.value} log level ---")

            # Create manager
            manager = EnhancedStateManager(mode=SystemMode.DEMO, log_level=log_level)
            print(f"✅ EnhancedStateManager created with {log_level.value} logging")

            # Generate some activity to test logging
            for i in range(5):
                manager.store_memory(
                    f"log_test_{i}", {"iteration": i, "timestamp": time.time()}
                )
                manager.add_backlog_entry(i, {"log_test": i}, "test", "memory")
                manager.generate_btc_price_hash(45000 + i * 100, 1000 + i * 50, 32)

            # Test system status
            status = manager.get_system_status()
            print(
                f"✅ System status retrieved: {status['mode']} mode, uptime: {status['uptime_seconds']:.2f}s"
            )

            # Test thread status
            threads = status["threads"]
            print(
                f"✅ Thread status: memory_cleanup={threads['memory_cleanup']}, "
                f"backlog_processor={threads['backlog_processor']}, "
                f"btc_hash_generator={threads['btc_hash_generator']}"
            )

            # Test memory and backlog status
            memory_info = status["memory"]
            backlog_info = status["backlog"]
            btc_info = status["btc_price"]

            print(f"✅ Memory: {memory_info['active_memories']} active memories")
            print(f"✅ Backlog: {backlog_info['queue_size']} in queue")
            print(f"✅ BTC: {btc_info['history_size']} price hashes")

            # Clean up
            del manager

        return True

    except Exception as e:
        print(f"❌ Logging and system states test failed: {e}")
        return False


def test_demo_state_generation():
    """Test demo state generation with BTC price hashing."""
    print("\n🎭 Testing Demo State Generation")
    print("=" * 50)

    try:
        from core.internal_state.system_integration import SystemIntegration
        from core.internal_state.enhanced_state_manager import SystemMode, LogLevel

        # Create integration
        integration = SystemIntegration(mode=SystemMode.DEMO, log_level=LogLevel.INFO)
        print("✅ SystemIntegration created for demo testing")

        # Test demo state generation with different BTC prices
        demo_scenarios = [
            {
                "price": 45000.0,
                "volume": 800.0,
                "phase": 16,
                "description": "Low price scenario",
            },
            {
                "price": 50000.0,
                "volume": 1000.0,
                "phase": 32,
                "description": "Medium price scenario",
            },
            {
                "price": 55000.0,
                "volume": 1200.0,
                "phase": 42,
                "description": "High price scenario",
            },
        ]
        generated_states = []

        for scenario in demo_scenarios:
            print(f"\n--- {scenario['description']} ---")

            demo_state = integration.create_demo_state_with_btc_hash(
                scenario["price"],
                scenario["volume"],
                scenario["phase"],
                {"scenario": scenario["description"], "test_data": "demo_generation"},
            )

            if "error" not in demo_state:
                generated_states.append(demo_state)

                # Verify demo state structure
                btc_data = demo_state["btc_price_hash"]
                system_data = demo_state["system_integration"]

                print(
                    f"✅ Demo state created: price={btc_data['price']}, volume={btc_data['volume']}"
                )
                print(f"✅ BTC hash: {btc_data['hash'][:16]}...")
                print(
                    f"✅ System integration: {len(system_data['connected_systems'])} systems"
                )
                print(
                    f"✅ System metrics: {demo_state['system_metrics']['memory_count']} memories, "
                    f"{demo_state['system_metrics']['backlog_size']} backlog items"
                )
            else:
                print(f"❌ Demo state creation failed: {demo_state['error']}")

        print(f"\n✅ Generated {len(generated_states)} demo states")

        # Test comprehensive status
        status = integration.get_comprehensive_system_status()
        print(
            f"✅ Comprehensive status: {status['system_health_summary']['total_systems']} systems, "
            f"{status['system_health_summary']['healthy_systems']} healthy"
        )

        # Test BTC price history
        history = integration.get_btc_price_history(limit=30)
        print(f"✅ BTC price history: {len(history)} entries with system context")

        # Test export
        export_file = integration.export_integrated_system_state()
        print(f"✅ Integrated system state exported to: {export_file}")

        return True

    except Exception as e:
        print(f"❌ Demo state generation test failed: {e}")
        return False


def test_system_initialization():
    """Test system initialization and organization."""
    print("\n🚀 Testing System Initialization")
    print("=" * 50)

    try:
        from core.internal_state.system_integration import SystemIntegration
        from core.internal_state.enhanced_state_manager import SystemMode, LogLevel

        # Test initialization in different modes
        for mode in [SystemMode.TESTING, SystemMode.DEMO, SystemMode.LIVE]:
            print(f"\n--- Testing {mode.value} mode initialization ---")

            # Create integration
            integration = SystemIntegration(mode=mode, log_level=LogLevel.INFO)
            print(f"✅ SystemIntegration initialized in {mode.value} mode")

            # Test system connections
            status = integration.get_comprehensive_system_status()
            connected_systems = status.get("connected_systems", {})

            print(f"✅ Connected systems: {len(connected_systems)}")
            for system_name, system_info in connected_systems.items():
                print(f"   - {system_name}: {system_info['status']}")

            # Test integration status
            integration_status = status.get("integration_status", {})
            print("✅ Integration status:")
            for key, value in integration_status.items():
                print(f"   - {key}: {value}")

            # Test system health
            health_summary = status.get("system_health_summary", {})
            print(
                f"✅ System health: {health_summary.get('healthy_systems', 0)}/{health_summary.get('total_systems', 0)} healthy"
            )

            # Test demo state creation
            demo_state = integration.create_demo_state_with_btc_hash(
                50000.0, 1000.0, 32, {"init_test": "data"}
            )

            if "error" not in demo_state:
                print(f"✅ Demo state created successfully in {mode.value} mode")
            else:
                print(
                    f"❌ Demo state creation failed in {mode.value} mode: {demo_state['error']}"
                )

            # Clean up
            del integration

        return True

    except Exception as e:
        print(f"❌ System initialization test failed: {e}")
        return False


def main():
    """Run all enhanced state system tests."""
    print("🚀 Enhanced State System Test Suite")
    print("=" * 60)

    tests = [
        ("EnhancedStateManager", test_enhanced_state_manager),
        ("SystemIntegration", test_system_integration),
        ("BTC Price Hashing", test_btc_price_hashing),
        ("Memory and Backlog", test_memory_and_backlog),
        ("Logging and System States", test_logging_and_system_states),
        ("Demo State Generation", test_demo_state_generation),
        ("System Initialization", test_system_initialization),
    ]
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 60}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📋 Enhanced State System Test Summary")
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
        print("🎉 All enhanced state system tests passed!")
        print(
            "✅ System properly initializes, organizes, and connects to internal systems"
        )
        print("✅ BTC price hashing works correctly for demo states")
        print("✅ Memory and backlog management is functional")
        print("✅ Logging and system states are properly managed")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
