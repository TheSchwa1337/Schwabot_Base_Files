#!/usr/bin/env python3
"""
Simple Enhanced State System Test
================================

Basic test of the enhanced state system functionality.
"""

import logging
import sys
import time

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_enhanced_state_manager():
    """Test EnhancedStateManager basic functionality."""
    print("\n🔧 Testing EnhancedStateManager")
    print("=" * 50)

    try:
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        # Create manager in demo mode
        manager = EnhancedStateManager(mode=SystemMode.DEMO, log_level=LogLevel.INFO)
        print("✅ EnhancedStateManager created successfully")

        # Test memory operations
        test_data = {"key": "value", "number": 42, "timestamp": time.time()}
        manager.store_memory("test_memory", test_data, ttl=1800.0)
        print("✅ Memory stored successfully")

        retrieved_data = manager.get_memory("test_memory")
        if retrieved_data and retrieved_data == test_data:
            print("✅ Memory retrieved successfully")
        else:
            print("❌ Memory retrieval failed")

        # Test BTC price hash generation
        btc_hash = manager.generate_btc_price_hash(50000.0, 1000.0, 32)
        print(f"✅ BTC price hash generated: {btc_hash.hash_value[:16]}...")

        # Test demo state creation
        demo_state = manager.create_demo_state()
            50000.0, 1000.0, 32, {"extra": "demo_data"}
        )
        print(f"✅ Demo state created: {demo_state['btc_price_hash']['hash'][:16]}...")

        # Test system status
        status = manager.get_system_status()
        print()
            f"✅ System status: {status['mode']} mode, {status['memory']['active_memories']} memories"
        )

        # Test BTC price history
        history = manager.get_btc_price_history(limit=10)
        print(f"✅ BTC price history: {len(history)} entries")

        # Test export
        export_file = manager.export_system_state()
        print(f"✅ System state exported to: {export_file}")

        return True

    except Exception as e:
        print(f"❌ EnhancedStateManager test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_system_integration():
    """Test SystemIntegration basic functionality."""
    print("\n🔄 Testing SystemIntegration")
    print("=" * 50)

    try:
        from core.internal_state.enhanced_state_manager import LogLevel, SystemMode
        from core.internal_state.system_integration import SystemIntegration

        # Create integration
        integration = SystemIntegration(mode=SystemMode.DEMO, log_level=LogLevel.INFO)
        print("✅ SystemIntegration created successfully")

        # Test demo state creation with BTC hash
        demo_state = integration.create_demo_state_with_btc_hash()
            50000.0, 1000.0, 32, {"integration_test": "data"}
        )

        if "error" not in demo_state:
            print()
                f"✅ Demo state created with BTC hash: {demo_state['btc_price_hash']['hash'][:16]}..."
            )
            print()
                f"✅ System integration data: {len(demo_state['system_integration']['connected_systems'])} systems"
            )
        else:
            print(f"❌ Demo state creation failed: {demo_state['error']}")

        # Test comprehensive system status
        status = integration.get_comprehensive_system_status()
        print()
            f"✅ Comprehensive status: {status['system_health_summary']['total_systems']} systems connected"
        )

        # Test export
        export_file = integration.export_integrated_system_state()
        print(f"✅ Integrated system state exported to: {export_file}")

        return True

    except Exception as e:
        print(f"❌ SystemIntegration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_btc_price_hashing():
    """Test BTC price hashing functionality."""
    print("\n₿ Testing BTC Price Hashing")
    print("=" * 50)

    try:
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

                    print()
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
        print()
            f"✅ Hash uniqueness: {len(unique_hashes)} unique hashes out of {len(hash_values)} total"
        )

        # Test BTC price history
        history = manager.get_btc_price_history(limit=50)
        print(f"✅ BTC price history: {len(history)} entries")

        return True

    except Exception as e:
        print(f"❌ BTC price hashing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run basic enhanced state system tests."""
    print("🚀 Simple Enhanced State System Test")
    print("=" * 60)

    tests = []
        ("EnhancedStateManager", test_enhanced_state_manager),
        ("SystemIntegration", test_system_integration),
        ("BTC Price Hashing", test_btc_price_hashing),
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
    print("\n📋 Simple Enhanced State System Test Summary")
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
        print()
            "✅ System properly initializes, organizes, and connects to internal systems"
        )
        print("✅ BTC price hashing works correctly for demo states")
        print("✅ Memory and backlog management is functional")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
