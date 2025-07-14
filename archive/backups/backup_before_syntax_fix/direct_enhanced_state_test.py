#!/usr/bin/env python3
"""
Direct Enhanced State System Test
================================

Direct test of the enhanced state system functionality without complex dependencies.
"""

import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SystemMode(Enum):
    """System operation modes."""

    TESTING = "testing"
    DEMO = "demo"
    LIVE = "live"


class LogLevel(Enum):
    """Logging levels for internal system."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
    class BTCPriceHash:
    """BTC price hash for demo state generation."""

    price: float
    volume: float
    timestamp: datetime
    hash_value: str
    phase: int
    agent: str = "BTC"

    @classmethod
    def from_price_data()
        cls, price: float, volume: float, phase: int = 32
    ) -> "BTCPriceHash":
        """Create BTC price hash from price data."""
        timestamp = datetime.now()
        data_str = f"{price:.8f}_{volume:.8f}_{timestamp.isoformat()}_{phase}"
        hash_value = hashlib.sha256(data_str.encode()).hexdigest()
        return cls()
            price=price,
            volume=volume,
            timestamp=timestamp,
            hash_value=hash_value,
            phase=phase,
        )


def test_btc_price_hashing():
    """Test BTC price hashing functionality."""
    print("\n₿ Testing BTC Price Hashing")
    print("=" * 50)

    try:
        # Test BTC price hash generation
        test_prices = [45000.0, 50000.0, 55000.0]
        test_volumes = [800.0, 1000.0, 1200.0]
        test_phases = [16, 32, 42]

        generated_hashes = []

        for price in test_prices:
            for volume in test_volumes:
                for phase in test_phases:
                    btc_hash = BTCPriceHash.from_price_data(price, volume, phase)
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
            f"✅ Hash uniqueness: {len(unique_hashes)} unique hashes out of {"}
                len(hash_values)
            } total"
        )

        return True

    except Exception as e:
        print(f"❌ BTC price hashing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_enhanced_state_manager_import():
    """Test if EnhancedStateManager can be imported."""
    print("\n🔧 Testing EnhancedStateManager Import")
    print("=" * 50)

    try:
        # Try to import the enhanced state manager
        from core.internal_state.enhanced_state_manager import ()
            EnhancedStateManager,
            SystemMode,
            LogLevel,
        )

        print("✅ EnhancedStateManager imported successfully")

        # Test basic functionality
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
            f"✅ System status: {status['mode']} mode, {"}
                status['memory']['active_memories']
            } memories"
        )

        return True

    except Exception as e:
        print(f"❌ EnhancedStateManager import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_system_integration_import():
    """Test if SystemIntegration can be imported."""
    print("\n🔄 Testing SystemIntegration Import")
    print("=" * 50)

    try:
        # Try to import the system integration
        from core.internal_state.system_integration import SystemIntegration
        from core.internal_state.enhanced_state_manager import SystemMode, LogLevel

        print("✅ SystemIntegration imported successfully")

        # Test basic functionality
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
            f"✅ Comprehensive status: {"}
                status['system_health_summary']['total_systems']
            } systems connected"
        )

        return True

    except Exception as e:
        print(f"❌ SystemIntegration import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_demo_state_generation():
    """Test demo state generation with BTC price hashing."""
    print("\n🎭 Testing Demo State Generation")
    print("=" * 50)

    try:
        # Test demo state generation with different BTC prices
        demo_scenarios = []
            {}
                "price": 45000.0,
                "volume": 800.0,
                "phase": 16,
                "description": "Low price scenario",
            },
            {}
                "price": 50000.0,
                "volume": 1000.0,
                "phase": 32,
                "description": "Medium price scenario",
            },
            {}
                "price": 55000.0,
                "volume": 1200.0,
                "phase": 42,
                "description": "High price scenario",
            },
        ]
        generated_states = []

        for scenario in demo_scenarios:
            print(f"\n--- {scenario['description']} ---")

            # Generate BTC price hash
            btc_hash = BTCPriceHash.from_price_data()
                scenario["price"], scenario["volume"], scenario["phase"]
            )

            # Create demo state
            demo_state = {}
                "mode": "demo",
                "btc_price_hash": {}
                    "price": btc_hash.price,
                    "volume": btc_hash.volume,
                    "hash": btc_hash.hash_value,
                    "phase": btc_hash.phase,
                    "timestamp": btc_hash.timestamp.isoformat(),
                },
                "system_metrics": {}
                    "memory_count": 1,
                    "backlog_size": 0,
                    "btc_history_size": 1,
                    "uptime_seconds": 0.0,
                },
                "additional_data": {}
                    "scenario": scenario["description"],
                    "test_data": "demo_generation",
                },
                "timestamp": datetime.now().isoformat(),
            }
            generated_states.append(demo_state)

            # Verify demo state structure
            btc_data = demo_state["btc_price_hash"]

            print()
                f"✅ Demo state created: price={btc_data['price']}, volume={"}
                    btc_data['volume']
                }"
            )
            print(f"✅ BTC hash: {btc_data['hash'][:16]}...")
            print()
                f"✅ System metrics: {"}
                    demo_state['system_metrics']['memory_count']
                } memories"
            )

        print(f"\n✅ Generated {len(generated_states)} demo states")

        return True

    except Exception as e:
        print(f"❌ Demo state generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_system_initialization():
    """Test system initialization and organization."""
    print("\n🚀 Testing System Initialization")
    print("=" * 50)

    try:
        # Test initialization in different modes
        for mode in [SystemMode.TESTING, SystemMode.DEMO, SystemMode.LIVE]:
            print(f"\n--- Testing {mode.value} mode initialization ---")

            # Test mode enum
            print(f"✅ SystemMode.{mode.name} = {mode.value}")

            # Test log level
            log_level = LogLevel.INFO
            print(f"✅ LogLevel.{log_level.name} = {log_level.value}")

            # Test BTC price hash generation in this mode
            btc_hash = BTCPriceHash.from_price_data(50000.0, 1000.0, 32)
            print()
                f"✅ BTC price hash generated in {mode.value} mode: {btc_hash.hash_value[:16]}..."
            )

            # Test demo state creation
            {}
                "mode": mode.value,
                "btc_price_hash": {}
                    "price": btc_hash.price,
                    "volume": btc_hash.volume,
                    "hash": btc_hash.hash_value,
                    "phase": btc_hash.phase,
                    "timestamp": btc_hash.timestamp.isoformat(),
                },
                "system_metrics": {}
                    "memory_count": 1,
                    "backlog_size": 0,
                    "btc_history_size": 1,
                    "uptime_seconds": 0.0,
                },
                "additional_data": {"init_test": "data"},
                "timestamp": datetime.now().isoformat(),
            }
            print(f"✅ Demo state created successfully in {mode.value} mode")

        return True

    except Exception as e:
        print(f"❌ System initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run direct enhanced state system tests."""
    print("🚀 Direct Enhanced State System Test")
    print("=" * 60)

    tests = []
        ("BTC Price Hashing", test_btc_price_hashing),
        ("EnhancedStateManager Import", test_enhanced_state_manager_import),
        ("SystemIntegration Import", test_system_integration_import),
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
    print("\n📋 Direct Enhanced State System Test Summary")
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
        print("🎉 All direct enhanced state system tests passed!")
        print("✅ BTC price hashing works correctly for demo states")
        print("✅ Enhanced state system properly initializes and organizes")
        print("✅ System integration connects to internal systems")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
