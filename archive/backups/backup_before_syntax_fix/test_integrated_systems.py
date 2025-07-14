import asyncio
import logging
import os
import sys
import time

from core.balance_loader import get_balance_statistics, update_load_metrics
from core.multi_bit_btc_processor import MultiBitBTCProcessor
from core.tick_management_system import get_tick_statistics, run_tick_cycle

#!/usr/bin/env python3
"""
Integrated Systems Test
======================

Comprehensive test script demonstrating the integration of:
- Tick Management System
- Balance Loader
- Ghost Trigger Manager
- BTC Processor

This test shows how all systems work together to create a cohesive
ALIF/ALEPH coordination system with balance loading and ghost trigger management.
"""


# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_tick_management_system():
    """Test the tick management system."""
    print("\n🧠 Testing Tick Management System")
    print("=" * 50)

    try:
            get_tick_manager,
            run_tick_cycle,
            get_tick_statistics,
            register_tick_callback,
            TickContext,
        )

        # Get the tick manager
        tick_manager = get_tick_manager()
        print(f"✅ Tick manager initialized: {tick_manager.tick_count} ticks")

        # Register a callback to log tick events
        def tick_callback(tick_context: TickContext):
            print()
                f"   Tick {tick_context.tick_id}: {tick_context.compression_mode.value} "
                f"(entropy: {tick_context.entropy:.3f}, echo: {tick_context.echo_strength:.3f})"
            )

        register_tick_callback(tick_callback)

        # Run several tick cycles
        print("\n🔄 Running tick cycles...")
        for i in range(5):
            tick_context = run_tick_cycle()
            if tick_context:
                time.sleep(0.1)  # Small delay between ticks

        # Get statistics
        stats = get_tick_statistics()
        print("\n📊 Tick Statistics:")
        print(f"   Total ticks: {stats['total_ticks']}")
        print(f"   Valid ticks: {stats['valid_ticks']}")
        print(f"   Hollow ticks: {stats['hollow_ticks']}")
        print(f"   Compressed ticks: {stats['compressed_ticks']}")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Current mode: {stats['current_compression_mode']}")

        return True

    except Exception as e:
        print(f"❌ Tick management test failed: {e}")
        return False


def test_balance_loader():
    """Test the balance loader system."""
    print("\n⚖️ Testing Balance Loader System")
    print("=" * 50)

    try:
            get_balance_loader,
            update_load_metrics,
            get_balance_statistics,
            get_optimal_route,
            monitor_float_decay,
        )

        # Get the balance loader
        get_balance_loader()
        print("✅ Balance loader initialized")

        # Test load metric updates
        print("\n🔄 Testing load metric updates...")

        # Simulate different load scenarios
        scenarios = []
            (15.0, 10.0, 0.7, 0.3, 0.0),  # ALIF heavy
            (8.0, 12.0, 0.4, 0.6, 0.0),  # ALEPH heavy
            (12.0, 11.0, 0.5, 0.5, 0.0),  # Balanced
            (18.0, 16.0, 0.8, 0.2, 0.5),  # High load with decay
        ]
        for i, ()
            alif_load,
            aleph_load,
            gpu_entropy,
            cpu_entropy,
            float_decay,
        ) in enumerate(scenarios):
            metrics = update_load_metrics()
                alif_load, aleph_load, gpu_entropy, cpu_entropy, float_decay
            )
            optimal_route = get_optimal_route(alif_load, aleph_load)

            print()
                f"   Scenario {i + 1}: ALIF={alif_load:.1f}, ALEPH={aleph_load:.1f} → {optimal_route}"
            )
            print(f"      Balance needed: {metrics.balance_needed}")
            print(f"      Compression ratio: {metrics.compression_ratio:.3f}")

        # Test float decay monitoring
        print("\n⏱️ Testing float decay monitoring...")
        decay_detected = monitor_float_decay(1.0, 1.5)  # 50ms decay
        print(f"   Float decay detected: {decay_detected}")

        # Get statistics
        stats = get_balance_statistics()
        print("\n📊 Balance Statistics:")
        print(f"   Current mode: {stats['current_mode']}")
        print(f"   Total adjustments: {stats['total_adjustments']}")
        print(f"   Success rate: {stats['success_rate']:.2%}")
        print(f"   Balance needed: {stats['balance_needed']}")
        print(f"   Compression ratio: {stats['compression_ratio']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Balance loader test failed: {e}")
        return False


def test_ghost_trigger_manager():
    """Test the ghost trigger manager."""
    print("\n👻 Testing Ghost Trigger Manager")
    print("=" * 50)

    try:
            get_ghost_trigger_manager,
            create_ghost_trigger,
            create_fallback_trigger,
            add_profit_vector,
            get_trigger_performance,
            get_profit_mapping_suggestions,
            TriggerType,
            AnchorStatus,
        )

        # Get the ghost trigger manager
        get_ghost_trigger_manager()
        print("✅ Ghost trigger manager initialized")

        # Create different types of triggers
        print("\n🔄 Creating ghost triggers...")

        # Anchored trigger (real, block)
        anchored_trigger = create_ghost_trigger()
            trigger_hash="anchored_1234567890abcdef","
            origin="btc_block_processor",
            anchor_status=AnchorStatus.ANCHORED,
            confidence=0.85,
            trigger_type=TriggerType.REAL_BLOCK,
            entropy_score=0.6,
            echo_strength=0.8,
            drift_score=0.02,
        )
        print(f"   Created anchored trigger: {anchored_trigger.trigger_hash[:16]}...")

        # Unanchored trigger (simulated)
        unanchored_trigger = create_ghost_trigger()
            trigger_hash="unanchored_abcdef1234567890",
            origin="alif_entropy_core",
            anchor_status=AnchorStatus.UNANCHORED,
            confidence=0.65,
            trigger_type=TriggerType.ALIF_ENTROPY,
            entropy_score=0.8,
            echo_strength=0.4,
            drift_score=0.15,
        )
        print()
            f"   Created unanchored trigger: {unanchored_trigger.trigger_hash[:16]}..."
        )

        # Create a fallback trigger
        fallback_trigger = create_fallback_trigger(unanchored_trigger, "4bit")
        print(f"   Created fallback trigger: {fallback_trigger.trigger_hash[:16]}...")

        # Add profit vectors
        print("\n💰 Adding profit vectors...")
        add_profit_vector()
            trigger_hash=anchored_trigger.trigger_hash,
            entry_price=65000.0,
            exit_price=65500.0,
            volume=1.0,
            confidence=0.85,
        )

        add_profit_vector()
            trigger_hash=unanchored_trigger.trigger_hash,
            entry_price=65000.0,
            exit_price=64800.0,
            volume=0.5,
            confidence=0.65,
        )

        add_profit_vector()
            trigger_hash=fallback_trigger.trigger_hash,
            entry_price=65000.0,
            exit_price=65200.0,
            volume=0.3,
            confidence=0.5,
        )

        # Get performance statistics
        performance = get_trigger_performance()
        print("\n📊 Trigger Performance:")
        print(f"   Total triggers: {performance['total_triggers']}")
        print(f"   Anchored triggers: {performance['anchored_triggers']}")
        print(f"   Unanchored triggers: {performance['unanchored_triggers']}")
        print(f"   Fallback triggers: {performance['fallback_triggers']}")
        print(f"   Total profit: {performance['total_profit']:.4f}")
        print(f"   Anchored profit: {performance['anchored_profit']:.4f}")
        print(f"   Unanchored profit: {performance['unanchored_profit']:.4f}")
        print(f"   Fallback profit: {performance['fallback_profit']:.4f}")

        # Get profit mapping suggestions
        suggestions = get_profit_mapping_suggestions()
        print("\n💡 Profit Mapping Suggestions:")
        print(f"   Prefer anchored: {suggestions['prefer_anchored']}")
        print(f"   Prefer fallback: {suggestions['prefer_fallback']}")
        print(f"   Compression needed: {suggestions['compression_needed']}")
        print(f"   Recommended actions: {suggestions['recommended_actions']}")

        return True

    except Exception as e:
        print(f"❌ Ghost trigger manager test failed: {e}")
        return False


async def test_btc_processor_integration():
    """Test BTC processor integration."""
    print("\n₿ Testing BTC Processor Integration")
    print("=" * 50)

    try:

        # Initialize BTC processor
        processor = MultiBitBTCProcessor()
        print("✅ BTC processor initialized")

        # Test with simulated data
        print("\n🔄 Testing BTC data processing...")

        test_data = []
            (65000.0, 100.0),
            (65100.0, 95.0),
            (65200.0, 110.0),
            (65150.0, 105.0),
            (65300.0, 120.0),
        ]
        for i, (price, volume) in enumerate(test_data):
            is_allowed, profit_vector = await processor.process_btc_data(price, volume)

            if is_allowed and profit_vector:
                print(f"   Tick {i + 1}: Price=${price:.0f}, Volume={volume:.0f}")
                print(f"      Class: {profit_vector['class']}")
                print(f"      Risk: {profit_vector['risk']:.3f}")
                print(f"      Coherence: {profit_vector['triplet_coherence']:.3f}")
                print()
                    f"      U_r Score: {profit_vector['asrl_unified_reflex_score']:.3f}"
                )
            else:
                print(f"   Tick {i + 1}: Processing failed or not allowed")

            await asyncio.sleep(0.1)

        return True

    except Exception as e:
        print(f"❌ BTC processor integration test failed: {e}")
        return False


def test_integrated_workflow():
    """Test the complete integrated workflow."""
    print("\n🔄 Testing Integrated Workflow")
    print("=" * 50)

    try:
            create_ghost_trigger,
            add_profit_vector,
            AnchorStatus,
            TriggerType,
        )

        print("🔄 Running integrated workflow simulation...")

        # Simulate a complete trading cycle
        for cycle in range(3):
            print(f"\n   Cycle {cycle + 1}:")

            # 1. Run tick cycle
            tick_context = run_tick_cycle()
            if tick_context:
                print()
                    f"      Tick {tick_context.tick_id}: {tick_context.compression_mode.value}"
                )

                # 2. Update balance metrics
                metrics = update_load_metrics()
                    tick_context.alif_score,
                    tick_context.aleph_score,
                    tick_context.entropy * 0.7,  # GPU entropy
                    tick_context.entropy * 0.3,  # CPU entropy
                    tick_context.drift_score,
                )
                print(f"      Balance needed: {metrics.balance_needed}")

                # 3. Create ghost trigger based on tick
                if tick_context.validated:
                    trigger = create_ghost_trigger()
                        trigger_hash=f"integrated_{tick_context.tick_id}_{int(time.time())}",
                        origin="integrated_workflow",
                        anchor_status=AnchorStatus.ANCHORED
                        if tick_context.echo_strength > 0.6
                        else AnchorStatus.UNANCHORED,
                        confidence=tick_context.echo_strength,
                        trigger_type=TriggerType.REAL_BLOCK
                        if tick_context.echo_strength > 0.6
                        else TriggerType.ALIF_ENTROPY,
                        entropy_score=tick_context.entropy,
                        echo_strength=tick_context.echo_strength,
                        drift_score=tick_context.drift_score,
                    )
                    print()
                        f"      Created trigger: {trigger.trigger_hash[:16]}... ({trigger.anchor_status.value})"
                    )

                    # 4. Simulate profit (if conditions are, good)
                    if tick_context.echo_strength > 0.6 and tick_context.entropy < 0.8:
                        entry_price = 65000.0 + (cycle * 100)
                        exit_price = entry_price + 200  # Simulate profit
                        add_profit_vector()
                            trigger.trigger_hash,
                            entry_price,
                            exit_price,
                            1.0,
                            tick_context.echo_strength,
                        )
                        print()
                            f"      Added profit vector: +{((exit_price - entry_price) / entry_price):.2%}"
                        )

            time.sleep(0.2)  # Simulate processing time

        # Get final statistics
        print("\n📊 Final Statistics:")

        tick_stats = get_tick_statistics()
        print(f"   Tick success rate: {tick_stats['success_rate']:.2%}")

        balance_stats = get_balance_statistics()
        print(f"   Balance mode: {balance_stats['current_mode']}")

        return True

    except Exception as e:
        print(f"❌ Integrated workflow test failed: {e}")
        return False


async def main():
    """Run all integrated system tests."""
    print("🚀 Starting Integrated Systems Test Suite")
    print("=" * 60)

    tests = []
        ("Tick Management System", test_tick_management_system),
        ("Balance Loader", test_balance_loader),
        ("Ghost Trigger Manager", test_ghost_trigger_manager),
        ("BTC Processor Integration", test_btc_processor_integration),
        ("Integrated Workflow", test_integrated_workflow),
    ]
    passed = 0
    total = len(tests)

    for name, test_func in tests:
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {name} test passed")
            else:
                print(f"❌ {name} test failed")
        except Exception as e:
            print(f"❌ {name} test failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integrated systems tests passed!")
        print("\n✅ Integration Summary:")
        print()
            "   - Tick Management System: Coordinating ALIF/ALEPH with compression modes"
        )
        print("   - Balance Loader: Managing GPU/CPU load balancing and float decay")
        print("   - Ghost Trigger Manager: Handling anchored vs unanchored triggers")
        print("   - BTC Processor: Processing real-time market data")
        print("   - Integrated Workflow: All systems working together seamlessly")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
