#!/usr/bin/env python3
"""
🧪 SIMPLE 2-GRAM SYSTEM TEST
============================

Quick test to verify the 2-gram pattern detection system is working.
"""

import asyncio
import logging
import time
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_2gram_detector():
    """Test the 2-gram detector in isolation."""
    print("🧬 Testing 2-Gram Detector")
    print("=" * 40)

    try:
        # Import the detector
        from core.two_gram_detector import create_two_gram_detector

        # Create detector
        detector = create_two_gram_detector({)}
            "window_size": 50,
            "burst_threshold": 1.5,
            "similarity_threshold": 0.85,
            "t_cell_sensitivity": 0.3,
            "enable_fractal_memory": True
        })

        print("✅ 2-Gram detector created successfully")

        # Test pattern analysis
        test_sequences = []
            "UDUDUD",  # Volatility pattern
            "BEBEBE",  # Swap pattern
            "UUUUUU",  # Trend pattern
            "AAAAAA",  # Flatline pattern
            "EEEEEE"   # Entropy spike
        ]

        total_signals = 0
        for sequence in test_sequences:
            signals = await detector.analyze_sequence(sequence, {"test_context": True})
            total_signals += len(signals)
            print(f"  {sequence}: {len(signals)} signals detected")

        print(f"✅ Total signals detected: {total_signals}")

        # Test statistics
        stats = await detector.get_pattern_statistics()
        print(f"✅ Statistics: {stats.get('active_patterns', 0)} active patterns")

        # Test health check
        health = await detector.health_check()
        print(f"✅ Health status: {health.get('overall_status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Error testing 2-gram detector: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_strategy_router():
    """Test the strategy trigger router in isolation."""
    print("\n🎯 Testing Strategy Trigger Router")
    print("=" * 40)

    try:
        # Import the router
        from core.strategy_trigger_router import create_strategy_trigger_router

        # Create router
        router = create_strategy_trigger_router({)}
            "execution_mode": "demo"
        })

        print("✅ Strategy router created successfully")

        # Test market data processing
        market_data = {}
            "BTC": {}
                "price": 50000.0,
                "price_change_24h": 3.5,
                "volume": 1000000,
                "volume_change_24h": 25.0
            },
            "ETH": {}
                "price": 3000.0,
                "price_change_24h": -1.8,
                "volume": 800000,
                "volume_change_24h": -10.0
            }
        }

        triggers = await router.process_market_data(market_data)
        print(f"✅ Generated {len(triggers)} triggers")

        # Test router statistics
        stats = await router.get_router_statistics()
        print(f"✅ Router status: {stats.get('router_status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Error testing strategy router: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_portfolio_balancer():
    """Test the portfolio balancer in isolation."""
    print("\n⚖️ Testing Portfolio Balancer")
    print("=" * 40)

    try:
        # Import the balancer
        from core.algorithmic_portfolio_balancer import create_portfolio_balancer

        # Create balancer
        balancer = create_portfolio_balancer({)}
            "rebalancing_strategy": "phantom_adaptive",
            "rebalance_threshold": 0.5,
            "max_rebalance_frequency": 3600
        })

        print("✅ Portfolio balancer created successfully")

        # Test portfolio state update
        market_data = {}
            "BTC": {"price": 50000.0, "volume": 1000000},
            "ETH": {"price": 3000.0, "volume": 800000},
            "USDC": {"price": 1.0, "volume": 5000000}
        }

        await balancer.update_portfolio_state(market_data)
        print(f"✅ Portfolio value: ${balancer.portfolio_state.total_value:,.2f}")

        # Test performance metrics
        performance = await balancer.calculate_performance_metrics()
        print(f"✅ Performance metrics calculated: {len(performance)} metrics")

        return True

    except Exception as e:
        print(f"❌ Error testing portfolio balancer: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_btc_usdc_integration():
    """Test the BTC/USDC integration in isolation."""
    print("\n💱 Testing BTC/USDC Integration")
    print("=" * 40)

    try:
        # Import the integration
        from core.btc_usdc_trading_integration import create_btc_usdc_integration

        # Create integration
        integration = create_btc_usdc_integration({)}
            "btc_usdc_config": {}
                "order_size_btc": 0.01,
                "max_daily_trades": 10,
                "risk_limit": 0.2
            }
        })

        print("✅ BTC/USDC integration created successfully")

        # Test market data processing
        market_data = {}
            "BTC": {}
                "price": 50000.0,
                "volume": 2000000,
                "timestamp": time.time()
            }
        }

        decision = await integration.process_market_data(market_data)
        print(f"✅ Market data processed: {'Decision generated' if decision else 'No decision'}")

        # Test system status
        status = await integration.get_system_status()
        print(f"✅ System status: {status.get('status', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Error testing BTC/USDC integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_visual_execution_node():
    """Test the visual execution node in isolation."""
    print("\n🖥️ Testing Visual Execution Node")
    print("=" * 40)

    try:
        # Import the visual node
        from core.visual_execution_node import GUIMode, VisualizationTheme, create_visual_execution_node

        # Create visual node
        visual_node = create_visual_execution_node({)}
            "gui_mode": GUIMode.DEMO_MODE,
            "theme": VisualizationTheme.SCHWABOT_CLASSIC,
            "update_interval_ms": 1000
        })

        print("✅ Visual execution node created successfully")

        # Test statistics
        stats = await visual_node.get_visualization_statistics()
        print(f"✅ GUI available: {stats.get('gui_available', False)}")
        print(f"✅ Theme: {stats.get('theme', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Error testing visual execution node: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all simple tests."""
    print("🚀 Starting Simple 2-Gram System Tests")
    print("=" * 60)

    tests = []
        test_2gram_detector,
        test_strategy_router,
        test_portfolio_balancer,
        test_btc_usdc_integration,
        test_visual_execution_node
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! 2-Gram system is operational.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

    return passed == total

if __name__ == "__main__":
    asyncio.run(main()) 