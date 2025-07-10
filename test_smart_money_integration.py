import logging
from typing import Any, Dict

import numpy as np

from core.enhanced_strategy_framework import EnhancedStrategyFramework
from core.smart_money_integration import SmartMoneyIntegrationFramework, enhance_wall_street_with_smart_money

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Money Integration Test Suite.

Test suite demonstrating the integration of Wall Street strategies
with smart money metrics, showcasing institutional-grade analysis.

Features Tested:
- Smart Money Metrics (OBV, VWAP, CVD, DPI, etc.)
- Wall Street Strategy Enhancement
- Correlation Analysis
- Risk-Adjusted Position Sizing
- Execution Optimization
"""

logger = logging.getLogger(__name__)


def safe_print():-> None:
    """Safe print function that handles Unicode characters."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


class SmartMoneyIntegrationTester:
    """Test suite for smart money integration with Wall Street strategies."""

    def __init__():-> None:
        """Initialize test suite."""
        self.test_results = {}
        self.overall_success_rate = 0.0

    def test_smart_money_metrics_calculation():-> Dict[str, Any]:
        """Test smart money metrics calculation."""
        safe_print("💰 Testing Smart Money Metrics Calculation...")

        try:
            # Initialize framework
            smart_money = SmartMoneyIntegrationFramework()

            # Test data
            asset = "BTC/USDT"
            price_data = []
                50000,
                50100,
                50050,
                50200,
                50150,
                50300,
                50250,
                50400,
                50350,
                50500,
            ]
            volume_data = [1000, 1200, 800, 1500, 900, 1800, 1100, 2000, 1300, 2500]

            # Analyze smart money metrics
            sm_signals = smart_money.analyze_smart_money_metrics()
                asset=asset, price_data=price_data, volume_data=volume_data
            )

            success = len(sm_signals) > 0

            return {}
                "component": "Smart Money Metrics Calculation",
                "success": success,
                "details": {}
                    "signals_generated": len(sm_signals),
                    "metrics_types": [signal.metric.value for signal in sm_signals],
                    "avg_institutional_confidence": ()
                        np.mean([s.institutional_confidence for s in sm_signals])
                        if sm_signals
                        else 0.0
                    ),
                    "whale_activity_detected": any()
                        s.whale_activity for s in sm_signals
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Smart money metrics test failed: {e}")
            return {}
                "component": "Smart Money Metrics Calculation",
                "success": False,
                "error": str(e),
            }

    def test_wall_street_smart_money_integration():-> Dict[str, Any]:
        """Test integration between Wall Street strategies and smart money metrics."""
        safe_print("🏛️ Testing Wall Street-Smart Money Integration...")

        try:
            # Initialize framework
            enhanced_framework = EnhancedStrategyFramework()

            # Test data
            asset = "BTC/USDT"
            price_data = []
                50000,
                50100,
                50050,
                50200,
                50150,
                50300,
                50250,
                50400,
                50350,
                50500,
            ]
            volume_data = [1000, 1200, 800, 1500, 900, 1800, 1100, 2000, 1300, 2500]

            # Order book simulation
            order_book_data = {}
                "bids": [[49950, 100], [49900, 200], [49850, 150]],
                "asks": [[50050, 120], [50100, 180], [50150, 90]],
            }

            # Enhance with smart money
            integration_result = enhance_wall_street_with_smart_money()
                enhanced_framework=enhanced_framework,
                asset=asset,
                price_data=price_data,
                volume_data=volume_data,
                order_book_data=order_book_data,
            )

            success = integration_result.get("success", False)

            return {}
                "component": "Wall Street-Smart Money Integration",
                "success": success,
                "details": {}
                    "integration_result": integration_result,
                    "integration_quality": integration_result.get()
                        "integration_quality", 0.0
                    ),
                    "signals_correlated": integration_result.get()
                        "integrated_signals", 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Wall Street smart money integration test failed: {e}")
            return {}
                "component": "Wall Street-Smart Money Integration",
                "success": False,
                "error": str(e),
            }

    def test_order_flow_analysis():-> Dict[str, Any]:
        """Test order flow imbalance analysis."""
        safe_print("📊 Testing Order Flow Analysis...")

        try:
            # Initialize framework
            smart_money = SmartMoneyIntegrationFramework()

            # Simulate order book with imbalance
            order_book_data = {}
                "bids": [[49950, 500], [49900, 300], [49850, 200]],  # Strong bid side
                "asks": [[50050, 100], [50100, 80], [50150, 60]],  # Weak ask side
            }

            # Calculate order flow imbalance
            ofi_signal = smart_money._calculate_order_flow_imbalance()
                "BTC/USDT", order_book_data
            )

            success = ofi_signal is not None and ofi_signal.order_flow_imbalance > 0

            return {}
                "component": "Order Flow Analysis",
                "success": success,
                "details": {}
                    "signal_generated": ofi_signal is not None,
                    "order_flow_imbalance": ofi_signal.order_flow_imbalance
                    if ofi_signal
                    else 0.0,
                    "bid_pressure_detected": ofi_signal.volume_signature.get()
                        "bid_pressure", False
                    )
                    if ofi_signal
                    else False,
                    "execution_urgency": ofi_signal.execution_urgency
                    if ofi_signal
                    else "unknown",
                },
            }

        except Exception as e:
            logger.error(f"Order flow analysis test failed: {e}")
            return {}
                "component": "Order Flow Analysis",
                "success": False,
                "error": str(e),
            }

    def test_whale_detection():-> Dict[str, Any]:
        """Test whale activity detection."""
        safe_print("🐋 Testing Whale Activity Detection...")

        try:
            # Initialize framework
            smart_money = SmartMoneyIntegrationFramework()

            # Simulate whale activity with large volume spike
            normal_volume = [1000, 1100, 900, 1200, 1050]
            whale_volume = normal_volume + [5000]  # Large spike

            price_data = [50000, 50100, 50050, 50200, 50150, 50500]  # Price impact

            # Detect whale activity
            whale_signal = smart_money._detect_whale_activity()
                "BTC/USDT", whale_volume, price_data
            )

            success = whale_signal is not None and whale_signal.whale_activity

            return {}
                "component": "Whale Activity Detection",
                "success": success,
                "details": {}
                    "whale_detected": whale_signal.whale_activity
                    if whale_signal
                    else False,
                    "volume_spike": whale_signal.volume_signature.get()
                        "volume_spike", 0.0
                    )
                    if whale_signal
                    else 0.0,
                    "volume_usd": whale_signal.volume_signature.get("volume_usd", 0.0)
                    if whale_signal
                    else 0.0,
                    "execution_urgency": whale_signal.execution_urgency
                    if whale_signal
                    else "unknown",
                },
            }

        except Exception as e:
            logger.error(f"Whale detection test failed: {e}")
            return {}
                "component": "Whale Activity Detection",
                "success": False,
                "error": str(e),
            }

    def test_vwap_analysis():-> Dict[str, Any]:
        """Test VWAP-based smart money analysis."""
        safe_print("📈 Testing VWAP Analysis...")

        try:
            # Initialize framework
            smart_money = SmartMoneyIntegrationFramework()

            # Test data with clear VWAP deviation
            price_data = [50000, 50100, 50200, 50300, 50400]  # Trending up
            volume_data = [1000, 1200, 1500, 1800, 2000]  # Increasing volume

            # Calculate VWAP signal
            vwap_signal = smart_money._calculate_vwap_signal()
                "BTC/USDT", price_data, volume_data
            )

            success = vwap_signal is not None and vwap_signal.signal_strength > 0

            return {}
                "component": "VWAP Analysis",
                "success": success,
                "details": {}
                    "signal_generated": vwap_signal is not None,
                    "vwap_value": vwap_signal.volume_signature.get("vwap", 0.0)
                    if vwap_signal
                    else 0.0,
                    "price_deviation": vwap_signal.volume_signature.get()
                        "deviation", 0.0
                    )
                    if vwap_signal
                    else 0.0,
                    "institutional_confidence": vwap_signal.institutional_confidence
                    if vwap_signal
                    else 0.0,
                },
            }

        except Exception as e:
            logger.error(f"VWAP analysis test failed: {e}")
            return {"component": "VWAP Analysis", "success": False, "error": str(e)}

    def test_dark_pool_detection():-> Dict[str, Any]:
        """Test dark pool activity detection."""
        safe_print("🌑 Testing Dark Pool Detection...")

        try:
            # Initialize framework
            smart_money = SmartMoneyIntegrationFramework()

            # Simulate dark pool activity with high volume variance
            high_variance_volume = [1000, 500, 2000, 800, 1800, 600, 2200]

            # Calculate dark pool index
            dpi_signal = smart_money._calculate_dark_pool_index()
                "BTC/USDT", high_variance_volume
            )

            success = dpi_signal is not None and dpi_signal.dark_pool_activity > 0

            return {}
                "component": "Dark Pool Detection",
                "success": success,
                "details": {}
                    "signal_generated": dpi_signal is not None,
                    "dark_pool_estimate": dpi_signal.volume_signature.get()
                        "dark_pool_estimate", 0.0
                    )
                    if dpi_signal
                    else 0.0,
                    "dark_pool_activity": dpi_signal.dark_pool_activity
                    if dpi_signal
                    else 0.0,
                    "institutional_confidence": dpi_signal.institutional_confidence
                    if dpi_signal
                    else 0.0,
                },
            }

        except Exception as e:
            logger.error(f"Dark pool detection test failed: {e}")
            return {}
                "component": "Dark Pool Detection",
                "success": False,
                "error": str(e),
            }

    def run_all_tests():-> Dict[str, Any]:
        """Run all smart money integration tests."""
        safe_print("💎 Smart Money Integration Test Suite")
        safe_print("=" * 60)

        test_methods = []
            self.test_smart_money_metrics_calculation,
            self.test_wall_street_smart_money_integration,
            self.test_order_flow_analysis,
            self.test_whale_detection,
            self.test_vwap_analysis,
            self.test_dark_pool_detection,
        ]

        results = []

        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)

                status = "✅" if result["success"] else "❌"
                component = result["component"]
                safe_print()
                    f"  {status} {component}: {'PASS' if result['success'] else 'FAIL'}"
                )

                if not result["success"] and "error" in result:
                    safe_print(f"    Error: {result['error']}")
                elif result["success"] and "details" in result:
                    # Show key metrics for successful tests
                    details = result["details"]
                    if "signals_generated" in details:
                        safe_print()
                            f"    Signals Generated: {details['signals_generated']}"
                        )
                    if "whale_detected" in details:
                        safe_print(f"    Whale Activity: {details['whale_detected']}")
                    if "integration_quality" in details:
                        safe_print()
                            f"    Integration Quality: {details['integration_quality']:.2f}"
                        )

            except Exception as e:
                safe_print(f"  ❌ {test_method.__name__}: CRITICAL FAILURE")
                safe_print(f"    Error: {e}")
                results.append()
                    {}
                        "component": test_method.__name__,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Calculate overall success rate
        successful_tests = sum(1 for r in results if r["success"])
        total_tests = len(results)
        success_rate = ()
            (successful_tests / total_tests) * 100 if total_tests > 0 else 0.0
        )

        safe_print("\n" + "=" * 60)
        safe_print("📊 SMART MONEY INTEGRATION TEST RESULTS")
        safe_print("=" * 60)
        safe_print(f"🎯 Overall Success Rate: {success_rate:.1f}%")
        safe_print(f"✅ Tests Passed: {successful_tests}/{total_tests}")

        if success_rate >= 90:
            safe_print("🎉 Excellent! Smart money integration is working perfectly!")
        elif success_rate >= 70:
            safe_print("✅ Good! Most smart money features are working correctly.")
        elif success_rate >= 50:
            safe_print()
                "⚠️  Partial success. Some smart money components need attention."
            )
        else:
            safe_print("❌ Multiple smart money components need debugging.")

        return {}
            "overall_success_rate": success_rate,
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "detailed_results": results,
        }


def main():-> Dict[str, Any]:
    """Run smart money integration tests."""
    tester = SmartMoneyIntegrationTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    results = main()
    safe_print()
        f"\nSmart Money Testing completed with {results['overall_success_rate']:.1f}% success rate"
    )
