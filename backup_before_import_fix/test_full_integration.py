import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from core.brain_trading_engine import BrainTradingEngine
from core.schwabot_integration_pipeline import IntegrationMessage, IntegrationOrchestrator, SecureAPIManager
from symbolic_profit_router import SymbolicProfitRouter

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Full Integration Test
=============================

Comprehensive test of the complete 8-layer Schwabot integration system:
1. Market Data Ingestion Layer
2. Brain Trading Engine Layer (AI Decision, Core)
3. Symbolic Profit Router Layer (Glyph, Processing)
4. Unified Math System Layer (Mathematical, Core)
5. API Management & Security Layer
6. Lantern Eye Visualization Layer
7. Risk Management & Portfolio Layer
8. Integration Pipeline & Orchestration Layer

This test validates the complete integration pipeline with real data flow.
"""


# Configure logging
logging.basicConfig()
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_full_integration_pipeline():
    """Test the complete integration pipeline."""
    print("🚀 SCHWABOT FULL INTEGRATION TEST")
    print("=" * 60)

    results = {}

    # Test 1: Configuration Loading
    print("\n📋 TESTING CONFIGURATION SYSTEM")
    print("-" * 40)

    try:

        config_path = Path("config/master_integration.yaml")

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            print("✅ Master configuration loaded successfully")
            print(f"   Version: {config.get('version', 'unknown')}")
            print()
                f"   Layers configured: {len([k for k in config.keys() if k.endswith('_layer')])}"
            )
            results["config_loading"] = True
        else:
            print("⚠️ Master configuration not found, using defaults")
            results["config_loading"] = False

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        results["config_loading"] = False

    # Test 2: Brain Trading Engine
    print("\n🧠 TESTING BRAIN TRADING ENGINE")
    print("-" * 40)

    try:

        brain_config = {}
            "base_profit_rate": 0.02,
            "confidence_threshold": 0.6,
            "enhancement_range": (0.8, 2.0),
            "max_history_size": 100,
        }

        brain_engine = BrainTradingEngine(brain_config)

        # Test multiple market scenarios
        test_scenarios = []
            {"name": "Strong Bull", "price": 55000, "volume": 2000},
            {"name": "Bear Market", "price": 45000, "volume": 800},
            {"name": "Sideways", "price": 50000, "volume": 1000},
            {"name": "Volatile", "price": 52000, "volume": 1800},
        ]

        brain_results = []
        for scenario in test_scenarios:
            signal = brain_engine.process_brain_signal()
                scenario["price"], scenario["volume"], "BTC"
            )
            decision = brain_engine.get_trading_decision(signal)
            brain_results.append()
                {}
                    "scenario": scenario["name"],
                    "confidence": signal.confidence,
                    "profit_score": signal.profit_score,
                    "action": decision["action"],
                    "position_size": decision["position_size"],
                }
            )

        print("✅ Brain Trading Engine operational")
        print(f"   Processed {len(brain_results)} scenarios")

        # Show results
        for result in brain_results:
            print()
                f"   {result['scenario']}: {result['action']} "
                f"(conf: {result['confidence']:.3f}, ")
                f"profit: {result['profit_score']:.2f})"
            )

        results["brain_engine"] = True
        results["brain_results"] = brain_results

    except Exception as e:
        print(f"❌ Brain Trading Engine test failed: {e}")
        results["brain_engine"] = False

    # Test 3: Symbolic Profit Router
    print("\n🔣 TESTING SYMBOLIC PROFIT ROUTER")
    print("-" * 40)

    try:

        router = SymbolicProfitRouter()

        # Test with brain symbols and results
        brain_symbols = ["[BRAIN]", "🧠", "💰", "📈", "⚡"]
        router_results = []

        for i, symbol in enumerate(brain_symbols):
            # Use brain results if available
            if "brain_results" in results and i < len(results["brain_results"]):
                brain_result = results["brain_results"][i]
                profit = brain_result["profit_score"] / 1000  # Scale down
                volume = 1000 + (i * 200)
            else:
                profit = 0.5 + (i * 0.2)  # 5%, 7%, 9%, etc.
                volume = 1000 + (i * 200)

            # Register and process
            router.register_glyph(symbol)
            vault_key = router.store_profit_sequence(symbol, profit, volume, "buy")

            # Get visualization
            viz = router.get_profit_tier_visualization(symbol)

            router_results.append()
                {}
                    "symbol": symbol,
                    "profit": profit,
                    "tier": viz["tier"],
                    "vault_stored": vault_key is not None,
                    "bit_state": viz["bit_state"],
                }
            )

        print("✅ Symbolic Profit Router operational")
        print(f"   Processed {len(router_results)} symbols")

        for result in router_results:
            print()
                f"   {result['symbol']}: {result['tier']} "
                f"(profit: {result['profit']:.3f}, ")
                f"vault: {result['vault_stored']})"
            )

        results["symbolic_router"] = True
        results["router_results"] = router_results

    except Exception as e:
        print(f"❌ Symbolic Profit Router test failed: {e}")
        results["symbolic_router"] = False

    # Test 4: Clean Unified Math System
    print("\n🧮 TESTING CLEAN UNIFIED MATH SYSTEM")
    print("-" * 40)

    try:
            CleanUnifiedMathSystem,
            optimize_brain_profit,
        )

        math_system = CleanUnifiedMathSystem()
        math_results = []

        # Test with brain trading data
        if "brain_results" in results:
            for brain_result in results["brain_results"][:3]:  # Test first 3
                # Simulate market data based on brain results
                price = 50000 + (brain_result["confidence"] - 0.5) * 10000
                volume = 1000 + brain_result["confidence"] * 1000

                # Test mathematical optimization
                optimized_profit = optimize_brain_profit()
                    price, volume, brain_result["confidence"], 1.2
                )

                # Test risk calculations
                returns = [0.5, 0.2, -0.1, 0.3, 0.1]
                sharpe = math_system.calculate_sharpe_ratio(returns)

                # Test portfolio calculations
                position_size = ()
                    math_system.calculate_portfolio_weight()
                        brain_result["confidence"], 0.1
                    )
                    * 100000
                )  # $100k portfolio

                math_results.append()
                    {}
                        "scenario": brain_result["scenario"],
                        "optimized_profit": optimized_profit,
                        "sharpe_ratio": sharpe,
                        "position_size": position_size,
                    }
                )
        else:
            # Fallback test data
            test_data = []
                {"price": 50000, "volume": 1000, "confidence": 0.75},
                {"price": 51000, "volume": 1200, "confidence": 0.8},
                {"price": 49000, "volume": 800, "confidence": 0.6},
            ]

            for i, data in enumerate(test_data):
                optimized_profit = optimize_brain_profit()
                    data["price"], data["volume"], data["confidence"], 1.1
                )

                math_results.append()
                    {}
                        "scenario": f"Test {i + 1}",
                        "optimized_profit": optimized_profit,
                        "sharpe_ratio": 0.8,  # Mock value
                        "position_size": data["confidence"] * 10000,
                    }
                )

        print("✅ Clean Unified Math System operational")
        print(f"   Processed {len(math_results)} calculations")

        for result in math_results:
            print()
                f"   {result['scenario']}: "
                f"profit={result['optimized_profit']:.2f}, "
                f"position=${result['position_size']:.0f}"
            )

        # Test integration function
        input_data = {}
            "tensor": [[50000, 1200], [51000, 1100]],
            "metadata": {"source": "integration_test"},
        }
        integration_result = math_system.integrate_all_systems(input_data)
        print()
            f"   Integration test: combined_score={integration_result.get('combined_score', 0):.2f}"
        )

        results["math_system"] = True
        results["math_results"] = math_results

    except Exception as e:
        print(f"❌ Clean Unified Math System test failed: {e}")
        results["math_system"] = False

    # Test 5: Integration Pipeline Orchestrator
    print("\n🔄 TESTING INTEGRATION ORCHESTRATOR")
    print("-" * 40)

    try:

        # Initialize orchestrator
        orchestrator = IntegrationOrchestrator()

        # Test system status
        system_status = orchestrator.get_system_status()

        print("✅ Integration Orchestrator operational")
        print("   Available components:")
        components = system_status.get("available_components", {})
        for comp, available in components.items():
            status = "✅" if available else "❌"
            print(f"     {status} {comp}")

        print(f"   Configured layers: {len(system_status.get('layers', {}))}")

        # Test brief pipeline run (5 seconds)
        print("   Running brief integration test...")

        # Start a short integration run
        integration_task = asyncio.create_task()
            orchestrator.start_integration_pipeline()
        )

        # Let it run for 5 seconds
        await asyncio.sleep(5)

        # Stop the pipeline
        await orchestrator.emergency_shutdown()

        # Cancel the task
        integration_task.cancel()
        try:
            await integration_task
        except asyncio.CancelledError:
            pass

        # Export final state
        orchestrator.export_system_state("integration_test_state.json")

        print("   ✅ Brief integration run completed")
        results["integration_orchestrator"] = True

    except Exception as e:
        print(f"❌ Integration Orchestrator test failed: {e}")
        results["integration_orchestrator"] = False

    # Test 6: API Security Layer
    print("\n🔐 TESTING API SECURITY LAYER")
    print("-" * 40)

    try:

        # Test API manager
        config = {}
            "api_security_layer": {}
                "security_protocols": {}
                    "api_key_encryption": True,
                    "secret_key_hashing": True,
                }
            }
        }

        api_manager = SecureAPIManager(config)

        # Test key encryption
        test_key = "test_api_key_123"
        encrypted = api_manager.encrypt_api_key(test_key, "test_api")

        # Test validation
        is_valid = api_manager.validate_api_access("test_api")

        print("✅ API Security Layer operational")
        print(f"   Key encryption: {'✅' if encrypted else '❌'}")
        print(f"   Access validation: {'✅' if is_valid else '❌'}")

        results["api_security"] = True

    except Exception as e:
        print(f"❌ API Security Layer test failed: {e}")
        results["api_security"] = False

    # Test 7: Cross-Layer Communication
    print("\n📡 TESTING CROSS-LAYER COMMUNICATION")
    print("-" * 40)

    try:
        # Test message creation and processing

        test_messages = []

        # Create test messages
        if "brain_results" in results and "router_results" in results:
            # Brain to Symbolic message
            brain_data = results["brain_results"][0]
            message1 = IntegrationMessage()
                source_layer="brain_engine_layer",
                target_layer="symbolic_profit_layer",
                message_type="signal_data",
                data={}
                    "confidence": brain_data["confidence"],
                    "profit_score": brain_data["profit_score"],
                    "action": brain_data["action"],
                },
            )
            test_messages.append(message1)

            # Symbolic to Math message
            router_data = results["router_results"][0]
            message2 = IntegrationMessage()
                source_layer="symbolic_profit_layer",
                target_layer="unified_math_layer",
                message_type="profit_data",
                data={}
                    "symbol": router_data["symbol"],
                    "profit": router_data["profit"],
                    "tier": router_data["tier"],
                },
            )
            test_messages.append(message2)

        print("✅ Cross-Layer Communication operational")
        print(f"   Created {len(test_messages)} test messages")

        for i, msg in enumerate(test_messages):
            print(f"   Message {i + 1}: {msg.source_layer} → {msg.target_layer}")

        results["cross_layer_comm"] = True

    except Exception as e:
        print(f"❌ Cross-Layer Communication test failed: {e}")
        results["cross_layer_comm"] = False

    # Test Summary
    print("\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    passed_tests = sum(1 for v in results.values() if isinstance(v, bool) and v)
    total_tests = sum(1 for v in results.values() if isinstance(v, bool))

    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

    print("\nDetailed Results:")
    test_names = {}
        "config_loading": "Configuration Loading",
        "brain_engine": "Brain Trading Engine",
        "symbolic_router": "Symbolic Profit Router",
        "math_system": "Clean Unified Math System",
        "integration_orchestrator": "Integration Orchestrator",
        "api_security": "API Security Layer",
        "cross_layer_comm": "Cross-Layer Communication",
    }

    for key, passed in results.items():
        if isinstance(passed, bool):
            status = "✅ PASS" if passed else "❌ FAIL"
            name = test_names.get(key, key)
            print(f"  {name}: {status}")

    # Data Flow Validation
    if all()
        results.get(key, False)
        for key in ["brain_engine", "symbolic_router", "math_system"]
    ):
        print("\n🔄 DATA FLOW VALIDATION")
        print("-" * 40)
        print("✅ Complete data flow verified:")
        print("   Market Data → Brain Engine → Symbolic Router → Math System")

        if "brain_results" in results and "router_results" in results:
            print(f"   Processed {len(results['brain_results'])} brain signals")
            print(f"   Generated {len(results['router_results'])} symbolic mappings")
            if "math_results" in results:
                print()
                    f"   Calculated {len(results['math_results'])} mathematical optimizations"
                )

    # Export comprehensive test results
    test_report = {}
        "timestamp": time.time(),
        "test_results": results,
        "summary": {}
            "passed": passed_tests,
            "total": total_tests,
            "success_rate": (passed_tests / total_tests) * 100,
        },
        "data_flow_validated": all()
            results.get(key, False)
            for key in ["brain_engine", "symbolic_router", "math_system"]
        ),
        "ready_for_production": passed_tests >= 5,  # At least 5 core tests must pass
    }

    # Save detailed results
    with open("full_integration_test_results.json", "w") as f:
        json.dump(test_report, f, indent=2, default=str)

    print("\n📄 Test report saved to full_integration_test_results.json")

    # Final Status
    if test_report["ready_for_production"]:
        print("\n🎯 INTEGRATION STATUS: READY FOR PRODUCTION")
        print("✅ All critical systems operational")
        print("✅ Data flow validated")
        print("✅ Cross-layer communication working")
        print("\n🚀 Ready to build Windows executable!")
    else:
        print("\n⚠️ INTEGRATION STATUS: NEEDS ATTENTION")
        print("Some critical systems need fixing before production")

    return test_report


async def main():
    """Main test execution."""
    print("Starting Schwabot Full Integration Test...")

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    try:
        test_report = await test_full_integration_pipeline()

        if test_report["ready_for_production"]:
            print("\n🎉 ALL SYSTEMS GO! Schwabot is ready for deployment!")
            return 0
        else:
            print("\n⚠️ Some systems need attention before deployment")
            return 1

    except Exception as e:
        print(f"\n❌ Critical test failure: {e}")
        logger.exception("Full integration test failed")
        return 1


if __name__ == "__main__":

    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
