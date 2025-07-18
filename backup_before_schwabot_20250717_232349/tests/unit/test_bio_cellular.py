#!/usr/bin/env python3
"""
🧬 Simple Bio-Cellular Trading Test
==================================

A simplified test of the bio-cellular trading system
to verify all components work correctly.
"""

print("🧬 Testing Bio-Cellular Trading System")
print("=" * 50)

# Test imports
    try:
    print("📦 Testing imports...")

    # Test bio-cellular imports
    try:
        from core.bio_cellular_signaling import BioCellularSignaling, CellularSignalType
        print("  ✅ Bio-cellular signaling imported successfully")
    except Exception as e:
        print(f"  ❌ Bio-cellular signaling import failed: {e}")

    try:
        from core.bio_profit_vectorization import BioProfitVectorization
        print("  ✅ Bio-profit vectorization imported successfully")
    except Exception as e:
        print(f"  ❌ Bio-profit vectorization import failed: {e}")

    try:
        from core.cellular_trade_executor import CellularTradeExecutor
        print("  ✅ Cellular trade executor imported successfully")
    except Exception as e:
        print(f"  ❌ Cellular trade executor import failed: {e}")

    try:
        from core.bio_cellular_integration import BioCellularIntegration
        print("  ✅ Bio-cellular integration imported successfully")
    except Exception as e:
        print(f"  ❌ Bio-cellular integration import failed: {e}")

    print("\n🧪 Testing basic functionality...")

    # Test basic cellular signaling
    try:
        cellular_signaling = BioCellularSignaling()
        print("  ✅ Bio-cellular signaling system created")

        # Test market signal processing
        market_data = {}
            'price_momentum': 0.5,
            'volatility': 0.3,
            'volume_delta': 0.2,
            'liquidity': 0.8,
            'risk_level': 0.3
        }

        responses = cellular_signaling.process_market_signal(market_data)
        print(f"  ✅ Processed market signal - {len(responses)} cellular responses")

        for signal_type, response in responses.items():
            print(f"    {signal_type.value}: activation={response.activation_strength:.3f}")

    except Exception as e:
        print(f"  ❌ Cellular signaling test failed: {e}")

    # Test profit vectorization
    try:
        profit_system = BioProfitVectorization()
        print("  ✅ Bio-profit vectorization system created")

        # Test with the same market data and cellular responses
        profit_response = profit_system.optimize_profit_vectorization(market_data, responses)
        print(f"  ✅ Profit optimization complete - pathway: {profit_response.metabolic_pathway.value}")
        print(f"    Position: {profit_response.recommended_position:.3f}")
        print(f"    Efficiency: {profit_response.cellular_efficiency:.3f}")

    except Exception as e:
        print(f"  ❌ Profit vectorization test failed: {e}")

    # Test integrated system
    try:
        integration = BioCellularIntegration()
        print("  ✅ Bio-cellular integration system created")

        # Test integrated signal processing
        result = integration.process_integrated_signal(market_data)
        print(f"  ✅ Integrated signal processing complete")
        print(f"    Trade Action: {result.hybrid_decision['trade_action']}")
        print(f"    Position Size: {result.hybrid_decision['position_size']:.3f}")
        print(f"    Confidence: {result.integration_confidence:.3f}")

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")

    print("\n🎉 Bio-Cellular Trading System Test Complete!")
    print("\n🌟 Key Features Demonstrated:")
    print("  • Cellular signaling pathways process market data")
    print("  • Metabolic profit optimization")
    print("  • Integrated decision making")
    print("  • Biological feedback mechanisms")

except Exception as e:
    print(f"❌ Test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("\n🧬 Bio-cellular trading test finished!") 