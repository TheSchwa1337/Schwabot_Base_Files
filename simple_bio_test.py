#!/usr/bin/env python3
"""
🧬 Minimal Bio-Cellular Test
===========================

A minimal test to isolate import issues.
"""

print("🧬 Starting minimal bio-cellular test...")

# Test each bio-cellular file individually
print("Testing bio_cellular_signaling...")
    try:
    from core.bio_cellular_signaling import BioCellularSignaling
    print("✅ bio_cellular_signaling imported successfully")
    except Exception as e:
    print(f"❌ bio_cellular_signaling failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting bio_profit_vectorization...")
    try:
    from core.bio_profit_vectorization import BioProfitVectorization
    print("✅ bio_profit_vectorization imported successfully")
    except Exception as e:
    print(f"❌ bio_profit_vectorization failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting cellular_trade_executor...")
    try:
    from core.cellular_trade_executor import CellularTradeExecutor
    print("✅ cellular_trade_executor imported successfully")
    except Exception as e:
    print(f"❌ cellular_trade_executor failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting bio_cellular_integration...")
    try:
    from core.bio_cellular_integration import BioCellularIntegration
    print("✅ bio_cellular_integration imported successfully")
    except Exception as e:
    print(f"❌ bio_cellular_integration failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🧬 Minimal test complete!") 