from utils.safe_print import safe_print, info, warn, error, success, debug
#!/usr/bin/env python3
"""
Simple Component Validation Script
=================================

This script validates that the new mathematical components can be imported
and basic functionality works.
"""

import sys
import traceback


def test_imports():
    """Test importing the mathematical components."""
    safe_print("Testing component imports...")

    try:
        from core.phantom_lag_model import PhantomLagModel
        safe_print("\\u2705 Phantom Lag Model imported successfully")

        # Test basic instantiation
        model = PhantomLagModel()
        safe_print("\\u2705 Phantom Lag Model instantiated successfully")

        # Test basic calculation
        penalty = model.calculate_phantom_lag_penalty(1000.0, 0.3, 70000.0)
        safe_print(f"\\u2705 Phantom Lag penalty calculation: {penalty:.6f}")

    except Exception as e:
        safe_print(f"\\u274c Phantom Lag Model import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.meta_layer_ghost_bridge import MetaLayerGhostBridge
        safe_print("\\u2705 Meta-Layer Ghost Bridge imported successfully")

        # Test basic instantiation
        bridge = MetaLayerGhostBridge()
        safe_print("\\u2705 Meta-Layer Ghost Bridge instantiated successfully")

        # Test basic functionality
        import time
        ghost_price = bridge.update_exchange_data("test", "BTC/USD", 50000.0, 1000.0, time.time())
        safe_print(f"\\u2705 Ghost price calculation: {ghost_price:.2f}")

    except Exception as e:
        safe_print(f"\\u274c Meta-Layer Ghost Bridge import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from core.fallback_logic_router import FallbackLogicRouter
        safe_print("\\u2705 Fallback Logic Router imported successfully")

        # Test basic instantiation
        router = FallbackLogicRouter()
        safe_print("\\u2705 Fallback Logic Router instantiated successfully")

    except Exception as e:
        safe_print(f"\\u274c Fallback Logic Router import failed: {e}")
        traceback.print_exc()
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of the components."""
    safe_print("\\nTesting basic functionality...")

    try:
        from core.phantom_lag_model import PhantomLagModel
        from core.meta_layer_ghost_bridge import MetaLayerGhostBridge

        # Test Phantom Lag Model
        model = PhantomLagModel()

        # Test various scenarios
        scenarios = [
            (1000.0, 0.3, 70000.0),  # Small opportunity, low entropy
            (5000.0, 0.7, 70000.0),  # Large opportunity, high entropy
            (0.0, 0.5, 70000.0),     # No opportunity
        ]

        for delta_price, entropy, max_price in scenarios:
            penalty = model.calculate_phantom_lag_penalty(delta_price, entropy, max_price)
            safe_print(f"  Delta: ${delta_price}, Entropy: {entropy}, Penalty: {penalty:.6f}")

        # Test Meta-Layer Ghost Bridge
        bridge = MetaLayerGhostBridge()

        # Test exchange data updates
        import time
        current_time = time.time()

        exchanges = [
            ("binance", 50000.0, 1000.0),
            ("coinbase", 50100.0, 1200.0),
            ("kraken", 49900.0, 800.0),
        ]

        for exchange, price, volume in exchanges:
            ghost_price = bridge.update_exchange_data(exchange, "BTC/USD", price, volume, current_time)
            safe_print(f"  {exchange}: ${price} -> Ghost: ${ghost_price:.2f}")

        # Test meta vector
        meta_vector = bridge.get_meta_vector("BTC/USD")
        safe_print(f"  Meta vector: {meta_vector:.6f}")

        return True

    except Exception as e:
        safe_print(f"\\u274c Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    safe_print("\\u1f9e0 Schwabot Mathematical Components Validation")
    safe_print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()

        if functionality_ok:
            safe_print("\\n\\u2705 All components validated successfully!")
            return 0
        else:
            safe_print("\\n\\u274c Basic functionality tests failed")
            return 1
    else:
        safe_print("\\n\\u274c Import tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""