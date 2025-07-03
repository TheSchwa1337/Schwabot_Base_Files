from pathlib import Path
from typing import Dict, List, Any
import asyncio
import json
import logging
import sys
import time

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schwabot Brain Integration Test
=============================

Comprehensive test of brain trading functionality with working implementations.
This replaces placeholders with functional brain trading algorithms.
"""


# Add core directory to path for robust imports
core_dir = Path(__file__).resolve().parent.parent.parent.parent / "core"
sys.path.insert(0, str(core_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_confidence():-> float:
    """
    Computes trading confidence based on profit score and volatility.
    """
    return min(1.0, (profit_score / (volatility + 1e-5)) * 0.01)


def test_brain_trading_engine():
# ... existing code ...
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

        # Export data
        engine.export_signals("test_brain_signals.json")
        print("   📄 Data exported to test_brain_signals.json")

        return True, engine, results

    except Exception as e:
        print(f"❌ Brain Trading Engine test failed: {e}")
        return False, None, None


def test_confidence_calculation():
    """Test the injected confidence calculation logic."""
    print("\n🧠 TESTING CONFIDENCE CALCULATION")
    print("=" * 50)
    try:
        # Mock data validation
        mock_profit = 1480
        mock_volatility = 225
        confidence = compute_confidence(mock_profit, mock_volatility)
        print(f"  Mock Data -> Profit: {mock_profit}, Volatility: {mock_volatility}")
        print(f"  Computed Confidence: {confidence:.4f}")
        assert 0 < confidence < 1.0, "Confidence score should be between 0 and 1"

        # BONUS: Real Data Injection Example
        print("\n  BONUS: Real Data Injection Example")
        # These would be live calls in the actual pipeline
        # from some_api import fetch_btc_price
        # btc_price = fetch_btc_price()  # From CCXT
        btc_volatility = 225           # Sample from API or internal tick history
        profit_estimate = 1480         # From Ferris loop exit logic

        live_confidence = compute_confidence(profit_estimate, btc_volatility)
        print(f"  Live Data -> Profit Estimate: {profit_estimate}, Volatility: {btc_volatility}")
        print(f"  🧠 Trading Confidence: {live_confidence:.4f}")
        assert live_confidence == confidence, "Confidence should be identical with identical inputs"

        print("\n✅ Confidence calculation test passed.")
        return True
    except Exception as e:
        print(f"❌ Confidence calculation test failed: {e}")
        return False


def test_mathematical_functions():
# ... existing code ...
    if not test_mathematical_functions():
        print("\n🚨 MATHEMATICAL FUNCTIONS TEST FAILED - ABORTING")
        return

    # Run confidence calculation test
    if not test_confidence_calculation():
        print("\n🚨 CONFIDENCE CALCULATION TEST FAILED - ABORTING")
        return

    # Run symbol processing test
    if not test_symbol_processing():
        print("\n🚨 SYMBOL PROCESSING TEST FAILED - ABORTING")
# ... existing code ... 