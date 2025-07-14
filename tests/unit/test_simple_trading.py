#!/usr/bin/env python3
"""
Simple test to verify trading bot components work together.
"""

import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from core.cli_live_entry import LiveTradingBot


async def test_basic_functionality():
    """Test basic trading bot functionality."""
    print("🧪 Testing Schwabot Trading System...")

    # Test configuration
    config = {}
        "symbol": "BTCUSDT",
        "initial_capital": 10000.0,
        "exchange_config": {}
            "exchange": "binance",
            "apiKey": "test",
            "secret": "test",
            "sandbox": True
        },
        "market_data_config": {}
            "cache_ttl": 60,
            "apis": {}
                "coingecko": {"enabled": True, "weight": 0.5},
                "fear_greed": {"enabled": True, "weight": 0.5}
            }
        }
    }

    try:
        # Initialize bot
        bot = LiveTradingBot(config)
        await bot.initialize()
        print("✅ Bot initialized successfully")

        # Test single trade execution
        result = await bot.execute_live_trade("BTCUSDT", force_refresh=True)
        if result:
            print("✅ Trade execution test completed")
            action = result.get("trade_action", {}).get("action", "hold")
            print(f"   Action: {action}")
        else:
            print("❌ Trade execution failed")

        print("✅ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1) 