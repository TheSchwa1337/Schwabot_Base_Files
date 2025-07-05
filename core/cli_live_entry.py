# !/usr/bin/env python3
"""
CLI Live Entry - Live Trading Bot Command Interface
Connects to real APIs and executes actual trades
"""
from core.unified_market_data_pipeline import create_unified_pipeline
from core.soulprint_registry import SoulprintRegistry
from core.clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline
from core.ccxt_trading_executor import CCXTTradingExecutor
import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class LiveTradingBot:
    """Live trading bot that executes real trades through API connections."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_pipeline = None
        self.ccxt_executor = None
        self.registry = None
        self.is_running = False

    async def initialize(self):
        """Initialize all trading components."""
        # Initialize trading pipeline with live configuration
        self.trading_pipeline = create_trading_pipeline(
            symbol=self.config.get("symbol", "BTCUSDT"),
            initial_capital=self.config.get("initial_capital", 10000.0),
            safe_mode=self.config.get("safe_mode", False)
        )

        # Configure pipeline with market data settings
        if "market_data_config" in self.config:
            self.trading_pipeline.pipeline_config.update(
                self.config["market_data_config"])

        # Initialize CCXT executor with real exchange connection
        if "exchange_config" in self.config:
            self.ccxt_executor = CCXTTradingExecutor(
                self.config["exchange_config"])

        # Initialize registry for trade logging
        if "registry_file" in self.config:
            self.registry = SoulprintRegistry(self.config["registry_file"])

    async def execute_live_trade(
    self,
    symbol: str,
     force_refresh: bool = False):
        """Execute live trade based on current market conditions."""
        if not self.trading_pipeline:
            return {"error": "Trading pipeline not initialized"}

        # Process market data and execute trade
        trade_result = await self.trading_pipeline.process_market_data(
            market_data=None,  # Use live API data
            force_refresh=force_refresh
        )

        if trade_result and trade_result.get(
    "trade_action", {}).get("action") != "hold":
            # Execute actual trade through CCXT
            if self.ccxt_executor:
                execution_result = await self._execute_ccxt_trade(trade_result)
                trade_result["execution_result"] = execution_result

            # Log to registry
            if self.registry:
                self._log_trade_result(trade_result)

        return trade_result

    async def _execute_ccxt_trade(
        self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade through CCXT on real exchange."""
        try:
            trade_action = trade_result.get("trade_action", {})
            action = trade_action.get("action")
            symbol = trade_result.get("symbol")

            if action == "buy":
                result = await self.ccxt_executor.place_market_buy_order(
                    symbol=symbol,
                    amount=trade_action.get("position_size", 0.01)
                )
            elif action == "sell":
                result = await self.ccxt_executor.place_market_sell_order(
                    symbol=symbol,
                    amount=trade_action.get("position_size", 0.01)
                )
            else:
                return {"error": "Invalid trade action"}

        return result

    except Exception as e:
            return {"error": f"Trade execution failed: {e}"}

    def _log_trade_result(self, trade_result: Dict[str, Any]):
        """Log trade result to registry."""
        if not self.registry:
            return

        market_packet = trade_result.get("market_packet")
        if not market_packet:
            return

        try:
            # Extract key data for registry
            schwafit_info = {
                "symbol": trade_result.get("symbol"),
                "price": market_packet.get("price", 0),
                "signal_strength": trade_result.get("signals", {}).get("signal_strength", 0),
                "confidence": trade_result.get("signals", {}).get("confidence", 0),
                "action": trade_result.get("trade_action", {}).get("action", "hold")
            }

            # Log trigger
            self.registry.log_trigger(
                asset=trade_result.get("symbol", "UNKNOWN"),
                phase=market_packet.get("bb_position", 0.5),
                drift=market_packet.get("momentum_score", 0.0),
                schwafit_info=schwafit_info,
                trade_result=trade_result.get("execution_result", {})
            )

    except Exception as e:
            print(f"Failed to log trade: {e}")

    async def start_automated_trading(self, interval: int = 60):
        """Start automated trading loop."""
        self.is_running = True
        print(f"🚀 Starting automated trading (interval: {interval}s)")

        while self.is_running:
            try:
                trade_result = await self.execute_live_trade(
                    self.config.get("symbol", "BTCUSDT"),
                    force_refresh=True
                )

                if trade_result:
                    action = trade_result.get(
    "trade_action", {}).get(
        "action", "hold")
                    if action != "hold":
                        print(
    f"⚡ Trade executed: {action} {
        self.config.get('symbol')}")

                await asyncio.sleep(interval)

            except Exception as e:
                print(f"❌ Trading error: {e}")
                await asyncio.sleep(interval)

    def stop_trading(self):
        """Stop automated trading."""
        self.is_running = False
        print("🛑 Trading stopped")


def main():
    parser = argparse.ArgumentParser(description="Live Trading Bot CLI")
    parser.add_argument("--mode", choices=[
        "trade", "start-bot", "stop-bot", "execute-single",
        "log-trigger", "best-phase", "profit-vector",
        "cross-asset-best", "last-triggers"
    ], default="trade", help="Trading operation mode")

    parser.add_argument(
    "--config",
    type=str,
    required=True,
     help="Trading bot configuration file")
    parser.add_argument(
    "--symbol",
    type=str,
    default="BTCUSDT",
     help="Trading symbol")
    parser.add_argument("--interval", type=int, default=60,
                        help="Trading interval in seconds")
    parser.add_argument(
    "--force-refresh",
    action="store_true",
     help="Force refresh market data")
    parser.add_argument(
    "--safe-mode",
    action="store_true",
     help="Run in safe mode, skipping external plugins.")

    # Registry operations
    parser.add_argument("--registry-file", type=str, help="Registry file path")
    parser.add_argument(
    "--asset",
    type=str,
     help="Asset for registry operations")
    parser.add_argument(
    "--phase",
    type=float,
     help="Phase value for trigger logging")
    parser.add_argument(
    "--drift",
    type=float,
     help="Drift value for trigger logging")
    parser.add_argument(
    "--limit",
    type=int,
    default=10,
     help="Limit for query results")

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Handle safe mode
    if args.safe_mode:
        print("\033[91mMINIMAL SAFE MODE\033[0m")  # Red text
        config["safe_mode"] = True
    else:
        config["safe_mode"] = False

    # Override config with CLI args
    if args.symbol:
        config["symbol"] = args.symbol
    if args.registry_file:
        config["registry_file"] = args.registry_file

    # Execute based on mode
    if args.mode == "trade":
        return asyncio.run(execute_single_trade(config, args))
    elif args.mode == "start-bot":
        return asyncio.run(start_automated_trading(config, args))
    elif args.mode == "execute-single":
        return asyncio.run(execute_single_trade(config, args))
    elif args.mode in ["log-trigger", "best-phase", "profit-vector", "cross-asset-best", "last-triggers"]:
        return handle_registry_operation(config, args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


async def execute_single_trade(config: Dict[str, Any], args):
    """Execute a single trade operation."""
    try:
        bot = LiveTradingBot(config)
        await bot.initialize()

        trade_result = await bot.execute_live_trade(
            args.symbol,
            force_refresh=args.force_refresh
        )

        if trade_result:
            action = trade_result.get("trade_action", {}).get("action", "hold")
            print(f"✅ Trade completed: {action} {args.symbol}")

            if "execution_result" in trade_result:
                execution = trade_result["execution_result"]
                if "error" not in execution:
                    print(f"💰 Order ID: {execution.get('id', 'N/A')}")
                    print(f"💵 Amount: {execution.get('amount', 0)}")
                    print(f"💲 Price: {execution.get('price', 0)}")
        else:
                    print(f"❌ Execution error: {execution['error']}")

            return 0
        else:
            print("❌ Trade execution failed")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

async def start_automated_trading(config: Dict[str, Any], args):
    """Start automated trading bot."""
    try:
        bot = LiveTradingBot(config)
        await bot.initialize()
        
        await bot.start_automated_trading(args.interval)
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Trading bot stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

def handle_registry_operation(config: Dict[str, Any], args):
    """Handle registry operations."""
    registry_file = config.get("registry_file") or args.registry_file
    if not registry_file:
        print("❌ Registry file required for this operation")
        return 1
    
    registry = SoulprintRegistry(registry_file)
    
    if args.mode == "log-trigger":
        if not (args.asset and args.phase is not None and args.drift is not None):
            print("❌ log-trigger requires --asset, --phase, --drift")
            return 1
        # Log trigger would be called from trading execution, not manually
        print("⚠️  Triggers are logged automatically during trading")
        return 0
    
    elif args.mode == "best-phase":
        if not args.asset:
            print("❌ best-phase requires --asset")
            return 1
        best = registry.get_best_phase(args.asset)
        print(json.dumps(best, indent=2))
        return 0
    
    elif args.mode == "profit-vector":
        if not args.asset:
            print("❌ profit-vector requires --asset")
            return 1
        profits = registry.get_profit_vector(args.asset, phase=args.phase, drift=args.drift)
        print(json.dumps(profits, indent=2))
        return 0
    
    elif args.mode == "cross-asset-best":
        best = registry.get_cross_asset_best()
        print(json.dumps(best, indent=2))
        return 0
    
    elif args.mode == "last-triggers":
        if not args.asset:
            print("❌ last-triggers requires --asset")
            return 1
        last = registry.get_last_triggers(args.asset, n=args.limit)
        print(json.dumps(last, indent=2))
        return 0

if __name__ == "__main__":
    exit(main())