"""Module for Schwabot trading system."""

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List

from core.clean_trading_pipeline import CleanTradingPipeline, create_trading_pipeline
from core.soulprint_registry import SoulprintRegistry

# Adjust path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    async def run_backtest(config: Dict[str, Any]):
    """
    Runs a back-test using historical candle data, processing each candle
    through the pipeline and logging the outcome.
    """
    dataset_path = config.get("dataset")
        if not dataset_path or not os.path.exists(dataset_path):
        print("❌ Error: Historical dataset not found at '{0}'".format(dataset_path))
    return

    # 1. Initialize pipeline and registry
    print("🔧 Initializing trading pipeline for back-testing...")
    pipeline: CleanTradingPipeline = create_trading_pipeline(
    symbol=config.get("symbol", "BTC/USDT"),
    initial_capital=config.get("initial_capital", 10000.0),
    registry_file=config.get("registry_file"),
    )
    pipeline.set_mode("demo")  # Set to demo mode for back-testing

    # The pipeline creates its own registry instance if a file is provided
    soulprint_registry = pipeline.registry
        if not soulprint_registry:
        print("⚠️ Warning: No registry file configured. Signals will not be logged.")

        # 2. Load historical data
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                historical_candles = json.load(f)
                print("✅ Loaded {0} candles from '{1}'.".format(len(historical_candles), os.path.basename(dataset_path)))
                    except json.JSONDecodeError:
                    print("❌ Error: Could not decode JSON from '{0}'.".format(dataset_path))
                return
                    except Exception as e:
                    print("❌ Error loading dataset: {0}".format(e))
                return

                # 3. Process candles
                print("⏳ Processing historical candles...")
                trade_signals = []
                    for i, candle in enumerate(historical_candles):
                    print("   - Processing candle {0}/{1}".format(i + 1, len(historical_candles)), end="\r")
                    signal = await pipeline.process_candle(candle)

                        if signal and not signal.get("blocked"):
                        trade_signals.append(signal)
                            if soulprint_registry:
                            # Per user request: log timestamp, asset, mode, hash_id, signal
                            # vector, projected gain
                            market_context = signal.get("market_context", {})
                            entry_price = signal.get("entry_price", 0)
                            take_profit = signal.get("take_profit", 0)

                                if signal.get("action") == "buy" and take_profit > entry_price:
                                projected_gain = take_profit - entry_price
                                    elif signal.get("action") == "sell" and entry_price > take_profit:
                                    projected_gain = entry_price - take_profit
                                        else:
                                        projected_gain = 0

                                        log_data = {
                                        "timestamp": signal.get("timestamp"),
                                        "asset": signal.get("symbol"),
                                        "mode": "demo",
                                        "hash_id": signal.get("trade_id"),
                                        "signal_vector": {
                                        "action": signal.get("action"),
                                        "confidence": market_context.get("confidence"),
                                        "signal_strength": market_context.get("signal_strength"),
                                        },
                                        "projected_gain": projected_gain,
                                        "trade_details": signal,  # Store original signal for full context
                                        }
                                        # This method will be added to SoulprintRegistry in the next
                                        # step
                                            if hasattr(soulprint_registry, "log_backtest_signal"):
                                            soulprint_registry.log_backtest_signal(log_data)

                                            print("\n✅ Back-test complete.")
                                            print("   - Trades generated: {0}".format(len(trade_signals)))
                                            print("   - Final portfolio value (simulated)")


                                                def main():
                                                """Main entry point for the back-test driver."""
                                                parser = argparse.ArgumentParser(
                                                description="Schwabot Backtest Driver",
                                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                                )
                                                parser.add_argument("--dataset", type=str, required=True, help="Path to historical candle data JSON file.")
                                                parser.add_argument("--config", type=str, required=True, help="Path to the trading bot configuration file.")
                                                parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol for the back-test.")
                                                parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital for the simulation.")
                                                parser.add_argument(
                                                "--registry-file",
                                                type=str,
                                                default="data/logs/backtest_registry.json",
                                                help="File to log back-test signals.",
                                                )

                                                args = parser.parse_args()

                                                # Load base configuration
                                                    try:
                                                        with open(args.config, "r") as f:
                                                        config = json.load(f)
                                                            except FileNotFoundError:
                                                            print("❌ Error: Base config file not found at '{0}'".format(args.config))
                                                            sys.exit(1)
                                                                except json.JSONDecodeError:
                                                                print("❌ Error: Could not decode JSON from '{0}'.".format(args.config))
                                                                sys.exit(1)

                                                                # Override config with CLI arguments for the back-test
                                                                config["dataset"] = args.dataset
                                                                config["symbol"] = args.symbol
                                                                config["initial_capital"] = args.capital
                                                                config["registry_file"] = args.registry_file

                                                                # Ensure log directory exists
                                                                log_dir = os.path.dirname(args.registry_file)
                                                                    if log_dir:
                                                                    os.makedirs(log_dir, exist_ok=True)

                                                                    asyncio.run(run_backtest(config))


                                                                        if __name__ == "__main__":
                                                                        # Add project root to path to allow direct execution
                                                                        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                                                                            if project_root not in sys.path:
                                                                            sys.path.insert(0, project_root)

                                                                            main()
