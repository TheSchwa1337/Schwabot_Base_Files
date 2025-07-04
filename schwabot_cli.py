import argparse
import sys
import os
import numpy as np
from core.schwafit_core import SchwafitCore
from core.strategy_bit_mapper import StrategyBitMapper


def main():
    parser = argparse.ArgumentParser(description="Schwabot CLI - Schwafit, Matrix, Bit Mapper, Live Handler, Ferris Wheel, and more.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Run Schwafit on a price series.")
    fit_parser.add_argument("--prices", type=str, required=True, help="CSV file or comma-separated price list.")
    fit_parser.add_argument("--window", type=int, default=64, help="Window size for fit.")
    fit_parser.add_argument("--show-math", action="store_true", help="Show all math output.")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run a mock Schwafit test.")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show Schwafit memory state.")

    # Select strategy command
    select_parser = subparsers.add_parser("select-strategy", help="Select strategy using Schwafit and Bit Mapper.")
    select_parser.add_argument("--hash", type=str, required=True, help="Comma-separated hash vector (floats).")
    select_parser.add_argument("--asset", type=str, default=None, help="Asset hint (optional).")
    select_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")

    # Match matrix command
    match_parser = subparsers.add_parser("match-matrix", help="Match hash to matrix using Schwafit and Matrix Mapper.")
    match_parser.add_argument("--hash", type=str, required=True, help="Comma-separated hash vector (floats).")
    match_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")

    # Live handler status command
    live_parser = subparsers.add_parser("live-status", help="Show live handler status from Bit Mapper.")
    live_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")

    # Ferris wheel spin command
    ferris_parser = subparsers.add_parser("ferris-spin", help="Spin Ferris wheel and calculate tensor/profit vectors.")
    ferris_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")
    ferris_parser.add_argument("--ticks", type=int, default=24, help="Number of ticks to spin.")

    # Live tick command
    tick_parser = subparsers.add_parser("live-tick", help="Simulate a live tick: update tensors, profit, and fit.")
    tick_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")
    tick_parser.add_argument("--price", type=float, required=True, help="Current price.")
    tick_parser.add_argument("--asset", type=str, default="BTC/USDC", help="Asset symbol.")

    # Entry/exit command
    entry_parser = subparsers.add_parser("entry-exit", help="Calculate entry/exit using Schwafit and Bit Mapper.")
    entry_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")
    entry_parser.add_argument("--hash", type=str, required=True, help="Comma-separated hash vector (floats).")
    entry_parser.add_argument("--market-state", type=str, required=False, help="Market state as JSON string.")

    # Ghost trade command
    ghost_parser = subparsers.add_parser("ghost-trade", help="Simulate ghost (BTC/USDC) trade using Schwafit-driven logic.")
    ghost_parser.add_argument("--matrix-dir", type=str, required=True, help="Path to matrix directory.")
    ghost_parser.add_argument("--hash", type=str, required=True, help="Comma-separated hash vector (floats).")
    ghost_parser.add_argument("--price", type=float, required=True, help="Current BTC price.")
    ghost_parser.add_argument("--usdc-balance", type=float, default=1000.0, help="USDC balance.")
    ghost_parser.add_argument("--btc-balance", type=float, default=0.0, help="BTC balance.")

    # AI command (stub)
    ai_parser = subparsers.add_parser("ai", help="Launch AI/CUDA model for fit/override (stub).")

    # Config command (stub)
    config_parser = subparsers.add_parser("config", help="Show or edit config (stub).")

    args = parser.parse_args()

    if args.command == "fit":
        schwafit = SchwafitCore(window=args.window)
        if os.path.isfile(args.prices):
            prices = np.loadtxt(args.prices, delimiter=",")
        else:
            prices = np.array([float(x) for x in args.prices.split(",")])
        pattern_library = []
        profit_scores = []
        for i in range(len(prices) - args.window - 2):
            v = schwafit.delta2(prices[i:i+args.window+2])
            v_norm = schwafit.normalize(v)
            pattern_library.append(v_norm)
            profit_scores.append(float(prices[i+args.window+1] - prices[i+args.window]))
        result = schwafit.fit_vector(prices, pattern_library, profit_scores)
        print("Schwafit Fit Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if args.show_math:
            print("\nPattern Library Size:", len(pattern_library))
            print("Top Scores:", result["top_scores"])
            print("Top Profits:", result["top_profits"])
            print("Entropy:", result["entropy"])
    elif args.command == "test":
        schwafit = SchwafitCore()
        prices = np.cumsum(np.random.randn(200)) + 100
        pattern_library = []
        profit_scores = []
        for i in range(len(prices) - schwafit.window - 2):
            v = schwafit.delta2(prices[i:i+schwafit.window+2])
            v_norm = schwafit.normalize(v)
            pattern_library.append(v_norm)
            profit_scores.append(float(prices[i+schwafit.window+1] - prices[i+schwafit.window]))
        result = schwafit.fit_vector(prices, pattern_library, profit_scores)
        print("Mock Schwafit Fit Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.command == "status":
        schwafit = SchwafitCore()
        print("Schwafit Memory:")
        for entry in schwafit.fit_memory():
            print(entry)
    elif args.command == "select-strategy":
        hash_vec = np.array([float(x) for x in args.hash.split(",")])
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        result = bit_mapper.select_strategy(hash_vec, asset_hint=args.asset)
        print("Selected Strategy Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.command == "match-matrix":
        hash_vec = np.array([float(x) for x in args.hash.split(",")])
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        matrix_name, entry, score, schwafit_info = bit_mapper.match_hash_to_matrix(hash_vec)
        print("Matrix Match Result:")
        print(f"  matrix_name: {matrix_name}")
        print(f"  score: {score}")
        print(f"  schwafit_info: {schwafit_info}")
    elif args.command == "live-status":
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        status = bit_mapper.get_live_handler_status()
        print("Live Handler Status:")
        for k, v in status.items():
            print(f"  {k}: {v}")
    elif args.command == "ferris-spin":
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        print("Ferris Wheel Spin Results:")
        for tick in range(args.ticks):
            # Use tensor-weighted expansion and log profit vector
            expanded_id = bit_mapper.expand_strategy_bits(tick, target_bits=8, mode="ferris_wheel")
            print(f"Tick {tick}: Expanded ID: {expanded_id}")
        print("Tensor Weights:", bit_mapper.tensor_weights)
    elif args.command == "live-tick":
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        # Simulate a live tick: update tensors, profit, and fit
        # For demo, just update tensor weights and print
        api_data = {"price_history": [args.price]}
        bit_mapper.update_tensor_weights_from_api_data(api_data)
        print("Live Tick: Updated tensor weights.")
        print("Tensor Weights:", bit_mapper.tensor_weights)
    elif args.command == "entry-exit":
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        hash_vec = np.array([float(x) for x in args.hash.split(",")])
        market_state = {}
        if args.market_state:
            import json
            market_state = json.loads(args.market_state)
        # Use select_strategy to get signal
        result = bit_mapper.select_strategy(hash_vec)
        print("Entry/Exit Calculation:")
        print(result)
        # Optionally, trigger entry/exit logic if implemented
        # bit_mapper.trigger_entry_exit(result, market_state, ccxt_executor=None)
    elif args.command == "ghost-trade":
        bit_mapper = StrategyBitMapper(args.matrix_dir)
        hash_vec = np.array([float(x) for x in args.hash.split(",")])
        # Use Schwafit-driven logic to decide trade
        result = bit_mapper.select_strategy(hash_vec, asset_hint="BTC/USDC")
        print("Ghost Trade Decision:")
        print(result)
        # Simulate trade logic
        usdc = args.usdc_balance
        btc = args.btc_balance
        price = args.price
        if result["schwafit"] and result["schwafit"]["decision"]:
            # Buy BTC with USDC
            btc_bought = usdc / price
            usdc = 0
            btc += btc_bought
            print(f"Executed BUY: Bought {btc_bought:.6f} BTC at {price}")
        else:
            # Sell BTC for USDC
            usdc += btc * price
            print(f"Executed SELL: Sold {btc:.6f} BTC at {price}")
            btc = 0
        print(f"Balances after trade: USDC={usdc:.2f}, BTC={btc:.6f}")
    elif args.command == "ai":
        print("[AI/CUDA model integration coming soon]")
    elif args.command == "config":
        print("[Config management coming soon]")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 