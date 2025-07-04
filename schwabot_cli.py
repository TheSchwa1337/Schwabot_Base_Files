import argparse
import sys
import os
import numpy as np
from core.schwafit_core import SchwafitCore


def main():
    parser = argparse.ArgumentParser(description="Schwabot CLI - Schwafit, Matrix, Fractal, AI, and more.")
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

    # AI command (stub)
    ai_parser = subparsers.add_parser("ai", help="Launch AI/CUDA model for fit/override (stub).")

    # Config command (stub)
    config_parser = subparsers.add_parser("config", help="Show or edit config (stub).")

    args = parser.parse_args()

    schwafit = SchwafitCore(window=args.window if hasattr(args, 'window') else 64)

    if args.command == "fit":
        # Load prices
        if os.path.isfile(args.prices):
            prices = np.loadtxt(args.prices, delimiter=",")
        else:
            prices = np.array([float(x) for x in args.prices.split(",")])
        # Mock pattern library: use rolling windows from prices
        pattern_library = []
        profit_scores = []
        for i in range(len(prices) - args.window - 2):
            v = schwafit.delta2(prices[i:i+args.window+2])
            v_norm = schwafit.normalize(v)
            pattern_library.append(v_norm)
            # Mock profit: random or based on price diff
            profit_scores.append(float(prices[i+args.window+1] - prices[i+args.window]))
        # Fit
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
        # Generate mock price data
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
        print("Schwafit Memory:")
        for entry in schwafit.fit_memory():
            print(entry)
    elif args.command == "ai":
        print("[AI/CUDA model integration coming soon]")
    elif args.command == "config":
        print("[Config management coming soon]")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 