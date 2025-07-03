#!/usr/bin/env python3
"""
CLI Live Entry - Command line interface for live trading operations
"""
import argparse
import json
import sys
import os
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.integration_orchestrator import orchestrate_trade
from core.strategy_loader import load_strategy
from core.matrix_mapper import match_hash_to_matrix

def run_trade_by_hash(hash_vector: list, strategy_name: str = "momentum", matrix_dir: str = None):
    """Run a trade using a hash vector."""
    if matrix_dir is None:
        matrix_dir = os.path.join(os.path.dirname(__file__), "data")
    
    print(f"🔍 Processing hash vector (length: {len(hash_vector)})")
    print(f"🎯 Strategy: {strategy_name}")
    
    try:
        result = orchestrate_trade(hash_vector, matrix_dir, strategy_name)
        
        print("✅ Trade orchestration completed!")
        print(f"📊 Matrix file: {result['matrix_file']}")
        print(f"💼 Trade result: {json.dumps(result['trade_result'], indent=2)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def run_trade_by_strategy(strategy_name: str, market_data: Dict[str, Any]):
    """Run a trade using a specific strategy."""
    print(f"🎯 Loading strategy: {strategy_name}")
    
    try:
        strategy = load_strategy(strategy_name)
        if not strategy:
            print(f"❌ Strategy '{strategy_name}' not found")
            return None
        
        print("✅ Strategy loaded successfully")
        print(f"📊 Market data: {json.dumps(market_data, indent=2)}")
        
        result = strategy(market_data)
        
        print("✅ Strategy execution completed!")
        print(f"💼 Decision: {json.dumps(result, indent=2)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_available_strategies():
    """List all available strategies."""
    print("📋 Available Strategies:")
    print("========================")
    
    # This would need to be implemented based on your strategy registry
    strategies = ["momentum", "mean_reversion", "entropy_driven"]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy}")
    
    return strategies

def generate_test_hash(length: int = 10) -> list:
    """Generate a test hash vector."""
    return np.random.rand(length).tolist()

def main():
    parser = argparse.ArgumentParser(description="Schwabot CLI Live Entry")
    parser.add_argument("--mode", choices=["hash", "strategy", "list"], 
                       default="hash", help="Operation mode")
    parser.add_argument("--hash", type=str, help="Hash vector as JSON array")
    parser.add_argument("--strategy", type=str, default="momentum", 
                       help="Strategy name")
    parser.add_argument("--market-data", type=str, 
                       help="Market data as JSON object")
    parser.add_argument("--matrix-dir", type=str, 
                       help="Matrix directory path")
    parser.add_argument("--generate-hash", type=int, metavar="LENGTH",
                       help="Generate test hash vector of specified length")
    
    args = parser.parse_args()
    
    print("🌀 Schwabot CLI Live Entry")
    print("=" * 40)
    
    if args.mode == "list":
        list_available_strategies()
        return
    
    if args.mode == "hash":
        if args.generate_hash:
            hash_vector = generate_test_hash(args.generate_hash)
            print(f"🔧 Generated test hash vector (length: {args.generate_hash})")
        elif args.hash:
            try:
                hash_vector = json.loads(args.hash)
            except json.JSONDecodeError:
                print("❌ Invalid JSON format for hash vector")
                return
        else:
            print("❌ Please provide --hash or --generate-hash")
            return
        
        run_trade_by_hash(hash_vector, args.strategy, args.matrix_dir)
    
    elif args.mode == "strategy":
        if not args.market_data:
            # Use default market data
            market_data = {
                "symbol": "BTC/USDC",
                "price": 65000.0,
                "volume": 1000000000,
                "timestamp": 1234567890
            }
            print("🔧 Using default market data")
        else:
            try:
                market_data = json.loads(args.market_data)
            except json.JSONDecodeError:
                print("❌ Invalid JSON format for market data")
                return
        
        run_trade_by_strategy(args.strategy, market_data)

if __name__ == "__main__":
    main() 