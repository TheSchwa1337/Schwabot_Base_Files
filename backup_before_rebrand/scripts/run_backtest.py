#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtesting_integration import BacktestConfig, BacktestingIntegration
from utils.safe_print import safe_print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Run Schwabot backtesting')
    
    # Basic settings
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)',
                      default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)',
                      default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                      help='Initial capital in USDC')
    parser.add_argument('--trading-pairs', type=str, nargs='+',
                      default=['BTC/USDC'],
                      help='Trading pairs to backtest')
    
    # System settings  
    parser.add_argument('--use-gpu', action='store_true', default=True,
                      help='Enable GPU acceleration')
    parser.add_argument('--risk-profile', type=str, choices=['conservative', 'moderate', 'aggressive'],
                      default='moderate', help='Risk management profile')
    parser.add_argument('--enable-visualization', action='store_true', default=True,
                      help='Enable result visualization')
    parser.add_argument('--results-dir', type=str, default='backtest_results',
                      help='Directory to save results')
    
    # Advanced settings
    parser.add_argument('--gpu-batch-size', type=int, default=1024,
                      help='GPU processing batch size')
    parser.add_argument('--price-interval', type=str, default='1m',
                      choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                      help='Price data interval')
    parser.add_argument('--max-positions', type=int, default=5,
                      help='Maximum number of open positions')
    parser.add_argument('--max-leverage', type=float, default=1.0,
                      help='Maximum leverage')
    
    return parser.parse_args()

async def main():
    args = parse_args()
    
    try:
        # Create config
        config = BacktestConfig(
            start_date=datetime.strptime(args.start_date, '%Y-%m-%d'),
            end_date=datetime.strptime(args.end_date, '%Y-%m-%d'),
            initial_capital=Decimal(str(args.initial_capital)),
            trading_pairs=args.trading_pairs,
            use_gpu=args.use_gpu,
            risk_profile=args.risk_profile,
            enable_visualization=args.enable_visualization,
            results_dir=args.results_dir,
            gpu_batch_size=args.gpu_batch_size,
            price_data_interval=args.price_interval,
            max_open_positions=args.max_positions,
            max_leverage=args.max_leverage
        )
        
        # Initialize backtesting system
        safe_print("🚀 Initializing backtesting system...")
        backtester = BacktestingIntegration(config)
        
        if not await backtester.initialize():
            safe_print("❌ Failed to initialize backtesting system")
            return
            
        # Run backtest
        safe_print(f"📈 Running backtest from {args.start_date} to {args.end_date}")
        safe_print(f"Trading pairs: {', '.join(args.trading_pairs)}")
        safe_print(f"Initial capital: ${args.initial_capital:,.2f} USDC")
        
        results = await backtester.run_backtest()
        
        if "error" in results:
            safe_print(f"❌ Backtest failed: {results['error']}")
            return
            
        # Print summary
        safe_print("\n=== Backtest Results ===")
        safe_print(f"Total Return: {results['metrics']['total_return']:.2%}")
        safe_print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        safe_print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
        safe_print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
        safe_print(f"Total Trades: {len(results['trade_history'])}")
        
        safe_print("\n✅ Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        safe_print(f"❌ Error: {e}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}") 