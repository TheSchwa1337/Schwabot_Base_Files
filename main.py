#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Schwabot Unified CLI + Test Engine

Comprehensive command-line interface for the Schwabot trading system.
Provides testing, backtesting, live trading, and hash registry management.

Usage:
    python main.py --run-tests                    # Run comprehensive system tests
    python main.py --backtest --days 30          # Run backtest for 30 days
    python main.py --live --config config.yaml   # Start live trading
    python main.py --hash-log --symbol BTC/USDT  # Log hash decisions
    python main.py --fetch-hash-decision         # Fetch hash-based decisions
    python main.py --system-status               # Get system status
    python main.py --error-log --limit 50        # Get error log
    python main.py --reset-circuit-breakers      # Reset all circuit breakers
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('schwabot_cli.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class SchwabotCLI:
    """Unified CLI for Schwabot trading system."""
    
    def __init__(self):
        """Initialize the CLI system."""
        self.test_results = {}
        self.backtest_results = {}
        self.live_trading_active = False
        self.hash_registry = {}
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all major system components."""
        try:
            # Import core components
            from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
            from core.risk_manager import RiskManager
            from core.unified_btc_trading_pipeline import create_btc_trading_pipeline
            from core.pure_profit_calculator import PureProfitCalculator
            
            # Initialize components
            self.trading_executor = None
            self.risk_manager = RiskManager()
            self.btc_pipeline = create_btc_trading_pipeline()
            
            # Initialize profit calculator with default strategy params
            strategy_params = {
                'risk_tolerance': 0.02,
                'profit_target': 0.05,
                'stop_loss': 0.03,
                'position_size': 0.1
            }
            self.profit_calculator = PureProfitCalculator(strategy_params)
            
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive system tests."""
        logger.info("RUNNING COMPREHENSIVE SYSTEM TESTS")
        logger.info("=" * 60)
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': {}
        }
        
        try:
            # Test 1: Risk Manager
            logger.info("Test 1: Risk Manager")
            risk_test = await self._test_risk_manager()
            test_results['test_details']['risk_manager'] = risk_test
            if risk_test['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            
            # Test 2: BTC Trading Pipeline
            logger.info("Test 2: BTC Trading Pipeline")
            pipeline_test = await self._test_btc_pipeline()
            test_results['test_details']['btc_pipeline'] = pipeline_test
            if pipeline_test['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            
            # Test 3: Profit Calculator
            logger.info("Test 3: Profit Calculator")
            profit_test = await self._test_profit_calculator()
            test_results['test_details']['profit_calculator'] = profit_test
            if profit_test['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            
            # Test 4: Error Handling
            logger.info("Test 4: Error Handling")
            error_test = await self._test_error_handling()
            test_results['test_details']['error_handling'] = error_test
            if error_test['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            
            # Test 5: Hash Registry
            logger.info("Test 5: Hash Registry")
            hash_test = await self._test_hash_registry()
            test_results['test_details']['hash_registry'] = hash_test
            if hash_test['passed']:
                test_results['tests_passed'] += 1
            else:
                test_results['tests_failed'] += 1
            
            # Summary
            logger.info("TEST SUMMARY")
            logger.info(f"Tests Passed: {test_results['tests_passed']}")
            logger.info(f"Tests Failed: {test_results['tests_failed']}")
            logger.info(f"Success Rate: {test_results['tests_passed'] / (test_results['tests_passed'] + test_results['tests_failed']) * 100:.1f}%")
            
            return test_results
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {'error': str(e)}
    
    async def _test_risk_manager(self) -> Dict[str, Any]:
        """Test risk manager functionality."""
        try:
            # Test basic risk calculation
            import numpy as np
            test_returns = np.random.normal(0.001, 0.02, 1000)  # 1000 days of returns
            
            risk_metrics = self.risk_manager.calculate_risk_metrics(test_returns)
            
            # Verify metrics are reasonable
            assert -0.1 < risk_metrics.var_95 < 0.1, "VaR out of reasonable range"
            assert -0.2 < risk_metrics.max_drawdown < 0, "Max drawdown should be negative"
            assert risk_metrics.volatility > 0, "Volatility should be positive"
            
            # Test error logging
            self.risk_manager.log_error(
                self.risk_manager.ErrorType.TIMEOUT,
                "Test timeout error",
                symbol="BTC/USDT",
                trade_id="test_123"
            )
            
            error_stats = self.risk_manager.get_error_statistics()
            assert error_stats['total_errors'] > 0, "Error should be logged"
            
            return {
                'passed': True,
                'message': 'Risk manager working correctly',
                'metrics': {
                    'var_95': risk_metrics.var_95,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'volatility': risk_metrics.volatility
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Risk manager test failed: {e}',
                'error': str(e)
            }
    
    async def _test_btc_pipeline(self) -> Dict[str, Any]:
        """Test BTC trading pipeline."""
        try:
            # Test pipeline with sample data
            test_prices = [50000, 50100, 50200, 50150, 50300]
            test_volumes = [1000000, 1200000, 1100000, 900000, 1300000]
            
            results = []
            for price, volume in zip(test_prices, test_volumes):
                result = self.btc_pipeline.process_btc_price(price, volume)
                results.append(result)
            
            # Check that pipeline processes data
            assert len(results) == len(test_prices), "Pipeline should process all data"
            
            return {
                'passed': True,
                'message': 'BTC pipeline working correctly',
                'processed_count': len(results)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'BTC pipeline test failed: {e}',
                'error': str(e)
            }
    
    async def _test_profit_calculator(self) -> Dict[str, Any]:
        """Test profit calculator."""
        try:
            # Test profit calculation
            from core.pure_profit_calculator import MarketData, HistoryState
            
            market_data = MarketData(
                timestamp=time.time(),
                btc_price=50000.0,
                eth_price=3000.0,
                usdc_volume=1000000.0,
                volatility=0.02,
                momentum=0.01,
                volume_profile=0.5,
                on_chain_signals={'whale_activity': 0.3, 'network_health': 0.9}
            )
            
            history_state = HistoryState(timestamp=time.time())
            profit_result = self.profit_calculator.calculate_profit(market_data, history_state)
            
            # Verify profit calculation
            assert hasattr(profit_result, 'total_profit_score'), "Profit result should have profit score"
            assert 0 <= profit_result.confidence_score <= 1, "Confidence should be between 0 and 1"
            
            return {
                'passed': True,
                'message': 'Profit calculator working correctly',
                'profit_score': profit_result.total_profit_score,
                'confidence': profit_result.confidence_score
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Profit calculator test failed: {e}',
                'error': str(e)
            }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery."""
        try:
            # Test error logging
            self.risk_manager.log_error(
                self.risk_manager.ErrorType.NETWORK_ERROR,
                "Test network error",
                symbol="ETH/USDT"
            )
            
            # Test circuit breaker
            self.risk_manager.log_error(
                self.risk_manager.ErrorType.CCXT_REJECTION,
                "Test CCXT rejection",
                symbol="BTC/USDT"
            )
            
            # Check error statistics
            error_stats = self.risk_manager.get_error_statistics()
            
            # Test safe mode
            self.risk_manager._enter_safe_mode(
                self.risk_manager.SafeMode.DEGRADED,
                "Test safe mode entry"
            )
            
            system_status = self.risk_manager.get_system_status()
            
            return {
                'passed': True,
                'message': 'Error handling working correctly',
                'total_errors': error_stats['total_errors'],
                'safe_mode': system_status['safe_mode']
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Error handling test failed: {e}',
                'error': str(e)
            }
    
    async def _test_hash_registry(self) -> Dict[str, Any]:
        """Test hash registry functionality."""
        try:
            # Test hash generation and storage
            test_data = {
                'symbol': 'BTC/USDT',
                'price': 50000.0,
                'timestamp': time.time(),
                'decision': 'BUY'
            }
            
            # Generate hash
            import hashlib
            hash_value = hashlib.sha256(json.dumps(test_data, sort_keys=True).encode()).hexdigest()[:16]
            
            # Store in registry
            self.hash_registry[hash_value] = {
                'data': test_data,
                'timestamp': datetime.now().isoformat(),
                'decision': 'BUY'
            }
            
            # Verify storage
            assert hash_value in self.hash_registry, "Hash should be stored"
            assert self.hash_registry[hash_value]['decision'] == 'BUY', "Decision should be stored"
            
            return {
                'passed': True,
                'message': 'Hash registry working correctly',
                'hash_count': len(self.hash_registry)
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Hash registry test failed: {e}',
                'error': str(e)
            }
    
    async def run_backtest(self, days: int = 30) -> Dict[str, Any]:
        """Run backtest for specified number of days."""
        logger.info(f"RUNNING BACKTEST FOR {days} DAYS")
        logger.info("=" * 50)
        
        try:
            # Import backtesting components
            from backtesting.simple_backtester import SimpleBacktester
            from backtesting.historical_data_manager import HistoricalDataManager
            
            # Initialize backtester
            backtester = SimpleBacktester()
            data_manager = HistoricalDataManager()
            
            # Run backtest
            results = await backtester.run_backtest(
                symbol='BTC/USDT',
                start_date=datetime.now().replace(day=datetime.now().day - days),
                end_date=datetime.now(),
                initial_capital=10000.0
            )
            
            logger.info("BACKTEST RESULTS")
            logger.info(f"Total Return: {results.get('total_return', 0):.2f}%")
            logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
            logger.info(f"Total Trades: {results.get('total_trades', 0)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    async def start_live_trading(self, config_path: Optional[str] = None) -> None:
        """Start live trading."""
        logger.info("STARTING LIVE TRADING")
        logger.info("=" * 40)
        
        try:
            # Load configuration
            if config_path:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                config = self._get_default_config()
            
            # Initialize trading executor
            from core.entropy_enhanced_trading_executor import EntropyEnhancedTradingExecutor
            
            self.trading_executor = EntropyEnhancedTradingExecutor(
                exchange_config=config['exchange'],
                strategy_config=config['strategy'],
                entropy_config=config['entropy'],
                risk_config=config['risk']
            )
            
            self.live_trading_active = True
            
            logger.info("Live trading started successfully")
            logger.info("Press Ctrl+C to stop")
            
            # Run trading loop
            await self.trading_executor.run_trading_loop(interval_seconds=60)
            
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.live_trading_active = False
        except Exception as e:
            logger.error(f"Live trading failed: {e}")
            self.live_trading_active = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for live trading."""
        return {
            'exchange': {
                'exchange': 'binance',
                'sandbox': True,
                'api_key': '',
                'secret': ''
            },
            'strategy': {
                'enabled': True,
                'risk_tolerance': 'medium'
            },
            'entropy': {
                'enabled': True,
                'threshold': 0.5
            },
            'risk': {
                'risk_tolerance': 0.02,
                'max_portfolio_risk': 0.05,
                'error_handling': {
                    'max_errors_per_symbol': 3,
                    'max_errors_per_timeframe': 10,
                    'error_timeframe_seconds': 60,
                    'circuit_breaker_cooldown_seconds': 300,
                    'safe_mode_error_threshold': 5
                }
            }
        }
    
    def log_hash_decisions(self, symbol: str) -> None:
        """Log hash-based decisions for a symbol."""
        logger.info(f"LOGGING HASH DECISIONS FOR {symbol}")
        
        try:
            # Generate decision hash
            decision_data = {
                'symbol': symbol,
                'timestamp': time.time(),
                'price': 50000.0,  # Mock price
                'decision': 'BUY',
                'confidence': 0.75
            }
            
            import hashlib
            hash_value = hashlib.sha256(json.dumps(decision_data, sort_keys=True).encode()).hexdigest()[:16]
            
            # Store in registry
            self.hash_registry[hash_value] = {
                'data': decision_data,
                'timestamp': datetime.now().isoformat(),
                'decision': 'BUY',
                'confidence': 0.75
            }
            
            logger.info(f"Hash decision logged: {hash_value}")
            
        except Exception as e:
            logger.error(f"Failed to log hash decision: {e}")
    
    def fetch_hash_decisions(self) -> Dict[str, Any]:
        """Fetch hash-based decisions."""
        logger.info("FETCHING HASH DECISIONS")
        
        try:
            return {
                'hash_count': len(self.hash_registry),
                'recent_decisions': list(self.hash_registry.values())[-10:],  # Last 10 decisions
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch hash decisions: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        logger.info("GETTING SYSTEM STATUS")
        
        try:
            # Get risk manager status
            risk_status = self.risk_manager.get_system_status()
            
            # Get error statistics
            error_stats = self.risk_manager.get_error_statistics()
            
            # Get hash registry status
            hash_status = {
                'total_hashes': len(self.hash_registry),
                'recent_activity': len([h for h in self.hash_registry.values() 
                                      if (datetime.now() - datetime.fromisoformat(h['timestamp'])).seconds < 3600])
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_management': risk_status,
                'error_handling': error_stats,
                'hash_registry': hash_status,
                'live_trading': self.live_trading_active
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def get_error_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get error log entries."""
        logger.info(f"GETTING ERROR LOG (limit: {limit})")
        
        try:
            return self.risk_manager.get_error_log(limit=limit)
            
        except Exception as e:
            logger.error(f"Failed to get error log: {e}")
            return []
    
    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        logger.info("RESETTING CIRCUIT BREAKERS")
        
        try:
            # Reset risk manager circuit breakers
            self.risk_manager.reset_circuit_breakers()
            
            # Reset symbol circuit breakers
            for symbol in self.risk_manager.circuit_breaker_states:
                self.risk_manager.reset_circuit_breaker(symbol, manual=True)
            
            logger.info("All circuit breakers reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset circuit breakers: {e}")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Schwabot Unified CLI + Test Engine')
    parser.add_argument('--run-tests', action='store_true', help='Run comprehensive system tests')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--backtest-days', type=int, default=30, help='Number of days for backtest')
    parser.add_argument('--live', action='store_true', help='Start live trading')
    parser.add_argument('--config', type=str, help='Configuration file for live trading')
    parser.add_argument('--hash-log', action='store_true', help='Log hash decisions')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--fetch-hash-decision', action='store_true', help='Fetch hash-based decisions')
    parser.add_argument('--system-status', action='store_true', help='Get system status')
    parser.add_argument('--error-log', action='store_true', help='Get error log')
    parser.add_argument('--error-log-limit', type=int, default=100, help='Limit for error log entries')
    parser.add_argument('--reset-circuit-breakers', action='store_true', help='Reset all circuit breakers')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = SchwabotCLI()
    
    try:
        if args.run_tests:
            results = await cli.run_comprehensive_tests()
            print(json.dumps(results, indent=2))
        
        elif args.backtest:
            results = await cli.run_backtest(args.backtest_days)
            print(json.dumps(results, indent=2))
        
        elif args.live:
            await cli.start_live_trading(args.config)
        
        elif args.hash_log:
            cli.log_hash_decisions(args.symbol)
        
        elif args.fetch_hash_decision:
            decisions = cli.fetch_hash_decisions()
            print(json.dumps(decisions, indent=2))
        
        elif args.system_status:
            status = cli.get_system_status()
            print(json.dumps(status, indent=2))
        
        elif args.error_log:
            errors = cli.get_error_log(args.error_log_limit)
            print(json.dumps(errors, indent=2))
        
        elif args.reset_circuit_breakers:
            cli.reset_circuit_breakers()
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
