"""
Integrated Trading Test Suite
Comprehensive testing framework for the enhanced trading engine.
Integrates with visualization and live pipeline components.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.clean_unified_math import clean_unified_math
from core.trading_engine_integration import (
    ErrorSeverity,
    TradeExecution,
    TradeSignal,
    TradingError,
    generate_trade_signal,
)
from core.unified_trade_router import UnifiedTradeRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integrated_test_suite.log')
    ]
)

logger = logging.getLogger(__name__)


class IntegratedTradingTestSuite:
    """
    Comprehensive test suite for the enhanced trading system.
    Integrates mathematical validation, performance testing, and pipeline integration.
    """
    
    def __init__(self):
        self.router = UnifiedTradeRouter()
        self.test_results = []
        self.performance_data = []
        
    def test_mathematical_integration(self) -> Dict[str, Any]:
        """Test the mathematical integration with the trading engine."""
        
        logger.info("🧮 Testing Mathematical Integration...")
        
        test_cases = [
            {"price": 50000, "volume": 1.0, "expected_positive": True},
            {"price": 45000, "volume": 0.5, "expected_positive": False},
            {"price": 60000, "volume": 2.0, "expected_positive": True},
        ]
        
        results = {
            "test_name": "mathematical_integration",
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        for i, case in enumerate(test_cases):
            try:
                signal = generate_trade_signal(
                    asset="BTC/USDT",
                    price=case["price"],
                    volume=case["volume"]
                )
                
                # Validate mathematical score
                if case["expected_positive"] and signal.mathematical_score > 0:
                    results["passed"] += 1
                    results["details"].append({
                        "case": i + 1,
                        "status": "PASS",
                        "mathematical_score": signal.mathematical_score,
                        "risk_score": signal.risk_score
                    })
                elif not case["expected_positive"] and signal.mathematical_score <= 0:
                    results["passed"] += 1
                    results["details"].append({
                        "case": i + 1,
                        "status": "PASS",
                        "mathematical_score": signal.mathematical_score,
                        "risk_score": signal.risk_score
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "case": i + 1,
                        "status": "FAIL",
                        "expected": case["expected_positive"],
                        "actual": signal.mathematical_score
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "case": i + 1,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        logger.info(f"✅ Mathematical Integration: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_signal_generation(self) -> Dict[str, Any]:
        """Test signal generation with various market conditions."""
        
        logger.info("📡 Testing Signal Generation...")
        
        # Simulate different market scenarios
        market_scenarios = [
            {"name": "bullish", "price": 55000, "volume": 1.5, "trend": "up"},
            {"name": "bearish", "price": 45000, "volume": 0.8, "trend": "down"},
            {"name": "sideways", "price": 50000, "volume": 1.0, "trend": "stable"},
            {"name": "high_volatility", "price": 52000, "volume": 2.5, "trend": "volatile"},
        ]
        
        results = {
            "test_name": "signal_generation",
            "passed": 0,
            "failed": 0,
            "signals": []
        }
        
        for scenario in market_scenarios:
            try:
                signal = self.router.route_trade_signal(
                    price=scenario["price"],
                    volume=scenario["volume"],
                    metadata={"scenario": scenario["name"], "trend": scenario["trend"]}
                )
                
                # Validate signal properties
                if (signal.price > 0 and 
                    signal.volume > 0 and 
                    signal.confidence >= 0 and 
                    signal.confidence <= 1):
                    
                    results["passed"] += 1
                    results["signals"].append({
                        "scenario": scenario["name"],
                        "signal_id": signal.id,
                        "mathematical_score": signal.mathematical_score,
                        "confidence": signal.confidence,
                        "order_side": signal.order_side.value
                    })
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Signal generation failed for {scenario['name']}: {e}")
        
        logger.info(f"✅ Signal Generation: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_execution_pipeline(self) -> Dict[str, Any]:
        """Test the complete execution pipeline."""
        
        logger.info("⚡ Testing Execution Pipeline...")
        
        results = {
            "test_name": "execution_pipeline",
            "passed": 0,
            "failed": 0,
            "executions": []
        }
        
        # Generate test signals
        test_signals = []
        for i in range(5):
            try:
                signal = self.router.route_trade_signal(
                    price=50000 + (i * 1000),
                    volume=1.0 + (i * 0.1)
                )
                test_signals.append(signal)
            except Exception as e:
                logger.error(f"Failed to generate test signal {i}: {e}")
                continue
        
        # Test executions
        for signal in test_signals:
            try:
                execution = self.router.route_trade_execution(
                    signal=signal,
                    execution_price=signal.price + 50,  # Simulate slippage
                    execution_latency=0.03
                )
                
                # Validate execution
                if (execution.signal_id == signal.id and
                    execution.asset == signal.asset and
                    execution.volume == signal.volume):
                    
                    results["passed"] += 1
                    results["executions"].append({
                        "signal_id": signal.id,
                        "execution_id": execution.id,
                        "realized_profit": execution.realized_profit,
                        "performance_score": execution.performance_score
                    })
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Execution failed for signal {signal.id}: {e}")
        
        logger.info(f"✅ Execution Pipeline: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid inputs."""
        
        logger.info("🛡️ Testing Error Handling...")
        
        invalid_cases = [
            {"price": -100, "volume": 1.0, "description": "Negative price"},
            {"price": 50000, "volume": 0, "description": "Zero volume"},
            {"price": float('inf'), "volume": 1.0, "description": "Infinite price"},
            {"price": float('nan'), "volume": 1.0, "description": "NaN price"},
        ]
        
        results = {
            "test_name": "error_handling",
            "passed": 0,
            "failed": 0,
            "errors": []
        }
        
        for case in invalid_cases:
            try:
                signal = self.router.route_trade_signal(
                    price=case["price"],
                    volume=case["volume"]
                )
                # If we get here, the error handling failed
                results["failed"] += 1
                results["errors"].append({
                    "case": case["description"],
                    "status": "FAIL",
                    "message": "Expected error but got success"
                })
                
            except TradingError:
                # Expected error
                results["passed"] += 1
                results["errors"].append({
                    "case": case["description"],
                    "status": "PASS",
                    "message": "Correctly caught TradingError"
                })
            except Exception as e:
                # Unexpected error
                results["failed"] += 1
                results["errors"].append({
                    "case": case["description"],
                    "status": "FAIL",
                    "message": f"Unexpected error: {type(e).__name__}"
                })
        
        logger.info(f"✅ Error Handling: {results['passed']} passed, {results['failed']} failed")
        return results
    
    def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics calculation."""
        
        logger.info("📊 Testing Performance Metrics...")
        
        # Generate some test data
        for i in range(10):
            try:
                signal = self.router.route_trade_signal(
                    price=50000 + (i * 500),
                    volume=1.0 + (i * 0.1)
                )
                execution = self.router.route_trade_execution(signal)
            except Exception as e:
                logger.error(f"Failed to generate test data {i}: {e}")
        
        # Get performance metrics
        metrics = self.router.get_performance_metrics()
        
        results = {
            "test_name": "performance_metrics",
            "metrics": metrics,
            "validation": {}
        }
        
        # Validate metrics
        required_fields = [
            "total_signals", "total_executions", "success_count", 
            "error_count", "success_rate_percent"
        ]
        
        for field in required_fields:
            if field in metrics and metrics[field] is not None:
                results["validation"][field] = "PASS"
            else:
                results["validation"][field] = "FAIL"
        
        logger.info(f"✅ Performance Metrics: {sum(1 for v in results['validation'].values() if v == 'PASS')} valid")
        return results
    
    async def test_live_simulation(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Simulate live trading conditions."""
        
        logger.info(f"🔄 Starting Live Simulation for {duration_seconds} seconds...")
        
        start_time = time.time()
        signals_generated = 0
        executions_completed = 0
        
        results = {
            "test_name": "live_simulation",
            "duration_seconds": duration_seconds,
            "signals_generated": 0,
            "executions_completed": 0,
            "errors": 0,
            "performance_data": []
        }
        
        while time.time() - start_time < duration_seconds:
            try:
                # Simulate market data
                current_price = 50000 + (time.time() - start_time) * 10  # Simulate price movement
                current_volume = 1.0 + (time.time() - start_time) * 0.01
                
                # Generate signal
                signal = self.router.route_trade_signal(
                    price=current_price,
                    volume=current_volume,
                    metadata={"live_simulation": True, "timestamp": time.time()}
                )
                signals_generated += 1
                
                # Execute trade
                execution = self.router.route_trade_execution(signal)
                executions_completed += 1
                
                # Record performance data
                results["performance_data"].append({
                    "timestamp": time.time(),
                    "price": current_price,
                    "signal_strength": signal.signal_strength,
                    "mathematical_score": signal.mathematical_score,
                    "performance_score": execution.performance_score
                })
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
                
            except Exception as e:
                results["errors"] += 1
                logger.error(f"Live simulation error: {e}")
                await asyncio.sleep(0.1)
        
        results["signals_generated"] = signals_generated
        results["executions_completed"] = executions_completed
        
        logger.info(f"✅ Live Simulation: {signals_generated} signals, {executions_completed} executions, {results['errors']} errors")
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        
        logger.info("🚀 Starting Comprehensive Trading System Test...")
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_suite_version": "1.0.0",
            "results": {}
        }
        
        # Run synchronous tests
        test_results["results"]["mathematical_integration"] = self.test_mathematical_integration()
        test_results["results"]["signal_generation"] = self.test_signal_generation()
        test_results["results"]["execution_pipeline"] = self.test_execution_pipeline()
        test_results["results"]["error_handling"] = self.test_error_handling()
        test_results["results"]["performance_metrics"] = self.test_performance_metrics()
        
        # Run async live simulation
        try:
            loop = asyncio.get_event_loop()
            test_results["results"]["live_simulation"] = loop.run_until_complete(
                self.test_live_simulation(duration_seconds=10)
            )
        except Exception as e:
            logger.error(f"Live simulation failed: {e}")
            test_results["results"]["live_simulation"] = {"error": str(e)}
        
        # Calculate overall statistics
        total_tests = len(test_results["results"])
        passed_tests = sum(1 for result in test_results["results"].values() 
                          if "passed" in result and result.get("failed", 0) == 0)
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        logger.info(f"🎯 Comprehensive Test Complete: {passed_tests}/{total_tests} tests passed")
        return test_results
    
    def export_test_results(self, results: Dict[str, Any], filename: str = "test_results.json"):
        """Export test results to JSON file."""
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📁 Test results exported to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export test results: {e}")


def main():
    """Main function to run the integrated test suite."""
    
    test_suite = IntegratedTradingTestSuite()
    
    try:
        # Run comprehensive test
        results = test_suite.run_comprehensive_test()
        
        # Export results
        test_suite.export_test_results(results)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 INTEGRATED TRADING TEST SUITE RESULTS")
        print("="*60)
        
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed Tests: {summary['passed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        print("\n📊 Detailed Results:")
        for test_name, test_result in results["results"].items():
            if "passed" in test_result and "failed" in test_result:
                print(f"  {test_name}: {test_result['passed']} passed, {test_result['failed']} failed")
            else:
                print(f"  {test_name}: Completed")
        
        print(f"\n📁 Results saved to: test_results.json")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Test suite interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main() 