"""
Enhanced Demo Trade Pipeline
Demonstrates the integrated trading system with mathematical insights and error handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from core.unified_trade_router import UnifiedTradeRouter
from core.trading_engine_integration import TradingError, ErrorSeverity
from core.clean_unified_math import clean_unified_math
import logging

# Configure logging for better visibility
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('demo_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def run_demo_simulation():
    """Run the enhanced demo trade pipeline simulation."""
    
    router = UnifiedTradeRouter()
    
    # Enhanced BTC price points with more realistic data
    btc_price_points = [
        {"price": 65250.0, "volume": 0.68, "metadata": {"source": "demo", "tick": 1}},
        {"price": 65403.3, "volume": 0.712, "metadata": {"source": "demo", "tick": 2}},
        {"price": 65542.1, "volume": 0.677, "metadata": {"source": "demo", "tick": 3}},
        {"price": 65320.5, "volume": 0.723, "metadata": {"source": "demo", "tick": 4}},
        {"price": 65680.2, "volume": 0.691, "metadata": {"source": "demo", "tick": 5}},
    ]
    
    logger.info("🚀 Starting Enhanced Demo Trade Pipeline Simulation...")
    logger.info(f"Processing {len(btc_price_points)} price points")
    
    successful_signals = 0
    successful_executions = 0
    
    for i, tick in enumerate(btc_price_points):
        try:
            logger.info(f"\n📊 Processing Tick {i+1}: Price=${tick['price']:,.2f}, Volume={tick['volume']}")
            
            # Generate trade signal with enhanced metadata
            signal = router.route_trade_signal(
                price=tick["price"], 
                volume=tick["volume"],
                asset="BTC/USDT",
                metadata=tick.get("metadata", {})
            )
            successful_signals += 1
            
            # Generate trade execution
            execution = router.route_trade_execution(signal)
            successful_executions += 1
            
            # Display enhanced signal information
            print(f"\n🔹 SIGNAL #{i+1}:")
            print(f"  ID: {signal.id}")
            print(f"  Asset: {signal.asset}")
            print(f"  Price: ${signal.price:,.2f}")
            print(f"  Volume: {signal.volume}")
            print(f"  Signal Strength: {signal.signal_strength:.4f}")
            print(f"  Mathematical Score: {signal.mathematical_score:.4f}")
            print(f"  Risk Score: {signal.risk_score:.4f}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Order Side: {signal.order_side.value}")
            print(f"  Order Type: {signal.order_type.value}")
            print(f"  Entropy: {signal.entropy:.4f}")
            print(f"  Volatility: {signal.volatility:.4f}")
            
            # Display enhanced execution information
            print(f"\n🔸 EXECUTION #{i+1}:")
            print(f"  ID: {execution.id}")
            print(f"  Signal ID: {execution.signal_id}")
            print(f"  Execution Price: ${execution.execution_price:,.2f}")
            print(f"  Volume: {execution.volume}")
            print(f"  Latency: {execution.latency:.3f}s")
            print(f"  Order Side: {execution.order_side.value}")
            print(f"  Order Type: {execution.order_type.value}")
            
            if execution.realized_profit is not None:
                print(f"  Realized Profit: ${execution.realized_profit:,.2f}")
            if execution.performance_score is not None:
                print(f"  Performance Score: {execution.performance_score:.4f}")
            
            print(f"  Timestamp: {execution.timestamp}")
            
        except TradingError as te:
            logger.error(f"❌ Trading error on tick {i+1}: {te}")
            continue
        except Exception as e:
            logger.error(f"❌ Unexpected error on tick {i+1}: {e}")
            continue
    
    # Display final performance metrics
    print(f"\n📈 FINAL PERFORMANCE METRICS:")
    print("=" * 50)
    
    metrics = router.get_performance_metrics()
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Simulation Summary:")
    print(f"  Successful Signals: {successful_signals}/{len(btc_price_points)}")
    print(f"  Successful Executions: {successful_executions}/{len(btc_price_points)}")
    print(f"  Success Rate: {(successful_signals/len(btc_price_points)*100):.1f}%")
    
    logger.info("🎯 Enhanced Demo Trade Pipeline Simulation completed successfully!")


def run_stress_test():
    """Run a stress test with various edge cases."""
    
    logger.info("🧪 Starting Stress Test...")
    
    router = UnifiedTradeRouter()
    
    # Test cases including edge cases
    test_cases = [
        {"price": 50000.0, "volume": 1.0, "description": "Normal case"},
        {"price": 0.01, "volume": 0.001, "description": "Very low values"},
        {"price": 1000000.0, "volume": 1000.0, "description": "Very high values"},
        {"price": 50000.0, "volume": 0.0, "description": "Zero volume (should fail)"},
        {"price": -100.0, "volume": 1.0, "description": "Negative price (should fail)"},
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"\n🧪 Test Case {i+1}: {test_case['description']}")
        
        try:
            signal = router.route_trade_signal(
                price=test_case["price"],
                volume=test_case["volume"]
            )
            logger.info(f"✅ Test case {i+1} passed")
            
        except TradingError as te:
            logger.warning(f"⚠️ Expected error in test case {i+1}: {te}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in test case {i+1}: {e}")
    
    logger.info("🧪 Stress test completed!")


def export_results(router: UnifiedTradeRouter, filename: str = "demo_results.json"):
    """Export simulation results to JSON file."""
    
    try:
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": router.get_performance_metrics(),
            "signal_history": router.get_signal_history(),
            "execution_log": router.get_execution_log()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Results exported to {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to export results: {e}")


if __name__ == "__main__":
    try:
        # Run the main demo simulation
        run_demo_simulation()
        
        # Run stress test
        run_stress_test()
        
        # Export results
        router = UnifiedTradeRouter()  # Create fresh instance for export
        export_results(router)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Demo pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo pipeline failed: {e}")
        raise 