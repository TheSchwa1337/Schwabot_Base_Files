#!/usr/bin/env python3
"""
Hash Recollection System Demo
Demonstrates the pattern recognition and trading signal generation capabilities.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import HashRecollectionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TradingSignalHandler:
    """Handles trading signals from the Hash Recollection System"""
    
    def __init__(self):
        self.signals_received = 0
        self.entry_signals = 0
        self.exit_signals = 0
        
    def handle_signal(self, signal):
        """Process trading signal"""
        self.signals_received += 1
        
        if signal['action'] == 'entry':
            self.entry_signals += 1
            logger.info(
                f"🟢 ENTRY SIGNAL: Price=${signal['price']:.2f}, "
                f"Confidence={signal['confidence']:.3f}, "
                f"Tier={signal['tier']}, "
                f"Size={signal.get('size', 0):.3f} BTC"
            )
        elif signal['action'] == 'exit':
            self.exit_signals += 1
            logger.info(
                f"🔴 EXIT SIGNAL: Price=${signal['price']:.2f}, "
                f"Confidence={signal['confidence']:.3f}"
            )
        else:
            logger.info(
                f"ℹ️  {signal['action'].upper()} SIGNAL: "
                f"Price=${signal['price']:.2f}, "
                f"Confidence={signal['confidence']:.3f}"
            )
        
        # Print signal details
        print(f"   Reasons: {', '.join(signal.get('reasons', []))}")
        print(f"   Hash: {signal['hash_value']}")
        print(f"   Similarity: {signal.get('similarity_score', 0):.3f}")
        print()


def generate_realistic_market_data(duration_minutes=5):
    """Generate realistic BTC price and volume data"""
    ticks_per_minute = 60  # 1 tick per second
    total_ticks = duration_minutes * ticks_per_minute
    
    base_price = 35000  # Starting BTC price
    price = base_price
    
    for i in range(total_ticks):
        # Add trend component (slow drift)
        trend = np.sin(i * 0.001) * 100
        
        # Add volatility (random walk)
        volatility = np.random.randn() * 50
        
        # Add microstructure noise
        noise = np.random.randn() * 5
        
        # Update price
        price_change = trend + volatility + noise
        price = max(price + price_change, 1000)  # Prevent negative prices
        
        # Generate volume (correlated with volatility)
        volume_base = 1.0
        volume_volatility = abs(volatility) / 100 + np.random.exponential(0.5)
        volume = max(volume_base + volume_volatility, 0.1)
        
        yield price, volume, time.time()
        
        # Control tick rate
        time.sleep(0.1)


def main():
    """Main demo function"""
    print("🚀 Hash Recollection System Demo")
    print("=" * 50)
    print()
    
    # Initialize the system
    logger.info("Initializing Hash Recollection System...")
    hrs = HashRecollectionSystem()
    
    # Create signal handler
    signal_handler = TradingSignalHandler()
    hrs.register_signal_callback(signal_handler.handle_signal)
    
    # Start the system
    logger.info("Starting Hash Recollection System...")
    hrs.start()
    
    print("📊 Processing market data...")
    print("   (Watch for pattern detection and trading signals)")
    print()
    
    try:
        # Generate and process market data
        tick_count = 0
        last_report_time = time.time()
        
        for price, volume, timestamp in generate_realistic_market_data(duration_minutes=3):
            # Process tick
            hrs.process_tick(price, volume, timestamp)
            tick_count += 1
            
            # Print periodic status
            if time.time() - last_report_time >= 10:  # Every 10 seconds
                metrics = hrs.get_pattern_metrics()
                print(f"📈 Status Update (Tick {tick_count}):")
                print(f"   Price: ${price:.2f}")
                print(f"   Hash DB: {metrics['hash_count']} entries")
                print(f"   Patterns: {metrics['patterns_detected']}")
                print(f"   Signals: {signal_handler.signals_received}")
                print(f"   GPU: {metrics['gpu_utilization']:.1%}")
                print()
                
                last_report_time = time.time()
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    finally:
        # Stop the system
        logger.info("Stopping Hash Recollection System...")
        hrs.stop()
        
        # Print final report
        print("\n" + "=" * 50)
        print("📊 Final Report")
        print("=" * 50)
        
        report = hrs.get_system_report()
        
        print(f"Uptime: {report['summary']['uptime']}")
        print(f"Ticks Processed: {report['summary']['ticks_processed']}")
        print(f"Patterns Detected: {report['summary']['patterns_detected']}")
        print(f"Hash Database Size: {report['summary']['hash_database_size']}")
        print(f"Final Price: ${report['summary']['current_price']:.2f}")
        print()
        
        print("Signal Summary:")
        print(f"  Total Signals: {signal_handler.signals_received}")
        print(f"  Entry Signals: {signal_handler.entry_signals}")
        print(f"  Exit Signals: {signal_handler.exit_signals}")
        print()
        
        # Pattern metrics
        pattern_metrics = report['patterns']
        print("Pattern Metrics:")
        print(f"  Pattern Confidence: {pattern_metrics['pattern_confidence']:.3f}")
        print(f"  Collision Rate: {pattern_metrics['collision_rate']:.3f}")
        print(f"  Tetragram Density: {pattern_metrics['tetragram_density']:.3f}")
        
        if 'bit_pattern_strength' in pattern_metrics:
            print(f"  Bit Pattern Strength: {pattern_metrics['bit_pattern_strength']:.3f}")
            print(f"  Current Tier: {pattern_metrics['current_tier']}")
        
        print()
        print("✅ Demo completed successfully!")


if __name__ == "__main__":
    main() 