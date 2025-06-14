#!/usr/bin/env python3
"""
GAN Tick Processing Example
===========================

Demonstrates the complete tick-by-tick GAN filter system with:
- Real-time market data processing
- Mathematical validation and parent-child correction
- Profit signal routing and accumulation
- Integration with Schwabot components

Usage:
    python examples/gan_tick_example.py
"""

import numpy as np
import time
import logging
from typing import List
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tick_processor import TickProcessor, TickProfile, create_tick_processor

def generate_sample_tick_data() -> List[TickProfile]:
    """Generate sample market tick data for demonstration"""
    ticks = []
    base_time = time.time()
    
    symbols = ['BTC', 'ETH', 'AAPL', 'TSLA', 'GOOGL']
    
    for i in range(100):
        # Simulate realistic market data
        symbol = symbols[i % len(symbols)]
        price = 50000 + np.random.normal(0, 1000) if symbol == 'BTC' else \
                3000 + np.random.normal(0, 200) if symbol == 'ETH' else \
                150 + np.random.normal(0, 10)
        
        volume = np.random.exponential(1000)
        
        # Create some intentionally problematic vectors to test correction
        if i % 15 == 0:
            # Low entropy vector (should trigger correction)
            raw_vector = np.ones(64) * 0.1 + np.random.normal(0, 0.01, 64)
        elif i % 17 == 0:
            # High spectral deviation (should trigger correction)
            raw_vector = np.random.normal(0, 10, 64)
        else:
            # Normal vector
            raw_vector = np.random.normal(0, 1, 64)
            
        tick = TickProfile(
            timestamp=base_time + i * 0.1,  # 100ms intervals
            symbol=symbol,
            price=max(0.01, price),  # Ensure positive price
            volume=max(1.0, volume),  # Ensure positive volume
            raw_vector=raw_vector,
            metadata={'sequence': i}
        )
        
        ticks.append(tick)
        
    return ticks

def custom_profit_router(signals: dict) -> None:
    """Custom profit routing handler"""
    total_profit = signals.get('anomaly_profit', 0) + signals.get('correction_bonus', 0)
    if total_profit > 0.5:  # Significant profit signal
        print(f"🎯 PROFIT SIGNAL: {signals['symbol']} - Total: {total_profit:.3f}")
        print(f"   └─ Anomaly: {signals.get('anomaly_profit', 0):.3f}")
        print(f"   └─ Correction: {signals.get('correction_bonus', 0):.3f}")
        print(f"   └─ Parent Boost: {signals.get('parent_boost', 1.0):.3f}")

def custom_fault_handler(error: Exception, tick: TickProfile) -> None:
    """Custom fault handling"""
    print(f"⚠️  FAULT: {tick.symbol} at {tick.timestamp} - {type(error).__name__}: {error}")

def custom_dashboard_updater(data: dict) -> None:
    """Custom dashboard updates for significant events"""
    if data['is_anomaly'] or data['correction_applied']:
        print(f"📊 {data['symbol']}: Anomaly={data['is_anomaly']}, "
              f"Score={data['anomaly_score']:.3f}, "
              f"Corrected={data['correction_applied']}")

async def main():
    """Main demonstration"""
    print("🚀 Starting GAN Tick Processing Demo\n")
    
    # Create tick processor with custom configuration
    gan_config = {
        'latent_dim': 32,
        'hidden_dim': 64, 
        'num_layers': 3,
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_epochs': 50,  # Reduced for demo
        'anomaly_threshold': 0.7,
        'use_gpu': False,  # Use CPU for demo compatibility
        'entropy_weight': 0.15,
        'phase_weight': 0.1
    }
    
    processor = TickProcessor(gan_config, input_dim=64)
    
    # Register custom handlers
    processor.register_profit_router(custom_profit_router)
    processor.register_fault_handler(custom_fault_handler)
    processor.register_dashboard_updater(custom_dashboard_updater)
    
    # Generate sample data
    print("📈 Generating sample tick data...")
    tick_data = generate_sample_tick_data()
    
    # Train the GAN filter on historical data (simulate bootstrap)
    print("🧠 Training GAN filter on sample data...")
    training_data = tick_data[:50]  # Use first 50 ticks for training
    processor.train_from_tick_history(training_data)
    
    print(f"✅ Training complete! Created {len(processor.gan_filter.cluster_db)} matrix clusters\n")
    
    # Process remaining ticks in real-time simulation
    print("⚡ Processing live tick stream...\n")
    live_data = tick_data[50:]  # Use remaining ticks for live processing
    
    results = []
    for i, tick in enumerate(live_data):
        result = processor.process_tick(tick)
        if result:
            results.append(result)
            
        # Simulate real-time delay
        await asyncio.sleep(0.05)  # 50ms processing interval
        
        # Show progress every 10 ticks
        if (i + 1) % 10 == 0:
            stats = processor.get_processing_stats()
            profit_summary = processor.get_profit_summary()
            print(f"\n📊 Progress: {i+1}/{len(live_data)} ticks processed")
            print(f"   Anomalies: {stats['anomalies_detected']}")
            print(f"   Corrections: {stats['corrections_applied']}")
            print(f"   Avg Processing Time: {stats['avg_processing_time']*1000:.1f}ms")
            print(f"   Total Profit Signals: {profit_summary['active_signals']}")
            print()
            
    # Final summary
    print("\n" + "="*60)
    print("🎉 PROCESSING COMPLETE - FINAL SUMMARY")
    print("="*60)
    
    final_stats = processor.get_processing_stats()
    profit_summary = processor.get_profit_summary()
    health = processor.health_check()
    
    print(f"📈 Processed Ticks: {final_stats['total_ticks']}")
    print(f"🚨 Anomalies Detected: {final_stats['anomalies_detected']}")
    print(f"🔧 Corrections Applied: {final_stats['corrections_applied']}")
    print(f"💰 Profit Signals Generated: {final_stats['profit_signals_generated']}")
    print(f"🏭 Matrix Clusters Created: {final_stats['cluster_count']}")
    print(f"⚡ Avg Processing Time: {final_stats['avg_processing_time']*1000:.1f}ms")
    print(f"🎯 Total Anomaly Profit: {profit_summary['total_anomaly_profit']:.3f}")
    print(f"🔧 Total Correction Bonus: {profit_summary['total_correction_bonus']:.3f}")
    print(f"📊 System Health: {health['status'].upper()}")
    
    # Show some detailed results
    print(f"\n🔍 SAMPLE PROCESSING RESULTS:")
    print("-" * 40)
    
    anomaly_results = [r for r in results if r.anomaly_metrics.is_anomaly][:3]
    correction_results = [r for r in results if r.correction_applied][:3]
    
    if anomaly_results:
        print("🚨 Anomalies Detected:")
        for result in anomaly_results:
            print(f"   Score: {result.anomaly_metrics.anomaly_score:.3f}, "
                  f"Confidence: {result.anomaly_metrics.confidence:.3f}")
                  
    if correction_results:
        print("🔧 Corrections Applied:")
        for result in correction_results:
            print(f"   Hash: {result.tick_hash[:8]}..., "
                  f"Profit: {result.profit_signal.get('anomaly_profit', 0):.3f}")
    
    print(f"\n✨ Demo complete! The GAN filter successfully processed {len(results)} ticks")
    print("   with mathematical validation, parent-child correction, and profit routing.\n")

if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main()) 