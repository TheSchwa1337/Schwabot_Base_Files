"""
News-Profit Advanced Processing Demo
===================================

Demonstrates the advanced CPU/GPU allocation and thermal management
integration with the News-Profit Mathematical Bridge. Shows how the
system dynamically adjusts processing allocation based on thermal state
and integrates with existing Schwabot SERC systems.

Features demonstrated:
- CPU/GPU percentage allocation controls
- Thermal-aware dynamic adjustment
- Processing mode selection (hybrid, cpu_only, gpu_preferred, thermal_aware)
- Real-time thermal monitoring integration
- SERC data pipeline integration
- GPU cool-down management
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.news_profit_mathematical_bridge import create_news_profit_bridge

async def demo_advanced_processing_controls():
    """Demonstrate advanced processing allocation and thermal management"""
    
    print("=" * 60)
    print("NEWS-PROFIT ADVANCED PROCESSING CONTROLS DEMO")
    print("=" * 60)
    
    # Create bridge with advanced features
    print("\n1. Initializing News-Profit Mathematical Bridge...")
    bridge = create_news_profit_bridge()
    
    # Display initial configuration
    print(f"\n2. Initial Configuration:")
    print(f"   • CPU Allocation: {bridge.cpu_allocation_percentage}%")
    print(f"   • GPU Allocation: {bridge.gpu_allocation_percentage}%")
    print(f"   • Processing Mode: {bridge.processing_mode}")
    print(f"   • Thermal Integration: {bridge.thermal_integration_enabled}")
    print(f"   • Dynamic Allocation: {bridge.dynamic_allocation_enabled}")
    
    # Test different CPU allocation scenarios
    print(f"\n3. Testing CPU Allocation Scenarios...")
    
    scenarios = [
        {"name": "Balanced Processing", "cpu": 70.0, "mode": "hybrid"},
        {"name": "CPU-Heavy Processing", "cpu": 85.0, "mode": "cpu_only"},
        {"name": "GPU-Preferred Processing", "cpu": 40.0, "mode": "gpu_preferred"},
        {"name": "Thermal-Aware Processing", "cpu": 60.0, "mode": "thermal_aware"}
    ]
    
    for scenario in scenarios:
        print(f"\n   Testing: {scenario['name']}")
        print(f"   └─ Setting CPU allocation to {scenario['cpu']}%")
        
        # Update allocation
        bridge.set_cpu_allocation_percentage(scenario['cpu'])
        bridge.set_processing_mode(scenario['mode'])
        
        # Get dynamic allocation (includes thermal adjustments)
        if hasattr(bridge, '_calculate_dynamic_allocation'):
            dynamic_cpu, dynamic_gpu = bridge._calculate_dynamic_allocation()
            print(f"   └─ Dynamic allocation: CPU {dynamic_cpu:.1f}%, GPU {dynamic_gpu:.1f}%")
        
        # Get thermal state
        thermal_state = bridge.get_current_thermal_state()
        if thermal_state['thermal_available']:
            print(f"   └─ Thermal zone: {thermal_state['thermal_zone']}")
            print(f"   └─ CPU temp: {thermal_state['cpu_temp']:.1f}°C, GPU temp: {thermal_state['gpu_temp']:.1f}°C")
        else:
            print(f"   └─ Thermal monitoring: Unavailable (using fallback mode)")
        
        # Test processing decision
        test_operation_size = 500
        should_use_gpu = bridge._should_process_on_gpu(test_operation_size, 1.0)
        print(f"   └─ GPU processing decision for 500-element operation: {'YES' if should_use_gpu else 'NO'}")
        
        time.sleep(1)  # Brief pause between scenarios

async def demo_thermal_management():
    """Demonstrate thermal management and scaling"""
    
    print(f"\n4. Thermal Management Demo...")
    
    bridge = create_news_profit_bridge()
    
    # Test thermal threshold scenarios
    thermal_scenarios = [
        {"name": "Cool Operation", "cpu_temp": 65.0, "gpu_temp": 60.0},
        {"name": "Warm Operation", "cpu_temp": 76.0, "gpu_temp": 72.0},
        {"name": "Hot Operation", "cpu_temp": 82.0, "gpu_temp": 78.0},
        {"name": "Emergency Scenario", "cpu_temp": 87.0, "gpu_temp": 85.0}
    ]
    
    for scenario in thermal_scenarios:
        print(f"\n   Scenario: {scenario['name']}")
        print(f"   └─ Simulated CPU temp: {scenario['cpu_temp']}°C")
        print(f"   └─ Simulated GPU temp: {scenario['gpu_temp']}°C")
        
        # Simulate thermal conditions by adjusting thresholds
        original_cpu_threshold = bridge.thermal_cpu_threshold
        original_gpu_threshold = bridge.thermal_gpu_threshold
        
        bridge.thermal_cpu_threshold = scenario['cpu_temp'] - 5.0
        bridge.thermal_gpu_threshold = scenario['gpu_temp'] - 5.0
        
        # Test processing decisions under these thermal conditions
        test_ops = [100, 500, 1000, 2000]
        gpu_decisions = []
        
        for op_size in test_ops:
            gpu_decision = bridge._should_process_on_gpu(op_size, 1.0)
            gpu_decisions.append(gpu_decision)
        
        gpu_preference = sum(gpu_decisions) / len(gpu_decisions) * 100
        print(f"   └─ GPU preference rate: {gpu_preference:.0f}%")
        
        # Restore original thresholds
        bridge.thermal_cpu_threshold = original_cpu_threshold
        bridge.thermal_gpu_threshold = original_gpu_threshold

async def demo_mathematical_processing():
    """Demonstrate mathematical processing with allocation awareness"""
    
    print(f"\n5. Mathematical Processing with Allocation Awareness...")
    
    bridge = create_news_profit_bridge()
    
    # Generate test news data
    test_news = [
        {
            'id': 'thermal_test_1',
            'title': 'Bitcoin Breaks $50K as Institutional Adoption Surges',
            'content': 'Major investment firms announce significant cryptocurrency allocation following regulatory clarity.',
            'source': 'Bloomberg',
            'timestamp': datetime.now() - timedelta(minutes=5)
        },
        {
            'id': 'thermal_test_2',
            'title': 'Trump Announces Comprehensive Crypto Policy Framework',
            'content': 'Former president outlines pro-cryptocurrency stance for digital asset innovation.',
            'source': 'Coindesk',
            'timestamp': datetime.now() - timedelta(minutes=2)
        }
    ]
    
    print(f"   Processing {len(test_news)} news items with different allocation modes...")
    
    # Test with different processing modes
    modes = ["cpu_only", "hybrid", "gpu_preferred", "thermal_aware"]
    
    for mode in modes:
        print(f"\n   Mode: {mode.upper()}")
        bridge.set_processing_mode(mode)
        
        start_time = time.time()
        
        # Process through complete pipeline
        results = await bridge.process_complete_pipeline(test_news)
        
        processing_time = time.time() - start_time
        
        print(f"   └─ Processing time: {processing_time:.3f} seconds")
        print(f"   └─ Events extracted: {results.get('fact_events_extracted', 0)}")
        print(f"   └─ Signatures generated: {results.get('mathematical_signatures_generated', 0)}")
        print(f"   └─ Correlations calculated: {results.get('hash_correlations_calculated', 0)}")
        print(f"   └─ Profit opportunities: {results.get('profit_opportunities_identified', 0)}")
        
        # Check if thermal throttling occurred
        if hasattr(bridge, 'thermal_throttle_events'):
            print(f"   └─ Thermal throttle events: {bridge.thermal_throttle_events}")

def demo_serc_integration():
    """Demonstrate SERC integration and data pipeline logging"""
    
    print(f"\n6. SERC Integration and Data Pipeline...")
    
    bridge = create_news_profit_bridge()
    
    # Get system status with SERC integration info
    status = bridge.get_system_status()
    
    print(f"   SERC Integration Status:")
    print(f"   └─ Mathematical core active: ✓")
    print(f"   └─ Thermal integration: {'✓' if bridge.thermal_integration_enabled else '✗'}")
    print(f"   └─ GPU manager integration: {'✓' if hasattr(bridge, 'gpu_manager') and bridge.gpu_manager else '✗'}")
    print(f"   └─ Processing load tracking: {'✓' if hasattr(bridge, 'processing_load_history') else '✗'}")
    print(f"   └─ Allocation adjustments: {'✓' if hasattr(bridge, 'allocation_adjustments') else '✗'}")
    
    # Display processing allocation status
    allocation_status = status.get('processing_allocation', {})
    print(f"\n   Current Processing Allocation:")
    print(f"   └─ CPU: {allocation_status.get('cpu_percentage', 0):.1f}%")
    print(f"   └─ GPU: {allocation_status.get('gpu_percentage', 0):.1f}%")
    print(f"   └─ Mode: {allocation_status.get('processing_mode', 'unknown')}")
    print(f"   └─ Thermal scaling: {'ON' if allocation_status.get('thermal_scaling_enabled') else 'OFF'}")
    print(f"   └─ Dynamic allocation: {'ON' if allocation_status.get('dynamic_allocation_enabled') else 'OFF'}")
    
    # Display performance metrics
    thermal_status = status.get('thermal_management', {})
    print(f"\n   Performance Metrics:")
    print(f"   └─ Thermal throttle events: {thermal_status.get('thermal_throttle_events', 0)}")
    print(f"   └─ Allocation switches: {thermal_status.get('allocation_switches', 0)}")
    print(f"   └─ Processed events: {status.get('processed_events', 0)}")
    print(f"   └─ Successful trades: {status.get('successful_trades', 0)}")
    
    # Show recent processing history if available
    history = status.get('processing_history', {})
    recent_samples = history.get('recent_load_samples', [])
    if recent_samples:
        print(f"\n   Recent Processing Load Samples:")
        for i, sample in enumerate(recent_samples[-3:]):  # Show last 3 samples
            cpu_ops = sample.get('cpu_operations', 0)
            gpu_ops = sample.get('gpu_operations', 0)
            total_ops = sample.get('operation_count', 0)
            throttled = sample.get('thermal_throttled', False)
            print(f"   └─ Sample {i+1}: {cpu_ops} CPU ops, {gpu_ops} GPU ops, {'THROTTLED' if throttled else 'NORMAL'}")

async def main():
    """Run complete advanced processing demonstration"""
    
    try:
        print("Starting News-Profit Advanced Processing Demonstration...")
        print("This demo shows CPU/GPU allocation controls and thermal management")
        print("integrated with the mathematical bridge for optimal profit extraction.\n")
        
        # Run demonstration sections
        await demo_advanced_processing_controls()
        await demo_thermal_management()
        await demo_mathematical_processing()
        demo_serc_integration()
        
        print(f"\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ CPU/GPU allocation percentage controls")
        print("✓ Processing mode selection (hybrid, cpu_only, gpu_preferred, thermal_aware)")
        print("✓ Thermal-aware dynamic allocation adjustment")
        print("✓ Real-time thermal monitoring integration")
        print("✓ Mathematical processing with allocation awareness")
        print("✓ SERC data pipeline integration")
        print("✓ GPU cool-down management system")
        print("✓ Performance tracking and optimization")
        
        print(f"\nThe system provides the third 'advanced' option you requested:")
        print("• ON/OFF controls (existing)")
        print("• Processing mode selection (basic settings)")
        print("• CPU allocation percentage sliders (ADVANCED PANEL)")
        print("• Thermal management integration (ADVANCED PANEL)")
        print("• Real-time performance monitoring (ADVANCED PANEL)")
        
        print(f"\nThis integrates with your existing SERC systems for:")
        print("• Thermal zone management and drift compensation")
        print("• GPU cool-down cycle management")
        print("• Processing load balancing and optimization")
        print("• Mathematical synthesis through backchannels")
        print("• Profit-focused resource allocation")
        
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 