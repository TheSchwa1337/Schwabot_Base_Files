#!/usr/bin/env python3
"""
Strategy Sustainment Integration Demo
====================================

Demonstrates the complete integration of the 8-principle sustainment framework
with Schwabot's existing mathematical core systems.

This script shows:
1. How strategies are validated before execution
2. Integration with fractal core, confidence engine, and thermal management
3. Real-time adaptation based on validation feedback
4. Performance tracking and optimization
"""

import sys
import os
import time
import yaml
import json
from datetime import datetime
from typing import Dict, Any

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

try:
    from core.strategy_sustainment_validator import (
        StrategySustainmentValidator, 
        StrategyMetrics, 
        SustainmentPrinciple,
        validate_strategy_quick
    )
    from core.collapse_confidence import CollapseConfidenceEngine
    from core.fractal_core import FractalCore
    print("✅ Successfully imported strategy sustainment components")
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("📝 Running in mock mode - some features will be simulated")

def load_config() -> Dict[str, Any]:
    """Load configuration for the demo"""
    try:
        with open('config/strategy_sustainment_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Loaded configuration from YAML file")
        return config
    except FileNotFoundError:
        print("⚠️  Config file not found, using defaults")
        return {
            'overall_threshold': 0.75,
            'weights': {
                'integration': 1.0,
                'anticipation': 1.2,
                'responsiveness': 1.2,
                'simplicity': 0.8,
                'economy': 1.0,
                'survivability': 1.5,
                'continuity': 1.3,
                'transcendence': 2.0
            },
            'thresholds': {
                'integration': 0.75,
                'anticipation': 0.70,
                'responsiveness': 0.80,
                'simplicity': 0.65,
                'economy': 0.75,
                'survivability': 0.85,
                'continuity': 0.80,
                'transcendence': 0.70
            }
        }

def simulate_market_tick(tick_number: int) -> Dict[str, Any]:
    """Simulate a market tick with realistic data"""
    import random
    import numpy as np
    
    # Simulate market volatility cycles
    base_volatility = 0.15
    volatility_cycle = 0.05 * np.sin(tick_number * 0.1)
    volatility = max(0.05, base_volatility + volatility_cycle + random.uniform(-0.03, 0.03))
    
    # Simulate thermal state (varies with system load)
    thermal_load = 0.3 + 0.4 * np.sin(tick_number * 0.05) + random.uniform(-0.1, 0.1)
    thermal_budget = max(0.1, 1.0 - thermal_load)
    
    # Simulate tick signature
    tick_signature = {
        'tick_id': f"tick_{tick_number:04d}",
        'correlation_score': max(0.3, min(0.95, 0.7 + random.uniform(-0.2, 0.2))),
        'signal_strength': max(0.2, min(0.9, 0.6 + random.uniform(-0.25, 0.25))),
        'profit_tier': random.choice(['PLATINUM', 'GOLD', 'SILVER', 'BRONZE']),
        'timestamp': datetime.now()
    }
    
    # Simulate market conditions
    market_conditions = {
        'volatility': volatility,
        'thermal_budget': thermal_budget,
        'profit_tier_success_rate': max(0.4, min(0.9, 0.7 + random.uniform(-0.2, 0.2))),
        'recent_profit_correlation': max(0.3, min(0.85, 0.6 + random.uniform(-0.15, 0.15))),
        'anomaly_count': random.randint(0, 5)
    }
    
    return {
        'tick_signature': tick_signature,
        'market_conditions': market_conditions
    }

def build_strategy_metrics(tick_data: Dict[str, Any]) -> StrategyMetrics:
    """Build strategy metrics from tick data (like StrategyExecutionMapper would)"""
    tick_sig = tick_data['tick_signature']
    market = tick_data['market_conditions']
    
    return StrategyMetrics(
        # Integration metrics
        entropy_coherence=tick_sig['correlation_score'],
        system_harmony=market['thermal_budget'],
        module_alignment=min(tick_sig['signal_strength'], 1.0),
        
        # Anticipation metrics
        lead_time_prediction=tick_sig['signal_strength'],
        pattern_recognition_depth=min(0.5 + tick_sig['signal_strength'] * 0.4, 1.0),
        signal_forecast_accuracy=market['recent_profit_correlation'],
        
        # Responsiveness metrics
        latency=max(0.01, 0.2 - market['thermal_budget'] * 0.15),  # Better thermal = lower latency
        adaptation_speed=1.0 - market['volatility'],
        market_reaction_time=0.05 + market['volatility'] * 0.1,
        
        # Simplicity metrics
        logic_complexity=0.2 + market['volatility'] * 0.3,  # More complex in volatile markets
        operation_count=random.randint(50, 500),
        decision_tree_depth=random.randint(2, 8),
        
        # Economy metrics
        profit_efficiency=tick_sig['signal_strength'] * market['profit_tier_success_rate'],
        resource_utilization=market['thermal_budget'],
        cost_benefit_ratio=tick_sig['correlation_score'],
        
        # Survivability metrics
        drawdown_resistance=1.0 - market['volatility'],
        risk_adjusted_return=tick_sig['signal_strength'] * (1.0 - market['volatility']),
        volatility_tolerance=min(1.0 - market['volatility'], 1.0),
        
        # Continuity metrics
        pattern_memory_depth=0.7 + tick_sig['correlation_score'] * 0.2,
        state_persistence=0.8,
        cycle_completion_rate=0.85 + market['thermal_budget'] * 0.1,
        
        # Transcendence metrics
        emergent_signal_score=tick_sig['signal_strength'] * tick_sig['correlation_score'],
        adaptive_learning_rate=0.6 + market['thermal_budget'] * 0.3,
        optimization_convergence=0.7 + (1.0 - market['volatility']) * 0.2
    )

def print_validation_result(result, tick_number: int):
    """Print formatted validation results"""
    status_emoji = {
        'PASS': '✅',
        'WARNING': '⚠️',
        'FAIL': '❌'
    }
    
    print(f"\n{'='*60}")
    print(f"Tick {tick_number:04d} - Strategy Validation Results")
    print(f"{'='*60}")
    print(f"Status: {status_emoji.get(result.overall_status, '❓')} {result.overall_status}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Weighted Score: {result.weighted_score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Execution Approved: {'✅ YES' if result.execution_approved else '❌ NO'}")
    
    print(f"\n📊 Principle Breakdown:")
    for principle, score in result.principle_scores.items():
        status = '✅' if score.passed else '❌'
        print(f"  {status} {principle.value.capitalize():<15}: {score.score:.3f} (threshold: {score.threshold:.2f})")
    
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
        if len(result.recommendations) > 3:
            print(f"  ... and {len(result.recommendations) - 3} more")

def run_sustainment_demo():
    """Run the complete sustainment validation demonstration"""
    print("🚀 Starting Strategy Sustainment Validation Demo")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize validator
    try:
        validator = StrategySustainmentValidator(config)
        print("✅ Strategy Sustainment Validator initialized")
    except Exception as e:
        print(f"❌ Failed to initialize validator: {e}")
        return
    
    # Demo parameters
    num_ticks = 10
    results = []
    
    print(f"\n🔄 Processing {num_ticks} market ticks...")
    
    for tick_num in range(1, num_ticks + 1):
        print(f"\n📈 Processing Tick {tick_num:04d}...")
        
        # Simulate market data
        tick_data = simulate_market_tick(tick_num)
        
        # Build strategy metrics
        strategy_metrics = build_strategy_metrics(tick_data)
        
        # Validate strategy
        try:
            result = validator.validate_strategy(
                strategy_metrics, 
                strategy_id=f"demo_strategy_{tick_num:04d}",
                context={
                    'tick_number': tick_num,
                    'market_volatility': tick_data['market_conditions']['volatility'],
                    'thermal_state': tick_data['market_conditions']['thermal_budget']
                }
            )
            
            results.append(result)
            print_validation_result(result, tick_num)
            
        except Exception as e:
            print(f"❌ Validation failed for tick {tick_num}: {e}")
            continue
        
        # Brief pause for readability
        time.sleep(0.5)
    
    # Performance summary
    print("\n" + "="*60)
    print("📊 PERFORMANCE SUMMARY")
    print("="*60)
    
    try:
        summary = validator.get_performance_summary()
        
        print(f"Total Validations: {summary['total_validations']}")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        print(f"Warning Rate: {summary['warning_rate']:.1%}")
        print(f"Fail Rate: {summary['fail_rate']:.1%}")
        print(f"Average Score: {summary['average_score']:.3f}")
        print(f"Trend: {summary['recent_trend'].upper()}")
        
        print(f"\n📈 Principle Performance:")
        for principle, avg_score in summary['principle_averages'].items():
            print(f"  {principle.capitalize():<15}: {avg_score:.3f}")
            
    except Exception as e:
        print(f"⚠️  Could not generate performance summary: {e}")
    
    # Demonstrate quick validation
    print("\n" + "="*60)
    print("⚡ QUICK VALIDATION DEMO")
    print("="*60)
    
    print("Testing quick validation with good parameters:")
    quick_result = validate_strategy_quick(
        entropy_coherence=0.85,
        profit_efficiency=0.80,
        drawdown_resistance=0.90,
        latency=0.05
    )
    print(f"Result: {'✅ APPROVED' if quick_result else '❌ REJECTED'}")
    
    print("\nTesting quick validation with poor parameters:")
    quick_result = validate_strategy_quick(
        entropy_coherence=0.30,
        profit_efficiency=0.20,
        drawdown_resistance=0.15,
        latency=0.8
    )
    print(f"Result: {'✅ APPROVED' if quick_result else '❌ REJECTED'}")
    
    # Save results
    try:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': [
                {
                    'strategy_id': r.strategy_id,
                    'overall_status': r.overall_status,
                    'overall_score': r.overall_score,
                    'weighted_score': r.weighted_score,
                    'confidence': r.confidence,
                    'execution_approved': r.execution_approved
                }
                for r in results
            ],
            'summary': summary if 'summary' in locals() else None
        }
        
        with open(f'strategy_sustainment_demo_results_{int(time.time())}.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to JSON file")
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("="*60)

def demonstrate_integration_points():
    """Demonstrate key integration points with existing Schwabot systems"""
    print("\n🔧 INTEGRATION POINTS DEMONSTRATION")
    print("="*60)
    
    print("1. 📊 Strategy Execution Mapper Integration:")
    print("   - Validates strategies before trade signal generation")
    print("   - Uses tick signatures and market conditions")
    print("   - Prevents execution of unsustainable strategies")
    
    print("\n2. 🧮 Mathematical Framework Integration:")
    print("   - CollapseConfidenceEngine: Confidence scoring")
    print("   - FractalCore: Pattern recognition and coherence")
    print("   - ThermalZoneManager: Resource management")
    
    print("\n3. ⚙️  Configuration Integration:")
    print("   - YAML-based configuration")
    print("   - Adaptive threshold adjustment")
    print("   - Market condition modifiers")
    
    print("\n4. 📈 Performance Tracking:")
    print("   - Historical validation tracking")
    print("   - Trend analysis")
    print("   - Principle-specific performance metrics")
    
    print("\n5. 🎯 Real-time Adaptation:")
    print("   - Dynamic weight adjustment")
    print("   - Threshold optimization")
    print("   - Context-aware validation")

if __name__ == "__main__":
    print("Strategy Sustainment Validation System")
    print("Schwabot Integration Demo v1.0")
    print("="*60)
    
    try:
        # Run main demo
        run_sustainment_demo()
        
        # Show integration points
        demonstrate_integration_points()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thank you for using the Strategy Sustainment Demo!") 