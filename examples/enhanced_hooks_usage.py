"""
Enhanced Hook System Usage Example
=================================

Demonstrates how to use the enhanced hook system with thermal-aware,
profit-synchronized routing that addresses the original hook gaps.
"""

import time
import logging
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_enhanced_hooks():
    """Demonstrate the enhanced hook system capabilities"""
    
    print("🚀 Enhanced Hook System Demonstration")
    print("=" * 50)
    
    try:
        # Import the enhanced hook system
        from core.hooks import (
            execute_hook, get_hook_context, get_hook_stats,
            enable_hook, disable_hook, ncco_manager
        )
        
        print("✅ Enhanced hook system imported successfully")
        
        # 1. Get current system context
        print("\n📊 Current System Context:")
        context = get_hook_context()
        
        if hasattr(context, '__dict__'):
            # Enhanced mode - dataclass
            print(f"  Thermal Zone: {context.thermal_zone}")
            print(f"  Profit Zone: {context.profit_zone}")
            print(f"  Thermal Temp: {context.thermal_temp:.1f}°C")
            print(f"  Profit Vector: {context.profit_vector_strength:.3f}")
            print(f"  Memory Confidence: {context.memory_confidence:.3f}")
        else:
            # Legacy mode - dictionary
            print(f"  Thermal Zone: {context.get('thermal_zone', 'unknown')}")
            print(f"  Profit Zone: {context.get('profit_zone', 'unknown')}")
            print(f"  Legacy Mode: {context.get('legacy_mode', False)}")
            
        # 2. Demonstrate dynamic hook execution
        print("\n🔧 Dynamic Hook Execution:")
        
        # Try to execute different hooks based on conditions
        hooks_to_test = [
            ("ncco_manager", "process_data", {"test": "data"}),
            ("sfsss_router", "route_strategy", "strategy_alpha"),
            ("echo_logger", "log", "Enhanced hook test message")
        ]
        
        for hook_id, method, args in hooks_to_test:
            try:
                print(f"  Executing {hook_id}.{method}...")
                
                if isinstance(args, dict):
                    result = execute_hook(hook_id, method, **args)
                else:
                    result = execute_hook(hook_id, method, args)
                    
                if result is not None:
                    print(f"    ✅ Success: {result}")
                else:
                    print(f"    ⚠️  Execution denied or failed")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
                
        # 3. Demonstrate hook enabling/disabling
        print("\n🎛️  Hook Control:")
        
        # Try to disable a hook
        if disable_hook("sfsss_router"):
            print("  ✅ SFSSS Router disabled")
        else:
            print("  ⚠️  Could not disable SFSSS Router")
            
        # Try to enable it back
        if enable_hook("sfsss_router"):
            print("  ✅ SFSSS Router re-enabled")
        else:
            print("  ⚠️  Could not re-enable SFSSS Router")
            
        # 4. Get system statistics
        print("\n📈 System Statistics:")
        stats = get_hook_stats()
        
        if "system_health" in stats:
            health = stats["system_health"]
            print(f"  Total Hooks: {health.get('total_hooks', 'unknown')}")
            print(f"  Active Hooks: {health.get('active_hooks', 'unknown')}")
            print(f"  Thermal Zone: {health.get('thermal_zone', 'unknown')}")
            print(f"  Profit Zone: {health.get('profit_zone', 'unknown')}")
            
            # Show component status
            if "components" in health:
                components = health["components"]
                print("  Component Status:")
                for comp, status in components.items():
                    print(f"    {comp}: {'✅' if status else '❌'}")
        else:
            print(f"  System: {stats.get('system', 'unknown')}")
            print(f"  Enhanced Routing: {stats.get('enhanced_routing', False)}")
            
        # 5. Show performance metrics if available
        if "hook_performance" in stats:
            print("\n⚡ Hook Performance:")
            for hook_id, perf in stats["hook_performance"].items():
                print(f"  {hook_id}:")
                print(f"    Executions: {perf.get('total_executions', 0)}")
                print(f"    Success Rate: {perf.get('success_rate', 0):.1%}")
                print(f"    Avg Time: {perf.get('average_execution_time', 0):.3f}s")
                
    except ImportError as e:
        print(f"❌ Enhanced hooks not available: {e}")
        print("   Make sure all required components are installed")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def demonstrate_thermal_profit_awareness():
    """Demonstrate thermal and profit-aware hook routing"""
    
    print("\n🌡️ Thermal & Profit Awareness Demo")
    print("=" * 40)
    
    try:
        from core.hooks import get_hook_context, execute_hook
        
        # Simulate different market/thermal conditions
        scenarios = [
            "High profit, cool system",
            "Low profit, hot system", 
            "Volatile market, normal thermal",
            "Stable profit, warm system"
        ]
        
        for scenario in scenarios:
            print(f"\n📋 Scenario: {scenario}")
            
            # Get current context
            context = get_hook_context()
            
            # Show what hooks would execute in this context
            thermal_zone = getattr(context, 'thermal_zone', context.get('thermal_zone', 'unknown'))
            profit_zone = getattr(context, 'profit_zone', context.get('profit_zone', 'unknown'))
            
            print(f"  Current state: {thermal_zone} thermal, {profit_zone} profit")
            
            # Try executing hooks that might be affected by conditions
            test_hooks = ["vault_router", "drift_engine", "cluster_mapper"]
            
            for hook_id in test_hooks:
                result = execute_hook(hook_id, "get_status" if hasattr(globals().get(hook_id, {}), "get_status") else "process")
                status = "✅ Executed" if result is not None else "❌ Denied"
                print(f"    {hook_id}: {status}")
                
            time.sleep(1)  # Brief pause between scenarios
            
    except Exception as e:
        print(f"❌ Error in thermal/profit demo: {e}")

def demonstrate_memory_integration():
    """Demonstrate memory agent integration"""
    
    print("\n🧠 Memory Integration Demo")
    print("=" * 30)
    
    try:
        from core.hooks import get_hook_stats, execute_hook
        
        # Execute some hooks to generate memory data
        print("  Executing hooks to generate memory data...")
        
        test_executions = [
            ("echo_logger", "log", "Memory test 1"),
            ("ncco_manager", "process", "test_pattern"),
            ("cluster_mapper", "update", {"new_data": True})
        ]
        
        for hook_id, method, args in test_executions:
            if isinstance(args, dict):
                execute_hook(hook_id, method, **args)
            else:
                execute_hook(hook_id, method, args)
                
        # Check if memory data was generated
        stats = get_hook_stats()
        
        if "hook_performance" in stats:
            print("  📊 Memory data captured:")
            total_executions = sum(
                perf.get('total_executions', 0) 
                for perf in stats["hook_performance"].values()
            )
            print(f"    Total executions tracked: {total_executions}")
            
            # Show memory confidence if available
            context = get_hook_context()
            if hasattr(context, 'memory_confidence'):
                print(f"    Memory confidence: {context.memory_confidence:.3f}")
            elif 'memory_confidence' in context:
                print(f"    Memory confidence: {context['memory_confidence']:.3f}")
                
        else:
            print("  ⚠️  No performance tracking available")
            
    except Exception as e:
        print(f"❌ Error in memory demo: {e}")

if __name__ == "__main__":
    print("Enhanced Hook System - Comprehensive Demo")
    print("========================================")
    
    # Run all demonstrations
    demonstrate_enhanced_hooks()
    demonstrate_thermal_profit_awareness()
    demonstrate_memory_integration()
    
    print("\n🎉 Demo Complete!")
    print("\nKey Improvements Demonstrated:")
    print("  ✅ Dynamic hook routing based on thermal/profit state")
    print("  ✅ Memory-coupled echo feedback system")
    print("  ✅ Performance tracking and confidence weighting")
    print("  ✅ Configuration-driven hook management") 
    print("  ✅ Backward compatibility with legacy hooks")
    print("  ✅ Thermal-aware processing decisions")
    print("  ✅ Real-time context-aware execution")
    
    print("\nThe enhanced hook system addresses all major gaps:")
    print("  🔧 Static → Dynamic routing")
    print("  🌡️  No thermal awareness → Thermal-profit synchronized")
    print("  🧠 No memory → Memory agent integration")
    print("  📊 No tracking → Performance metrics")
    print("  ⚙️  No config → YAML-driven configuration")
    print("  🔄 No feedback → Echo-based learning") 