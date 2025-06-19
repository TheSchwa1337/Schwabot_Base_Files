"""
Optimized Constants Wrapper - Seamless Integration
================================================

This wrapper provides seamless integration of optimized magic numbers throughout
the system while maintaining complete backwards compatibility.

Usage Examples:
    from core.optimized_constants_wrapper import OPTIMIZED_CONSTANTS
    
    # Use exactly like original constants
    if correlation > OPTIMIZED_CONSTANTS.core_thresholds.MIN_HASH_CORRELATION_THRESHOLD:
        # This now uses the mathematically optimized value!
        
    # Or enable optimization dynamically
    OPTIMIZED_CONSTANTS.enable_optimizations()
    
    # Check optimization status
    if OPTIMIZED_CONSTANTS.is_optimized():
        print("Using optimized magic numbers with mathematical enhancement!")
"""

from typing import Any, Dict, Optional
import threading
import time
from dataclasses import dataclass, field

from .system_constants import SYSTEM_CONSTANTS
from .magic_number_optimization_engine import MagicNumberOptimizationEngine
from .zbe_temperature_tensor import ZBETemperatureTensor

class OptimizedConstantsManager:
    """
    Manager that provides optimized constants with seamless fallback to originals
    """
    
    def __init__(self):
        self._optimization_engine: Optional[MagicNumberOptimizationEngine] = None
        self._is_optimized = False
        self._optimization_results: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._last_optimization_time = 0
        self._optimization_cache: Dict[str, float] = {}
        
    def initialize_optimization(self, force_refresh: bool = False) -> bool:
        """Initialize the optimization engine and perform optimization"""
        with self._lock:
            try:
                # Check if we need to refresh
                current_time = time.time()
                if (not force_refresh and 
                    self._optimization_engine and 
                    current_time - self._last_optimization_time < 300):  # 5 minute cache
                    return True
                
                # Create ZBE tensor and optimization engine
                zbe_tensor = ZBETemperatureTensor()
                self._optimization_engine = MagicNumberOptimizationEngine(zbe_tensor)
                
                # Perform optimization
                print("🔥 Initializing magic number optimization...")
                self._optimization_results = self._optimization_engine.optimize_all_categories()
                
                # Update cache
                self._optimization_cache.clear()
                for context, factor in self._optimization_engine.active_optimizations.items():
                    self._optimization_cache[context] = factor.optimized_value
                
                self._is_optimized = True
                self._last_optimization_time = current_time
                
                # Print summary
                total_optimized = sum(r.total_constants_optimized for r in self._optimization_results.values())
                avg_improvement = sum(r.average_improvement for r in self._optimization_results.values()) / len(self._optimization_results)
                
                print(f"✅ Optimization complete: {total_optimized} constants optimized")
                print(f"📈 Average improvement: {avg_improvement:.1%}")
                
                return True
                
            except Exception as e:
                print(f"❌ Optimization initialization failed: {e}")
                self._is_optimized = False
                return False
    
    def enable_optimizations(self, force_refresh: bool = False) -> bool:
        """Enable optimized constants"""
        return self.initialize_optimization(force_refresh)
    
    def disable_optimizations(self):
        """Disable optimizations, fall back to original constants"""
        with self._lock:
            self._is_optimized = False
            print("⚪ Optimizations disabled, using original constants")
    
    def is_optimized(self) -> bool:
        """Check if optimizations are currently active"""
        return self._is_optimized
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics"""
        if not self._is_optimized or not self._optimization_results:
            return {
                'optimized': False,
                'message': 'Optimizations not active'
            }
        
        total_constants = sum(r.total_constants_optimized for r in self._optimization_results.values())
        avg_improvement = sum(r.average_improvement for r in self._optimization_results.values()) / len(self._optimization_results)
        
        return {
            'optimized': True,
            'total_constants_optimized': total_constants,
            'categories_optimized': len(self._optimization_results),
            'average_improvement': avg_improvement,
            'last_optimization_time': self._last_optimization_time,
            'optimization_results': self._optimization_results
        }
    
    def get_optimized_value(self, category: str, attribute: str) -> float:
        """Get optimized value for a specific constant"""
        if not self._is_optimized:
            # Fall back to original value
            category_mapping = {
                'Core System Thresholds': 'core',
                'Performance Constants': 'performance', 
                'Visualization Constants': 'visualization',
                'Trading Constants': 'trading',
                'Mathematical Constants': 'mathematical',
                'Thermal Constants': 'thermal',
                'Fault Detection Constants': 'fault_detection',
                'Intelligent Thresholds': 'intelligent',
                'Phase Gate Constants': 'phase_gate',
                'Sustainment Constants': 'sustainment',
                'Profit Routing Constants': 'profit_routing',
                'Configuration Ranges': 'configuration'
            }
            
            attr_name = category_mapping.get(category)
            if attr_name:
                category_obj = getattr(SYSTEM_CONSTANTS, attr_name, None)
                if category_obj:
                    return getattr(category_obj, attribute, 0.0)
            return 0.0
        
        # Try to get optimized value
        context = f"{category}.{attribute}"
        if context in self._optimization_cache:
            return self._optimization_cache[context]
        
        # Fall back to original if optimization not found
        category_mapping = {
            'Core System Thresholds': 'core',
            'Performance Constants': 'performance', 
            'Visualization Constants': 'visualization',
            'Trading Constants': 'trading',
            'Mathematical Constants': 'mathematical',
            'Thermal Constants': 'thermal',
            'Fault Detection Constants': 'fault_detection',
            'Intelligent Thresholds': 'intelligent',
            'Phase Gate Constants': 'phase_gate',
            'Sustainment Constants': 'sustainment',
            'Profit Routing Constants': 'profit_routing',
            'Configuration Ranges': 'configuration'
        }
        
        attr_name = category_mapping.get(category)
        if attr_name:
            category_obj = getattr(SYSTEM_CONSTANTS, attr_name, None)
            if category_obj:
                return getattr(category_obj, attribute, 0.0)
        return 0.0

class OptimizedConstantCategory:
    """Dynamic category that provides optimized values"""
    
    def __init__(self, category_name: str, original_category: Any, manager: OptimizedConstantsManager):
        self._category_name = category_name
        self._original_category = original_category
        self._manager = manager
    
    def __getattr__(self, name: str) -> Any:
        # Check if it's a constant (float/int attribute)
        if hasattr(self._original_category, name):
            original_value = getattr(self._original_category, name)
            if isinstance(original_value, (int, float)):
                # Return optimized value if available
                return self._manager.get_optimized_value(self._category_name, name)
        
        # Fall back to original attribute
        return getattr(self._original_category, name)

class OptimizedSystemConstants:
    """
    Drop-in replacement for SYSTEM_CONSTANTS with optimization capabilities
    """
    
    def __init__(self):
        self._manager = OptimizedConstantsManager()
        
        # Create optimized category wrappers
        self.core_thresholds = OptimizedConstantCategory(
            "Core System Thresholds", 
            SYSTEM_CONSTANTS.core, 
            self._manager
        )
        
        self.performance = OptimizedConstantCategory(
            "Performance Constants",
            SYSTEM_CONSTANTS.performance,
            self._manager
        )
        
        self.visualization = OptimizedConstantCategory(
            "Visualization Constants",
            SYSTEM_CONSTANTS.visualization,
            self._manager
        )
        
        self.trading = OptimizedConstantCategory(
            "Trading Constants",
            SYSTEM_CONSTANTS.trading,
            self._manager
        )
        
        self.mathematical = OptimizedConstantCategory(
            "Mathematical Constants",
            SYSTEM_CONSTANTS.mathematical,
            self._manager
        )
        
        self.thermal = OptimizedConstantCategory(
            "Thermal Constants",
            SYSTEM_CONSTANTS.thermal,
            self._manager
        )
        
        self.fault_detection = OptimizedConstantCategory(
            "Fault Detection Constants",
            SYSTEM_CONSTANTS.fault_detection,
            self._manager
        )
        
        self.intelligent_thresholds = OptimizedConstantCategory(
            "Intelligent Thresholds",
            SYSTEM_CONSTANTS.intelligent,
            self._manager
        )
        
        self.phase_gate = OptimizedConstantCategory(
            "Phase Gate Constants",
            SYSTEM_CONSTANTS.phase_gate,
            self._manager
        )
        
        self.sustainment = OptimizedConstantCategory(
            "Sustainment Constants",
            SYSTEM_CONSTANTS.sustainment,
            self._manager
        )
        
        self.profit_routing = OptimizedConstantCategory(
            "Profit Routing Constants",
            SYSTEM_CONSTANTS.profit_routing,
            self._manager
        )
        
        self.configuration = OptimizedConstantCategory(
            "Configuration Ranges",
            SYSTEM_CONSTANTS.configuration,
            self._manager
        )
    
    def enable_optimizations(self, force_refresh: bool = False) -> bool:
        """Enable mathematical optimizations for all constants"""
        return self._manager.enable_optimizations(force_refresh)
    
    def disable_optimizations(self):
        """Disable optimizations, use original constants"""
        self._manager.disable_optimizations()
    
    def is_optimized(self) -> bool:
        """Check if optimizations are currently active"""
        return self._manager.is_optimized()
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get optimization status and performance statistics"""
        return self._manager.get_optimization_status()
    
    def print_optimization_report(self):
        """Print comprehensive optimization report"""
        if self._manager._optimization_engine:
            report = self._manager._optimization_engine.generate_optimization_report()
            print(report)
        else:
            print("❌ No optimization engine available. Call enable_optimizations() first.")

# Global optimized constants instance
OPTIMIZED_CONSTANTS = OptimizedSystemConstants()

# Convenience functions for easy integration
def enable_magic_number_optimizations(force_refresh: bool = False) -> bool:
    """
    Enable magic number optimizations globally
    
    Args:
        force_refresh: Force refresh of optimizations even if cached
        
    Returns:
        bool: True if optimization successful
    """
    return OPTIMIZED_CONSTANTS.enable_optimizations(force_refresh)

def disable_magic_number_optimizations():
    """Disable magic number optimizations globally"""
    OPTIMIZED_CONSTANTS.disable_optimizations()

def get_magic_number_optimization_status() -> Dict[str, Any]:
    """Get current optimization status"""
    return OPTIMIZED_CONSTANTS.get_optimization_status()

def print_magic_number_optimization_report():
    """Print comprehensive optimization report"""
    OPTIMIZED_CONSTANTS.print_optimization_report()

# Auto-enable optimizations on import (can be disabled if needed)
# Disabled by default for stability - call enable_magic_number_optimizations() manually
# try:
#     if enable_magic_number_optimizations():
#         print("🌟 Magic number optimizations enabled automatically!")
#         print("   Use OPTIMIZED_CONSTANTS instead of SYSTEM_CONSTANTS for enhanced performance.")
#         print("   Call disable_magic_number_optimizations() to revert to original values.")
# except Exception as e:
#     print(f"⚠️ Auto-optimization failed: {e}")
#     print("   Call enable_magic_number_optimizations() manually when ready.")

print("💡 Magic Number Optimization System Ready!")
print("   Call enable_magic_number_optimizations() to activate mathematical enhancements.")
print("   Use OPTIMIZED_CONSTANTS.enable_optimizations() for per-instance control.") 