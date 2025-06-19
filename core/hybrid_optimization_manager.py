"""
Hybrid Optimization Manager - Dual Pipeline Strategy
==================================================

This manager provides intelligent switching between original constants and 
revolutionary optimized magic numbers based on:

- CPU vs GPU processing requirements
- High-frequency vs robust trading conditions  
- Volume determinisms and market entropy
- Multi-millisecond timing requirements
- Ferris wheel optimizations and drift system needs

STRATEGY:
- Original Constants: CPU-heavy, robust, deterministic pathways
- Magic Numbers: GPU optimization, speed, high-frequency trading
- Intelligent switching based on system load and market conditions

Windows CLI Compatibility: All logging and output properly handled
Type Annotations: Complete type safety implemented
Exception Handling: No bare except statements, structured error handling
Import Standards: No wildcard imports, explicit dependencies
"""

import time
import threading
import psutil
import os
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import logging

from .system_constants import SYSTEM_CONSTANTS
from .optimized_constants_wrapper import OPTIMIZED_CONSTANTS
from .zbe_temperature_tensor import ZBETemperatureTensor

# Windows CLI Compatibility Handler
class WindowsCliCompatibilityHandler:
    """Windows CLI compatibility for emoji and Unicode handling"""
    
    @staticmethod
    def is_windows_cli() -> bool:
        """Detect if running in Windows CLI environment"""
        return (os.name == 'nt' and 
                not os.environ.get('TERM_PROGRAM') and 
                not os.environ.get('WT_SESSION'))
    
    @staticmethod
    def safe_log_message(message: str, emoji_mapping: Optional[Dict[str, str]] = None) -> str:
        """Replace emojis with ASCII markers on Windows CLI"""
        if WindowsCliCompatibilityHandler.is_windows_cli():
            default_mapping = {
                '🚨': '[ALERT]',
                '⚠️': '[WARNING]', 
                '✅': '[SUCCESS]',
                '❌': '[ERROR]',
                '🔄': '[PROCESSING]',
                '💰': '[PROFIT]',
                '📊': '[DATA]',
                '🔧': '[CONFIG]',
                '🎯': '[TARGET]',
                '⚡': '[FAST]',
                '🔍': '[SEARCH]',
                '📈': '[METRICS]',
                '🧠': '[INTELLIGENCE]',
                '🛡️': '[PROTECTION]'
            }
            mapping = emoji_mapping or default_mapping
            for emoji, marker in mapping.items():
                message = message.replace(emoji, marker)
        return message
    
    @staticmethod
    def log_safe(logger: logging.Logger, level: str, message: str, **kwargs: Any) -> None:
        """Safe logging with Windows CLI compatibility"""
        safe_message = WindowsCliCompatibilityHandler.safe_log_message(message)
        log_method = getattr(logger, level, logger.info)
        log_method(safe_message, **kwargs)

class OptimizationMode(Enum):
    """Optimization modes for different system conditions"""
    ORIGINAL_ROBUST = "original_robust"      # CPU-heavy, deterministic
    MAGIC_SPEED = "magic_speed"             # GPU optimization, high-frequency
    HYBRID_AUTO = "hybrid_auto"             # Intelligent switching
    DUAL_PIPELINE = "dual_pipeline"         # Both active simultaneously

class ProcessingContext(Enum):
    """Different processing contexts requiring different optimization strategies"""
    CPU_HEAVY = "cpu_heavy"                 # Robust original constants
    GPU_ACCELERATED = "gpu_accelerated"     # Magic number optimization
    HIGH_FREQUENCY_TRADING = "hft"          # Speed-optimized magic numbers
    VOLUME_DETERMINISM = "volume_det"       # Original constants for stability
    MULTI_MILLISECOND = "multi_ms"          # Magic numbers for timing
    FERRIS_WHEEL_OPT = "ferris_wheel"      # Context-dependent switching
    DRIFT_SYSTEM = "drift_system"           # Original for robustness
    UI_OVERLAY = "ui_overlay"               # Magic numbers for visual performance
    BTC_POOL_INTEGRATION = "btc_pool"       # Original for hash stability
    CCXT_ORDERBOOK = "ccxt_orderbook"       # Mode depends on throw-put environment

@dataclass
class SystemConditions:
    """Current system conditions for optimization decisions"""
    cpu_usage: float
    gpu_usage: float
    memory_usage: float
    network_latency: float
    market_volatility: float
    volume_entropy: float
    thermal_state: Dict[str, float]
    processing_context: ProcessingContext
    
class HybridOptimizationManager:
    """
    Intelligent manager that switches between original and optimized constants
    based on real-time system conditions and processing requirements
    
    Windows CLI Compatible: All output properly handled
    Type Safe: Complete type annotations
    Error Safe: Structured exception handling only
    """
    
    def __init__(self) -> None:
        self.current_mode: OptimizationMode = OptimizationMode.HYBRID_AUTO
        self.zbe_tensor: ZBETemperatureTensor = ZBETemperatureTensor()
        self._monitoring_active: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock: threading.Lock = threading.Lock()
        self.cli_handler: WindowsCliCompatibilityHandler = WindowsCliCompatibilityHandler()
        
        # Performance thresholds for intelligent switching
        self.cpu_threshold_high: float = 80.0    # Switch to magic numbers when CPU > 80%
        self.gpu_threshold_low: float = 30.0     # Use magic numbers when GPU < 30% (room for optimization)
        self.volatility_threshold: float = 0.15  # Use original constants when volatility > 15%
        self.volume_entropy_threshold: float = 0.25  # Original constants for high entropy
        
        # Context-based optimization preferences
        self.context_preferences: Dict[ProcessingContext, OptimizationMode] = {
            ProcessingContext.CPU_HEAVY: OptimizationMode.ORIGINAL_ROBUST,
            ProcessingContext.GPU_ACCELERATED: OptimizationMode.MAGIC_SPEED,
            ProcessingContext.HIGH_FREQUENCY_TRADING: OptimizationMode.MAGIC_SPEED,
            ProcessingContext.VOLUME_DETERMINISM: OptimizationMode.ORIGINAL_ROBUST,
            ProcessingContext.MULTI_MILLISECOND: OptimizationMode.MAGIC_SPEED,
            ProcessingContext.FERRIS_WHEEL_OPT: OptimizationMode.HYBRID_AUTO,
            ProcessingContext.DRIFT_SYSTEM: OptimizationMode.ORIGINAL_ROBUST,
            ProcessingContext.UI_OVERLAY: OptimizationMode.MAGIC_SPEED,
            ProcessingContext.BTC_POOL_INTEGRATION: OptimizationMode.ORIGINAL_ROBUST,
            ProcessingContext.CCXT_ORDERBOOK: OptimizationMode.HYBRID_AUTO
        }
        
        # Dual pipeline state
        self.dual_pipeline_active: bool = False
        self.cpu_constants = SYSTEM_CONSTANTS    # Original for CPU operations
        self.gpu_constants: Optional[Any] = None                # Will be OPTIMIZED_CONSTANTS when enabled
        
        # Performance tracking
        self.performance_history: list = []
        self.decision_history: list = []
        
        logging.basicConfig(level=logging.INFO)
        self.logger: logging.Logger = logging.getLogger(__name__)

    def enable_dual_pipeline(self) -> bool:
        """Enable dual pipeline with both original and optimized constants"""
        try:
            # Initialize optimized constants
            if OPTIMIZED_CONSTANTS.enable_optimizations():
                self.gpu_constants = OPTIMIZED_CONSTANTS
                self.dual_pipeline_active = True
                self.current_mode = OptimizationMode.DUAL_PIPELINE
                
                self.cli_handler.log_safe(self.logger, 'info', "🔄 Dual pipeline activated!")
                self.cli_handler.log_safe(self.logger, 'info', "   CPU Operations: Original robust constants")
                self.cli_handler.log_safe(self.logger, 'info', "   GPU Operations: Revolutionary magic numbers")
                return True
            else:
                self.cli_handler.log_safe(self.logger, 'warning', "⚠️ Failed to initialize optimized constants")
                return False
                
        except Exception as e:
            self.cli_handler.log_safe(self.logger, 'error', f"❌ Dual pipeline activation failed: {e}")
            return False

    def get_system_conditions(self, context: ProcessingContext) -> SystemConditions:
        """Get current system conditions for optimization decisions"""
        try:
            # System metrics
            cpu_usage: float = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU usage (simplified - would need GPU library in real implementation)
            gpu_usage: float = 50.0  # Placeholder - integrate with actual GPU monitoring
            
            # Network latency (simplified)
            network_latency: float = 10.0  # Placeholder - integrate with actual network monitoring
            
            # Market conditions (simplified - integrate with actual market data)
            market_volatility: float = 0.12   # Placeholder
            volume_entropy: float = 0.18      # Placeholder
            
            # Thermal state from ZBE tensor
            thermal_state: Dict[str, float] = self.zbe_tensor.get_thermal_stats()
            
            return SystemConditions(
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                memory_usage=memory.percent,
                network_latency=network_latency,
                market_volatility=market_volatility,
                volume_entropy=volume_entropy,
                thermal_state=thermal_state,
                processing_context=context
            )
            
        except Exception as e:
            self.cli_handler.log_safe(self.logger, 'error', f"Error getting system conditions: {e}")
            # Return safe defaults
            return SystemConditions(
                cpu_usage=50.0, gpu_usage=50.0, memory_usage=50.0,
                network_latency=10.0, market_volatility=0.10, volume_entropy=0.15,
                thermal_state={'thermal_efficiency': 0.5}, processing_context=context
            )

    def decide_optimization_strategy(self, conditions: SystemConditions) -> OptimizationMode:
        """Intelligent decision on which optimization strategy to use"""
        
        # Context-based preference
        preferred_mode: OptimizationMode = self.context_preferences.get(
            conditions.processing_context, 
            OptimizationMode.HYBRID_AUTO
        )
        
        # If we have a strong context preference, use it unless conditions are extreme
        if preferred_mode != OptimizationMode.HYBRID_AUTO:
            # Check for extreme conditions that override context preference
            if conditions.cpu_usage > 90.0:  # Extreme CPU load - use magic numbers
                return OptimizationMode.MAGIC_SPEED
            elif conditions.volume_entropy > 0.30:  # Extreme entropy - use originals
                return OptimizationMode.ORIGINAL_ROBUST
            else:
                return preferred_mode
        
        # Hybrid auto decision logic
        decision_factors: list = []
        
        # CPU load factor
        if conditions.cpu_usage > self.cpu_threshold_high:
            decision_factors.append(("high_cpu", OptimizationMode.MAGIC_SPEED, 0.8))
        
        # GPU utilization factor  
        if conditions.gpu_usage < self.gpu_threshold_low:
            decision_factors.append(("low_gpu", OptimizationMode.MAGIC_SPEED, 0.6))
        
        # Market volatility factor
        if conditions.market_volatility > self.volatility_threshold:
            decision_factors.append(("high_volatility", OptimizationMode.ORIGINAL_ROBUST, 0.7))
        
        # Volume entropy factor
        if conditions.volume_entropy > self.volume_entropy_threshold:
            decision_factors.append(("high_entropy", OptimizationMode.ORIGINAL_ROBUST, 0.9))
        
        # Thermal efficiency factor
        thermal_eff: float = conditions.thermal_state.get('thermal_efficiency', 0.5)
        if thermal_eff > 0.8:  # High thermal efficiency - can use magic numbers
            decision_factors.append(("high_thermal", OptimizationMode.MAGIC_SPEED, 0.5))
        elif thermal_eff < 0.3:  # Low thermal efficiency - use robust originals
            decision_factors.append(("low_thermal", OptimizationMode.ORIGINAL_ROBUST, 0.6))
        
        # Network latency factor
        if conditions.network_latency < 5.0:  # Low latency - can use fast magic numbers
            decision_factors.append(("low_latency", OptimizationMode.MAGIC_SPEED, 0.4))
        
        # Calculate weighted decision
        magic_weight: float = sum(w for _, mode, w in decision_factors if mode == OptimizationMode.MAGIC_SPEED)
        robust_weight: float = sum(w for _, mode, w in decision_factors if mode == OptimizationMode.ORIGINAL_ROBUST)
        
        # Decision with logging
        if magic_weight > robust_weight:
            decision: OptimizationMode = OptimizationMode.MAGIC_SPEED
            reason: str = f"Magic numbers (weight: {magic_weight:.1f} vs {robust_weight:.1f})"
        elif robust_weight > magic_weight:
            decision = OptimizationMode.ORIGINAL_ROBUST  
            reason = f"Original constants (weight: {robust_weight:.1f} vs {magic_weight:.1f})"
        else:
            # Tie or no strong factors - use context or default
            decision = OptimizationMode.MAGIC_SPEED if conditions.processing_context in [
                ProcessingContext.GPU_ACCELERATED, ProcessingContext.HIGH_FREQUENCY_TRADING,
                ProcessingContext.MULTI_MILLISECOND, ProcessingContext.UI_OVERLAY
            ] else OptimizationMode.ORIGINAL_ROBUST
            reason = f"Context-based default ({decision.value})"
        
        # Log decision factors
        self.cli_handler.log_safe(self.logger, 'debug', f"Decision factors: {decision_factors}")
        self.cli_handler.log_safe(self.logger, 'info', f"Optimization decision: {reason}")
        
        # Store decision history
        self.decision_history.append({
            'timestamp': time.time(),
            'conditions': conditions,
            'decision': decision,
            'factors': decision_factors,
            'reason': reason
        })
        
        return decision

    def get_constant(self, category: str, attribute: str, context: ProcessingContext) -> float:
        """
        Get the appropriate constant value based on current optimization strategy
        
        Args:
            category: Constant category (e.g., 'core', 'trading', 'visualization')
            attribute: Constant attribute name
            context: Processing context for decision making
            
        Returns:
            float: Optimized constant value
        """
        if not self.dual_pipeline_active:
            # Single pipeline mode - use whatever is active
            if OPTIMIZED_CONSTANTS.is_optimized():
                return self._get_optimized_constant(category, attribute)
            else:
                return self._get_original_constant(category, attribute)
        
        # Dual pipeline mode - intelligent selection
        conditions: SystemConditions = self.get_system_conditions(context)
        strategy: OptimizationMode = self.decide_optimization_strategy(conditions)
        
        if strategy == OptimizationMode.MAGIC_SPEED:
            return self._get_optimized_constant(category, attribute)
        else:  # ORIGINAL_ROBUST
            return self._get_original_constant(category, attribute)

    def _get_original_constant(self, category: str, attribute: str) -> float:
        """Get original constant value"""
        try:
            category_obj = getattr(self.cpu_constants, category, None)
            if category_obj:
                return getattr(category_obj, attribute, 0.0)
        except Exception as e:
            self.cli_handler.log_safe(self.logger, 'error', f"Error getting original constant {category}.{attribute}: {e}")
        return 0.0

    def _get_optimized_constant(self, category: str, attribute: str) -> float:
        """Get optimized magic number value"""
        try:
            if self.gpu_constants and hasattr(self.gpu_constants, category):
                category_obj = getattr(self.gpu_constants, category)
                return getattr(category_obj, attribute, 0.0)
        except Exception as e:
            self.cli_handler.log_safe(self.logger, 'error', f"Error getting optimized constant {category}.{attribute}: {e}")
        
        # Fallback to original
        return self._get_original_constant(category, attribute)

    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance recommendations based on current conditions"""
        recommendations: Dict[str, Any] = {
            'current_mode': self.current_mode.value,
            'dual_pipeline_active': self.dual_pipeline_active,
            'recommendations': []
        }
        
        # Analyze recent decisions
        if len(self.decision_history) > 10:
            recent_decisions = self.decision_history[-10:]
            magic_count: int = sum(1 for d in recent_decisions if d['decision'] == OptimizationMode.MAGIC_SPEED)
            robust_count: int = len(recent_decisions) - magic_count
            
            if magic_count > 7:
                recommendations['recommendations'].append({
                    'type': 'optimization_bias',
                    'message': 'System heavily favoring magic numbers - consider GPU workload optimization',
                    'action': 'Review GPU utilization and thermal management'
                })
            elif robust_count > 7:
                recommendations['recommendations'].append({
                    'type': 'stability_bias', 
                    'message': 'System heavily favoring original constants - check for high entropy conditions',
                    'action': 'Monitor market volatility and volume patterns'
                })
        
        return recommendations

    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start background monitoring of system conditions"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.cli_handler.log_safe(self.logger, 'info', f"🔍 Started hybrid optimization monitoring (interval: {interval}s)")

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.cli_handler.log_safe(self.logger, 'info', "⏹️ Stopped hybrid optimization monitoring")

    def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Sample different contexts and log decisions
                contexts = [
                    ProcessingContext.CPU_HEAVY,
                    ProcessingContext.GPU_ACCELERATED,
                    ProcessingContext.HIGH_FREQUENCY_TRADING,
                    ProcessingContext.UI_OVERLAY
                ]
                
                for context in contexts:
                    conditions: SystemConditions = self.get_system_conditions(context)
                    strategy: OptimizationMode = self.decide_optimization_strategy(conditions)
                    
                    # Log significant changes
                    if len(self.decision_history) > 0:
                        last_decision: OptimizationMode = self.decision_history[-1]['decision']
                        if strategy != last_decision:
                            self.cli_handler.log_safe(
                                self.logger, 'info', 
                                f"🔄 Strategy change for {context.value}: {last_decision.value} → {strategy.value}"
                            )
                
                time.sleep(interval)
                
            except Exception as e:
                self.cli_handler.log_safe(self.logger, 'error', f"Error in monitoring loop: {e}")
                time.sleep(interval)

# Global hybrid manager instance
HYBRID_MANAGER: HybridOptimizationManager = HybridOptimizationManager()

# Convenience functions for easy integration
def enable_hybrid_optimization() -> bool:
    """Enable hybrid optimization with dual pipeline"""
    return HYBRID_MANAGER.enable_dual_pipeline()

def get_smart_constant(category: str, attribute: str, context: ProcessingContext) -> float:
    """Get intelligently optimized constant based on context"""
    return HYBRID_MANAGER.get_constant(category, attribute, context)

def start_smart_monitoring(interval: float = 30.0) -> None:
    """Start intelligent optimization monitoring"""
    HYBRID_MANAGER.start_monitoring(interval)

def get_optimization_recommendations() -> Dict[str, Any]:
    """Get performance optimization recommendations"""
    return HYBRID_MANAGER.get_performance_recommendations() 