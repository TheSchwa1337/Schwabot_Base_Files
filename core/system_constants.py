"""
Comprehensive System Constants for Schwabot Visual Synthesis System

This module consolidates all magic numbers from across the system into named constants
for better maintainability, clarity, and future adaptive capabilities.

All values are preserved exactly as they were - this is purely an organizational improvement.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any
import os

# =============================================================================
# CORE SYSTEM THRESHOLDS
# =============================================================================

@dataclass
class CoreSystemThresholds:
    """Core system-wide threshold constants"""
    
    # Hash Correlation Thresholds (from quantum_btc_intelligence_core.py)
    MIN_HASH_CORRELATION_THRESHOLD: float = 0.25
    MIN_PRESSURE_DIFFERENTIAL_THRESHOLD: float = 0.15
    MIN_PROFIT_VECTOR_MAGNITUDE_THRESHOLD: float = 0.1
    MIN_STABILITY_INDEX_THRESHOLD: float = 0.7
    MIN_DETERMINISTIC_CONFIDENCE_THRESHOLD: float = 0.8
    MIN_SUSTAINMENT_INDEX_THRESHOLD: float = 0.65  # Critical sustainment threshold
    MIN_INTEGRATION_SCORE_THRESHOLD: float = 0.6
    MIN_SURVIVABILITY_SCORE_THRESHOLD: float = 0.7
    
    # Altitude/Pressure Calculation Parameters
    BASE_PRESSURE: float = 1.0
    ALTITUDE_FACTOR: float = 0.33  # From altitude adjustment theory
    VELOCITY_FACTOR: float = 2.0   # Speed compensation
    DENSITY_THRESHOLD: float = 0.15 # Market density threshold
    SUSTAINMENT_FACTOR: float = 0.25  # Sustainment influence on pressure

# =============================================================================
# PERFORMANCE AND RESOURCE LIMITS
# =============================================================================

@dataclass
class PerformanceConstants:
    """Performance monitoring and resource management constants"""
    
    # CPU/GPU Utilization Thresholds
    MAX_CPU_PERCENT_THRESHOLD: float = 80.0
    MAX_MEMORY_PERCENT_THRESHOLD: float = 70.0
    HIGH_CPU_UTILIZATION_THRESHOLD: float = 0.9
    HIGH_GPU_UTILIZATION_THRESHOLD: float = 0.9
    HIGH_MEMORY_UTILIZATION_THRESHOLD: float = 0.85
    
    # Visual Architecture Performance
    FRAME_RATE_LOW_THRESHOLD: float = 30.0
    FRAME_RATE_MEDIUM_THRESHOLD: float = 60.0
    FRAME_RATE_HIGH_THRESHOLD: float = 90.0
    FRAME_RATE_ULTRA_THRESHOLD: float = 120.0
    
    # Render Performance
    TARGET_RENDER_TIME_MS: float = 16.67
    BASE_ANIMATION_SPEED: float = 0.02  # 2% per frame at 60 FPS
    SLOW_ANIMATION_MULTIPLIER: float = 0.5
    FAST_ANIMATION_MULTIPLIER: float = 2.0
    
    # Memory and Buffer Limits
    MAX_HF_DISPLAY_ALLOCATIONS: int = 10000  # Maximum allocations to visualize
    MAX_ALLOCATION_HISTORY: int = 100  # High-frequency threshold
    MAX_OPTIMIZATION_HISTORY: int = 100
    ALLOCATION_BUFFER_SIZE: int = 100  # Send only recent 100 for performance

# =============================================================================
# VISUALIZATION AND UI CONSTANTS
# =============================================================================

@dataclass
class VisualizationConstants:
    """Visual synthesis and UI-related constants"""
    
    # Bit Level Visualization
    BIT_LEVEL_4: int = 4
    BIT_LEVEL_8: int = 8
    BIT_LEVEL_16: int = 16
    BIT_LEVEL_42: int = 42  # Phaser level
    BIT_LEVEL_81: int = 81
    
    # Visualization Parameters
    BASE_RADIUS: int = 100
    RADIUS_EXPANSION_PER_BIT: int = 5  # Expand radius with bit level
    
    # Tick Management
    DEFAULT_TICK_FREQUENCY_HZ: float = 60.0
    LOW_PERFORMANCE_TICK_FREQUENCY: float = 30.0
    HIGH_PERFORMANCE_TICK_FREQUENCY_LIMIT: float = 120.0
    
    # Processing and Flow
    TRANSITION_PROGRESS_MAX: float = 1.0
    PROCESSING_INTENSITY_DEFAULT: float = 0.5
    THERMAL_INFLUENCE_DEFAULT: float = 0.0
    SMOOTHING_FACTOR_DEFAULT: float = 0.8
    FLOW_SPEED_DEFAULT: float = 1.0
    
    # Drift Compensation
    DRIFT_THRESHOLD: float = 0.1   # Drift compensation threshold
    DRIFT_CAP_PERCENTAGE: float = 0.5  # Cap at 50% compensation
    DRIFT_RATE_DIVISOR: float = 10000  # For drift calculation
    
    # WebSocket Configuration
    DEFAULT_WEBSOCKET_PORT: int = 8765

# =============================================================================
# TRADING AND PROFIT CONSTANTS
# =============================================================================

@dataclass
class TradingConstants:
    """Trading, profit, and market-related constants"""
    
    # Position Sizing and Risk Management
    MAX_POSITION_SIZE: float = 0.1   # 10% max position
    DEFAULT_POSITION_SIZE: float = 0.05
    BASE_POSITION_SIZE_STRATEGIC: float = 0.15
    MAX_LOSS_PER_TRADE: float = 0.02  # 2% max loss
    MIN_CONFIDENCE_THRESHOLD: float = 0.7  # 70% minimum confidence
    
    # Profit Targets and Expectations
    PROFIT_TARGET_DEFAULT: float = 0.06    # 6% profit target
    STOP_LOSS_DEFAULT: float = 0.02  # 2% stop loss
    EXPECTED_PROFIT_DEFAULT: float = 0.02  # 2% profit assumption
    PROFIT_POTENTIAL_RESIDUAL: float = 0.03  # 3% profit potential
    
    # Price Movement and Volatility
    EXPECTED_PRICE_SWING_PERCENT: float = 0.01  # 1% expected swing
    PRICE_SLIPPAGE_RANGE: float = 0.001  # ±0.1% slippage range
    VOLATILITY_TOLERANCE_DEFAULT: float = 0.1
    VOLATILITY_TOLERANCE_CONSERVATIVE: float = 0.05
    
    # Market Conditions
    BASE_BTC_PRICE: float = 47500.0
    BTC_PRICE_FALLBACK: float = 48000.0
    BTC_EMERGENCY_FALLBACK: float = 10000.0
    SINE_COMPONENT_AMPLITUDE: float = 2000  # ±$2000 movement
    ENTROPY_ADJUSTMENT_RANGE: float = 1000  # Entropy-based adjustment
    
    # Volume Calculations
    BASE_VOLUME_MIN: float = 15000.0
    BASE_VOLUME_RANGE: float = 5000.0  # 15k-20k base
    ENTROPY_VOLUME_MULTIPLIER: float = 0.5  # Up to 50% increase
    DEFAULT_VOLUME_FALLBACK: float = 12000.0
    EMERGENCY_VOLUME_FALLBACK: float = 10000.0

# =============================================================================
# MATHEMATICAL AND ALGORITHMIC CONSTANTS
# =============================================================================

@dataclass
class MathematicalConstants:
    """Mathematical algorithm parameters and precision constants"""
    
    # Entropy Calculations
    ENTROPY_EPSILON: float = 1e-10  # Prevent log(0) in entropy calculations
    ENTROPY_EPSILON_ALT: float = 1e-8   # Alternative epsilon for different contexts
    ENTROPY_EPSILON_MICRO: float = 1e-6 # Micro epsilon for high precision
    
    # Numerical Precision and Convergence
    CONVERGENCE_THRESHOLD: float = 1e-6
    WEIGHT_SUM_TOLERANCE: float = 1e-6  # For weight normalization checks
    SOFTMAX_TOLERANCE: float = 1e-10
    SIGMOID_PRECISION: float = 1e-10
    NUMERICAL_STABILITY_EPSILON: float = 1e-9
    
    # Time and Scaling Factors
    MICROSECOND_SCALING: float = 1e-6   # Microsecond scaling factor
    MILLISECOND_CONVERSION: float = 1000  # Convert seconds to milliseconds
    HOURLY_CYCLE_SECONDS: float = 3600  # Hourly cycle
    FOUR_HOUR_CYCLE_SECONDS: float = 14400  # 4-hour cycle
    
    # Hash and Encoding
    HASH_DIVISOR: float = 2**256  # For hash normalization
    CHUNK_SIZE_BYTES: int = 8192  # File reading chunk size
    SHA_HASH_PREFIX_LENGTH: int = 16  # SHA hash prefix for display

# =============================================================================
# THERMAL AND RESOURCE MANAGEMENT
# =============================================================================

@dataclass
class ThermalConstants:
    """Thermal management and resource monitoring constants"""
    
    # Temperature Thresholds
    MAX_GPU_TEMPERATURE: float = 75.0
    THERMAL_CRITICAL_THRESHOLD: float = 0.4
    THERMAL_HIGH_THRESHOLD: float = 0.2
    THERMAL_REDUCTION_STEP: float = 0.05
    
    # Resource Utilization Bands
    CPU_NORMAL_MAX: float = 60.0
    CPU_WARM_RANGE: Tuple[float, float] = (60.0, 70.0)
    CPU_HOT_RANGE: Tuple[float, float] = (70.0, 80.0)
    CPU_CRITICAL_RANGE: Tuple[float, float] = (80.0, 100.0)
    
    MEMORY_NORMAL_MAX: float = 65.0
    MEMORY_WARM_RANGE: Tuple[float, float] = (65.0, 75.0)
    MEMORY_HOT_RANGE: Tuple[float, float] = (75.0, 85.0)
    MEMORY_CRITICAL_RANGE: Tuple[float, float] = (85.0, 100.0)
    
    GPU_NORMAL_MAX: float = 60.0
    GPU_WARM_RANGE: Tuple[float, float] = (60.0, 75.0)
    GPU_HOT_RANGE: Tuple[float, float] = (75.0, 85.0)
    GPU_CRITICAL_RANGE: Tuple[float, float] = (85.0, 100.0)
    
    # Execution Time Hints
    THERMAL_RESPONSE_TIME: float = 0.05  # Fast thermal response
    PROFIT_ANALYSIS_TIME: float = 0.2  # Moderate time for profit analysis
    BITMAP_OPERATION_TIME: float = 0.3  # Slower bitmap operations
    GPU_OPERATION_TIME: float = 0.5  # GPU operations can be slower
    FALLBACK_OPERATION_TIME: float = 0.01  # Very fast fallback

# =============================================================================
# FAULT DETECTION AND RECOVERY
# =============================================================================

@dataclass
class FaultDetectionConstants:
    """Fault detection, anomaly detection, and recovery constants"""
    
    # Default System Values
    DEFAULT_WEIGHT_MATRIX_VALUE: float = 0.9
    MAX_QUEUE_SIZE: float = 50.0
    NORMALIZATION_FACTOR: float = 1.0
    DEFAULT_INTERVAL: float = 0.1
    MAX_PROFIT_THRESHOLD: float = 100.0
    
    # Anomaly Detection
    ANOMALY_Z_SCORE_THRESHOLD: float = 2.5  # Statistically significant
    ANOMALY_NORMALIZATION_DIVISOR: float = 5.0  # Normalize z-score to [0,1]
    ANOMALY_CLUSTER_THRESHOLD: int = 3  # Multiple anomalies for JuMBO behavior
    ANOMALY_HISTORY_WINDOW: int = 10  # Check last 10 for clustering
    
    # Correlation Analysis
    CORRELATION_DECAY_FACTOR: float = 0.95
    MIN_CORRELATION_THRESHOLD: float = 0.3
    CORRELATION_CONFIDENCE_DIVISOR: float = 10.0
    PREDICTIVE_CORRELATION_THRESHOLD: float = 0.5
    
    # Memory Management
    RECOVERY_EVENT_MEMORY_LIMIT: int = 1000
    TEMPORAL_EXECUTION_HISTORY_LIMIT: int = 5000
    CORRELATION_BUFFER_LIMIT: int = 1000
    PIPELINE_CORRECTION_HISTORY_LIMIT: int = 500
    RECENT_EXECUTION_ANALYSIS_LIMIT: int = 100
    
    # Async Threshold and Scoring
    ASYNC_THRESHOLD_DEFAULT: float = 0.5
    SEVERITY_SCORE_BASE: float = 0.3
    URGENCY_SCORE_BASE: float = 0.25
    SYSTEM_LOAD_PENALTY: float = -0.2
    PROFIT_OPPORTUNITY_SCORE: float = 0.2

# =============================================================================
# INTELLIGENT THRESHOLDS AND ADAPTATION
# =============================================================================

@dataclass
class IntelligentThresholds:
    """Adaptive and intelligent threshold management constants"""
    
    # DLT Waveform Engine Thresholds
    ENTROPY_MAXIMUM_THRESHOLD: float = 4.0
    LATENCY_MAXIMUM_THRESHOLD: float = 100.0
    CPU_MAXIMUM_THRESHOLD: float = 80.0
    GPU_MAXIMUM_THRESHOLD: float = 85.0
    PROFIT_LOSS_PREVENTION_THRESHOLD: float = -0.05
    
    # Improvement and Adaptation
    MINIMUM_IMPROVEMENT_THRESHOLD: float = 0.1  # If not improving significantly
    ADAPTIVE_THRESHOLD_MULTIPLIER: float = 0.95  # For threshold adjustments
    ENTROPY_THRESHOLD_MULTIPLIER: float = 1.1
    ENTROPY_THRESHOLD_MAX: float = 6.0
    LATENCY_TIMEOUT_MULTIPLIER: float = 1.2
    LATENCY_TIMEOUT_MAX: float = 200.0
    
    # Resolution Improvements
    GPU_THERMAL_IMPROVEMENT: float = 0.3
    CPU_BOTTLENECK_IMPROVEMENT: float = 0.25
    ENTROPY_SPIKE_IMPROVEMENT: float = 0.4
    LATENCY_OPTIMIZATION_IMPROVEMENT: float = 0.35
    BATCH_ORDER_IMPROVEMENT: float = 0.2
    TICK_TIMING_IMPROVEMENT: float = 0.15
    MEMORY_CLEANUP_IMPROVEMENT: float = 0.3
    PROFIT_THRESHOLD_IMPROVEMENT: float = 0.25
    
    # Conservative Adjustment Factors
    PROFIT_CONSERVATIVE_MULTIPLIER: float = 0.8  # More conservative approach

# =============================================================================
# PHASE GATE AND EXECUTION CONSTANTS
# =============================================================================

@dataclass
class PhaseGateConstants:
    """Phase gate and execution decision constants"""
    
    # Phase Gate Configurations
    PHASE_4B_EXECUTION_DELAY: float = 0.1
    PHASE_4B_MIN_CONFIDENCE: float = 0.85
    PHASE_8B_EXECUTION_DELAY: float = 1.0
    PHASE_8B_MIN_CONFIDENCE: float = 0.70
    PHASE_42B_EXECUTION_DELAY: float = 5.0
    PHASE_42B_MIN_CONFIDENCE: float = 0.60
    
    # Entropy Thresholds for Phase Selection
    ENTROPY_LOW_THRESHOLD: float = 0.25   # Low entropy -> micro gate (4b)
    ENTROPY_MEDIUM_THRESHOLD: float = 0.45  # Medium entropy -> harmonic gate (8b)
    ENTROPY_HIGH_THRESHOLD: float = 0.75   # High entropy -> strategic gate (42b)
    
    # Execution Decision Thresholds
    MIN_EXECUTION_CONFIDENCE: float = 0.65
    EXECUTION_DELAY_THRESHOLD: float = 0.80
    UNIFIED_CONFIDENCE_THRESHOLD: float = 0.65
    
    # Confidence Component Weights
    EXECUTION_CONFIDENCE_WEIGHT: float = 0.25
    PHASE_CONFIDENCE_WEIGHT: float = 0.25

# =============================================================================
# SUSTAINMENT FRAMEWORK CONSTANTS
# =============================================================================

@dataclass
class SustainmentConstants:
    """8-principle sustainment framework constants"""
    
    # Core Sustainment Thresholds
    COHERENCE_THRESHOLD: float = 0.70
    SUSTAINMENT_THRESHOLD: float = 0.60
    MINIMUM_SUSTAINMENT_INDEX: float = 0.65
    MAXIMUM_TOTAL_RISK: float = 0.10  # 10% maximum
    
    # Sustainment Principle Weights
    ENTRY_DELTA_WEIGHT: float = 0.3
    EXIT_VELOCITY_WEIGHT: float = 0.2
    ECHO_WEIGHT: float = 0.2
    CONFIDENCE_WEIGHT: float = 0.2
    MARKET_TREND_WEIGHT: float = 0.1
    
    # Performance Scoring Bands
    PERFORMANCE_EXCELLENT: float = 0.9
    PERFORMANCE_VERY_GOOD: float = 0.8
    PERFORMANCE_GOOD: float = 0.7
    PERFORMANCE_ACCEPTABLE: float = 0.6
    PERFORMANCE_MARGINAL: float = 0.5
    PERFORMANCE_POOR: float = 0.4
    PERFORMANCE_VERY_POOR: float = 0.3
    
    # Echo and Temporal Thresholds
    ECHO_RETENTION_DAYS: int = 180
    ECHO_WEIGHT_THRESHOLD: float = 0.2
    PHASE_SHORT_ECHOES: int = 5   # 5+ successful echoes in 30d
    PHASE_MID_ECHOES: int = 3     # 3+ successful echoes in 90d  
    PHASE_LONG_ECHOES: int = 10   # 10+ phase-aligned echoes

# =============================================================================
# PROFIT ROUTING AND STRATEGY CONSTANTS
# =============================================================================

@dataclass
class ProfitRoutingConstants:
    """Profit routing and strategy selection constants"""
    
    # Route-Specific Requirements
    MICRO_SCALP_SI_REQUIREMENT: float = 0.75
    HARMONIC_SWING_SI_REQUIREMENT: float = 0.65
    STRATEGIC_HOLD_SI_REQUIREMENT: float = 0.60
    BASE_SI_REQUIREMENT: float = 0.65
    
    # Profit/Risk Analysis
    MINIMUM_ALLOCATION_THRESHOLD: float = 0.15
    MAXIMUM_ALLOCATION_LIMIT: float = 1.0
    PROFIT_RISK_RATIO_EPSILON: float = 0.001  # Avoid division by zero
    
    # Route Performance Thresholds
    ROUTE_SUCCESS_RATE_HIGH: float = 0.6
    ROUTE_AVERAGE_RETURN_HIGH: float = 0.02
    ROUTE_SUCCESS_RATE_MEDIUM: float = 0.48
    ROUTE_AVERAGE_RETURN_MEDIUM: float = 0.01
    
    # Strategy Scoring Weights
    DURATION_SCORE_WEIGHT: float = 0.4
    SUCCESS_RATE_WEIGHT: float = 0.4
    AVERAGE_PROFIT_WEIGHT: float = 0.2
    
    # Mathematical Validity
    MATHEMATICAL_VALIDITY_THRESHOLD: float = 0.65

# =============================================================================
# CONFIGURATION RANGES AND VALIDATION
# =============================================================================

@dataclass
class ConfigurationRanges:
    """System configuration ranges for validation"""
    
    # Profit and Performance Ranges
    PROFIT_TREND_RANGE_MIN: float = 0.001
    STABILITY_RANGE_LOW: Tuple[float, float] = (0.7, 1.0)
    STABILITY_RANGE_MEDIUM: Tuple[float, float] = (0.5, 0.9)
    MEMORY_COHERENCE_RANGE_HIGH: Tuple[float, float] = (0.8, 1.0)
    MEMORY_COHERENCE_RANGE_MEDIUM: Tuple[float, float] = (0.6, 0.9)
    
    # Paradox and Entropy Ranges
    PARADOX_PRESSURE_RANGE_LOW: Tuple[float, float] = (0.0, 2.0)
    PARADOX_PRESSURE_RANGE_MEDIUM: Tuple[float, float] = (1.0, 3.0)
    PARADOX_PRESSURE_RANGE_HIGH: Tuple[float, float] = (2.0, 5.0)
    ENTROPY_RATE_RANGE_LOW: Tuple[float, float] = (0.0, 0.3)
    ENTROPY_RATE_RANGE_MEDIUM: Tuple[float, float] = (0.2, 0.5)
    ENTROPY_RATE_RANGE_HIGH: Tuple[float, float] = (0.4, 1.0)
    
    # Thermal State Ranges
    THERMAL_STATE_RANGE_LOW: Tuple[float, float] = (0.0, 0.6)
    THERMAL_STATE_RANGE_MEDIUM: Tuple[float, float] = (0.3, 0.8)
    THERMAL_STATE_RANGE_HIGH: Tuple[float, float] = (0.6, 1.0)
    
    # Trust Score Ranges
    TRUST_SCORE_RANGE_HIGH: Tuple[float, float] = (0.7, 1.0)
    TRUST_SCORE_RANGE_MEDIUM: Tuple[float, float] = (0.5, 0.9)
    TRUST_SCORE_RANGE_LOW: Tuple[float, float] = (0.0, 0.5)

# =============================================================================
# UNIFIED SYSTEM CONSTANTS ACCESS
# =============================================================================

class SystemConstants:
    """Unified access point for all system constants"""
    
    def __init__(self):
        self.core = CoreSystemThresholds()
        self.performance = PerformanceConstants()
        self.visualization = VisualizationConstants()
        self.trading = TradingConstants()
        self.mathematical = MathematicalConstants()
        self.thermal = ThermalConstants()
        self.fault_detection = FaultDetectionConstants()
        self.intelligent = IntelligentThresholds()
        self.phase_gate = PhaseGateConstants()
        self.sustainment = SustainmentConstants()
        self.profit_routing = ProfitRoutingConstants()
        self.configuration = ConfigurationRanges()
    
    def get_all_constants(self) -> Dict[str, Any]:
        """Get all constants as a dictionary for debugging/inspection"""
        return {
            'core_thresholds': self.core.__dict__,
            'performance': self.performance.__dict__,
            'visualization': self.visualization.__dict__,
            'trading': self.trading.__dict__,
            'mathematical': self.mathematical.__dict__,
            'thermal': self.thermal.__dict__,
            'fault_detection': self.fault_detection.__dict__,
            'intelligent_thresholds': self.intelligent.__dict__,
            'phase_gate': self.phase_gate.__dict__,
            'sustainment': self.sustainment.__dict__,
            'profit_routing': self.profit_routing.__dict__,
            'configuration_ranges': self.configuration.__dict__
        }
    
    def validate_environment_compatibility(self) -> Dict[str, bool]:
        """Validate constants for the current environment"""
        validations = {}
        
        # Check for Windows CLI compatibility
        validations['windows_cli_compatible'] = os.name == 'nt'
        
        # Validate mathematical constants
        validations['epsilon_values_valid'] = (
            self.mathematical.ENTROPY_EPSILON > 0 and
            self.mathematical.CONVERGENCE_THRESHOLD > 0
        )
        
        # Validate performance thresholds
        validations['performance_thresholds_valid'] = (
            0 < self.performance.MAX_CPU_PERCENT_THRESHOLD <= 100 and
            0 < self.performance.MAX_MEMORY_PERCENT_THRESHOLD <= 100
        )
        
        # Validate trading parameters
        validations['trading_parameters_valid'] = (
            0 < self.trading.MAX_POSITION_SIZE <= 1.0 and
            0 < self.trading.MAX_LOSS_PER_TRADE <= 1.0
        )
        
        return validations

# =============================================================================
# GLOBAL SYSTEM CONSTANTS INSTANCE
# =============================================================================

# Create the global instance for system-wide access
SYSTEM_CONSTANTS = SystemConstants()

# Export commonly used constant groups for convenience
CORE_THRESHOLDS = SYSTEM_CONSTANTS.core
PERFORMANCE_LIMITS = SYSTEM_CONSTANTS.performance
VISUALIZATION_PARAMS = SYSTEM_CONSTANTS.visualization
TRADING_PARAMS = SYSTEM_CONSTANTS.trading
MATH_CONSTANTS = SYSTEM_CONSTANTS.mathematical
THERMAL_LIMITS = SYSTEM_CONSTANTS.thermal
FAULT_CONSTANTS = SYSTEM_CONSTANTS.fault_detection
INTELLIGENT_THRESHOLDS = SYSTEM_CONSTANTS.intelligent
PHASE_GATE_CONFIG = SYSTEM_CONSTANTS.phase_gate
SUSTAINMENT_CONFIG = SYSTEM_CONSTANTS.sustainment
PROFIT_ROUTING_CONFIG = SYSTEM_CONSTANTS.profit_routing

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_constant_by_path(path: str) -> Any:
    """
    Get a constant by dot-notation path.
    
    Example:
        get_constant_by_path('core.MIN_HASH_CORRELATION_THRESHOLD')
        get_constant_by_path('trading.MAX_POSITION_SIZE')
    """
    parts = path.split('.')
    obj = SYSTEM_CONSTANTS
    
    for part in parts:
        obj = getattr(obj, part)
    
    return obj

def validate_all_constants() -> bool:
    """Validate all constants are reasonable and environment-compatible"""
    validation_results = SYSTEM_CONSTANTS.validate_environment_compatibility()
    return all(validation_results.values())

def print_constant_summary():
    """Print a summary of all constant categories (useful for debugging)"""
    print("[CONSTANTS] Schwabot System Constants Summary:")
    print("=====================================")
    
    constant_counts = {
        'Core System Thresholds': len(SYSTEM_CONSTANTS.core.__dict__),
        'Performance Constants': len(SYSTEM_CONSTANTS.performance.__dict__),
        'Visualization Constants': len(SYSTEM_CONSTANTS.visualization.__dict__),
        'Trading Constants': len(SYSTEM_CONSTANTS.trading.__dict__),
        'Mathematical Constants': len(SYSTEM_CONSTANTS.mathematical.__dict__),
        'Thermal Constants': len(SYSTEM_CONSTANTS.thermal.__dict__),
        'Fault Detection Constants': len(SYSTEM_CONSTANTS.fault_detection.__dict__),
        'Intelligent Thresholds': len(SYSTEM_CONSTANTS.intelligent.__dict__),
        'Phase Gate Constants': len(SYSTEM_CONSTANTS.phase_gate.__dict__),
        'Sustainment Constants': len(SYSTEM_CONSTANTS.sustainment.__dict__),
        'Profit Routing Constants': len(SYSTEM_CONSTANTS.profit_routing.__dict__),
        'Configuration Ranges': len(SYSTEM_CONSTANTS.configuration.__dict__)
    }
    
    total_constants = sum(constant_counts.values())
    
    for category, count in constant_counts.items():
        print(f"  {category}: {count} constants")
    
    print(f"\nTotal Constants Organized: {total_constants}")
    validation_passed = validate_all_constants()
    status = "[SUCCESS] PASSED" if validation_passed else "[ERROR] FAILED"
    print(f"Environment Validation: {status}")

if __name__ == "__main__":
    print_constant_summary() 