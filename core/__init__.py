# Core module for the 100% Advanced Dualistic Trading Execution System
# Import only essential working components for final implementation

# Essential mathematical components that are working
try:
    from .unified_math_system import UnifiedMathSystem, unified_math
    UNIFIED_MATH_AVAILABLE = True
except ImportError:
    UNIFIED_MATH_AVAILABLE = False

try:
    from .unified_profit_vectorization_system import UnifiedProfitVectorizationSystem, profit_vectorization_system
    PROFIT_VECTORIZATION_AVAILABLE = True
except ImportError:
    PROFIT_VECTORIZATION_AVAILABLE = False

try:
    from .dualistic_state_machine import DualisticStateMachine
    DUALISTIC_STATE_AVAILABLE = True
except ImportError:
    DUALISTIC_STATE_AVAILABLE = False

try:
    from .advanced_tensor_algebra import UnifiedTensorAlgebra
    TENSOR_ALGEBRA_AVAILABLE = True
except ImportError:
    TENSOR_ALGEBRA_AVAILABLE = False

try:
    from .ccxt_integration import CCXTIntegration
    CCXT_INTEGRATION_AVAILABLE = True
except ImportError:
    CCXT_INTEGRATION_AVAILABLE = False

try:
    from .phase_bit_integration import PhaseBitIntegration
    PHASE_BIT_AVAILABLE = True
except ImportError:
    PHASE_BIT_AVAILABLE = False

# Advanced dualistic trading system - 100% Complete Implementation
try:
    from .advanced_dualistic_trading_execution_system import (
        AdvancedDualisticTradingExecutionSystem,
        GhostTradeType,
        TriggerComplexity,
        advanced_trading_system
    )
    ADVANCED_SYSTEM_AVAILABLE = True
except ImportError:
    ADVANCED_SYSTEM_AVAILABLE = False

__all__ = [
    'UNIFIED_MATH_AVAILABLE',
    'PROFIT_VECTORIZATION_AVAILABLE', 
    'DUALISTIC_STATE_AVAILABLE',
    'TENSOR_ALGEBRA_AVAILABLE',
    'CCXT_INTEGRATION_AVAILABLE',
    'PHASE_BIT_AVAILABLE',
    'ADVANCED_SYSTEM_AVAILABLE'
]

if UNIFIED_MATH_AVAILABLE:
    __all__.extend(['UnifiedMathSystem', 'unified_math'])

if PROFIT_VECTORIZATION_AVAILABLE:
    __all__.extend(['UnifiedProfitVectorizationSystem', 'profit_vectorization_system'])

if DUALISTIC_STATE_AVAILABLE:
    __all__.extend(['DualisticStateMachine'])

if TENSOR_ALGEBRA_AVAILABLE:
    __all__.extend(['UnifiedTensorAlgebra'])

if CCXT_INTEGRATION_AVAILABLE:
    __all__.extend(['CCXTIntegration'])

if PHASE_BIT_AVAILABLE:
    __all__.extend(['PhaseBitIntegration'])

if ADVANCED_SYSTEM_AVAILABLE:
    __all__.extend([
        'AdvancedDualisticTradingExecutionSystem',
        'GhostTradeType', 
        'TriggerComplexity',
        'advanced_trading_system'
    ]) 