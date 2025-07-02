# This is a package initializer file for the 'core' directory.

from .basket_vector_linker import BasketVectorLinker
from .glyph_phase_resolver import GlyphPhaseResolver
from .profit_memory_echo import ProfitMemoryEcho
from .quantum_superpositional_trigger import QuantumSuperpositionalTrigger
from .warp_sync_core import WarpSyncCore
from .mathlib_v4 import MathLibV4
from .unified_trading_pipeline import UnifiedTradingPipeline
from .unified_math_system import UnifiedMathSystem
from .matrix_math_utils import analyze_price_matrix
from .risk_manager import RiskManager
from .strategy_logic import StrategyLogic

# Add backup integration components
try:
    from .ghost_flip_executor import GhostFlipExecutor
    from .profit_orbit_engine import ProfitOrbitEngine  
    from .pair_flip_orbit import PairFlipOrbit
    BACKUP_COMPONENTS_AVAILABLE = True
except ImportError:
    BACKUP_COMPONENTS_AVAILABLE = False

# Mathematical framework components
try:
    from .profit_vector_forecast import ProfitVectorForecastEngine
    from .brain_trading_engine import BrainTradingEngine
    from .unified_profit_vectorization_system import UnifiedProfitVectorizationSystem
    TRADING_COMPONENTS_AVAILABLE = True
except ImportError:
    TRADING_COMPONENTS_AVAILABLE = False

__all__ = [
    'MathLibV4',
    'UnifiedTradingPipeline', 
    'UnifiedMathSystem',
    'analyze_price_matrix',
    'RiskManager',
    'StrategyLogic',
    'BACKUP_COMPONENTS_AVAILABLE',
    'TRADING_COMPONENTS_AVAILABLE'
]

if BACKUP_COMPONENTS_AVAILABLE:
    __all__.extend(['GhostFlipExecutor', 'ProfitOrbitEngine', 'PairFlipOrbit'])

if TRADING_COMPONENTS_AVAILABLE:
    __all__.extend(['ProfitVectorForecastEngine', 'BrainTradingEngine', 'UnifiedProfitVectorizationSystem'])
