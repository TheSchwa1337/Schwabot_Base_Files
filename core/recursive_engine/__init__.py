from .math_functions import coherence_trigger, fractal_signal, recursive_output, recursive_sum
from .integrator import SignalIntegrator
from .primary_loop import RecursiveEngine
from .signal_processor import RecursiveSignalHandler
from .recursive_strategy_handler import RecursiveStrategyHandler, TradeSignal
from .profit_memory_vault import ProfitMemoryVault, PatternMemory
from .vault_reentry_pipeline import VaultReentryPipeline

__all__ = [
    # Core recursive math
    "coherence_trigger",
    "fractal_signal",
    "recursive_output",
    "recursive_sum",
    "SignalIntegrator",
    "RecursiveEngine",
    "RecursiveSignalHandler",
    
    # Trading components
    "RecursiveStrategyHandler",
    "TradeSignal",
    "ProfitMemoryVault",
    "PatternMemory",
    "VaultReentryPipeline",
]

def register_recursive_engine(bus):
    """Convenience helper to attach RecursiveSignalHandler to an EventBus.

    Example
    -------
    >>> from init.event_bus import EventBus
    >>> bus = EventBus()
    >>> register_recursive_engine(bus)
    """
    return RecursiveSignalHandler(bus) 