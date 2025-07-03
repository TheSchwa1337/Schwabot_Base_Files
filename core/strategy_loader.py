# !/usr/bin/env python3
"""
Strategy Loader - Loads and routes strategies by name or hash.
"""
import importlib
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "strategy")

# Dynamically load all strategy modules in the strategy/ folder
STRATEGY_REGISTRY: Dict[str, Callable] = {}

# Fallback strategy implementations
def momentum_strategy(data):
    """Default momentum strategy."""
    return {"action": "buy", "confidence": 0.8, "strategy": "momentum"}

def mean_reversion_strategy(data):
    """Default mean reversion strategy."""
    return {"action": "sell", "confidence": 0.7, "strategy": "mean_reversion"}

def entropy_driven_strategy(data):
    """Default entropy driven strategy."""
    return {"action": "hold", "confidence": 0.6, "strategy": "entropy_driven"}

# Add fallback strategies
STRATEGY_REGISTRY.update({
    "momentum": momentum_strategy,
    "mean_reversion": mean_reversion_strategy,
    "entropy_driven": entropy_driven_strategy
})

# Try to load actual strategy files if they exist
if os.path.exists(STRATEGY_DIR):
    for fname in os.listdir(STRATEGY_DIR):
        if fname.endswith(".py") and not fname.startswith("_"):
            mod_name = fname[:-3]
            try:
                mod = importlib.import_module(f"core.strategy.{mod_name}")
                if hasattr(mod, "execute"):
                    STRATEGY_REGISTRY[mod_name] = getattr(mod, "execute")
                    logger.info(f"Loaded strategy: {mod_name}")
            except ImportError as e:
                logger.warning(f"Could not import strategy {mod_name}: {e}")
            except Exception as e:
                logger.warning(f"Error loading strategy {mod_name}: {e}")

# Example hash mapping (expand as needed)
HASH_MAP = {
    "momentum": "momentum",
    "mean_reversion": "mean_reversion",
    "entropy_driven": "entropy_driven"
}

def load_strategy(name_or_hash: str) -> Optional[Callable]:
    """Load a strategy by name or hash."""
    key = HASH_MAP.get(name_or_hash, name_or_hash)
    strategy = STRATEGY_REGISTRY.get(key)

    if strategy:
        logger.info(f"Strategy loaded: {key}")
    else:
        logger.warning(f"Strategy not found: {key}, using momentum fallback")
        strategy = STRATEGY_REGISTRY.get("momentum")

    return strategy

__all__ = ["load_strategy", "STRATEGY_REGISTRY"]