"""Module for Schwabot trading system."""

import importlib
import logging
import os
from typing import Any, Callable, Dict, Optional

# !/usr/bin/env python3
"""Strategy Loader - Loads and routes strategies by name or hash."""
logger = logging.getLogger(__name__)

STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "strategy")

# Dynamically load all strategy modules in the strategy/ folder
STRATEGY_REGISTRY: Dict[str, Callable] = {}


# Fallback strategy implementations
    def momentum_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return default momentum strategy."""
return {"action": "buy", "confidence": 0.8, "strategy": "momentum"}


    def mean_reversion_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return default mean reversion strategy."""
return {"action": "sell", "confidence": 0.7, "strategy": "mean_reversion"}


    def entropy_driven_strategy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return default entropy driven strategy."""
return {"action": "hold", "confidence": 0.6, "strategy": "entropy_driven"}


# Add fallback strategies
STRATEGY_REGISTRY.update(
{
"momentum": momentum_strategy,
"mean_reversion": mean_reversion_strategy,
"entropy_driven": entropy_driven_strategy,
}
)

# Try to load actual strategy files if they exist
    if os.path.exists(STRATEGY_DIR):
        for fname in os.listdir(STRATEGY_DIR):
            if fname.endswith(".py") and not fname.startswith("_"):
            mod_name = fname[:-3]
                try:
                mod = importlib.import_module("core.strategy.{0}".format(mod_name))
                    if hasattr(mod, "execute"):
                    STRATEGY_REGISTRY[mod_name] = mod.execute
                    logger.info("Loaded strategy: {0}".format(mod_name))
                        except ImportError as e:
                        logger.warning("Could not import strategy {0}: {1}".format(mod_name, e))
                            except Exception as e:
                            logger.warning("Error loading strategy {0}: {1}".format(mod_name, e))

                            # Example hash mapping (expand as needed)
                            HASH_MAP = {
                            "momentum": "momentum",
                            "mean_reversion": "mean_reversion",
                            "entropy_driven": "entropy_driven",
                            }


                                def load_strategy(name_or_hash: str) -> Optional[Callable]:
                                """Load a strategy by name or hash."""
                                key = HASH_MAP.get(name_or_hash, name_or_hash)
                                strategy = STRATEGY_REGISTRY.get(key)

                                    if strategy:
                                    logger.info("Strategy loaded: {0}".format(key))
                                        else:
                                        logger.warning("Strategy not found: {0}, using momentum fallback".format(key))
                                        strategy = STRATEGY_REGISTRY.get("momentum")

                                    return strategy


                                    __all__ = ["load_strategy", "STRATEGY_REGISTRY"]
