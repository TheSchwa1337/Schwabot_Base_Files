"""
Ferris Wheel RDE Integration
===========================

Connects the Radial Dynamic Engine (RDE) with the Ferris Wheel architecture,
enabling dynamic bit mode selection and spin ID generation to influence
trading strategy selection and execution.

Features:
- RDE-driven bit mode selection for Ferris Wheel spins
- Performance-based mode adaptation
- Spin history integration
- Strategy mapping based on RDE biotypes
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from ncco_core.rde_core import RDEEngine
from ncco_core.ferris_wheel import FerrisWheel  # Assuming this exists

class FerrisRDE:
    """Integrates RDE engine with Ferris Wheel for dynamic strategy selection."""
    
    def __init__(self, rde_cfg: str | Path, ferris_cfg: str | Path):
        self.rde = RDEEngine(rde_cfg)
        self.ferris = FerrisWheel(ferris_cfg)
        
        # Strategy mapping based on bit modes
        self._mode_strategy_map = {
            4: ["hold", "stable_swap"],  # Low precision -> conservative strategies
            8: ["hedge", "flip"],        # Medium precision -> balanced strategies
            42: ["aggressive", "spec"]   # High precision -> aggressive strategies
        }
        
        # Performance tracking
        self._strategy_performance: Dict[str, float] = {}
        self._mode_weights: Dict[int, float] = {
            mode: 1.0 for mode in self.rde.bit_modes
        }
        
        # Integration state
        self._last_spin: Optional[Dict] = None
        self._spin_outcomes: List[Dict] = []

    def update_market_state(self, signals: Dict[str, float]) -> None:
        """Update both RDE and Ferris Wheel with new market signals."""
        # Update RDE first to determine bit mode
        self.rde.update_signals(signals)
        
        # Get current biotype and mode
        spin_tag = self.rde.compute_biotype()
        current_mode = self.rde.current_bit_mode
        
        # Map to Ferris Wheel strategies based on mode
        strategies = self._mode_strategy_map[current_mode]
        
        # Weight strategies based on performance
        weights = self._get_strategy_weights(strategies)
        
        # Update Ferris Wheel with weighted strategies
        self.ferris.update_strategies(strategies, weights)
        
        # Log the spin
        self._log_spin(spin_tag, current_mode, strategies, weights)

    def _get_strategy_weights(self, strategies: List[str]) -> List[float]:
        """Get performance-weighted strategy weights."""
        if not self._strategy_performance:
            return [1.0] * len(strategies)
            
        weights = []
        for strategy in strategies:
            perf = self._strategy_performance.get(strategy, 0.5)
            weights.append(perf)
            
        # Normalize weights
        total = sum(weights)
        if total == 0:
            return [1.0] * len(strategies)
        return [w/total for w in weights]

    def _log_spin(self, spin_tag: str, mode: int, strategies: List[str], 
                 weights: List[float]) -> None:
        """Log spin details for analysis."""
        spin_data = {
            "tag": spin_tag,
            "utc": datetime.now(timezone.utc).isoformat(),
            "bit_mode": mode,
            "strategies": strategies,
            "weights": weights,
            "mode_performance": self.rde.get_mode_performance(),
            "strategy_performance": self._strategy_performance
        }
        
        self._last_spin = spin_data
        self._spin_outcomes.append(spin_data)
        
        # Log to file
        log_dir = Path("~/Schwabot/init/ferris_rde_logs").expanduser()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{spin_tag}_ferris.json"
        log_file.write_text(json.dumps(spin_data, indent=2))

    def update_strategy_performance(self, strategy: str, performance: float) -> None:
        """Update performance metrics for a strategy."""
        self._strategy_performance[strategy] = performance
        
        # Update mode weights based on strategy performance
        self._update_mode_weights()

    def _update_mode_weights(self) -> None:
        """Update bit mode weights based on strategy performance."""
        mode_performance = {mode: 0.0 for mode in self.rde.bit_modes}
        
        # Aggregate performance by mode
        for mode, strategies in self._mode_strategy_map.items():
            perf_sum = 0.0
            count = 0
            for strategy in strategies:
                if strategy in self._strategy_performance:
                    perf_sum += self._strategy_performance[strategy]
                    count += 1
            if count > 0:
                mode_performance[mode] = perf_sum / count
        
        # Update weights with exponential smoothing
        alpha = 0.3  # Smoothing factor
        for mode, perf in mode_performance.items():
            self._mode_weights[mode] = (
                (1 - alpha) * self._mode_weights[mode] + 
                alpha * perf
            )

    def get_current_state(self) -> Dict:
        """Get current state of the integration."""
        return {
            "last_spin": self._last_spin,
            "current_mode": self.rde.current_bit_mode,
            "mode_weights": self._mode_weights,
            "strategy_performance": self._strategy_performance,
            "spin_count": len(self._spin_outcomes)
        }

    def get_spin_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get recent spin history."""
        if limit is None:
            return self._spin_outcomes
        return self._spin_outcomes[-limit:]

    def get_mode_performance(self) -> Dict[int, float]:
        """Get current performance metrics for each bit mode."""
        return self._mode_weights.copy()

    def get_strategy_performance(self) -> Dict[str, float]:
        """Get current performance metrics for each strategy."""
        return self._strategy_performance.copy()

    def reset_performance(self) -> None:
        """Reset all performance metrics."""
        self._strategy_performance.clear()
        self._mode_weights = {mode: 1.0 for mode in self.rde.bit_modes}
        self._spin_outcomes.clear()
        self._last_spin = None

# Example usage:
"""
from ncco_core.ferris_rde import FerrisRDE

# Initialize integration
ferris_rde = FerrisRDE(
    rde_cfg="ncco_core/rde_config.yaml",
    ferris_cfg="ncco_core/ferris_config.yaml"
)

# Update with market signals
signals = {
    "BTC_price_delta": 0.02,
    "BTC_volatility": 0.15,
    "ETH_price_delta": -0.01,
    "ETH_volatility": 0.12
}
ferris_rde.update_market_state(signals)

# Update strategy performance after execution
ferris_rde.update_strategy_performance("hold", 0.95)
ferris_rde.update_strategy_performance("flip", 0.82)

# Get current state
state = ferris_rde.get_current_state()
print(f"Current mode: {state['current_mode']}")
print(f"Mode weights: {state['mode_weights']}")
""" 