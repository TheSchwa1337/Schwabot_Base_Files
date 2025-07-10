import time
from enum import Enum  # Added missing import for Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

"""
LEGACY FILE - COMMENTED OUT DUE TO SYNTAX ERRORS

This file has been automatically commented out because it contains syntax errors
that prevent the Schwabot system from running properly.

Original file: core\strategy\multi_phase_strategy_weight_tensor.py
Date commented out: 2025-07-02 19:37:06

The clean implementation has been preserved in the following files:
- core/clean_math_foundation.py (mathematical foundation)
- core/clean_profit_vectorization.py (profit calculations)
- core/clean_trading_pipeline.py (trading logic)
- core/clean_unified_math.py (unified mathematics)

All core functionality has been reimplemented in clean, production-ready files.
"""

# ORIGINAL CONTENT COMMENTED OUT BELOW:
"""






Multi-Phase Strategy Weight Tensor Module
-----------------------------------------
Implements a multi-dimensional tensor to dynamically manage and adjust the
weights of various trading strategies across different market phases.
This module is critical for adaptive strategy orchestration, allowing
the system to prioritize or de-prioritize strategies based on prevailing
market conditions or internal performance metrics.

Key functionalities include:
- Definition and management of strategy weights as a tensor.
- Adaptive adjustment of weights based on phase indicators.
- Support for dif ferent market phases (e.g., trend, consolidation, volatility).
- Integration with other strategy modules for dynamic control.class MarketPhase(Enum):Defines different market phases.TREND = trendCONSOLIDATION =  consolidationVOLATILITY = volatilityREVERSAL =  reversalUNKNOWN = unknownclass MultiPhaseStrategyWeightTensor:Manages a multi-dimensional tensor of strategy weights, adapting them
based on identified market phases.def __init__():Initializes the MultiPhaseStrategyWeightTensor.

Args:
            strategy_ids: A list of all unique strategy identifiers that this tensor will manage.
initial_weights: Optional dictionary of initial weights for strategies. If None,
weights are initialized equally.
phase_sensitivity: How aggressively weights adapt to phase changes (0.0 to 1.0).
            decay_factor: Factor by which old weight influence decays (0.0 to 1.0).if not strategy_ids:
            raise ValueError(strategy_ids cannot be empty.)

# Ensure unique and ordered
self.strategy_ids = sorted(list(set(strategy_ids)))
self.num_strategies = len(self.strategy_ids)
self.strategy_to_index = {sid: i for i, sid in enumerate(self.strategy_ids)}

# Initialize weights tensor (strategies x phases)
        # For simplicity, we'll start with 2D: strategy_id x phase'
        # More complex could be strategy_id x phase x time_horizon, etc.
self.phases = [phase.value for phase in MarketPhase]
self.num_phases = len(self.phases)
self.phase_to_index = {phase_val: i for i, phase_val in enumerate(self.phases)}

# Initialize the weight tensor. Shape: (num_strategies, num_phases)
self.weight_tensor = (
            np.ones((self.num_strategies, self.num_phases)) / self.num_strategies
)

if initial_weights:
            for strategy, weight in initial_weights.items():
                if strategy in self.strategy_to_index: idx = self.strategy_to_index[strategy]
# Apply initial weight across all phases equally if not
# phase-specific
self.weight_tensor[idx, :] = weight
            # Normalize column-wise (per phase) to ensure sum is 1.0
self._normalize_weights()

self.phase_sensitivity = phase_sensitivity
self.decay_factor = decay_factor

self.current_phase: MarketPhase = MarketPhase.UNKNOWN
self.metrics: Dict[str, Any] = {last_update_time: None,total_updates: 0,phase_transitions: 0,active_phase": self.current_phase.value,
}

def _normalize_weights():Normalizes the weights within each phase column to sum to 1.0.
Avoids division by zero if a column sums to 0.col_sums = self.weight_tensor.sum(axis=0, keepdims=True)
# Prevent division by zero if a column is all zeros
col_sums[col_sums == 0] = 1.0
self.weight_tensor = self.weight_tensor / col_sums

def get_strategy_weights_for_phase(self, phase: MarketPhase): -> Dict[str, float]:
Retrieves the weights for all strategies given a specific market phase.

Args:
            phase: The market phase for which to retrieve weights.

Returns:
            A dictionary mapping strategy IDs to their corresponding weights.if phase.value not in self.phase_to_index:
            raise ValueError(fUnknown market phase: {phase.value})

phase_idx = self.phase_to_index[phase.value]
weights = self.weight_tensor[:, phase_idx]

        return {self.strategy_ids[i]: weights[i] for i in range(self.num_strategies)}

def update_weights():
Adjusts strategy weights based on the identified market phase and
performance feedback for each strategy.

Args:
            identified_phase: The currently identified market phase.
performance_feedback: A dictionary where keys are strategy IDs and values
are dictionaries containing performance metrics'
(e.g., {'strategy_A': {'pnl': 0.05, 'volatility': 0.01}}).'
                                  Expects a 'pnl' key for profit/loss.self.metrics[total_updates] += 1self.metrics[last_update_time] = time.time()

if identified_phase != self.current_phase:
            self.metrics[phase_transitions] += 1
self.current_phase = identified_phase
self.metrics[active_phase] = self.current_phase.value
print(f"Market phase transitioned to: {identified_phase.value})

phase_idx = self.phase_to_index[identified_phase.value]

# Apply decay to existing weights in the current phase
self.weight_tensor[:, phase_idx] *= self.decay_factor

# Adjust weights based on performance feedback
for strategy_id, feedback in performance_feedback.items():
            if strategy_id in self.strategy_to_index: strat_idx = self.strategy_to_index[strategy_id]
                pnl = feedback.get(pnl, 0.0)

# Simple adaptive logic: reward positive PnL, penalize negative
adjustment = pnl * self.phase_sensitivity
self.weight_tensor[strat_idx, phase_idx] += adjustment

# Ensure weights are non-negative after adjustment
self.weight_tensor = np.maximum(self.weight_tensor, 0.0)

# Re-normalize weights for the current phase column
self._normalize_weights()

def get_current_state():-> Dict[str, Any]:
Returns the current state of the tensor and related metrics.return {# Convert numpy array to list for readability
current_weight_tensor: self.weight_tensor.tolist(),strategy_ids: self.strategy_ids,phases: self.phases,current_phase": self.current_phase.value,metrics": self.metrics,
}

def reset():Resets the tensor to initial weights and clears metrics.self.weight_tensor = (
            np.ones((self.num_strategies, self.num_phases)) / self.num_strategies
)
if initial_weights:
            for strategy, weight in initial_weights.items():
                if strategy in self.strategy_to_index: idx = self.strategy_to_index[strategy]
                    self.weight_tensor[idx, :] = weight
self._normalize_weights()
self.current_phase = MarketPhase.UNKNOWN
self.metrics = {last_update_time: None,total_updates: 0,phase_transitions": 0,active_phase": self.current_phase.value,
}

def get_active_phase():-> MarketPhase:Returns the currently active market phase.return self.current_phase


if __name__ == __main__:
    print(--- Multi-Phase Strategy Weight Tensor Demo ---)

# Define some dummy strategy IDs
strategies = [EMA_Cross,RSI_Divergence,Bollinger_Squeeze,Volume_Breakout]
    tensor_manager = MultiPhaseStrategyWeightTensor(strategy_ids=strategies)

print(\nInitial Weights (across all phases):)print(tensor_manager.get_current_state()[current_weight_tensor])

# Simulate phase changes and performance feedback
# Scenario 1: Trend phase, EMA_Cross performs well
print(\n--- Scenario 1: Trend Phase, EMA_Cross performs well ---)
performance_trend = {EMA_Cross: {pnl: 0.02,volatility": 0.005},RSI_Divergence": {pnl: -0.005,volatility": 0.002},Bollinger_Squeeze": {pnl: 0.001,volatility": 0.001},Volume_Breakout": {pnl: 0.008,volatility": 0.003},
}
tensor_manager.update_weights(MarketPhase.TREND, performance_trend)
print(Weights for TREND phase after update:)
print(tensor_manager.get_strategy_weights_for_phase(MarketPhase.TREND))
    print(f"Current Active Phase: {tensor_manager.get_active_phase()})

# Scenario 2: Consolidation phase, Bollinger_Squeeze performs well
print(\n--- Scenario 2: Consolidation Phase, Bollinger_Squeeze performs well ---)
performance_consolidation = {EMA_Cross: {pnl: -0.01,volatility": 0.003},RSI_Divergence": {pnl: 0.002,volatility": 0.001},Bollinger_Squeeze": {pnl: 0.03,volatility": 0.001},Volume_Breakout": {pnl: -0.002,volatility": 0.004},
}
tensor_manager.update_weights(MarketPhase.CONSOLIDATION, performance_consolidation)
print(Weights for CONSOLIDATION phase after update:)
print(tensor_manager.get_strategy_weights_for_phase(MarketPhase.CONSOLIDATION))
    print(f"Current Active Phase: {tensor_manager.get_active_phase()})

# Scenario 3: Volatility phase, Volume_Breakout performs well
    print(\n--- Scenario 3: Volatility Phase, Volume_Breakout performs well ---)
performance_volatility = {EMA_Cross: {pnl: 0.00,volatility": 0.005},RSI_Divergence": {pnl: 0.01,volatility": 0.006},Bollinger_Squeeze": {pnl: -0.008,volatility": 0.002},Volume_Breakout": {pnl: 0.04,volatility": 0.01},
}
tensor_manager.update_weights(MarketPhase.VOLATILITY, performance_volatility)
print(Weights for VOLATILITY phase after update:)
print(tensor_manager.get_strategy_weights_for_phase(MarketPhase.VOLATILITY))
    print(f"Current Active Phase: {tensor_manager.get_active_phase()})
print(\n--- Current State and Metrics ---)
state = tensor_manager.get_current_state()'
print(fTotal Updates: {state['metrics']['total_updates']})'print(f"Phase Transitions: {state['metrics']['phase_transitions']})'print(f"Active Phase: {state['metrics']['active_phase']})print(Full Weight Tensor:)for row in state[current_weight_tensor]:
        print([f{x:.4f} for x in row])
print(\n--- Resetting the tensor ---)
    tensor_manager.reset()
print(Weights after reset(across all phases):)
print(tensor_manager.get_current_state()[current_weight_tensor])""'"
"""
