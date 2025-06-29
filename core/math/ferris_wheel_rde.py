import scipy as sp

"""
Ferris Wheel RDE (Recursive Dualistic Engine)
============================================

Mathematical implementation of the Ferris Wheel RDE system for Schwabot.
Implements recursive dualistic logic for trading strategy execution, phase detection,
strategy weight normalization, phase-influenced selection, and NCCO feedback.

Core Concepts:
- Recursive Dualistic Engine (RDE)
- Strategy weight normalization and smoothing
- Bit-mode strategy selection
- Ferris phase detection (cycle, phase angle, ascent/peak/descent/trough)
- Phase-influenced strategy weighting
- Softmax/argmax final selection
- NCCO feedback/memory update
- Extension points for mining and CoinMarketCap integration
- Strategy mutation feedback with trend-based evolution
- Multi-timeframe phase analysis with weighted Hilbert transforms
- Strategy reinforcement learning with reward updates

Mathematical Foundation:
- Recursive Functions: Self-referential mathematical structures
- Dualistic Logic: Binary state management
- Orbital Mechanics: Circular trading patterns
- Information Theory: Entropy and pattern recognition
- Quantum Simulation: Classical approximation of quantum behaviors
- Reinforcement Learning: Performance-based strategy adaptation
"""

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --- Data Classes ---


@dataclass
class FerrisState:
    """Represents a Ferris Wheel state."""

    phase: int
    bit_state: int
    rotation_count: int
    entropy_level: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRecord:
    """Represents a strategy's performance and weight."""

    name: str
    performance: float
    weight: float
    smoothed_weight: float
    last_update: float = field(default_factory=time.time)
    phase_modifier: float = 1.0


@dataclass
class NCCOHistoryRecord:
    """Represents a feedback record for NCCO memory."""

    timestamp: float
    strategy: str
    weight: float
    phase_modifier: float
    probability: float
    phase: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NCCO:
    """Neural Cycle Control Object with RDE context integration."""

    id: str
    price_delta: float
    bit_mode: int
    score: float
    rde_context: Optional[FerrisState] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# --- Ferris Wheel RDE Core ---


class FerrisWheelRDE:
    """
    Ferris Wheel RDE (Recursive Dualistic Engine) implementation.
    Implements full mathematical logic for phase-cycling, memory-weighted,
    bit-mode-influenced strategy selection with feedback.
    """

    def __init__(self, max_phases: int = 256):
        self.max_phases = max_phases
        self.current_phase = 0
        self.rotation_count = 0
        self.bit_phase = 4  # Default to 4-bit
        self.states: List[FerrisState] = []
        self.memory_bank: Dict[str, Any] = {}
        self.ncco_history: List[NCCOHistoryRecord] = []
        self.standard_bits = [4, 8, 42]
        self.alpha = 0.3  # Smoothing factor
        self.learning_rate = 0.05  # Strategy mutation learning rate
        self.reward_rate = 0.1  # Reinforcement learning update rate

        # Strategy configuration
        self.strategy_sets = {4: ["hold"], 8: ["stable_swap"], 42: ["flip", "exit"]}
        self.phase_modifiers = {
            ("hold", "ascent"): 1.2,
            ("flip", "descent"): 1.3,
            ("exit", "peak"): 1.5,
            ("entry", "trough"): 1.4,
        }

        # Strategy performance tracking with history
        self.strategy_performance: Dict[str, float] = {s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]}
        self.strategy_weights: Dict[str, float] = {s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]}
        self.strategy_smoothed: Dict[str, float] = {s: 1.0 for s in ["hold", "stable_swap", "flip", "exit", "entry"]}
        self.strategy_performance_history: Dict[str, List[float]] = {
            s: [] for s in ["hold", "stable_swap", "flip", "exit", "entry"]
        }

        # Multi-timeframe phase analysis
        self.phase_windows = [16, 32, 64]
        self.phase_weights = [0.2, 0.5, 0.3]

        # State tracking
        self.last_phase_angle = 0.0
        self.last_price = None
        self.last_time = None
        self.price_history: List[Tuple[float, float]] = []  # (timestamp, price)

        logger.info(f"🎡 Ferris Wheel RDE initialized with max_phases={max_phases}")

    # --- Strategy Weight Normalization ---
    def normalize_weights(self, performance: Dict[str, float]) -> Dict[str, float]:
        total = sum(performance.values())
        if total == 0:
            n = len(performance)
            return {k: 1.0 / n for k in performance}
        return {k: v / total for k, v in performance.items()}

    # --- Exponential Smoothing ---
    def smooth_weights(
        self, old_weights: Dict[str, float], performance: Dict[str, float], alpha: float
    ) -> Dict[str, float]:
        return {k: (1 - alpha) * old_weights.get(k, 1.0) + alpha * performance.get(k, 1.0) for k in performance}

    # --- Strategy Mutation Feedback ---
    def mutate_strategy_weights(self, success: Dict[str, float], learning_rate: float = None) -> None:
        """Mutate strategy weights based on success trends."""
        if learning_rate is None:
            learning_rate = self.learning_rate

        for strat, score in success.items():
            if strat not in self.strategy_performance_history:
                self.strategy_performance_history[strat] = []

            hist = self.strategy_performance_history[strat]
            hist.append(score)

            # Keep last 10 performance scores
            if len(hist) > 10:
                hist.pop(0)

            # Calculate trend using linear regression
            if len(hist) > 2:
                x = np.arange(len(hist))
                trend = np.polyfit(x, hist, 1)[0]  # Slope of linear fit
            else:
                trend = 0.0

            # Apply mutation: positive trend increases weight
            delta = learning_rate * trend
            current_perf = self.strategy_performance.get(strat, 1.0)
            self.strategy_performance[strat] = max(0.1, current_perf + delta)  # Ensure positive

            logger.debug(
                f"Strategy {strat} mutation: trend={trend:.4f}, delta={delta:.4f}, new_perf={self.strategy_performance[strat]:.4f}"
            )

    # --- Strategy Reinforcement Learning ---
    def update_strategy_reward(self, strategy: str, reward: float, reward_rate: float = None) -> None:
        """Update strategy performance based on reward/penalty."""
        if reward_rate is None:
            reward_rate = self.reward_rate

        current = self.strategy_performance.get(strategy, 1.0)
        new_performance = (1 - reward_rate) * current + reward_rate * reward

        # Ensure performance stays in reasonable bounds
        self.strategy_performance[strategy] = max(0.1, min(10.0, new_performance))

        logger.debug(
            f"Strategy {strategy} reward update: reward={reward:.3f}, old={current:.3f}, new={self.strategy_performance[strategy]:.3f}"
        )

    # --- Bit-Mode Strategy Selection ---
    def get_strategy_set(self, bit_mode: int) -> List[str]:
        return self.strategy_sets.get(bit_mode, ["hold"])

    # --- Entropy Calculation ---
    def _calculate_entropy(self, data: str) -> float:
        """Calculate entropy of a string using Shannon's formula."""
        if not data:
            return 0.0

        # Count character frequencies
        char_counts = {}
        for char in data:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Calculate probabilities and entropy
        length = len(data)
        entropy = 0.0

        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    # --- Ferris Phase Detection ---
    def update_price_history(self, price: float, timestamp: Optional[float] = None):
        t = timestamp if timestamp is not None else time.time()
        self.price_history.append((t, price))
        if len(self.price_history) > 1000:
            self.price_history.pop(0)

    def get_phase_angle(self, window: int = 32) -> float:
        """Get phase angle using Hilbert transform on single window."""
        if len(self.price_history) < 2:
            return 0.0

        try:
            from scipy.signal import hilbert

            times, prices = zip(*self.price_history[-window:])
            p = np.array(prices)
            # Detrend
            p = p - np.mean(p)
            # Estimate phase angle using Hilbert transform
            analytic_signal = hilbert(p)
            phase_angles = np.angle(analytic_signal)
            return float(phase_angles[-1]) % (2 * math.pi)
        except ImportError:
            # Fallback: use price delta
            return 0.0

    def get_multi_phase_angle(self, windows: List[int] = None, weights: List[float] = None) -> float:
        """Get weighted phase angle from multiple timeframes."""
        if windows is None:
            windows = self.phase_windows
        if weights is None:
            weights = self.phase_weights

        if len(windows) != len(weights):
            raise ValueError("Windows and weights must have same length")

        try:
            from scipy.signal import hilbert
        except ImportError:
            return self.get_phase_angle()

        phase_sum = 0.0
        total_weight = 0.0

        for w, a in zip(windows, weights):
            if len(self.price_history) >= w:
                times, prices = zip(*self.price_history[-w:])
                p = np.array(prices) - np.mean(prices)
                analytic = hilbert(p)
                angle = np.angle(analytic)[-1] % (2 * math.pi)
                phase_sum += a * angle
                total_weight += a

        return phase_sum / total_weight if total_weight > 0 else 0.0

    def get_phase_state(self, price: float, timestamp: Optional[float] = None) -> str:
        self.update_price_history(price, timestamp)
        phase_angle = self.get_multi_phase_angle()  # Use multi-timeframe analysis
        if len(self.price_history) < 2:
            return "ascent"
        # Calculate dP/dt
        (t1, p1), (t0, p0) = self.price_history[-1], self.price_history[-2]
        dpdt = (p1 - p0) / (t1 - t0) if t1 != t0 else 0.0
        # Phase logic
        if dpdt > 0 and 0 <= phase_angle < math.pi:
            return "ascent"
        elif abs(phase_angle - math.pi) < 0.1:
            return "peak"
        elif dpdt < 0 and math.pi <= phase_angle < 2 * math.pi:
            return "descent"
        elif abs(phase_angle - 2 * math.pi) < 0.1:
            return "trough"
        else:
            return "ascent"

    # --- Phase-Influenced Strategy Weighting ---
    def apply_phase_modifiers(self, weights: Dict[str, float], phase: str) -> Dict[str, float]:
        adjusted = {}
        for s, w in weights.items():
            phi = self.phase_modifiers.get((s, phase), 1.0)
            adjusted[s] = w * phi
        # Re-normalize
        total = sum(adjusted.values())
        if total == 0:
            n = len(adjusted)
            return {k: 1.0 / n for k in adjusted}
        return {k: v / total for k, v in adjusted.items()}

    # --- Softmax and Argmax Selection ---
    def softmax(self, weights: Dict[str, float]) -> Dict[str, float]:
        exp_w = {k: math.exp(v) for k, v in weights.items()}
        total = sum(exp_w.values())
        return {k: v / total for k, v in exp_w.items()}

    def select_strategy(self, weights: Dict[str, float], method: str = "softmax") -> Tuple[str, float]:
        if method == "softmax":
            probs = self.softmax(weights)
            strategy = max(probs, key=probs.get)
            return strategy, probs[strategy]
        else:
            strategy = max(weights, key=weights.get)
            total = sum(weights.values())
            prob = weights[strategy] / total if total > 0 else 1.0 / len(weights)
            return strategy, prob

    # --- Main Ferris RDE Cycle ---
    def ferris_rde_cycle(self, price: float, bit_mode: int = 4, timestamp: Optional[float] = None) -> Dict[str, Any]:
        # 1. Update phase state
        phase = self.get_phase_state(price, timestamp)
        # 2. Normalize and smooth strategy weights
        strat_set = self.get_strategy_set(bit_mode)
        perf = {s: self.strategy_performance.get(s, 1.0) for s in strat_set}
        norm = self.normalize_weights(perf)
        smoothed = self.smooth_weights(self.strategy_smoothed, norm, self.alpha)
        # 3. Apply phase modifiers
        adjusted = self.apply_phase_modifiers(smoothed, phase)
        # 4. Select strategy
        strategy, prob = self.select_strategy(adjusted, method="softmax")

        # 5. Generate entropy from strategy decision
        hash_input = f"{strategy}_{bit_mode}_{phase}_{prob:.4f}_{self.current_phase}"
        entropy_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        entropy_val = self._calculate_entropy(entropy_hash)

        # 6. Feedback: update memory
        self.strategy_smoothed.update(smoothed)
        self.strategy_weights.update(norm)
        self.ncco_history.append(
            NCCOHistoryRecord(
                timestamp=timestamp if timestamp is not None else time.time(),
                strategy=strategy,
                weight=smoothed[strategy],
                phase_modifier=self.phase_modifiers.get((strategy, phase), 1.0),
                probability=prob,
                phase=phase,
                meta={"bit_mode": bit_mode, "entropy": entropy_val},
            )
        )

        # 7. Update Ferris state with actual entropy
        self.current_phase = (self.current_phase + 1) % self.max_phases
        self.rotation_count += 1
        ferris_state = FerrisState(
            phase=self.current_phase,
            bit_state=bit_mode,
            rotation_count=self.rotation_count,
            entropy_level=entropy_val,
            metadata={
                "strategy": strategy,
                "probability": prob,
                "phase": phase,
                "weights": adjusted,
                "hash_input": hash_input,
                "entropy_hash": entropy_hash[:16],  # Store first 16 chars for debugging
            },
        )
        self.states.append(ferris_state)

        # 8. Return decision with RDE context
        return {
            "strategy": strategy,
            "probability": prob,
            "phase": phase,
            "weights": adjusted,
            "ferris_state": ferris_state,
            "ncco_history": self.ncco_history[-10:],  # last 10
            "rde_context": ferris_state,
        }

    # --- NCCO Integration ---
    def create_ncco_from_rde(self, price: float, bit_mode: int, previous_price: Optional[float] = None) -> NCCO:
        """Create NCCO object from RDE cycle result."""
        rde_result = self.ferris_rde_cycle(price, bit_mode)

        # Calculate price delta
        if previous_price is not None:
            price_delta = (price - previous_price) / previous_price
        else:
            price_delta = 0.0

        # Calculate score from RDE probability and weights
        strategy = rde_result["strategy"]
        prob = rde_result["probability"]
        weight = rde_result["weights"].get(strategy, 1.0)
        score = prob * weight

        ncco_id = f"ncco_{self.rotation_count:06d}_{strategy}_{bit_mode}"

        return NCCO(
            id=ncco_id,
            price_delta=price_delta,
            bit_mode=bit_mode,
            score=score,
            rde_context=rde_result["rde_context"],
            metadata={
                "strategy": strategy,
                "probability": prob,
                "phase": rde_result["phase"],
                "entropy": rde_result["ferris_state"].entropy_level,
            },
        )

    # --- Visual Debug ---
    def print_cycle_summary(self, last_n: int = 5) -> None:
        """Print summary of last N cycles for debugging."""
        print(f"\n🎡 Ferris Wheel RDE Cycle Summary (Last {last_n}):")
        print("-" * 80)
        for i, record in enumerate(self.ncco_history[-last_n:]):
            print(
                f"[{i+1:2d}] [{record.phase.upper():<7}] {record.strategy:<12} | "
                f"weight={record.weight:.3f} | prob={record.probability:.3f} | "
                f"entropy={record.meta.get('entropy', 0.0):.3f}"
            )
        print("-" * 80)

    # --- Mining and CoinMarketCap Extension Points ---
    def mining_hook(self, pool_state: Optional[Dict[str, Any]] = None) -> None:
        """Extension point for mining pool integration."""
        # Implement mining logic or call external miner here
        pass

    def coinmarketcap_hook(self, cmc_data: Optional[Dict[str, Any]] = None) -> None:
        """Extension point for CoinMarketCap integration."""
        # Implement CMC data pull/processing here
        pass

    # --- Save/Load State ---
    def save_rde_state(self, filepath: str) -> None:
        state_data = {
            "current_phase": self.current_phase,
            "rotation_count": self.rotation_count,
            "bit_phase": self.bit_phase,
            "states": [state.__dict__ for state in self.states],
            "memory_bank": self.memory_bank,
            "ncco_history": [record.__dict__ for record in self.ncco_history],
            "strategy_performance": self.strategy_performance,
            "strategy_performance_history": self.strategy_performance_history,
            "timestamp": time.time(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

    def load_rde_state(self, filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            state_data = json.load(f)
        self.current_phase = state_data.get("current_phase", 0)
        self.rotation_count = state_data.get("rotation_count", 0)
        self.bit_phase = state_data.get("bit_phase", 4)
        self.memory_bank = state_data.get("memory_bank", {})
        self.states = [FerrisState(**s) for s in state_data.get("states", [])]
        self.ncco_history = [NCCOHistoryRecord(**r) for r in state_data.get("ncco_history", [])]
        self.strategy_performance = state_data.get("strategy_performance", self.strategy_performance)
        self.strategy_performance_history = state_data.get(
            "strategy_performance_history", self.strategy_performance_history
        )

    # --- Statistics ---
    def get_rde_statistics(self) -> Dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "rotation_count": self.rotation_count,
            "bit_phase": self.bit_phase,
            "total_states": len(self.states),
            "memory_bank_size": len(self.memory_bank),
            "ncco_history_size": len(self.ncco_history),
            "max_phases": self.max_phases,
            "strategy_performance": self.strategy_performance,
            "avg_entropy": np.mean([s.entropy_level for s in self.states[-100:]]) if self.states else 0.0,
        }


# Global instance for easy access
ferris_rde = FerrisWheelRDE()

if __name__ == "__main__":
    print("Ferris Wheel RDE Module Demonstration")
    print("=" * 80)

    # Simulate price data
    import random

    prices = [10000 + 1000 * math.sin(i / 10.0) + random.uniform(-50, 50) for i in range(100)]

    print("Running Ferris RDE cycles with enhanced features...")
    for i, price in enumerate(prices):
        result = ferris_rde.ferris_rde_cycle(price, bit_mode=4 if i % 3 == 0 else 8 if i % 3 == 1 else 42)

        # Simulate strategy mutation every 10 cycles
        if i % 10 == 0 and i > 0:
            success_scores = {result["strategy"]: random.uniform(0.5, 1.5)}
            ferris_rde.mutate_strategy_weights(success_scores)

        # Simulate reinforcement learning every 5 cycles
        if i % 5 == 0 and i > 0:
            reward = random.uniform(-0.5, 1.0)
            ferris_rde.update_strategy_reward(result["strategy"], reward)

        if i % 20 == 0:  # Print every 20th cycle
            print(
                f"Step {i:03d}: Price={price:.2f} | Phase={result['phase']:<7} | "
                f"Strategy={result['strategy']:<12} | Prob={result['probability']:.3f} | "
                f"Entropy={result['ferris_state'].entropy_level:.3f}"
            )

    print("\n" + "=" * 80)
    print("Final RDE Statistics:")
    stats = ferris_rde.get_rde_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    ferris_rde.print_cycle_summary(10)

    # Test NCCO creation
    print("\n" + "=" * 80)
    print("Testing NCCO Integration:")
    ncco = ferris_rde.create_ncco_from_rde(10500.0, 8, 10000.0)
    print(f"Created NCCO: {ncco.id}")
    print(f"  Price Delta: {ncco.price_delta:.4f}")
    print(f"  Score: {ncco.score:.4f}")
    print(f"  RDE Context Entropy: {ncco.rde_context.entropy_level:.4f}")
