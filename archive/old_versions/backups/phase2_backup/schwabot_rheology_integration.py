"""Module for Schwabot trading system."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core.schwabot_rheology_integration
================================================

Rheological integration layer for the Schwabot trading system.

Key concepts
-------------
Rheology studies the flow of complex fluids.  When applied to trading we treat
• **market stress** (τ) – combined effect of volatility, slippage, order-book
imbalance …
• **viscosity**  (η) – resistance to changing strategy; higher ⇒ slower
switching.
• **shear rate** (γ̇) – rate of market *deformation* → normalised price change
per unit time.
• **profit gradient** (∇P) – derivative of cumulative PnL.
• **smoothing force** (Ω) – damping term ensuring stability.

This module converts raw market/strategy metrics into a *RheologicalState* that
other components (e.g. `matrix_mapper`, demo scripts) can query.  It also
provides helpers for tag generation, failure-reconvergence analysis, and a very
light quantum-tensor hook so we remain compatible with
`QuantumMathematicalBridge`.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

    try:
    # Optional – full bridge provides extra analytics;
    # we keep a fall-back so unit-tests do not fail when SciPy / CuPy not present.
    from .quantum_mathematical_bridge import QuantumMathematicalBridge, QuantumTensor
    except Exception:  # pragma: no cover – graceful degrade

        class QuantumMathematicalBridge:  # type: ignore:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        """Placeholder when quantum bridge is unavailable."""

        def fuse_tensors(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:  # noqa: D401
    return np.tensordot(a, b, axes=1)

    QuantumTensor = np.ndarray  # type: ignore


    logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------------------
    # Dataclasses / Enums
    # ---------------------------------------------------------------------------


    autogen_time = lambda: time.time()  # noqa: E731 – small lambda is ok here


    @dataclass
        class RheologicalState:
    """Class for Schwabot trading functionality."""
        """Class for Schwabot trading functionality."""
        stress: float = 0.0
        viscosity: float = 1.0
        shear_rate: float = 0.0
        memory_decay: float = 0.95
        profit_gradient: float = 0.0
        entropy: float = 0.5
        smoothing_force: float = 0.0
        phase_id: str = ""
        timestamp: float = field(default_factory=autogen_time)


            class RheologicalFlowType(str, Enum):
    """Class for Schwabot trading functionality."""
            """Class for Schwabot trading functionality."""
            NEWTONIAN = "newtonian"
            SHEAR_THINNING = "shear_thinning"
            SHEAR_THICKENING = "shear_thickening"
            BINGHAM_PLASTIC = "bingham_plastic"
            VISCOELASTIC = "viscoelastic"


            @dataclass
                class RheologicalTag:
    """Class for Schwabot trading functionality."""
                """Class for Schwabot trading functionality."""
                tag_id: str
                rheological_state: RheologicalState
                strategy_id: str
                profit_delta: float
                confidence: float
                failure_count: int = 0
                success_count: int = 0
                last_update: float = field(default_factory=autogen_time)
                metadata: Dict[str, Any] = field(default_factory=dict)


                # ---------------------------------------------------------------------------
                # Main integration class
                # ---------------------------------------------------------------------------


                    class SchwabotRheologyIntegration:
    """Class for Schwabot trading functionality."""
                    """Class for Schwabot trading functionality."""
                    """Implements rheological analytics for strategy orchestration."""

                    # Tunable constants – these can be surfaced through config later
                    _VISCOSITY_MIN = 0.1
                    _VISCOSITY_MAX = 10.0
                    _SMOOTHING_WEIGHTS = (0.4, 0.3, 0.3)  # (entropy, |∇P|, viscosity-offset)

                        def __init__(self, quantum_bridge: Optional[QuantumMathematicalBridge] = None) -> None:
                        self._bridge = quantum_bridge or QuantumMathematicalBridge()
                        self._state = RheologicalState()
                        self._state_lock = threading.Lock()
                        self._tag_lock = threading.Lock()
                        self._tags: Dict[str, RheologicalTag] = {}
                        # local history
                        self._profit_history: List[float] = []
                        self._gradient_history: List[float] = []
                        logger.info("🧬 Schwabot Rheological Integration initialised → ready")

                        # ------------------------------------------------------------------
                        # PUBLIC API
                        # ------------------------------------------------------------------

                        def calculate_rheological_state(
                        self,
                        market_data: Optional[Dict[str, Any]] = None,
                        strategy_perf: Optional[Dict[str, Any]] = None,
                            ) -> RheologicalState:
                            """Compute a new *RheologicalState* from market / performance snapshots."""
                            market_data = market_data or {}
                            strategy_perf = strategy_perf or {}

                            # Stress combines volatility, volume delta and price momentum
                            volatility = float(market_data.get("volatility", 0.3))
                            volume_delta = float(market_data.get("volume_delta", 0.0))
                            price_momentum = float(market_data.get("price_momentum", 0.0))
                            stress = math.sqrt(volatility ** 2 + volume_delta ** 2 + price_momentum ** 2)

                            # Viscosity rises with strategy switch frequency (more switches ⇒ thicker fluid)
                            switch_freq = float(strategy_perf.get("switch_frequency", 1.0))
                            viscosity = 1.0 + math.tanh(switch_freq / 5.0)
                            viscosity = float(np.clip(viscosity, self._VISCOSITY_MIN, self._VISCOSITY_MAX))

                            # Shear rate – relative price change per unit time
                            price_now = float(market_data.get("price", 1.0))
                            time_delta = float(market_data.get("time_delta", 1.0))
                            price_prev = getattr(self, "_last_price", price_now)
                            shear_rate = abs(price_now - price_prev) / max(price_prev, 1e-9) / max(time_delta, 1e-6)
                            self._last_price = price_now

                            # Profit gradient
                            profit_hist = strategy_perf.get("profit_history", [])
                                if profit_hist:
                                self._profit_history.extend(profit_hist)
                                    if len(self._profit_history) > 2:
                                    profit_gradient = float(np.gradient(self._profit_history)[-1])
                                        else:
                                        profit_gradient = 0.0
                                        self._gradient_history.append(profit_gradient)

                                        # Entropy simple proxy – bounded volatility + volume influence
                                        entropy = min(volatility + abs(volume_delta) * 0.5, 1.0)

                                        smoothing_force = self._calculate_smoothing_force(entropy, profit_gradient, viscosity)

                                        phase_id = self._gen_phase_id(stress, viscosity, shear_rate)

                                            with self._state_lock:
                                            self._state = RheologicalState(
                                            stress=stress,
                                            viscosity=viscosity,
                                            shear_rate=shear_rate,
                                            profit_gradient=profit_gradient,
                                            memory_decay=0.95,
                                            entropy=entropy,
                                            smoothing_force=smoothing_force,
                                            phase_id=phase_id,
                                            timestamp=time.time(),
                                            )
                                        return self._state

                                        # ------------------------------------------------------------------
                                        # Tagging / gradient helpers
                                        # ------------------------------------------------------------------

                                        def create_rheological_tag(
                                        self,
                                        strategy_id: str,
                                        profit_delta: float,
                                        confidence: float = 0.5,
                                        metadata: Optional[Dict[str, Any]] = None,
                                            ) -> RheologicalTag:
                                            """Attach a *RheologicalTag* to a strategy for later analytics."""
                                                with self._state_lock:
                                                state_snapshot = self._state
                                                tag_id = hashlib.sha1(f"{strategy_id}{time.time()}".encode()).hexdigest()[:12]
                                                tag = RheologicalTag(
                                                tag_id=tag_id,
                                                rheological_state=state_snapshot,
                                                strategy_id=strategy_id,
                                                profit_delta=profit_delta,
                                                confidence=confidence,
                                                metadata=metadata or {},
                                                )
                                                    with self._tag_lock:
                                                    self._tags[tag_id] = tag
                                                return tag

                                                    def optimize_profit_gradient_flow(self) -> Dict[str, float]:
                                                    """Perform a lightweight optimisation on recorded ∇P history."""
                                                        if len(self._gradient_history) < 5:
                                                    return {"status": "insufficient_data"}
                                                    arr = np.array(self._gradient_history[-100:])
                                                    trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
                                                    volatility = float(np.std(arr))
                                                    score = trend / (volatility + 1e-6)
                                                return {"trend": trend, "volatility": volatility, "score": score}

                                                # ------------------------------------------------------------------
                                                # Failure reconvergence
                                                # ------------------------------------------------------------------

                                                    def handle_failure_reconvergence(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
                                                    """Analyse a failure scenario and propose reconvergence plan."""
                                                    failure_type = scenario.get("failure_type", "unknown")
                                                    magnitude = float(scenario.get("magnitude", 1.0))
                                                    ctx = scenario.get("context", {})

                                                    # Simple heuristic – bigger magnitude ⇒ longer recovery, higher viscosity
                                                    recovery_time = magnitude * 30  # seconds
                                                    recovery_viscosity = float(np.clip(self._state.viscosity * (1 + magnitude / 5), 1.0, self._VISCOSITY_MAX))

                                                    plan = {
                                                    "failure_type": failure_type,
                                                    "estimated_recovery_time": recovery_time,
                                                    "recommended_viscosity": recovery_viscosity,
                                                    "context": ctx,
                                                    "timestamp": time.time(),
                                                    }
                                                return plan

                                                # ------------------------------------------------------------------
                                                # Quantum-tensor hook
                                                # ------------------------------------------------------------------

                                                    def quantum_rheological_integration(self, quantum_tensor: QuantumTensor) -> float:
                                                    """Combine quantum tensor with current rheological damping factor."""
                                                    damping = 1.0 / (1.0 + self._state.viscosity)
                                                    fused = self._bridge.fuse_tensors(quantum_tensor, quantum_tensor.T)  # simple example
                                                return float(np.sum(np.abs(fused)) * damping)

                                                # ------------------------------------------------------------------
                                                # Misc helpers
                                                # ------------------------------------------------------------------

                                                    def _determine_rheological_action(self) -> str:
                                                    """Choose an action given the current state – used by demo scripts."""
                                                        if self._state.stress > 2.0:
                                                    return "reduce_position"
                                                        if self._state.viscosity > 5.0:
                                                    return "hold_position"
                                                        if self._state.shear_rate > 1.0:
                                                    return "scalp"
                                                return "normal_trade"

                                                # ------------------------------------------------------------------
                                                # Public system status
                                                # ------------------------------------------------------------------

                                                    def get_system_status(self) -> Dict[str, Any]:
                                                        with self._state_lock:
                                                    return {
                                                    "state": self._state.__dict__,
                                                    "tag_count": len(self._tags),
                                                    "profit_gradient_history": self._gradient_history[-20:],
                                                    }

                                                    # ------------------------------------------------------------------
                                                    # Internal utilities
                                                    # ------------------------------------------------------------------

                                                    @staticmethod
                                                        def _calculate_smoothing_force(entropy: float, profit_grad: float, viscosity: float) -> float:
                                                        a, b, c = SchwabotRheologyIntegration._SMOOTHING_WEIGHTS
                                                    return float(np.clip(a * entropy + b * abs(profit_grad) + c * (viscosity - 1.0), 0.0, 1.0))

                                                    @staticmethod
                                                        def _gen_phase_id(stress: float, viscosity: float, shear_rate: float) -> str:
                                                        seed = f"{stress:.3f}{viscosity:.3f}{shear_rate:.3f}{int(time.time())}"
                                                    return hashlib.sha1(seed.encode()).hexdigest()[:10]
