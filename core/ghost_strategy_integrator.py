#!/usr/bin/env python3
"""Ghost Strategy Integrator – Unified pipeline integration for Schwabot.

This module provides the class-based architecture that integrates all ghost
mathematical components into Schwabot's strategy trigger pipeline following
the Ferris Wheel activation cycle:

    Hash Tick Check → Phase Sync → Glyph/Trigger Mapping → Execution

All modules are hard-linked and follow the standard data flow from core vectors
(BTC price, USDC flow, chart patterns) downstream to strategy_mapper,
profit_cycle_allocator, and matrix_fault_resolver.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Import all our mathematical modules
from .ghost_phase_integrator import GhostPhasePacket, compute_ghost_phase_packet
from .ghost_router import GhostRouter
from .ghost_strategy_matrix import (
    strategy_match_matrix, reward_matrix, dynamic_strategy_switch, update_strategy_matrix
)
from .phantom_entry_logic import phantom_entry_probability
from .phantom_exit_logic import phantom_exit_score
from .phantom_memory import PhantomMemory, GhostEvent
from .hash_tick_synchronizer import compute_tick_hash, sync_probability, hash_match_check
from .glyph_math_core import glyph_determinant, glyph_matrix, glyph_psi, glyph_tensor
from .btc_vector_aggregator import btc_vector, btc_eta, btc_xi
from .usdc_position_manager import usdc_position, usdc_trading, usdc_sigma, usdc_optimal_time

__all__: list[str] = [
    "GhostStrategyIntegrator",
    "StrategyTriggerPipeline", 
    "FerrisWheelActivator",
    "CoreVectorProcessor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CoreVectorData:
    """Upstream data from core vectors (BTC, USDC, patterns)."""
    
    btc_price: float
    btc_volume: float
    usdc_flow: float
    chart_patterns: List[str]
    timestamp: float
    market_sentiment: float = 0.0


@dataclass(slots=True)
class StrategyExecutionPacket:
    """Downstream packet for strategy_mapper and profit_cycle_allocator."""
    
    action: str  # "buy", "sell", "hold", "wait"
    volume: float
    confidence: float
    strategy_signature: str
    thermal_weight: float
    phase_sync: bool
    glyph_mapping: Dict[str, float]


# ---------------------------------------------------------------------------
# Ferris Wheel Activation Cycle
# ---------------------------------------------------------------------------


class FerrisWheelActivator:
    """Manages the Ferris Wheel activation cycle for strategy triggers."""
    
    def __init__(self, *, tick_tolerance: int = 2, sync_sigma: float = 1.0):
        """Initialize Ferris Wheel activator."""
        self.tick_tolerance = tick_tolerance
        self.sync_sigma = sync_sigma
        self.hash_registry: Dict[str, float] = {}
        self.cycle_position = 0
        
    def hash_tick_check(
        self,
        price: float,
        volume_delta: float,
        time_delta: float,
    ) -> tuple[str, bool]:
        """Step 1: Hash tick check with registry matching."""
        current_hash = compute_tick_hash(price, volume_delta, time_delta)
        is_match = hash_match_check(
            current_hash, 
            self.hash_registry, 
            tolerance=self.tick_tolerance
        )
        
        # Update registry
        if not is_match:
            self.hash_registry[current_hash] = time.time()
            
        return current_hash, is_match
    
    def phase_sync_check(
        self,
        tick_t1: float,
        tick_t2: float,
        xi_sync: bool,
    ) -> float:
        """Step 2: Phase synchronization probability."""
        return sync_probability(tick_t1, tick_t2, self.sync_sigma, xi_sync)
    
    def advance_cycle(self) -> int:
        """Advance Ferris Wheel position and return new position."""
        self.cycle_position = (self.cycle_position + 1) % 8
        return self.cycle_position


# ---------------------------------------------------------------------------
# Core Vector Processor
# ---------------------------------------------------------------------------


class CoreVectorProcessor:
    """Processes upstream core vector data (BTC, USDC, patterns)."""
    
    def __init__(self):
        """Initialize core vector processor."""
        self.btc_history: List[float] = []
        self.usdc_history: List[float] = []
        self.volume_history: List[float] = []
        
    def process_btc_vectors(
        self,
        exit_prices: Sequence[float],
        entry_prices: Sequence[float],
        volume_weights: Sequence[float],
        price_delta: float,
        time_delta: float,
    ) -> Dict[str, float]:
        """Process BTC price vectors into aggregated signals."""
        if not exit_prices or time_delta <= 0:
            return {"v_btc": 0.0, "eta_btc": 0.0, "xi_btc": 0.0}
            
        v_btc = btc_vector(exit_prices, entry_prices, volume_weights)
        eta_btc = btc_eta(price_delta, time_delta, volume_weights)
        xi_btc = btc_xi(v_btc, eta_btc)
        
        return {"v_btc": v_btc, "eta_btc": eta_btc, "xi_btc": xi_btc}
    
    def process_usdc_flows(
        self,
        holdings: Sequence[float],
        rates: Sequence[float],
        time_deltas: Sequence[float],
        alpha_entry: float,
        delta_buy: float,
        beta_exit: float,
        delta_sell: float,
    ) -> Dict[str, Any]:
        """Process USDC position and trading signals."""
        if not holdings:
            return {"position": 0.0, "trading": 0.0, "optimal_time": 0}
            
        position = usdc_position(holdings, rates, time_deltas)
        trading = usdc_trading(alpha_entry, delta_buy, beta_exit, delta_sell)
        
        # Compute sigma using dummy gradient (would use real gradient in practice)
        dummy_gradient = [0.1] * len(holdings)
        sigma_series = usdc_sigma(dummy_gradient, trading)
        optimal_time = usdc_optimal_time(sigma_series, theta_usdc=0.5)
        
        return {
            "position": position,
            "trading": trading,
            "sigma": sigma_series.tolist(),
            "optimal_time": optimal_time,
        }


# ---------------------------------------------------------------------------
# Strategy Trigger Pipeline
# ---------------------------------------------------------------------------


class StrategyTriggerPipeline:
    """Main strategy trigger pipeline that coordinates all components."""
    
    def __init__(
        self,
        *,
        max_memory_events: int = 1000,
        decay_lambda: float = 0.01,
    ):
        """Initialize strategy trigger pipeline."""
        self.ghost_router = GhostRouter()
        self.phantom_memory = PhantomMemory(
            max_events=max_memory_events,
            decay_lambda=decay_lambda,
        )
        self.ferris_wheel = FerrisWheelActivator()
        self.core_processor = CoreVectorProcessor()
        
        # Strategy matrix state
        self.strategy_matrix = np.zeros((4, 4), dtype=float)  # 4x4 default
        self.hash_edges = [0, 1000, 10000, 100000, 1000000]  # Hash bands
        self.zeta_edges = [-1.0, -0.5, 0.0, 0.5, 1.0]       # Zeta bands
        
    def process_trigger_cycle(
        self,
        core_data: CoreVectorData,
    ) -> StrategyExecutionPacket:
        """Execute complete trigger cycle following Ferris Wheel pattern."""
        
        # Step 1: Hash tick check
        current_hash, hash_match = self.ferris_wheel.hash_tick_check(
            core_data.btc_price,
            core_data.btc_volume,
            core_data.timestamp - time.time(),
        )
        
        # Step 2: Phase synchronization
        sync_prob = self.ferris_wheel.phase_sync_check(
            core_data.timestamp,
            time.time(),
            hash_match,
        )
        
        # Step 3: Glyph/Trigger mapping via ZPE and glyph math
        glyph_mapping = self._compute_glyph_mapping(core_data)
        
        # Step 4: Ghost phase integration
        phase_packet = self._compute_phase_integration(core_data, current_hash)
        
        # Step 5: Strategy matrix evaluation
        strategy_decision = self._evaluate_strategy_matrix(
            int(current_hash[:8], 16) % 1000000,  # Convert hash to int
            phase_packet.zeta_final,
        )
        
        # Step 6: Entry/Exit logic
        entry_prob = self._compute_entry_probability(core_data, phase_packet)
        exit_score = self._compute_exit_score(core_data, phase_packet)
        
        # Step 7: Final execution decision
        execution_packet = self._generate_execution_packet(
            strategy_decision,
            entry_prob,
            exit_score,
            sync_prob,
            glyph_mapping,
            phase_packet,
        )
        
        # Advance Ferris Wheel
        self.ferris_wheel.advance_cycle()
        
        return execution_packet
    
    def _compute_glyph_mapping(self, core_data: CoreVectorData) -> Dict[str, float]:
        """Compute glyph mapping using mathematical core functions."""
        # Simple price function for glyph determinant
        def price_func(x: float, y: float) -> float:
            return core_data.btc_price * (1 + 0.01 * x + 0.005 * y)
        
        # Compute glyph determinant at current price point
        g_det = glyph_determinant(price_func, 1.0, 1.0)
        
        # Compute glyph matrix with dummy weights
        g_matrix = glyph_matrix([g_det], [1.0])
        
        # Compute psi and tensor
        psi = glyph_psi(g_matrix, g_det)
        tensor = glyph_tensor([psi, psi * 0.5])  # Simple 2D gradient
        
        return {
            "determinant": g_det,
            "matrix": g_matrix,
            "psi": psi,
            "tensor_trace": float(np.trace(tensor)),
        }
    
    def _compute_phase_integration(
        self,
        core_data: CoreVectorData,
        current_hash: str,
    ) -> GhostPhasePacket:
        """Compute ghost phase integration packet."""
        # Get recent phantom memory events for echo
        recent_events = self.phantom_memory.get_recent_events(300.0)  # 5 min window
        h_echo = [current_hash] if not recent_events else [current_hash, current_hash]
        
        # Compute phase packet with dummy values (would use real signals)
        phase_packet = compute_ghost_phase_packet(
            H_t=current_hash,
            H_echo=h_echo,
            zeta_news_t=core_data.market_sentiment,
            lambda_sentiment_t=0.5,
            alpha_t=1.0,
            phi_fractal_t=core_data.btc_price / 50000.0,  # Normalized price
            nu_cycle_t=0.1,
            delta_alt_t=0.0,
            grad_phi_fractal_t=core_data.btc_volume / 1000.0,
            delta_nu_cycle_t=0.0,
            drift_t=0.05,
            q_exec_prev=100.0,
            q_exec_curr=core_data.btc_volume,
            delta_t=1.0,
        )
        
        # Store new phantom event
        phantom_event = GhostEvent(
            timestamp=core_data.timestamp,
            zeta=phase_packet.zeta_final,
            xi_ghost=phase_packet.mu_echo,
        )
        self.phantom_memory.add_event(phantom_event)
        
        return phase_packet
    
    def _evaluate_strategy_matrix(self, hash_int: int, zeta: float) -> int:
        """Evaluate strategy matrix for decision index."""
        # Create binary match matrix
        match_matrix = strategy_match_matrix(
            hash_int, zeta, self.hash_edges, self.zeta_edges
        )
        
        # Update strategy matrix with dummy reward
        dummy_reward = reward_matrix(
            match_matrix.astype(float),
            match_matrix.astype(float) * 0.1,  # Dummy gain
            match_matrix.astype(float) * zeta,
        )
        
        self.strategy_matrix = update_strategy_matrix(
            self.strategy_matrix,
            dummy_reward,
            match_matrix.astype(float),
        )
        
        # Dynamic strategy switch decision
        Q = np.ones(4)  # Quality scores
        T = np.ones(4) * 0.9  # Trust scores
        lam = np.ones(4)  # Lambda weights
        
        return dynamic_strategy_switch(Q, T, lam)
    
    def _compute_entry_probability(
        self,
        core_data: CoreVectorData,
        phase_packet: GhostPhasePacket,
    ) -> float:
        """Compute phantom entry probability."""
        return phantom_entry_probability(
            alpha_vec=[1.0, 0.8, 0.6],
            phi_vec=[0.5, phase_packet.mu_echo, phase_packet.zeta_final],
            zeta_final=phase_packet.zeta_final,
            mu_echo=phase_packet.mu_echo,
            price_now=core_data.btc_price,
            profit_band=(45000.0, 55000.0),  # Example bands
        )
    
    def _compute_exit_score(
        self,
        core_data: CoreVectorData,
        phase_packet: GhostPhasePacket,
    ) -> float:
        """Compute phantom exit score."""
        return phantom_exit_score(
            lambda_trust=phase_packet.C_t,
            profit_delta=100.0,  # Dummy profit delta
            zeta_derivative=0.1,  # Dummy derivative
        )
    
    def _generate_execution_packet(
        self,
        strategy_idx: int,
        entry_prob: float,
        exit_score: float,
        sync_prob: float,
        glyph_mapping: Dict[str, float],
        phase_packet: GhostPhasePacket,
    ) -> StrategyExecutionPacket:
        """Generate final execution packet for downstream processing."""
        
        # Determine action based on probabilities
        if entry_prob > 0.7 and sync_prob > 0.5:
            action = "buy"
            confidence = entry_prob * sync_prob
        elif exit_score > 0.6:
            action = "sell" 
            confidence = exit_score
        elif sync_prob < 0.3:
            action = "wait"
            confidence = 1.0 - sync_prob
        else:
            action = "hold"
            confidence = 0.5
        
        # Calculate volume and thermal weight
        volume = confidence * 100.0  # Scale with confidence
        thermal_weight = phase_packet.C_t
        
        # Generate strategy signature
        strategy_signature = f"ghost_strat_{strategy_idx}_{int(phase_packet.mu_echo * 1000):03d}"
        
        return StrategyExecutionPacket(
            action=action,
            volume=volume,
            confidence=confidence,
            strategy_signature=strategy_signature,
            thermal_weight=thermal_weight,
            phase_sync=(sync_prob > 0.5),
            glyph_mapping=glyph_mapping,
        )


# ---------------------------------------------------------------------------
# Main Ghost Strategy Integrator
# ---------------------------------------------------------------------------


class GhostStrategyIntegrator:
    """Main integrator class that coordinates all ghost strategy components."""
    
    def __init__(self, **kwargs):
        """Initialize ghost strategy integrator."""
        self.trigger_pipeline = StrategyTriggerPipeline(**kwargs)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def process_market_data(
        self,
        btc_price: float,
        btc_volume: float,
        usdc_flow: float,
        chart_patterns: Optional[List[str]] = None,
        market_sentiment: float = 0.0,
    ) -> StrategyExecutionPacket:
        """Main entry point for processing market data through ghost pipeline."""
        
        # Package core vector data
        core_data = CoreVectorData(
            btc_price=btc_price,
            btc_volume=btc_volume,
            usdc_flow=usdc_flow,
            chart_patterns=chart_patterns or [],
            timestamp=time.time(),
            market_sentiment=market_sentiment,
        )
        
        # Log input data
        self.logger.debug(f"Processing market data: BTC=${btc_price}, Vol={btc_volume}")
        
        # Execute trigger cycle
        execution_packet = self.trigger_pipeline.process_trigger_cycle(core_data)
        
        # Log output decision
        self.logger.info(
            f"Ghost strategy decision: {execution_packet.action} "
            f"(confidence={execution_packet.confidence:.3f}, "
            f"volume={execution_packet.volume:.2f})"
        )
        
        return execution_packet
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status for monitoring."""
        return {
            "ferris_wheel_position": self.trigger_pipeline.ferris_wheel.cycle_position,
            "phantom_memory_events": self.trigger_pipeline.phantom_memory.event_count,
            "hash_registry_size": len(self.trigger_pipeline.ferris_wheel.hash_registry),
            "strategy_matrix_shape": self.trigger_pipeline.strategy_matrix.shape,
        } 