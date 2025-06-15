"""
Fractal Controller - Master Orchestration System
===============================================

Central controller that orchestrates all fractal systems:
- Forever, Paradox, Eco, and Braid fractals
- Profit projection engine integration
- Dynamic fractal weighting
- Decision synthesis and execution

Mathematical Foundation:
Decision(t) = argmax[Σ w_i(t) · f_i(t) · P_projected(t+Δt)]
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Import our fractal systems
from .fractal_command_dispatcher import FractalCommandDispatcher, CommandType
from .braid_fractal import BraidFractal
from .profit_projection import ProfitProjectionEngine, ProfitHorizon
from .fractal_weights import FractalWeightBus

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    """Market tick data structure"""
    timestamp: float
    price: float
    volume: float
    volatility: float
    bid: float = 0.0
    ask: float = 0.0

@dataclass
class FractalDecision:
    """Decision output from fractal controller"""
    timestamp: float
    action: str  # "long", "short", "hold", "exit"
    confidence: float
    projected_profit: float
    hold_duration: int
    fractal_signals: Dict[str, float]
    fractal_weights: Dict[str, float]
    risk_assessment: Dict[str, Any]
    reasoning: str

class FractalController:
    """
    Master fractal controller for recursive profit optimization.
    
    Orchestrates all fractal systems to make optimal trading decisions
    based on mathematical synthesis of fractal signals and profit projections.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fractal controller.
        
        Args:
            config: Configuration dictionary for all subsystems
        """
        self.config = config or {}
        
        # Initialize core systems
        self.fractal_dispatcher = FractalCommandDispatcher()
        self.braid_fractal = BraidFractal()
        self.profit_engine = ProfitProjectionEngine()
        self.weight_bus = FractalWeightBus()
        
        # Market data tracking
        self.tick_history: deque = deque(maxlen=1000)
        self.decision_history: deque = deque(maxlen=200)
        
        # Fractal signal storage
        self.fractal_signals = {
            "forever": deque(maxlen=100),
            "paradox": deque(maxlen=100),
            "eco": deque(maxlen=100),
            "braid": deque(maxlen=100)
        }
        
        # Decision parameters
        self.confidence_threshold = 0.6
        self.min_profit_threshold = 10.0  # Minimum projected profit in basis points
        self.max_hold_duration = 50  # Maximum ticks to hold position
        
        # Threading for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.processing_lock = threading.Lock()
        
        # Performance tracking
        self.total_decisions = 0
        self.successful_decisions = 0
        self.current_position = None
        self.position_entry_time = None
        
        logger.info("Fractal Controller initialized with all subsystems")
    
    def process_tick(self, tick: MarketTick) -> FractalDecision:
        """
        Process new market tick and generate trading decision.
        
        Args:
            tick: New market tick data
            
        Returns:
            FractalDecision with recommended action
        """
        with self.processing_lock:
            # Store tick
            self.tick_history.append(tick)
            
            # Update all fractal systems in parallel
            fractal_futures = self._update_fractals_parallel(tick)
            
            # Wait for fractal updates to complete
            fractal_results = {}
            for fractal_name, future in fractal_futures.items():
                try:
                    fractal_results[fractal_name] = future.result(timeout=1.0)
                except Exception as e:
                    logger.error(f"Fractal {fractal_name} update failed: {e}")
                    fractal_results[fractal_name] = 0.0
            
            # Update braid fractal with all signals
            braid_signal = self._update_braid_fractal(fractal_results, tick)
            fractal_results["braid"] = braid_signal
            
            # Store fractal signals
            for name, signal in fractal_results.items():
                if name in self.fractal_signals:
                    self.fractal_signals[name].append(signal)
            
            # Update fractal weights based on performance
            self._update_fractal_weights(fractal_results, tick)
            
            # Generate profit projection
            profit_horizon = self._generate_profit_projection(tick)
            
            # Make trading decision
            decision = self._make_trading_decision(tick, fractal_results, profit_horizon)
            
            # Store decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            
            return decision
    
    def _update_fractals_parallel(self, tick: MarketTick) -> Dict[str, Any]:
        """Update all fractal systems in parallel."""
        futures = {}
        
        # Submit fractal update tasks
        futures["forever"] = self.executor.submit(
            self._update_forever_fractal, tick
        )
        futures["paradox"] = self.executor.submit(
            self._update_paradox_fractal, tick
        )
        futures["eco"] = self.executor.submit(
            self._update_eco_fractal, tick
        )
        
        return futures
    
    def _update_forever_fractal(self, tick: MarketTick) -> float:
        """Update Forever fractal system."""
        try:
            # Prepare tick data for Forever fractal
            tick_data = {
                "price": tick.price,
                "volume": tick.volume,
                "timestamp": tick.timestamp
            }
            
            # Send command to fractal dispatcher
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.CALCULATE, "TFF", tick_data
            )
            
            return result.get("tff_signal", 0.0)
            
        except Exception as e:
            logger.error(f"Forever fractal update error: {e}")
            return 0.0
    
    def _update_paradox_fractal(self, tick: MarketTick) -> float:
        """Update Paradox fractal system."""
        try:
            # Prepare tick data for Paradox fractal
            tick_data = {
                "price": tick.price,
                "volatility": tick.volatility,
                "timestamp": tick.timestamp
            }
            
            # Send command to fractal dispatcher
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.RESOLVE, "TPF", tick_data
            )
            
            return result.get("tpf_signal", 0.0)
            
        except Exception as e:
            logger.error(f"Paradox fractal update error: {e}")
            return 0.0
    
    def _update_eco_fractal(self, tick: MarketTick) -> float:
        """Update Eco fractal system."""
        try:
            # Prepare tick data for Eco fractal
            tick_data = {
                "price": tick.price,
                "volume": tick.volume,
                "volatility": tick.volatility,
                "timestamp": tick.timestamp
            }
            
            # Send command to fractal dispatcher
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.AMPLIFY, "TEF", tick_data
            )
            
            return result.get("tef_signal", 0.0)
            
        except Exception as e:
            logger.error(f"Eco fractal update error: {e}")
            return 0.0
    
    def _update_braid_fractal(self, fractal_results: Dict[str, float], tick: MarketTick) -> float:
        """Update Braid fractal with all fractal signals."""
        try:
            # Get recent fractal values
            f_vals = list(self.fractal_signals["forever"])[-5:] if self.fractal_signals["forever"] else [0.0]
            p_vals = list(self.fractal_signals["paradox"])[-5:] if self.fractal_signals["paradox"] else [0.0]
            e_vals = list(self.fractal_signals["eco"])[-5:] if self.fractal_signals["eco"] else [0.0]
            
            # Add current values
            f_vals.append(fractal_results.get("forever", 0.0))
            p_vals.append(fractal_results.get("paradox", 0.0))
            e_vals.append(fractal_results.get("eco", 0.0))
            
            # Create time range
            t_range = [i * 0.1 for i in range(len(f_vals))]
            
            # Update braid fractal
            braid_signal = self.braid_fractal.update(f_vals, p_vals, e_vals, t_range)
            
            return braid_signal
            
        except Exception as e:
            logger.error(f"Braid fractal update error: {e}")
            return 0.0
    
    def _update_fractal_weights(self, fractal_results: Dict[str, float], tick: MarketTick):
        """Update fractal weights based on current performance."""
        try:
            # Calculate performance metrics for each fractal
            for fractal_name, signal in fractal_results.items():
                if fractal_name == "braid":
                    continue
                    
                # Create performance feedback
                feedback = {
                    "profit_delta": self._estimate_profit_impact(signal, tick),
                    "prediction_accuracy": self._calculate_prediction_accuracy(fractal_name),
                    "volatility_handling": self._assess_volatility_handling(fractal_name, tick),
                    "success": signal > 0.1  # Simple success criterion
                }
                
                # Update weight bus
                self.weight_bus.update_performance(fractal_name, feedback)
                
        except Exception as e:
            logger.error(f"Weight update error: {e}")
    
    def _estimate_profit_impact(self, signal: float, tick: MarketTick) -> float:
        """Estimate profit impact of fractal signal."""
        # Simple heuristic: signal strength * volatility
        return signal * tick.volatility * 100  # Convert to basis points
    
    def _calculate_prediction_accuracy(self, fractal_name: str) -> float:
        """Calculate recent prediction accuracy for fractal."""
        # Simplified accuracy calculation
        recent_signals = list(self.fractal_signals[fractal_name])[-10:]
        if len(recent_signals) < 3:
            return 0.5
            
        # Check signal consistency (higher consistency = higher accuracy)
        signal_std = np.std(recent_signals)
        accuracy = np.exp(-signal_std)  # Lower std = higher accuracy
        return np.clip(accuracy, 0.0, 1.0)
    
    def _assess_volatility_handling(self, fractal_name: str, tick: MarketTick) -> float:
        """Assess how well fractal handles current volatility."""
        # Simple assessment based on signal stability during high volatility
        if tick.volatility > 0.5:  # High volatility
            recent_signals = list(self.fractal_signals[fractal_name])[-5:]
            if len(recent_signals) >= 2:
                signal_stability = 1.0 - np.std(recent_signals)
                return np.clip(signal_stability, 0.0, 1.0)
        
        return 0.7  # Default moderate handling
    
    def _generate_profit_projection(self, tick: MarketTick) -> ProfitHorizon:
        """Generate profit projection using current state."""
        try:
            # Get braid memory
            braid_memory = list(self.fractal_signals["braid"])
            
            # Get current fractal weights
            fractal_weights = self.weight_bus.get_weights()
            
            # Generate projection
            horizon = self.profit_engine.forecast_profit(
                braid_memory=braid_memory,
                tick_volatility=tick.volatility,
                fractal_weights=fractal_weights,
                current_profit=0.0  # Could be current P&L
            )
            
            return horizon
            
        except Exception as e:
            logger.error(f"Profit projection error: {e}")
            return self.profit_engine._empty_horizon()
    
    def _make_trading_decision(self, tick: MarketTick, fractal_results: Dict[str, float], 
                             profit_horizon: ProfitHorizon) -> FractalDecision:
        """Make final trading decision based on all inputs."""
        try:
            # Get current fractal weights
            weights = self.weight_bus.get_weights()
            
            # Calculate weighted fractal score
            weighted_score = 0.0
            for fractal_name, signal in fractal_results.items():
                weight = weights.get(fractal_name, 1.0)
                weighted_score += weight * signal
            
            # Normalize by total weight
            total_weight = sum(weights.values())
            if total_weight > 0:
                weighted_score /= total_weight
            
            # Get profit projection metrics
            max_projected_profit = max(profit_horizon.projected_profits) if profit_horizon.projected_profits else 0.0
            convergence_prob = profit_horizon.convergence_probability
            
            # Calculate overall confidence
            confidence = (
                0.4 * abs(weighted_score) +  # Signal strength
                0.3 * convergence_prob +      # Profit convergence probability
                0.3 * (1.0 - tick.volatility) # Stability factor
            )
            
            # Determine action
            action = "hold"
            reasoning = "Default hold"
            
            if confidence > self.confidence_threshold:
                if weighted_score > 0.2 and max_projected_profit > self.min_profit_threshold:
                    action = "long"
                    reasoning = f"Strong positive signals (score: {weighted_score:.3f}, profit: {max_projected_profit:.1f})"
                elif weighted_score < -0.2:
                    action = "short"
                    reasoning = f"Strong negative signals (score: {weighted_score:.3f})"
                else:
                    reasoning = f"Signals too weak (score: {weighted_score:.3f})"
            else:
                reasoning = f"Confidence too low ({confidence:.3f} < {self.confidence_threshold})"
            
            # Determine hold duration
            hold_duration = self.profit_engine.get_optimal_hold_duration(profit_horizon)
            hold_duration = min(hold_duration, self.max_hold_duration)
            
            # Create decision
            decision = FractalDecision(
                timestamp=tick.timestamp,
                action=action,
                confidence=confidence,
                projected_profit=max_projected_profit,
                hold_duration=hold_duration,
                fractal_signals=fractal_results.copy(),
                fractal_weights=weights.copy(),
                risk_assessment=self._assess_risk(tick, fractal_results),
                reasoning=reasoning
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            return self._default_decision(tick)
    
    def _assess_risk(self, tick: MarketTick, fractal_results: Dict[str, float]) -> Dict[str, Any]:
        """Assess risk factors for current decision."""
        return {
            "volatility_risk": "high" if tick.volatility > 0.7 else "moderate" if tick.volatility > 0.3 else "low",
            "fractal_divergence": np.std(list(fractal_results.values())),
            "braid_stability": self.braid_fractal.get_interference_summary().get("stability_index", 0.5),
            "overall_risk": "high" if tick.volatility > 0.7 or np.std(list(fractal_results.values())) > 0.5 else "moderate"
        }
    
    def _default_decision(self, tick: MarketTick) -> FractalDecision:
        """Create default decision for error cases."""
        return FractalDecision(
            timestamp=tick.timestamp,
            action="hold",
            confidence=0.0,
            projected_profit=0.0,
            hold_duration=0,
            fractal_signals={},
            fractal_weights={},
            risk_assessment={"overall_risk": "unknown"},
            reasoning="Error in decision processing - defaulting to hold"
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "total_decisions": self.total_decisions,
            "success_rate": self.successful_decisions / max(self.total_decisions, 1),
            "fractal_weights": self.weight_bus.get_weights(),
            "fractal_performance": self.weight_bus.get_performance_summary(),
            "braid_summary": self.braid_fractal.get_interference_summary(),
            "profit_projection_summary": self.profit_engine.get_projection_summary(),
            "recent_decisions": len(self.decision_history),
            "current_position": self.current_position
        }
    
    def update_position_outcome(self, profit_realized: float):
        """Update system with realized profit from position."""
        if self.total_decisions > 0:
            if profit_realized > 0:
                self.successful_decisions += 1
                
            # Update profit projection accuracy
            last_decision = self.decision_history[-1] if self.decision_history else None
            if last_decision:
                self.profit_engine.update_accuracy(
                    last_decision.projected_profit, profit_realized
                )
    
    def shutdown(self):
        """Shutdown fractal controller and cleanup resources."""
        self.executor.shutdown(wait=True)
        logger.info("Fractal Controller shutdown complete")

# Example usage
if __name__ == "__main__":
    # Test fractal controller
    controller = FractalController()
    
    # Simulate market tick
    tick = MarketTick(
        timestamp=time.time(),
        price=100.0,
        volume=1000,
        volatility=0.3,
        bid=99.9,
        ask=100.1
    )
    
    # Process tick
    decision = controller.process_tick(tick)
    
    print(f"Decision: {decision.action}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Projected profit: {decision.projected_profit:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    
    # Get system status
    status = controller.get_system_status()
    print(f"System status: {status}")
    
    controller.shutdown() 