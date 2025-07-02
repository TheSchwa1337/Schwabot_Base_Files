import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.correction_overlay_matrix import CorrectionOverlayMatrix
from core.drift_shell_engine import DriftShellEngine, ProfitVector, TimingMetrics
from core.dualistic_state_machine import (
from core.latency_compensator import DualisticState, LatencyCompensator
from core.live_execution_mapper import LiveExecutionMapper
from core.portfolio_tracker import PortfolioTracker
    from core.profit_vector_forecast import ProfitVectorForecastEngine
    from core.risk_manager import RiskManager
from core.trade_executor import TradeExecutor
import random
from typing import Tuple
from typing import Callable



# -*- coding: utf-8 -*-
"""Schwabot Unified Integration - Long-Term Trading System Architecture."

Implements the complete integration of Schwabot's advanced systems:'
- Drift Shell Engine with TDCF/BCOE/PVF/CIF mathematics
- ALEPH/ALIF dualistic state management
- Quantum Static Core verification
- Latency compensation and temporal drift correction
- Profit vector forecasting and correction overlays

This module serves as the central hub that orchestrates all subsystems
for self-correcting AI trading with quantum-aware timing alignment.:

Mathematical Foundation:
- Unified Decision: D(t) = DSM(ALEPH/ALIF) × DSE(timing) × LC(latency) × QSC(quantum)
- Profit Optimization: P(t) = ∫[PVF(vectors) × CIF(corrections) × Risk(management)] dt
- Temporal Coherence: TC(t) = Validity(memory) × Coherence(states) × Alignment(quantum)
"""

# Import core Schwabot systems
try:
        DualisticStateMachine,
StateType,
TransitionEvent,
)
except ImportError as e:
    logging.warning(f"Some core modules not available: {e}")

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDecision:
    """Represents a unified trading decision from all subsystems."""

timestamp: float
asset: str
price: float
volume: float

# Dualistic state info
current_state: str  # ALEPH or ALIF
state_confidence: float
nibble_score: float
rittle_score: float

# Timing and drift analysis
memory_validity: float
timing_coherence: float
latency_correction: float
drift_shell_radius: float

# Profit vector and corrections
    profit_vector: ProfitVector
correction_factors: Dict[str, float]
anomalies_detected: List[str]

# Final decision
should_trade: bool
trade_direction: str  # "long", "short", "hold"
position_size: float
confidence_score: float
risk_adjustment: float

# Integration metadata
processing_time_ms: float
subsystem_scores: Dict[str, float]
quantum_phase: float
entropy_level: float


@dataclass
class SystemHealth:
    """Overall system health metrics."""

drift_engine_health: float
state_machine_health: float
latency_compensator_health: float
profit_forecast_health: float
    correction_matrix_health: float
overall_health: float
last_health_check: float


class SchwabotUnifiedIntegration:
    """Unified integration hub for all Schwabot trading systems."""

def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified integration system."

Args:
            config: Configuration dictionary for all subsystems
"""
self.config = config or self._default_config()

# Initialize all core subsystems
self._initialize_subsystems()

# Integration state
self.is_running = False
self.last_decision_time = 0.0
self.decision_history = []
self.max_history_size = self.config.get("max_decision_history", 1000)

# Performance tracking
self.stats = {
"total_decisions": 0,
"successful_trades": 0,
"rejected_decisions": 0,
"avg_processing_time": 0.0,
"dualistic_transitions": 0,
"quantum_adjustments": 0,
"anomaly_corrections": 0,
"drift_compensations": 0,
}

# System health monitoring
self.system_health = SystemHealth(
drift_engine_health=1.0,
            state_machine_health=1.0,
            latency_compensator_health=1.0,
            profit_forecast_health=1.0,
            correction_matrix_health=1.0,
            overall_health=1.0,
last_health_check=time.time(),
)

# Callback system for external integration
self.decision_callbacks: List[Callable[[UnifiedDecision], None]] = []
self.health_callbacks: List[Callable[[SystemHealth], None]] = []

logger.info("🚀 Schwabot Unified Integration System initialized")

def _default_config(self) -> Dict[str, Any]:
        """Default configuration for all subsystems."""
return {
# Drift Shell Engine
"drift_shell_radius": 144.44,
"drift_memory_buffer_size": 512,
"drift_confidence_threshold": 0.7,
            "drift_timing_threshold_ms": 300.0,
# Dualistic State Machine
"dsm_entropy_threshold": 0.6,
            "dsm_quantum_phase_sensitivity": 0.3,
            "dsm_transition_cooldown_ms": 1000.0,
# Latency Compensator
"latency_max_acceptable_ms": 300.0,
            "latency_correction_alpha": 0.1,
            "latency_memory_decay": 0.05,
            # Profit Vector Forecast
"pvf_lookback_periods": 144,
"pvf_fibonacci_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
"pvf_volatility_window": 50,
# Correction Overlay Matrix
            "com_anomaly_sensitivity": 0.1,
"com_correction_weights": {
"quantum": 0.3,
                "tensor": 0.4,
                "smart_money": 0.3,
},
"com_max_correction_magnitude": 0.5,
# Trading execution
"initial_portfolio_cash": 100000.0,
"simulation_mode": True,
"enable_risk_management": True,
            "enable_portfolio_tracking": True,
# Integration settings
"decision_frequency_ms": 1000.0,
            "health_check_interval_s": 30.0,
"max_decision_history": 1000,
"enable_quantum_verification": True,
"enable_self_correction": True,
}

def _initialize_subsystems(self) -> None:
        """Initialize all core subsystems."""
try:
            # Drift Shell Engine
self.drift_engine = DriftShellEngine(
shell_radius=self.config["drift_shell_radius"],
memory_buffer_size=self.config["drift_memory_buffer_size"],
confidence_threshold=self.config["drift_confidence_threshold"],
                timing_threshold_ms=self.config["drift_timing_threshold_ms"],
)

# Dualistic State Machine
self.state_machine = DualisticStateMachine(
entropy_threshold=self.config["dsm_entropy_threshold"],
quantum_phase_sensitivity=self.config["dsm_quantum_phase_sensitivity"],
transition_cooldown_ms=self.config["dsm_transition_cooldown_ms"],
)

# Latency Compensator
self.latency_compensator = LatencyCompensator(
max_acceptable_latency_ms=self.config["latency_max_acceptable_ms"],
correction_alpha=self.config["latency_correction_alpha"],
memory_decay_factor=self.config["latency_memory_decay"],
)

# Profit Vector Forecast Engine
            self.profit_forecast = ProfitVectorForecastEngine(
lookback_periods=self.config["pvf_lookback_periods"],
fibonacci_levels=self.config["pvf_fibonacci_levels"],
volatility_window=self.config["pvf_volatility_window"],
)

# Correction Overlay Matrix
            self.correction_matrix = CorrectionOverlayMatrix(
anomaly_sensitivity=self.config["com_anomaly_sensitivity"],
correction_weights=self.config["com_correction_weights"],
max_correction_magnitude=self.config["com_max_correction_magnitude"],
)

# Trading execution components
self.live_execution = LiveExecutionMapper(
simulation_mode=self.config["simulation_mode"],
initial_portfolio_cash=self.config["initial_portfolio_cash"],
                enable_risk_manager=self.config["enable_risk_management"],
                enable_portfolio_tracker=self.config["enable_portfolio_tracking"],
)

# Set up integration callbacks
self._setup_integration_callbacks()

logger.info("✅ All subsystems initialized successfully")

except Exception as e:
            logger.error(f"❌ Failed to initialize subsystems: {e}")
raise

def _setup_integration_callbacks(self) -> None:
        """Set up callbacks for inter-subsystem communication."""

# State machine transition callback
def on_state_transition(event: TransitionEvent)::::
            self.stats["dualistic_transitions"] += 1

# Update latency compensator with new state
self.latency_compensator.update_dualistic_state(
state_type=event.to_state.value,
quantum_phase=self.state_machine.quantum_phase,
entropy_level=self.state_machine.entropy_level,
nibble_score=self.state_machine.nibble_score,
rittle_score=self.state_machine.rittle_score,
)

logger.info(
f"🔄 Integrated state transition: {
event.from_state.value} → {
event.to_state.value}""
)

self.state_machine.add_transition_callback(on_state_transition)

async def process_market_tick(
self, asset: str, price: float, volume: float, market_context: Dict[str, Any]
) -> UnifiedDecision:
        """Process a market tick through the complete unified system."

Args:
            asset: Trading asset (e.g., "BTC/USD")
price: Current price
volume: Current volume
market_context: Additional market context data

Returns:
            UnifiedDecision with complete analysis and trading recommendation
"""
start_time = time.time()
operation_id = f"market_tick_{int(time.time() * 1000)}"

# Start latency tracking
self.latency_compensator.start_operation(operation_id, "market_analysis")

try:
            # Step 1: Record memory in drift shell engine
tick_hash = self._generate_tick_hash(asset, price, volume, market_context)
memory_hash = self.drift_engine.record_memory(
tick_id=operation_id,
price=price,
volume=volume,
context_snapshot=market_context,
rsi=market_context.get("rsi", 50.0),
                momentum=market_context.get("momentum", 0.0),
)

# Step 2: Update dualistic state machine
self.state_machine.update_scores(
nibble_score=market_context.get("nibble_score", 0.5),
                rittle_score=market_context.get("rittle_score", 0.5),
                quantum_phase=market_context.get("quantum_phase", 0.0),
                entropy_level=market_context.get("entropy_level", 0.3),
                market_volatility=market_context.get("volatility", 0.02),
)

# Step 3: Evaluate temporal drift and memory validity
timing_metrics = TimingMetrics(
T_mem_read=0.02,
                T_hash_eval=0.01,
                T_AI_response=0.08,
                T_execute=0.04,
                total_latency=0.15,
)

drift_result = self.drift_engine.evaluate_drift(
current_price=price,
current_volume=volume,
current_hash=tick_hash,
timing_metrics=timing_metrics,
)

# Step 4: Generate profit vector forecast
timeframes = self._create_timeframe_data(market_context)
profit_vector = self.profit_forecast.generate_profit_vector(
current_price=price,
current_volume=volume,
current_rsi=market_context.get("rsi", 50.0),
                current_momentum=market_context.get("momentum", 0.0),
current_hash=tick_hash,
ghost_alignment=market_context.get("ghost_alignment", 0.0),
timeframes=timeframes,
)

# Step 5: Detect anomalies and apply corrections
anomalies = self.correction_matrix.detect_anomalies(
                current_vector=profit_vector,
current_price=price,
current_volume=volume,
current_hash=tick_hash,
market_context=market_context,
)

correction_factors = None
if anomalies:
                correction_factors = self.correction_matrix.apply_correction(
                    current_vector=profit_vector,
anomalies=anomalies,
market_context=market_context,
)
self.stats["anomaly_corrections"] += 1

# Step 6: Calculate bitmap confidence
bitmap_confidence = self.drift_engine.calculate_bitmap_confidence(
current_context=market_context,
profit_projection=profit_vector.magnitude,
)

# Step 7: Unified confidence validation
validation_result = self.drift_engine.unified_confidence_validator(
drift_result=drift_result,
bitmap_confidence=bitmap_confidence,
profit_vector=profit_vector,
correction_factors=correction_factors,
)

# Step 8: Apply quantum adjustments if enabled
quantum_coherence = market_context.get("quantum_coherence", 0.8)
if self.config["enable_quantum_verification"]:
                adjusted_latency = self.latency_compensator.apply_quantum_adjustment(
base_latency_ms=timing_metrics.total_latency * 1000,
quantum_coherence=quantum_coherence,
)
self.stats["quantum_adjustments"] += 1

# Step 9: Calculate final confidence and risk adjustment
final_confidence = validation_result["final_confidence"]
risk_adjustment = validation_result["risk_adjustment"]

# Step 10: Determine trade decision
should_trade = (
validation_result["should_activate"]
and final_confidence >= self.config["drift_confidence_threshold"]
and len(anomalies) < 3  # Don't trade during too many anomalies'
)

# Calculate position sizing
position_size = self._calculate_position_size(
profit_vector=profit_vector,
confidence=final_confidence,
risk_adjustment=risk_adjustment,
market_context=market_context,
)

# End latency tracking
latency_measurement = self.latency_compensator.end_operation(
operation_id, "market_analysis", tick_hash
)

# Create unified decision
decision = UnifiedDecision(
timestamp=time.time(),
asset=asset,
price=price,
volume=volume,
# Dualistic state
current_state=self.state_machine.current_state.value,
state_confidence=self.state_machine.calculate_coherence_score(),
nibble_score=self.state_machine.nibble_score,
rittle_score=self.state_machine.rittle_score,
# Timing analysis
memory_validity=(
max(r["validity"] for r in drift_result["valid_recalls"])
if drift_result["valid_recalls"]:
else 0.0
),
timing_coherence=final_confidence,
latency_correction=latency_measurement.correction_applied,
drift_shell_radius=self.drift_engine.shell_radius,
# Profit and corrections
                profit_vector=profit_vector,
correction_factors=(
correction_factors.confidence_weights if correction_factors else {}
),
anomalies_detected=[a.anomaly_type.value for a in anomalies],
# Final decision
should_trade=should_trade,
trade_direction=profit_vector.direction,
position_size=position_size,
confidence_score=final_confidence,
risk_adjustment=risk_adjustment,
# Integration metadata
processing_time_ms=(time.time() - start_time) * 1000,
subsystem_scores={
"drift_validity": (
max(r["validity"] for r in drift_result["valid_recalls"])
if drift_result["valid_recalls"]:
else 0.0
),
"state_coherence": self.state_machine.calculate_coherence_score(),
"profit_magnitude": profit_vector.magnitude,
"correction_applied": len(anomalies) > 0,
},
quantum_phase=self.state_machine.quantum_phase,
entropy_level=self.state_machine.entropy_level,
)

# Store decision
self.decision_history.append(decision)
if len(self.decision_history) > self.max_history_size:
                self.decision_history.pop(0)

# Update statistics
self.stats["total_decisions"] += 1
if should_trade:
                self.stats["successful_trades"] += 1
else:
                self.stats["rejected_decisions"] += 1

# Update average processing time
self._update_avg_processing_time((time.time() - start_time) * 1000)

# Call decision callbacks
for callback in self.decision_callbacks:
                try:
                    callback(decision)
except Exception as e:
                    logger.error(f"Error in decision callback: {e}")

logger.info(
f"🎯 Unified decision: {asset} {
decision.trade_direction} ""
f"(confidence={
final_confidence:.3f}, state={
decision.current_state})""
)

return decision

except Exception as e:
            logger.error(f"❌ Error processing market tick: {e}")
# End latency tracking even on error
self.latency_compensator.end_operation(
operation_id, "market_analysis", "error"
)
raise

async def execute_unified_trade(self, decision: UnifiedDecision)::: -> Dict[str, Any]:
        """Execute a trade based on unified decision."

Args:
            decision: UnifiedDecision from process_market_tick

Returns:
            Execution result with trade details
"""
if not decision.should_trade:
            return {"status": "skipped", "reason": "decision_rejected"}

try:
            # Execute through live execution mapper
            execution_state = self.live_execution.execute_glyph_trade(
glyph=f"{decision.current_state}_{decision.trade_direction}",
volume=decision.volume,
asset=decision.asset,
price=decision.price,
confidence_boost=decision.confidence_score
- 0.7,  # Adjust base confidence
)

logger.info(
f"🔄 Trade executed: {execution_state.trade_id} "
f"({execution_state.status})"
)

return {
"status": "executed",
"trade_id": execution_state.trade_id,
"execution_state": execution_state,
"decision": decision,
}

except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
return {"status": "failed", "error": str(e), "decision": decision}

async def run_continuous_trading(
self,
market_data_source: Callable[[], Dict[str, Any]],
stop_condition: Optional[Callable[[], bool]] = None,
) -> None:
        """Run continuous trading with the unified system."

Args:
            market_data_source: Function that returns market data
stop_condition: Optional function that returns True to stop trading
"""
self.is_running = True
decision_interval = self.config["decision_frequency_ms"] / 1000.0
health_check_interval = self.config["health_check_interval_s"]
last_health_check = time.time()

logger.info("🚀 Starting continuous unified trading")

try:
            while self.is_running:
                # Check stop condition
if stop_condition and stop_condition():
                    logger.info("🛑 Stop condition met, ending trading")
break

# Get market data
try:
                    market_data = market_data_source()

# Process market tick
decision = await self.process_market_tick(
asset=market_data.get("asset", "BTC/USD"),
price=market_data.get("price", 50000.0),
                        volume=market_data.get("volume", 1000000.0),
market_context=market_data.get("context", {}),
)

# Execute trade if recommended
if decision.should_trade:
                        await self.execute_unified_trade(decision)

except Exception as e:
                    logger.error(f"❌ Error in trading loop: {e}")

# Periodic health check
if time.time() - last_health_check > health_check_interval:
                    await self.perform_health_check()
last_health_check = time.time()

# Wait for next decision interval
await asyncio.sleep(decision_interval)

except KeyboardInterrupt:
            logger.info("🛑 Trading interrupted by user")
except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
finally:
            self.is_running = False
logger.info("🏁 Continuous trading stopped")

async def perform_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check."""
# Check each subsystem
drift_stats = self.drift_engine.get_performance_stats()
state_stats = self.state_machine.get_performance_stats()
latency_stats = self.latency_compensator.get_performance_stats()
forecast_stats = self.profit_forecast.get_performance_stats()
        correction_stats = self.correction_matrix.get_performance_stats()

# Calculate health scores (0.0 to 1.0)
self.system_health.drift_engine_health = min(
1.0,
drift_stats.get("valid_memory_recalls", 0)
/ max(drift_stats.get("total_evaluations", 1), 1),
)
self.system_health.state_machine_health = state_stats.get(
"coherence_score", 0.0
)
self.system_health.latency_compensator_health = 1.0 - min(
            1.0, latency_stats.get("avg_latency_ms", 0) / 1000.0
)
self.system_health.profit_forecast_health = min(
            1.0, forecast_stats.get("total_forecasts", 0) / 100.0
)
self.system_health.correction_matrix_health = 1.0 - min(
            1.0, correction_stats.get("anomalies_detected", 0) / 100.0
)

# Overall health
health_scores = [
self.system_health.drift_engine_health,
self.system_health.state_machine_health,
self.system_health.latency_compensator_health,
self.system_health.profit_forecast_health,
            self.system_health.correction_matrix_health,
]
self.system_health.overall_health = sum(health_scores) / len(health_scores)
self.system_health.last_health_check = time.time()

# Call health callbacks
for callback in self.health_callbacks:
            try:
                callback(self.system_health)
except Exception as e:
                logger.error(f"Error in health callback: {e}")

logger.info(
f"💚 System health check: {
self.system_health.overall_health:.3f}""
)
return self.system_health

def _generate_tick_hash(
self, asset: str, price: float, volume: float, context: Dict[str, Any]
) -> str:
        """Generate hash for market tick."""
hash_data = f"{asset}_{price:.2f}_{volume:.0f}_{time.time():.3f}"
        return f"tick_{hash(hash_data) % 1000000:06d}"

def _create_timeframe_data(
self, market_context: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
        """Create timeframe data for profit vector forecast."""
        base_rsi = market_context.get("rsi", 50.0)
        base_momentum = market_context.get("momentum", 0.0)
        base_volume = market_context.get("volume_ratio", 1.0)

return {
"1m": {
"rsi": base_rsi + 2,
"momentum": base_momentum * 1.1,
"volume": base_volume,
},
"5m": {
"rsi": base_rsi - 1,
"momentum": base_momentum * 0.9,
                "volume": base_volume * 0.95,
},
"15m": {
"rsi": base_rsi + 3,
"momentum": base_momentum * 1.2,
                "volume": base_volume * 1.05,
},
"1h": {
"rsi": base_rsi - 2,
"momentum": base_momentum * 0.8,
                "volume": base_volume * 0.9,
},
}

def _calculate_position_size(
self,
profit_vector: ProfitVector,
confidence: float,
risk_adjustment: float,
market_context: Dict[str, Any],
) -> float:
        """Calculate position size based on all factors."""
base_size = 1000.0  # Base position size

# Adjust for profit vector magnitude
        magnitude_factor = min(2.0, profit_vector.magnitude * 2)

# Adjust for confidence
confidence_factor = confidence

# Adjust for volatility
volatility = market_context.get("volatility", 0.02)
        volatility_factor = 1.0 / (1.0 + volatility * 10)

# Adjust for dualistic state
state_factor = (
1.2 if self.state_machine.current_state == StateType.ALEPH else 0.8
)

final_size = (
base_size
* magnitude_factor
* confidence_factor
* risk_adjustment
* volatility_factor
* state_factor
)

return max(10.0, min(10000.0, final_size))  # Clamp to reasonable range

def _update_avg_processing_time(self, new_time_ms: float)::: -> None:
        """Update average processing time metric."""
total_decisions = self.stats["total_decisions"]
current_avg = self.stats["avg_processing_time"]

if total_decisions == 1:
            self.stats["avg_processing_time"] = new_time_ms
else:
            self.stats["avg_processing_time"] = (
current_avg * (total_decisions - 1) + new_time_ms
) / total_decisions

def add_decision_callback(
self, callback: Callable[[UnifiedDecision], None]
) -> None:
        """Add callback for unified decisions."""
self.decision_callbacks.append(callback)

def add_health_callback(self, callback: Callable[[SystemHealth], None]) -> None:
        """Add callback for health checks."""
self.health_callbacks.append(callback)

def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
summary = self.stats.copy()

# Add subsystem stats
summary["subsystem_stats"] = {
"drift_engine": self.drift_engine.get_performance_stats(),
"state_machine": self.state_machine.get_performance_stats(),
"latency_compensator": self.latency_compensator.get_performance_stats(),
"profit_forecast": self.profit_forecast.get_performance_stats(),
            "correction_matrix": self.correction_matrix.get_performance_stats(),
}

# Add system health
summary["system_health"] = {
"overall_health": self.system_health.overall_health,
"last_health_check": self.system_health.last_health_check,
"individual_health": {
"drift_engine": self.system_health.drift_engine_health,
"state_machine": self.system_health.state_machine_health,
"latency_compensator": self.system_health.latency_compensator_health,
"profit_forecast": self.system_health.profit_forecast_health,
                "correction_matrix": self.system_health.correction_matrix_health,
},
}

# Calculate success rates
if summary["total_decisions"] > 0:
            summary["trade_success_rate"] = (
summary["successful_trades"] / summary["total_decisions"]
)
summary["rejection_rate"] = (
summary["rejected_decisions"] / summary["total_decisions"]
)

return summary

def stop_trading(self) -> None:
        """Stop continuous trading."""
self.is_running = False
logger.info("🛑 Trading stop requested")


async def main():
    """Demonstrate unified integration system."""
logging.basicConfig(level=logging.INFO)

print("🚀 Schwabot Unified Integration Demo")
print("=" * 60)

# Initialize unified system
integration = SchwabotUnifiedIntegration()

# Add callbacks
def on_decision(decision: UnifiedDecision)::::
        print(
f"  📊 Decision: {decision.asset} {decision.trade_direction} "
f"(confidence={decision.confidence_score:.3f})"
)

def on_health(health: SystemHealth)::::
        print(f"  💚 Health: {health.overall_health:.3f}")

integration.add_decision_callback(on_decision)
integration.add_health_callback(on_health)

# Simulate market data
def get_market_data():
        return {
"asset": "BTC/USD",
"price": 50000 + random.uniform(-1000, 1000),
"volume": 1000000 + random.uniform(-200000, 200000),
"context": {
"rsi": 45 + random.uniform(-10, 20),
"momentum": random.uniform(-0.1, 0.1),
                "volatility": 0.02 + random.uniform(-0.01, 0.02),
"quantum_phase": random.uniform(0, 1),
"entropy_level": random.uniform(0.2, 0.8),
                "nibble_score": random.uniform(0.3, 0.9),
                "rittle_score": random.uniform(0.3, 0.9),
                "quantum_coherence": random.uniform(0.7, 0.95),
},
}

# Process a few market ticks
print("\n📊 Processing market ticks...")
for i in range(5):
        market_data = get_market_data()
decision = await integration.process_market_tick(
asset=market_data["asset"],
price=market_data["price"],
volume=market_data["volume"],
market_context=market_data["context"],
)

print(
f"  Tick {i + 1}: {decision.current_state} state, "
f"confidence={decision.confidence_score:.3f}, "
f"should_trade={decision.should_trade}"
)

# Health check
print("\n💚 Performing health check...")
health = await integration.perform_health_check()
print(f"  Overall health: {health.overall_health:.3f}")

# Performance summary
print("\n📊 Performance Summary:")
summary = integration.get_performance_summary()
for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"    {sub_key}: {sub_value:.4f}")
else:
                    print(f"    {sub_key}: {sub_value}")
elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
else:
            print(f"  {key}: {value}")

print("\n✅ Unified Integration demo completed!")
print(
"🎯 System ready for quantum-aware, dualistic trading with temporal drift correction!"
)


if __name__ == "__main__":
    asyncio.run(main())

"""
"""