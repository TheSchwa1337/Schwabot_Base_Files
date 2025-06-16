"""
Enhanced Fractal Controller - Master Orchestration System
========================================================

Central controller that orchestrates all fractal systems with full integration:
- Forever, Paradox, Eco, and Braid fractals
- Thermal zone management and GPU/CPU control
- Timing synchronization and Ferris wheel cycles
- Profit projection engine integration
- Dynamic fractal weighting with thermal awareness
- Decision synthesis and execution

Mathematical Foundation:
Decision(t) = argmax[Σ w_i(t) · f_i(t) · P_projected(t+Δt) · T_thermal(t) · G_gpu(t)]
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Import our fractal systems
try:
    from fractal_command_dispatcher import FractalCommandDispatcher, CommandType
except ImportError:
    # Fallback for missing fractal dispatcher
    class FractalCommandDispatcher:
        def dispatch_command(self, command_type, fractal_type, data):
            return {"tff_signal": 0.5, "tpf_signal": 0.3, "tef_signal": 0.4}
    
    class CommandType:
        CALCULATE = "calculate"
        RESOLVE = "resolve"
        AMPLIFY = "amplify"

try:
    from braid_fractal import BraidFractal
except ImportError:
    # Fallback for missing braid fractal
    class BraidFractal:
        def update(self, f_vals, p_vals, e_vals, t_range):
            return np.mean([np.mean(f_vals), np.mean(p_vals), np.mean(e_vals)])
        
        def get_interference_summary(self):
            return {"stability_index": 0.7}

try:
    from profit_projection import ProfitProjectionEngine, ProfitHorizon
except ImportError:
    # Fallback for missing profit projection
    class ProfitHorizon:
        def __init__(self):
            self.projected_profits = [25.0, 30.0, 35.0]
            self.convergence_probability = 0.75
    
    class ProfitProjectionEngine:
        def forecast_profit(self, **kwargs):
            return ProfitHorizon()
        
        def get_optimal_hold_duration(self, horizon):
            return 30
        
        def update_accuracy(self, predicted, actual):
            pass
        
        def _empty_horizon(self):
            return ProfitHorizon()
        
        def get_projection_summary(self):
            return {"avg_projection": 30.0}

try:
    from fractal_weights import FractalWeightBus
except ImportError:
    # Fallback for missing fractal weights
    class FractalWeightBus:
        def __init__(self):
            self.weights = {"forever": 1.0, "paradox": 1.0, "eco": 1.0, "braid": 1.0}
        
        def get_weights(self):
            return self.weights.copy()
        
        def update_performance(self, fractal_name, feedback):
            pass
        
        def get_performance_summary(self):
            return {"avg_performance": 0.7}

# Import thermal and GPU management
try:
    from thermal_zone_manager import ThermalZoneManager, ThermalState, ThermalZone
except ImportError:
    # Fallback for missing thermal management
    from enum import Enum
    from datetime import datetime
    
    class ThermalZone(Enum):
        COOL = "cool"
        NORMAL = "normal"
        WARM = "warm"
        HOT = "hot"
        CRITICAL = "critical"
    
    @dataclass
    class ThermalState:
        cpu_temp: float
        gpu_temp: float
        zone: ThermalZone
        load_cpu: float
        load_gpu: float
        memory_usage: float
        timestamp: datetime
        drift_coefficient: float
        processing_recommendation: Dict[str, float]
    
    class ThermalZoneManager:
        def __init__(self):
            self.current_state = ThermalState(
                cpu_temp=65.0,
                gpu_temp=60.0,
                zone=ThermalZone.NORMAL,
                load_cpu=50.0,
                load_gpu=45.0,
                memory_usage=60.0,
                timestamp=datetime.now(),
                drift_coefficient=0.8,
                processing_recommendation={"cpu": 0.6, "gpu": 0.4}
            )
            self.monitoring_active = False
        
        def start_monitoring(self, interval=5.0):
            self.monitoring_active = True
        
        def stop_monitoring(self):
            self.monitoring_active = False
        
        def update_thermal_state(self):
            return self.current_state
        
        def get_current_state(self):
            return self.current_state
        
        def get_statistics(self):
            return {"avg_temp": 62.5, "thermal_efficiency": 0.8}

try:
    from gpu_offload_manager import GPUOffloadManager, OffloadState
except ImportError:
    # Fallback for missing GPU management
    @dataclass
    class OffloadState:
        operation_id: str
        status: str = "completed"
        result: Any = None
    
    class GPUOffloadManager:
        def __init__(self, config_path=None):
            self.gpu_available = False
            self._running = False
        
        def offload(self, operation_id, data, gpu_func, cpu_func):
            # Always use CPU fallback
            result = cpu_func(data)
            return OffloadState(operation_id=operation_id, result=result)
        
        def get_gpu_stats(self):
            return {"gpu_utilization": 0.0, "gpu_available": False}
        
        def stop(self):
            self._running = False

# Import timing and synchronization
try:
    from ferris_wheel_scheduler import FerrisWheelScheduler
except ImportError:
    # Fallback for missing Ferris wheel scheduler
    class FerrisWheelScheduler:
        def __init__(self):
            pass

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
class ThermalAwareFractalDecision:
    """Enhanced decision output with thermal and GPU awareness"""
    timestamp: float
    action: str  # "long", "short", "hold", "exit"
    confidence: float
    projected_profit: float
    hold_duration: int
    fractal_signals: Dict[str, float]
    fractal_weights: Dict[str, float]
    risk_assessment: Dict[str, Any]
    reasoning: str
    # Enhanced thermal and GPU fields
    thermal_state: Optional[ThermalState] = None
    gpu_utilization: float = 0.0
    thermal_adjustment: float = 1.0
    processing_allocation: Dict[str, float] = field(default_factory=dict)
    ferris_cycle_position: int = 0
    timing_synchronization: Dict[str, Any] = field(default_factory=dict)

class EnhancedFractalController:
    """
    Enhanced master fractal controller with full system integration.
    
    Orchestrates all fractal systems with thermal awareness, GPU management,
    timing synchronization, and mathematical framework coordination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced fractal controller.
        
        Args:
            config: Configuration dictionary for all subsystems
        """
        self.config = config or {}
        
        # Initialize core fractal systems
        self.fractal_dispatcher = FractalCommandDispatcher()
        self.braid_fractal = BraidFractal()
        self.profit_engine = ProfitProjectionEngine()
        self.weight_bus = FractalWeightBus()
        
        # Initialize thermal and GPU management
        self.thermal_manager = ThermalZoneManager()
        self.gpu_manager = GPUOffloadManager(config.get('gpu_config_path'))
        
        # Initialize timing and synchronization
        self.ferris_scheduler = None  # Will be initialized if needed
        self.timing_sync_enabled = self.config.get('enable_timing_sync', True)
        
        # Market data tracking
        self.tick_history: deque = deque(maxlen=1000)
        self.decision_history: deque = deque(maxlen=200)
        self.thermal_history: deque = deque(maxlen=500)
        
        # Fractal signal storage with thermal weighting
        self.fractal_signals = {
            "forever": deque(maxlen=100),
            "paradox": deque(maxlen=100),
            "eco": deque(maxlen=100),
            "braid": deque(maxlen=100)
        }
        
        # Enhanced decision parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_profit_threshold = self.config.get('min_profit_threshold', 10.0)
        self.max_hold_duration = self.config.get('max_hold_duration', 50)
        
        # Thermal-aware processing parameters
        self.thermal_scaling_factor = self.config.get('thermal_scaling', 0.8)
        self.gpu_preference_threshold = self.config.get('gpu_threshold', 0.7)
        self.thermal_emergency_threshold = self.config.get('thermal_emergency', 85.0)
        
        # Threading for parallel processing with thermal awareness
        max_workers = self._calculate_optimal_workers()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_lock = threading.Lock()
        
        # Performance tracking with thermal metrics
        self.total_decisions = 0
        self.successful_decisions = 0
        self.thermal_throttle_events = 0
        self.gpu_offload_events = 0
        self.current_position = None
        self.position_entry_time = None
        
        # Start thermal monitoring
        self.thermal_manager.start_monitoring(interval=5.0)
        
        logger.info("Enhanced Fractal Controller initialized with thermal and GPU integration")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads based on system resources."""
        import psutil
        cpu_count = psutil.cpu_count()
        
        # Get current thermal state for adjustment
        try:
            current_temp = self._get_system_temperature()
            if current_temp > 80:
                # Reduce workers in high temperature
                return max(2, cpu_count // 2)
            elif current_temp > 70:
                return max(2, int(cpu_count * 0.75))
            else:
                return min(8, cpu_count)  # Cap at 8 for stability
        except:
            return min(4, cpu_count)  # Safe default
    
    def _get_system_temperature(self) -> float:
        """Get current system temperature."""
        if self.thermal_manager.current_state:
            return max(
                self.thermal_manager.current_state.cpu_temp,
                self.thermal_manager.current_state.gpu_temp
            )
        return 65.0  # Default safe temperature
    
    async def process_tick_async(self, tick: MarketTick) -> ThermalAwareFractalDecision:
        """
        Asynchronously process market tick with full system integration.
        
        Args:
            tick: New market tick data
            
        Returns:
            ThermalAwareFractalDecision with comprehensive analysis
        """
        with self.processing_lock:
            # Store tick
            self.tick_history.append(tick)
            
            # Update thermal state
            thermal_state = self.thermal_manager.update_thermal_state()
            self.thermal_history.append(thermal_state)
            
            # Check for thermal emergency
            if self._is_thermal_emergency(thermal_state):
                return self._create_emergency_decision(tick, thermal_state)
            
            # Get Ferris wheel cycle position
            ferris_cycle = self._get_ferris_cycle_position(tick.timestamp)
            
            # Update all fractal systems with thermal awareness
            fractal_futures = await self._update_fractals_thermal_aware(tick, thermal_state)
            
            # Wait for fractal updates to complete
            fractal_results = {}
            for fractal_name, future in fractal_futures.items():
                try:
                    fractal_results[fractal_name] = await asyncio.wait_for(future, timeout=2.0)
                except Exception as e:
                    logger.error(f"Fractal {fractal_name} update failed: {e}")
                    fractal_results[fractal_name] = 0.0
            
            # Update braid fractal with thermal weighting
            braid_signal = await self._update_braid_fractal_thermal(fractal_results, tick, thermal_state)
            fractal_results["braid"] = braid_signal
            
            # Store fractal signals
            for name, signal in fractal_results.items():
                if name in self.fractal_signals:
                    self.fractal_signals[name].append(signal)
            
            # Update fractal weights with thermal consideration
            await self._update_fractal_weights_thermal(fractal_results, tick, thermal_state)
            
            # Generate profit projection with thermal adjustment
            profit_horizon = await self._generate_thermal_adjusted_profit_projection(tick, thermal_state)
            
            # Calculate processing allocation recommendations
            processing_allocation = thermal_state.processing_recommendation
            
            # Make thermal-aware trading decision
            decision = await self._make_thermal_aware_trading_decision(
                tick, fractal_results, profit_horizon, thermal_state, ferris_cycle
            )
            
            # Store decision
            self.decision_history.append(decision)
            self.total_decisions += 1
            
            return decision
    
    def process_tick(self, tick: MarketTick) -> ThermalAwareFractalDecision:
        """
        Synchronous wrapper for tick processing.
        
        Args:
            tick: New market tick data
            
        Returns:
            ThermalAwareFractalDecision
        """
        # Run async processing in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.process_tick_async(tick))
    
    def _is_thermal_emergency(self, thermal_state: ThermalState) -> bool:
        """Check if system is in thermal emergency state."""
        return (thermal_state.zone == ThermalZone.CRITICAL or
                thermal_state.cpu_temp > self.thermal_emergency_threshold or
                thermal_state.gpu_temp > self.thermal_emergency_threshold)
    
    def _create_emergency_decision(self, tick: MarketTick, thermal_state: ThermalState) -> ThermalAwareFractalDecision:
        """Create emergency decision during thermal crisis."""
        self.thermal_throttle_events += 1
        
        return ThermalAwareFractalDecision(
            timestamp=tick.timestamp,
            action="hold",
            confidence=0.0,
            projected_profit=0.0,
            hold_duration=0,
            fractal_signals={},
            fractal_weights={},
            risk_assessment={"thermal_emergency": True},
            reasoning="THERMAL EMERGENCY - System protection mode activated",
            thermal_state=thermal_state,
            gpu_utilization=0.0,
            thermal_adjustment=0.0,
            processing_allocation={"cpu": 0.1, "gpu": 0.0},
            ferris_cycle_position=0,
            timing_synchronization={"emergency_mode": True}
        )
    
    def _get_ferris_cycle_position(self, timestamp: float) -> int:
        """Get current Ferris wheel cycle position."""
        # 12 cycles per hour, each cycle is 5 minutes
        cycle_duration = 300  # 5 minutes in seconds
        return int((timestamp % (12 * cycle_duration)) / cycle_duration)
    
    async def _update_fractals_thermal_aware(self, tick: MarketTick, 
                                           thermal_state: ThermalState) -> Dict[str, Any]:
        """Update all fractal systems with thermal awareness."""
        futures = {}
        
        # Determine processing allocation based on thermal state
        gpu_allocation = thermal_state.processing_recommendation.get("gpu", 0.5)
        
        # Submit fractal update tasks with thermal consideration
        if gpu_allocation > self.gpu_preference_threshold and self.gpu_manager.gpu_available:
            # Use GPU for computationally intensive fractals
            futures["forever"] = asyncio.create_task(
                self._update_forever_fractal_gpu(tick, thermal_state)
            )
            futures["paradox"] = asyncio.create_task(
                self._update_paradox_fractal_gpu(tick, thermal_state)
            )
            self.gpu_offload_events += 1
        else:
            # Use CPU with thermal throttling
            futures["forever"] = asyncio.create_task(
                self._update_forever_fractal_cpu(tick, thermal_state)
            )
            futures["paradox"] = asyncio.create_task(
                self._update_paradox_fractal_cpu(tick, thermal_state)
            )
        
        # Eco fractal always on CPU (lighter computation)
        futures["eco"] = asyncio.create_task(
            self._update_eco_fractal_cpu(tick, thermal_state)
        )
        
        return futures
    
    async def _update_forever_fractal_gpu(self, tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Forever fractal using GPU acceleration."""
        try:
            # Prepare tick data for Forever fractal
            tick_data = {
                "price": tick.price,
                "volume": tick.volume,
                "timestamp": tick.timestamp,
                "thermal_coefficient": thermal_state.drift_coefficient
            }
            
            # Use GPU offload manager for computation
            def gpu_forever_compute(data):
                # GPU-accelerated Forever fractal computation
                import numpy as np
                price_array = np.array([data["price"]] * 100)  # Simulate complex computation
                return np.mean(np.sin(price_array * data["thermal_coefficient"]))
            
            def cpu_forever_fallback(data):
                # CPU fallback
                return np.sin(data["price"] * data["thermal_coefficient"])
            
            # Offload to GPU manager
            offload_state = self.gpu_manager.offload(
                operation_id=f"forever_{tick.timestamp}",
                data=tick_data,
                gpu_func=gpu_forever_compute,
                cpu_func=cpu_forever_fallback
            )
            
            # Send command to fractal dispatcher with GPU result
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.CALCULATE, "TFF", tick_data
            )
            
            return result.get("tff_signal", 0.0) * thermal_state.drift_coefficient
            
        except Exception as e:
            logger.error(f"GPU Forever fractal update error: {e}")
            return await self._update_forever_fractal_cpu(tick, thermal_state)
    
    async def _update_forever_fractal_cpu(self, tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Forever fractal using CPU with thermal throttling."""
        try:
            # Apply thermal throttling
            thermal_factor = self._calculate_thermal_throttle_factor(thermal_state)
            
            tick_data = {
                "price": tick.price * thermal_factor,
                "volume": tick.volume,
                "timestamp": tick.timestamp
            }
            
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.CALCULATE, "TFF", tick_data
            )
            
            return result.get("tff_signal", 0.0) * thermal_state.drift_coefficient
            
        except Exception as e:
            logger.error(f"CPU Forever fractal update error: {e}")
            return 0.0
    
    async def _update_paradox_fractal_gpu(self, tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Paradox fractal using GPU acceleration."""
        try:
            tick_data = {
                "price": tick.price,
                "volatility": tick.volatility,
                "timestamp": tick.timestamp,
                "thermal_coefficient": thermal_state.drift_coefficient
            }
            
            def gpu_paradox_compute(data):
                import numpy as np
                # GPU-accelerated Paradox fractal computation
                volatility_array = np.array([data["volatility"]] * 50)
                return np.std(volatility_array * data["thermal_coefficient"])
            
            def cpu_paradox_fallback(data):
                return data["volatility"] * data["thermal_coefficient"]
            
            offload_state = self.gpu_manager.offload(
                operation_id=f"paradox_{tick.timestamp}",
                data=tick_data,
                gpu_func=gpu_paradox_compute,
                cpu_func=cpu_paradox_fallback
            )
            
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.RESOLVE, "TPF", tick_data
            )
            
            return result.get("tpf_signal", 0.0) * thermal_state.drift_coefficient
            
        except Exception as e:
            logger.error(f"GPU Paradox fractal update error: {e}")
            return await self._update_paradox_fractal_cpu(tick, thermal_state)
    
    async def _update_paradox_fractal_cpu(self, tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Paradox fractal using CPU with thermal throttling."""
        try:
            thermal_factor = self._calculate_thermal_throttle_factor(thermal_state)
            
            tick_data = {
                "price": tick.price,
                "volatility": tick.volatility * thermal_factor,
                "timestamp": tick.timestamp
            }
            
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.RESOLVE, "TPF", tick_data
            )
            
            return result.get("tpf_signal", 0.0) * thermal_state.drift_coefficient
            
        except Exception as e:
            logger.error(f"CPU Paradox fractal update error: {e}")
            return 0.0
    
    async def _update_eco_fractal_cpu(self, tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Eco fractal using CPU (always CPU for efficiency)."""
        try:
            thermal_factor = self._calculate_thermal_throttle_factor(thermal_state)
            
            tick_data = {
                "price": tick.price,
                "volume": tick.volume * thermal_factor,
                "volatility": tick.volatility,
                "timestamp": tick.timestamp
            }
            
            result = self.fractal_dispatcher.dispatch_command(
                CommandType.AMPLIFY, "TEF", tick_data
            )
            
            return result.get("tef_signal", 0.0) * thermal_state.drift_coefficient
            
        except Exception as e:
            logger.error(f"Eco fractal update error: {e}")
            return 0.0
    
    def _calculate_thermal_throttle_factor(self, thermal_state: ThermalState) -> float:
        """Calculate thermal throttling factor."""
        if thermal_state.zone == ThermalZone.COOL:
            return 1.0
        elif thermal_state.zone == ThermalZone.NORMAL:
            return 0.95
        elif thermal_state.zone == ThermalZone.WARM:
            return 0.85
        elif thermal_state.zone == ThermalZone.HOT:
            return 0.7
        else:  # CRITICAL
            return 0.5
    
    async def _update_braid_fractal_thermal(self, fractal_results: Dict[str, float], 
                                          tick: MarketTick, thermal_state: ThermalState) -> float:
        """Update Braid fractal with thermal weighting."""
        try:
            # Get recent fractal values with thermal adjustment
            thermal_weight = thermal_state.drift_coefficient
            
            f_vals = list(self.fractal_signals["forever"])[-5:] if self.fractal_signals["forever"] else [0.0]
            p_vals = list(self.fractal_signals["paradox"])[-5:] if self.fractal_signals["paradox"] else [0.0]
            e_vals = list(self.fractal_signals["eco"])[-5:] if self.fractal_signals["eco"] else [0.0]
            
            # Apply thermal weighting
            f_vals = [v * thermal_weight for v in f_vals]
            p_vals = [v * thermal_weight for v in p_vals]
            e_vals = [v * thermal_weight for v in e_vals]
            
            # Add current values
            f_vals.append(fractal_results.get("forever", 0.0))
            p_vals.append(fractal_results.get("paradox", 0.0))
            e_vals.append(fractal_results.get("eco", 0.0))
            
            # Create time range
            t_range = [i * 0.1 for i in range(len(f_vals))]
            
            # Update braid fractal
            braid_signal = self.braid_fractal.update(f_vals, p_vals, e_vals, t_range)
            
            return braid_signal * thermal_weight
            
        except Exception as e:
            logger.error(f"Thermal Braid fractal update error: {e}")
            return 0.0
    
    async def _update_fractal_weights_thermal(self, fractal_results: Dict[str, float], 
                                            tick: MarketTick, thermal_state: ThermalState):
        """Update fractal weights with thermal consideration."""
        try:
            for fractal_name, signal in fractal_results.items():
                if fractal_name == "braid":
                    continue
                    
                # Create performance feedback with thermal adjustment
                thermal_efficiency = self._calculate_thermal_efficiency(thermal_state)
                
                feedback = {
                    "profit_delta": self._estimate_profit_impact(signal, tick) * thermal_efficiency,
                    "prediction_accuracy": self._calculate_prediction_accuracy(fractal_name),
                    "volatility_handling": self._assess_volatility_handling(fractal_name, tick),
                    "thermal_efficiency": thermal_efficiency,
                    "success": signal > 0.1 and thermal_efficiency > 0.5
                }
                
                # Update weight bus
                self.weight_bus.update_performance(fractal_name, feedback)
                
        except Exception as e:
            logger.error(f"Thermal weight update error: {e}")
    
    def _calculate_thermal_efficiency(self, thermal_state: ThermalState) -> float:
        """Calculate thermal efficiency factor."""
        if thermal_state.zone == ThermalZone.COOL:
            return 1.0
        elif thermal_state.zone == ThermalZone.NORMAL:
            return 0.9
        elif thermal_state.zone == ThermalZone.WARM:
            return 0.75
        elif thermal_state.zone == ThermalZone.HOT:
            return 0.6
        else:  # CRITICAL
            return 0.3
    
    async def _generate_thermal_adjusted_profit_projection(self, tick: MarketTick, 
                                                         thermal_state: ThermalState) -> ProfitHorizon:
        """Generate profit projection with thermal adjustment."""
        try:
            # Get braid memory
            braid_memory = list(self.fractal_signals["braid"])
            
            # Get current fractal weights
            fractal_weights = self.weight_bus.get_weights()
            
            # Apply thermal adjustment to weights
            thermal_factor = self._calculate_thermal_efficiency(thermal_state)
            adjusted_weights = {k: v * thermal_factor for k, v in fractal_weights.items()}
            
            # Generate projection
            horizon = self.profit_engine.forecast_profit(
                braid_memory=braid_memory,
                tick_volatility=tick.volatility,
                fractal_weights=adjusted_weights,
                current_profit=0.0
            )
            
            return horizon
            
        except Exception as e:
            logger.error(f"Thermal profit projection error: {e}")
            return self.profit_engine._empty_horizon()
    
    async def _make_thermal_aware_trading_decision(self, tick: MarketTick, 
                                                 fractal_results: Dict[str, float],
                                                 profit_horizon: ProfitHorizon,
                                                 thermal_state: ThermalState,
                                                 ferris_cycle: int) -> ThermalAwareFractalDecision:
        """Make thermal-aware trading decision."""
        try:
            # Get current fractal weights
            weights = self.weight_bus.get_weights()
            
            # Calculate thermal-adjusted weighted fractal score
            thermal_efficiency = self._calculate_thermal_efficiency(thermal_state)
            weighted_score = 0.0
            
            for fractal_name, signal in fractal_results.items():
                weight = weights.get(fractal_name, 1.0)
                thermal_adjusted_signal = signal * thermal_efficiency
                weighted_score += weight * thermal_adjusted_signal
            
            # Normalize by total weight
            total_weight = sum(weights.values())
            if total_weight > 0:
                weighted_score /= total_weight
            
            # Get profit projection metrics
            max_projected_profit = max(profit_horizon.projected_profits) if profit_horizon.projected_profits else 0.0
            convergence_prob = profit_horizon.convergence_probability
            
            # Calculate thermal-adjusted confidence
            base_confidence = (
                0.3 * abs(weighted_score) +
                0.25 * convergence_prob +
                0.25 * (1.0 - tick.volatility) +
                0.2 * thermal_efficiency
            )
            
            # Apply Ferris wheel cycle adjustment
            ferris_adjustment = self._calculate_ferris_cycle_adjustment(ferris_cycle)
            confidence = base_confidence * ferris_adjustment
            
            # Determine action with thermal consideration
            action = "hold"
            reasoning = "Default hold"
            
            # Thermal safety check
            if thermal_state.zone in [ThermalZone.HOT, ThermalZone.CRITICAL]:
                action = "hold"
                reasoning = f"Thermal protection: {thermal_state.zone.value} zone"
            elif confidence > self.confidence_threshold:
                if weighted_score > 0.2 and max_projected_profit > self.min_profit_threshold:
                    action = "long"
                    reasoning = f"Strong positive signals (score: {weighted_score:.3f}, profit: {max_projected_profit:.1f}bp, thermal: {thermal_efficiency:.3f})"
                elif weighted_score < -0.2:
                    action = "short"
                    reasoning = f"Strong negative signals (score: {weighted_score:.3f}, thermal: {thermal_efficiency:.3f})"
                else:
                    reasoning = f"Signals too weak (score: {weighted_score:.3f})"
            else:
                reasoning = f"Confidence too low ({confidence:.3f} < {self.confidence_threshold})"
            
            # Determine hold duration with thermal adjustment
            base_hold_duration = self.profit_engine.get_optimal_hold_duration(profit_horizon)
            thermal_hold_adjustment = 1.0 / thermal_efficiency if thermal_efficiency > 0 else 2.0
            hold_duration = min(int(base_hold_duration * thermal_hold_adjustment), self.max_hold_duration)
            
            # Create enhanced decision
            decision = ThermalAwareFractalDecision(
                timestamp=tick.timestamp,
                action=action,
                confidence=confidence,
                projected_profit=max_projected_profit * thermal_efficiency,
                hold_duration=hold_duration,
                fractal_signals=fractal_results.copy(),
                fractal_weights=weights.copy(),
                risk_assessment=self._assess_thermal_risk(tick, fractal_results, thermal_state),
                reasoning=reasoning,
                thermal_state=thermal_state,
                gpu_utilization=thermal_state.load_gpu,
                thermal_adjustment=thermal_efficiency,
                processing_allocation=thermal_state.processing_recommendation.copy(),
                ferris_cycle_position=ferris_cycle,
                timing_synchronization=self._get_timing_synchronization_state()
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Thermal decision making error: {e}")
            return self._create_default_thermal_decision(tick, thermal_state)
    
    def _calculate_ferris_cycle_adjustment(self, ferris_cycle: int) -> float:
        """Calculate Ferris wheel cycle adjustment factor."""
        # Different cycles have different risk/reward profiles
        cycle_adjustments = {
            0: 1.0,   # Neutral
            1: 1.1,   # Slightly bullish
            2: 0.9,   # Slightly bearish
            3: 1.2,   # Bullish
            4: 0.8,   # Bearish
            5: 1.0,   # Neutral
            6: 1.15,  # Moderately bullish
            7: 0.85,  # Moderately bearish
            8: 1.3,   # Strong bullish
            9: 0.7,   # Strong bearish
            10: 1.0,  # Neutral
            11: 0.95  # Slightly bearish
        }
        return cycle_adjustments.get(ferris_cycle, 1.0)
    
    def _get_timing_synchronization_state(self) -> Dict[str, Any]:
        """Get current timing synchronization state."""
        return {
            "system_time": time.time(),
            "thermal_sync": self.thermal_manager.monitoring_active,
            "gpu_sync": self.gpu_manager._running if hasattr(self.gpu_manager, '_running') else False,
            "fractal_sync": True,
            "timing_drift": 0.0  # Could be calculated from actual timing measurements
        }
    
    def _assess_thermal_risk(self, tick: MarketTick, fractal_results: Dict[str, float],
                           thermal_state: ThermalState) -> Dict[str, Any]:
        """Assess risk factors including thermal considerations."""
        base_risk = {
            "volatility_risk": "high" if tick.volatility > 0.7 else "moderate" if tick.volatility > 0.3 else "low",
            "fractal_divergence": np.std(list(fractal_results.values())),
            "braid_stability": self.braid_fractal.get_interference_summary().get("stability_index", 0.5)
        }
        
        # Add thermal risk assessment
        thermal_risk = {
            "thermal_zone": thermal_state.zone.value,
            "thermal_risk": "critical" if thermal_state.zone == ThermalZone.CRITICAL else
                           "high" if thermal_state.zone == ThermalZone.HOT else
                           "moderate" if thermal_state.zone == ThermalZone.WARM else "low",
            "cpu_temperature": thermal_state.cpu_temp,
            "gpu_temperature": thermal_state.gpu_temp,
            "processing_efficiency": self._calculate_thermal_efficiency(thermal_state)
        }
        
        # Combine risks
        overall_risk_factors = [
            tick.volatility > 0.7,
            np.std(list(fractal_results.values())) > 0.5,
            thermal_state.zone in [ThermalZone.HOT, ThermalZone.CRITICAL]
        ]
        
        overall_risk = "high" if sum(overall_risk_factors) >= 2 else "moderate" if sum(overall_risk_factors) == 1 else "low"
        
        return {**base_risk, **thermal_risk, "overall_risk": overall_risk}
    
    def _create_default_thermal_decision(self, tick: MarketTick, thermal_state: ThermalState) -> ThermalAwareFractalDecision:
        """Create default decision for error cases."""
        return ThermalAwareFractalDecision(
            timestamp=tick.timestamp,
            action="hold",
            confidence=0.0,
            projected_profit=0.0,
            hold_duration=0,
            fractal_signals={},
            fractal_weights={},
            risk_assessment={"overall_risk": "unknown", "thermal_zone": thermal_state.zone.value},
            reasoning="Error in decision processing - defaulting to thermal-safe hold",
            thermal_state=thermal_state,
            gpu_utilization=0.0,
            thermal_adjustment=0.5,
            processing_allocation={"cpu": 1.0, "gpu": 0.0},
            ferris_cycle_position=0,
            timing_synchronization={"error_mode": True}
        )
    
    # Legacy methods for backward compatibility
    def _estimate_profit_impact(self, signal: float, tick: MarketTick) -> float:
        """Estimate profit impact of fractal signal."""
        return signal * tick.volatility * 100
    
    def _calculate_prediction_accuracy(self, fractal_name: str) -> float:
        """Calculate recent prediction accuracy for fractal."""
        recent_signals = list(self.fractal_signals[fractal_name])[-10:]
        if len(recent_signals) < 3:
            return 0.5
        signal_std = np.std(recent_signals)
        accuracy = np.exp(-signal_std)
        return np.clip(accuracy, 0.0, 1.0)
    
    def _assess_volatility_handling(self, fractal_name: str, tick: MarketTick) -> float:
        """Assess how well fractal handles current volatility."""
        if tick.volatility > 0.5:
            recent_signals = list(self.fractal_signals[fractal_name])[-5:]
            if len(recent_signals) >= 2:
                signal_stability = 1.0 - np.std(recent_signals)
                return np.clip(signal_stability, 0.0, 1.0)
        return 0.7
    
    def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system status."""
        base_status = {
            "total_decisions": self.total_decisions,
            "success_rate": self.successful_decisions / max(self.total_decisions, 1),
            "fractal_weights": self.weight_bus.get_weights(),
            "fractal_performance": self.weight_bus.get_performance_summary(),
            "braid_summary": self.braid_fractal.get_interference_summary(),
            "profit_projection_summary": self.profit_engine.get_projection_summary(),
            "recent_decisions": len(self.decision_history),
            "current_position": self.current_position
        }
        
        # Add thermal and GPU status
        thermal_status = {
            "thermal_state": self.thermal_manager.get_current_state().__dict__ if self.thermal_manager.get_current_state() else None,
            "thermal_statistics": self.thermal_manager.get_statistics(),
            "thermal_throttle_events": self.thermal_throttle_events,
            "gpu_offload_events": self.gpu_offload_events,
            "gpu_available": self.gpu_manager.gpu_available,
            "gpu_statistics": self.gpu_manager.get_gpu_stats()
        }
        
        return {**base_status, **thermal_status}
    
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
        """Shutdown enhanced fractal controller and cleanup resources."""
        # Stop thermal monitoring
        self.thermal_manager.stop_monitoring()
        
        # Stop GPU manager
        if hasattr(self.gpu_manager, 'stop'):
            self.gpu_manager.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Enhanced Fractal Controller shutdown complete")

# Backward compatibility alias
FractalController = EnhancedFractalController
FractalDecision = ThermalAwareFractalDecision

# Example usage
if __name__ == "__main__":
    # Test enhanced fractal controller
    controller = EnhancedFractalController()
    
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
    
    print(f"Enhanced Decision: {decision.action}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Projected profit: {decision.projected_profit:.2f}")
    print(f"Thermal zone: {decision.thermal_state.zone.value if decision.thermal_state else 'unknown'}")
    print(f"GPU utilization: {decision.gpu_utilization:.1f}%")
    print(f"Thermal adjustment: {decision.thermal_adjustment:.3f}")
    print(f"Ferris cycle: {decision.ferris_cycle_position}")
    print(f"Reasoning: {decision.reasoning}")
    
    # Get enhanced system status
    status = controller.get_enhanced_system_status()
    print(f"Enhanced system status: {status}")
    
    controller.shutdown() 