"""
Profit Cycle Navigator
=====================

Implements mathematical profit cycle detection and navigation for Schwabot.
Uses JuMBO-style anomaly detection and fault-correlated profit optimization.
Prevents recursive loops while identifying genuine profit tiers.
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import hashlib
from core.fault_bus import FaultBus, FaultType, FaultBusEvent

class ProfitCycleState(Enum):
    """States in the profit cycle detection state machine"""
    SEEKING = "seeking"           # Looking for profit opportunity
    ENTERING = "entering"         # Detected opportunity, preparing entry
    RIDING = "riding"            # In profitable position
    PEAK_DETECTED = "peak_detected"  # Peak profit detected, prepare exit
    EXITING = "exiting"          # Exiting position
    COOLDOWN = "cooldown"        # Waiting before next cycle
    ANOMALY_DETECTED = "anomaly_detected"  # JuMBO-style profit anomaly

@dataclass
class ProfitVector:
    """Mathematical representation of a profit opportunity vector"""
    magnitude: float              # Expected profit magnitude
    direction: int               # 1 for long, -1 for short, 0 for hold
    confidence: float            # Confidence in prediction (0-1)
    entry_price: float           # Suggested entry price
    exit_price: float            # Suggested exit price
    volume_weight: float         # Suggested volume as fraction of portfolio
    sha_signature: str           # SHA signature for loop detection
    fault_correlations: Dict[FaultType, float]  # Correlated fault probabilities
    anomaly_strength: float      # JuMBO-style anomaly strength
    temporal_window: timedelta   # Expected duration of opportunity

class ProfitCycleDetector:
    """Mathematical detector for profit cycles using fault correlation"""
    
    def __init__(self, 
                 cycle_window: int = 100,
                 min_profit_threshold: float = 0.02,
                 anomaly_threshold: float = 2.5,
                 correlation_threshold: float = 0.5):
        self.cycle_window = cycle_window
        self.min_profit_threshold = min_profit_threshold
        self.anomaly_threshold = anomaly_threshold
        self.correlation_threshold = correlation_threshold
        
        # Profit cycle history and analysis
        self.profit_history: deque = deque(maxlen=cycle_window)
        self.cycle_patterns: Dict[str, Dict] = {}
        self.active_cycles: List[Dict] = []
        
        # Mathematical models
        self.fourier_coefficients = None
        self.trend_polynomials = None
        self.volatility_model = None
        
    def detect_cycle_phase(self, current_profit: float, timestamp: datetime) -> Tuple[str, float]:
        """
        Detect current phase of profit cycle using Fourier analysis
        Returns (phase_name, phase_confidence)
        """
        self.profit_history.append((current_profit, timestamp))
        
        if len(self.profit_history) < 20:
            return "insufficient_data", 0.0
        
        # Extract profit values for analysis
        profits = [p[0] for p in list(self.profit_history)]
        
        # Fourier analysis to detect cyclical patterns
        fft = np.fft.fft(profits)
        frequencies = np.fft.fftfreq(len(profits))
        
        # Find dominant frequency (strongest cycle)
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        cycle_strength = np.abs(fft[dominant_freq_idx]) / len(profits)
        
        # Determine phase within cycle
        phase_angle = np.angle(fft[dominant_freq_idx])
        phase_position = (phase_angle + np.pi) / (2 * np.pi)  # Normalize to [0,1]
        
        # Map phase position to cycle phase names
        if phase_position < 0.2:
            phase_name = "accumulation"
        elif phase_position < 0.4:
            phase_name = "markup"
        elif phase_position < 0.6:
            phase_name = "distribution"
        elif phase_position < 0.8:
            phase_name = "markdown"
        else:
            phase_name = "re_accumulation"
        
        return phase_name, cycle_strength
    
    def compute_profit_gradient(self, window: int = 5) -> np.ndarray:
        """Compute profit gradient (rate of change) over specified window"""
        if len(self.profit_history) < window:
            return np.array([0.0])
        
        recent_profits = [p[0] for p in list(self.profit_history)[-window:]]
        return np.gradient(recent_profits)
    
    def detect_anomaly_cluster(self, fault_context: Dict) -> Tuple[bool, float, str]:
        """
        Detect JuMBO-style profit anomaly clusters
        Returns (is_anomaly, strength, cluster_type)
        """
        if len(self.profit_history) < 10:
            return False, 0.0, "insufficient_data"
        
        profits = [p[0] for p in list(self.profit_history)]
        
        # Statistical analysis
        mean_profit = np.mean(profits)
        std_profit = np.std(profits)
        current_profit = profits[-1]
        
        if std_profit == 0:
            return False, 0.0, "no_variance"
        
        z_score = abs(current_profit - mean_profit) / std_profit
        
        # Check for anomaly clustering (multiple recent anomalies)
        recent_z_scores = []
        for i in range(min(10, len(profits))):
            z = abs(profits[-(i+1)] - mean_profit) / std_profit
            recent_z_scores.append(z)
        
        anomaly_count = sum(1 for z in recent_z_scores if z > 2.0)
        
        if z_score > self.anomaly_threshold and anomaly_count >= 3:
            # JuMBO-style cluster detected
            anomaly_strength = min(z_score / 5.0, 1.0)
            
            if current_profit > mean_profit:
                cluster_type = "profit_binary"  # Like JuMBO binary system
            else:
                cluster_type = "loss_binary"
            
            return True, anomaly_strength, cluster_type
        
        return False, 0.0, "normal"

class ProfitCycleNavigator:
    """Main navigation system for profit cycles with fault correlation"""
    
    def __init__(self, fault_bus: FaultBus, initial_portfolio_value: float = 10000.0):
        self.fault_bus = fault_bus
        self.portfolio_value = initial_portfolio_value
        self.current_state = ProfitCycleState.SEEKING
        self.detector = ProfitCycleDetector()
        
        # Navigation parameters
        self.max_position_size = 0.3  # Max 30% of portfolio in single position
        self.stop_loss_ratio = 0.02   # 2% stop loss
        self.take_profit_ratio = 0.06 # 6% take profit
        
        # Current position tracking
        self.current_position: Optional[Dict] = None
        self.entry_vector: Optional[ProfitVector] = None
        self.navigation_log: List[Dict] = []
        
        # Mathematical models
        self.volume_optimizer = VolumeOptimizer()
        self.risk_calculator = RiskCalculator()
        
    def update_market_state(self, 
                          current_price: float, 
                          current_volume: float,
                          timestamp: datetime) -> ProfitVector:
        """
        Update market state and compute optimal profit vector
        """
        # Calculate current profit delta
        profit_delta = 0.0
        if self.current_position:
            if self.current_position['direction'] == 1:  # Long
                profit_delta = (current_price - self.current_position['entry_price']) / self.current_position['entry_price']
            else:  # Short
                profit_delta = (self.current_position['entry_price'] - current_price) / self.current_position['entry_price']
        
        # Update fault bus with profit context
        self.fault_bus.update_profit_context(profit_delta, int(timestamp.timestamp()))
        
        # Detect cycle phase
        cycle_phase, phase_confidence = self.detector.detect_cycle_phase(profit_delta, timestamp)
        
        # Check for anomalies
        fault_context = {ft.value: 0.5 for ft in FaultType}  # Simplified fault context
        is_anomaly, anomaly_strength, cluster_type = self.detector.detect_anomaly_cluster(fault_context)
        
        # Get fault correlations for profit prediction
        profit_correlations = self.fault_bus.get_profit_correlations()
        
        # Compute profit vector
        profit_vector = self._compute_profit_vector(
            current_price, current_volume, profit_delta, 
            cycle_phase, phase_confidence, is_anomaly, 
            anomaly_strength, profit_correlations, timestamp
        )
        
        # Update navigation state
        self._update_navigation_state(profit_vector, timestamp)
        
        return profit_vector
    
    def _compute_profit_vector(self, 
                             current_price: float,
                             current_volume: float,
                             profit_delta: float,
                             cycle_phase: str,
                             phase_confidence: float,
                             is_anomaly: bool,
                             anomaly_strength: float,
                             correlations: List,
                             timestamp: datetime) -> ProfitVector:
        """Compute optimal profit vector using mathematical models"""
        
        # Base magnitude from cycle phase
        phase_multipliers = {
            "accumulation": 0.3,
            "markup": 0.8,
            "distribution": 0.2,
            "markdown": -0.6,
            "re_accumulation": 0.4,
            "insufficient_data": 0.0
        }
        
        base_magnitude = phase_multipliers.get(cycle_phase, 0.0) * phase_confidence
        
        # Anomaly adjustment (JuMBO-style)
        if is_anomaly:
            base_magnitude *= (1.0 + anomaly_strength * 2.0)  # Boost for anomalies
        
        # Fault correlation adjustment
        correlation_adjustment = 0.0
        fault_correlations = {}
        for corr in correlations:
            predicted_impact = self.fault_bus.predict_profit_from_fault(corr.fault_type)
            if predicted_impact:
                correlation_adjustment += predicted_impact * corr.confidence
                fault_correlations[corr.fault_type] = predicted_impact
        
        final_magnitude = base_magnitude + correlation_adjustment
        
        # Determine direction
        direction = 1 if final_magnitude > 0 else -1 if final_magnitude < -0.01 else 0
        
        # Calculate confidence
        confidence = min(phase_confidence + (anomaly_strength if is_anomaly else 0), 1.0)
        
        # Calculate entry/exit prices with volatility adjustment
        volatility = self._estimate_volatility()
        entry_price = current_price * (1 + direction * volatility * 0.1)
        exit_price = entry_price * (1 + direction * abs(final_magnitude))
        
        # Optimize volume
        volume_weight = self.volume_optimizer.calculate_optimal_volume(
            abs(final_magnitude), confidence, self.portfolio_value
        )
        
        # Generate SHA signature for loop detection
        state_string = f"{cycle_phase}_{final_magnitude:.4f}_{confidence:.4f}_{timestamp.isoformat()}"
        sha_signature = hashlib.sha256(state_string.encode()).hexdigest()[:16]
        
        return ProfitVector(
            magnitude=abs(final_magnitude),
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
            volume_weight=volume_weight,
            sha_signature=sha_signature,
            fault_correlations=fault_correlations,
            anomaly_strength=anomaly_strength if is_anomaly else 0.0,
            temporal_window=timedelta(minutes=30)  # Default 30-minute window
        )
    
    def _estimate_volatility(self) -> float:
        """Estimate current market volatility from profit history"""
        if len(self.detector.profit_history) < 5:
            return 0.02  # Default 2% volatility
        
        profits = [p[0] for p in list(self.detector.profit_history)[-20:]]
        return np.std(profits)
    
    def _update_navigation_state(self, profit_vector: ProfitVector, timestamp: datetime):
        """Update navigation state machine based on profit vector"""
        
        if profit_vector.magnitude < self.detector.min_profit_threshold:
            if self.current_state != ProfitCycleState.SEEKING:
                self.current_state = ProfitCycleState.SEEKING
                self._log_state_change("SEEKING", "Low profit magnitude", timestamp)
        
        elif profit_vector.confidence > 0.7 and self.current_state == ProfitCycleState.SEEKING:
            self.current_state = ProfitCycleState.ENTERING
            self.entry_vector = profit_vector
            self._log_state_change("ENTERING", f"High confidence: {profit_vector.confidence}", timestamp)
        
        elif profit_vector.anomaly_strength > 0.5:
            self.current_state = ProfitCycleState.ANOMALY_DETECTED
            self._log_state_change("ANOMALY_DETECTED", f"Anomaly strength: {profit_vector.anomaly_strength}", timestamp)
    
    def _log_state_change(self, new_state: str, reason: str, timestamp: datetime):
        """Log navigation state changes"""
        self.navigation_log.append({
            'timestamp': timestamp.isoformat(),
            'old_state': self.current_state.value,
            'new_state': new_state,
            'reason': reason
        })
        logging.info(f"Profit Navigator: {self.current_state.value} -> {new_state} ({reason})")
    
    def get_trade_signal(self) -> Optional[Dict]:
        """Get current trade signal based on navigation state"""
        if self.current_state == ProfitCycleState.ENTERING and self.entry_vector:
            return {
                'action': 'ENTER',
                'direction': 'LONG' if self.entry_vector.direction == 1 else 'SHORT',
                'entry_price': self.entry_vector.entry_price,
                'exit_price': self.entry_vector.exit_price,
                'volume_weight': self.entry_vector.volume_weight,
                'confidence': self.entry_vector.confidence,
                'stop_loss': self.entry_vector.entry_price * (1 - self.entry_vector.direction * self.stop_loss_ratio),
                'take_profit': self.entry_vector.entry_price * (1 + self.entry_vector.direction * self.take_profit_ratio),
                'sha_signature': self.entry_vector.sha_signature
            }
        
        elif self.current_state == ProfitCycleState.EXITING:
            return {
                'action': 'EXIT',
                'reason': 'Navigation state exiting'
            }
        
        return None
    
    def export_navigation_log(self, file_path: Optional[str] = None) -> str:
        """Export navigation log for analysis"""
        output = json.dumps(self.navigation_log, indent=2)
        if file_path:
            with open(file_path, 'w') as f:
                f.write(output)
        return output

class VolumeOptimizer:
    """Optimize trading volume based on profit magnitude and confidence"""
    
    def __init__(self, max_risk_per_trade: float = 0.02):
        self.max_risk_per_trade = max_risk_per_trade
    
    def calculate_optimal_volume(self, 
                               profit_magnitude: float, 
                               confidence: float, 
                               portfolio_value: float) -> float:
        """
        Calculate optimal volume using Kelly Criterion-inspired formula
        V = f* = (bp - q) / b
        where: b = odds, p = probability of win, q = probability of loss
        """
        # Adjust probability based on confidence and magnitude
        win_probability = min(0.5 + confidence * 0.3, 0.9)  # Max 90% probability
        loss_probability = 1 - win_probability
        
        # Expected return (odds)
        expected_return = profit_magnitude
        
        # Kelly fraction
        if expected_return > 0:
            kelly_fraction = (win_probability * expected_return - loss_probability) / expected_return
        else:
            kelly_fraction = 0.0
        
        # Apply risk constraints
        kelly_fraction = max(0.0, min(kelly_fraction, self.max_risk_per_trade))
        
        # Convert to volume weight
        return kelly_fraction

class RiskCalculator:
    """Calculate and manage trading risks"""
    
    def __init__(self):
        self.var_confidence = 0.95  # 95% Value at Risk
    
    def calculate_var(self, profit_history: List[float], confidence: float = None) -> float:
        """Calculate Value at Risk for given confidence level"""
        if not profit_history:
            return 0.0
        
        conf = confidence or self.var_confidence
        percentile = (1 - conf) * 100
        return np.percentile(profit_history, percentile)
    
    def calculate_sharpe_ratio(self, profits: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for performance evaluation"""
        if not profits or len(profits) < 2:
            return 0.0
        
        excess_returns = [p - risk_free_rate/252 for p in profits]  # Daily risk-free rate
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    fault_bus = FaultBus()
    navigator = ProfitCycleNavigator(fault_bus)
    
    # Simulate market updates
    import random
    base_price = 100.0
    
    for i in range(50):
        # Simulate price movement
        price_change = random.gauss(0, 0.02)  # 2% volatility
        current_price = base_price * (1 + price_change)
        current_volume = random.uniform(1000, 5000)
        timestamp = datetime.now()
        
        # Update navigator
        profit_vector = navigator.update_market_state(current_price, current_volume, timestamp)
        
        # Get trade signal
        signal = navigator.get_trade_signal()
        
        if signal:
            print(f"Tick {i}: {signal['action']} signal - Direction: {signal.get('direction', 'N/A')}, Confidence: {profit_vector.confidence:.3f}")
        
        base_price = current_price
    
    # Export logs
    print("\n=== Navigation Log ===")
    print(navigator.export_navigation_log())
    print("\n=== Fault Correlations ===")
    print(fault_bus.export_correlation_matrix()) 