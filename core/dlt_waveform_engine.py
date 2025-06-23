#!/usr/bin/env python3
"""
dlt_waveform_engine.py - Discrete Log-Time Waveform Engine for Schwabot.

Builds discrete logic-based waveform profiles to monitor volatility and detect
momentum-based changes across time ticks. Serves as a reactive core for
timing entries and exits.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
import json
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import threading
import time
from collections import deque
from enum import Enum

from core.utils.math_utils import (
    calculate_tick_acceleration,
    waveform_pattern_match,
    moving_average,
)

# Import comprehensive typing system
from core.type_defs import (
    GhostLogicState, FallbackSystem, IdentityState, IdentityTrace,
    hash_state, save_identity_trace
)

logger = logging.getLogger(__name__)


class WaveformType(Enum):
    """Waveform type enumeration"""
    PRICE_WAVEFORM = "price_waveform"
    VOLUME_WAVEFORM = "volume_waveform"
    VOLATILITY_WAVEFORM = "volatility_waveform"
    MOMENTUM_WAVEFORM = "momentum_waveform"
    FRACTAL_WAVEFORM = "fractal_waveform"


@dataclass
class WaveformData:
    """Waveform data structure"""
    waveform_id: str
    waveform_type: WaveformType
    timestamp: datetime
    data_points: List[float]
    metadata: Dict[str, Any]
    hash_signature: str


@dataclass
class WaveformAnalysis:
    """Waveform analysis result"""
    analysis_id: str
    waveform_id: str
    timestamp: datetime
    frequency_components: Dict[str, float]
    amplitude_spectrum: List[float]
    phase_spectrum: List[float]
    dominant_frequencies: List[float]
    pattern_matches: List[Dict[str, Any]]
    prediction_confidence: float
    next_state_probability: Dict[str, float]
    metadata: Dict[str, Any]


class DLTWaveformEngine:
    """Distributed Ledger Technology waveform analysis engine"""
    
    def __init__(self, history_size: int = 1000, analysis_window: int = 100):
        self.history_size = history_size
        self.analysis_window = analysis_window
        
        # Data storage
        self.price_history: deque = deque(maxlen=history_size)
        self.volume_history: deque = deque(maxlen=history_size)
        self.volatility_history: deque = deque(maxlen=history_size)
        self.momentum_history: deque = deque(maxlen=history_size)
        
        # Waveform storage
        self.waveforms: Dict[str, WaveformData] = {}
        self.analyses: Dict[str, WaveformAnalysis] = {}
        
        # Real-time analysis
        self.current_waveform_state = {
            "price_acceleration": 0.0,
            "volume_momentum": 0.0,
            "volatility_trend": 0.0,
            "momentum_strength": 0.0,
            "fractal_dimension": 1.5,
            "pattern_confidence": 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        self.analysis_thread = None
        
        # Initialize directories
        self._initialize_directories()
        
        # Load existing data
        self._load_waveform_data()
        
        # Start background analysis
        self.start_background_analysis()
    
    def _initialize_directories(self):
        """Initialize waveform-related directories"""
        waveform_dirs = [
            "core/waveform_data/",
            "core/waveform_analyses/",
            "core/waveform_patterns/",
            "core/waveform_predictions/"
        ]
        
        for dir_path in waveform_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _load_waveform_data(self):
        """Load existing waveform data from files"""
        try:
            # Load waveforms
            waveforms_file = Path("core/waveform_data/waveforms.json")
            if waveforms_file.exists():
                with open(waveforms_file, 'r') as f:
                    waveforms_data = json.load(f)
                    for waveform_id, data in waveforms_data.items():
                        data["waveform_type"] = WaveformType(data["waveform_type"])
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                        self.waveforms[waveform_id] = WaveformData(**data)
            
            # Load analyses
            analyses_file = Path("core/waveform_analyses/analyses.json")
            if analyses_file.exists():
                with open(analyses_file, 'r') as f:
                    analyses_data = json.load(f)
                    for analysis_id, data in analyses_data.items():
                        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                        self.analyses[analysis_id] = WaveformAnalysis(**data)
                        
        except Exception as e:
            print(f"Warning: Could not load waveform data: {e}")
    
    def _save_waveform_data(self):
        """Save waveform data to files"""
        try:
            # Save waveforms
            waveforms_data = {
                waveform_id: asdict(waveform) 
                for waveform_id, waveform in self.waveforms.items()
            }
            with open("core/waveform_data/waveforms.json", 'w') as f:
                json.dump(waveforms_data, f, indent=2, default=str)
            
            # Save analyses
            analyses_data = {
                analysis_id: asdict(analysis) 
                for analysis_id, analysis in self.analyses.items()
            }
            with open("core/waveform_analyses/analyses.json", 'w') as f:
                json.dump(analyses_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving waveform data: {e}")
    
    def update_tick_data(self, price: float, timestamp: float = None):
        """Update tick data with new price and volume information"""
        
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            # Update price history
            self.price_history.append(price)
            
            # Calculate volume (simulated)
            volume = 1000.0 + np.random.normal(0, 200)
            self.volume_history.append(volume)
            
            # Calculate volatility
            if len(self.price_history) > 1:
                returns = np.diff(list(self.price_history)) / list(self.price_history)[:-1]
                volatility = np.std(returns) if len(returns) > 0 else 0.0
                self.volatility_history.append(volatility)
            
            # Calculate momentum
            if len(self.price_history) > 10:
                short_ma = np.mean(list(self.price_history)[-10:])
                long_ma = np.mean(list(self.price_history)[-20:])
                momentum = (short_ma - long_ma) / long_ma
                self.momentum_history.append(momentum)
            
            # Update current waveform state
            self._update_current_state()
    
    def _update_current_state(self):
        """Update current waveform state based on recent data"""
        
        if len(self.price_history) < 20:
            return
        
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        
        # Calculate price acceleration
        if len(prices) >= 3:
            price_changes = np.diff(prices[-3:])
            self.current_waveform_state["price_acceleration"] = price_changes[-1] - price_changes[0]
        
        # Calculate volume momentum
        if len(volumes) >= 5:
            volume_ma = np.mean(volumes[-5:])
            current_volume = volumes[-1]
            self.current_waveform_state["volume_momentum"] = (current_volume - volume_ma) / volume_ma
        
        # Calculate volatility trend
        if len(self.volatility_history) >= 10:
            recent_vol = list(self.volatility_history)[-10:]
            self.current_waveform_state["volatility_trend"] = np.mean(recent_vol[-5:]) - np.mean(recent_vol[:5])
        
        # Calculate momentum strength
        if len(self.momentum_history) >= 5:
            momentum_values = list(self.momentum_history)[-5:]
            self.current_waveform_state["momentum_strength"] = np.mean(momentum_values)
        
        # Calculate fractal dimension (simplified)
        if len(prices) >= 20:
            # Simplified box-counting method
            price_range = max(prices[-20:]) - min(prices[-20:])
            price_std = np.std(prices[-20:])
            self.current_waveform_state["fractal_dimension"] = 1.0 + (price_std / price_range) if price_range > 0 else 1.5
        
        # Calculate pattern confidence
        pattern_matches = self._detect_patterns(prices)
        self.current_waveform_state["pattern_confidence"] = len(pattern_matches) * 0.2
    
    def _detect_patterns(self, prices: List[float]) -> List[Dict[str, Any]]:
        """Detect patterns in price data"""
        
        patterns = []
        
        if len(prices) < 10:
            return patterns
        
        # Detect trend patterns
        recent_prices = prices[-10:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if trend_slope > 0.01:
            patterns.append({
                "type": "uptrend",
                "confidence": min(abs(trend_slope) * 100, 0.9),
                "strength": abs(trend_slope)
            })
        elif trend_slope < -0.01:
            patterns.append({
                "type": "downtrend",
                "confidence": min(abs(trend_slope) * 100, 0.9),
                "strength": abs(trend_slope)
            })
        
        # Detect reversal patterns
        if len(prices) >= 15:
            recent_15 = prices[-15:]
            first_half = recent_15[:7]
            second_half = recent_15[7:]
            
            first_trend = np.polyfit(range(len(first_half)), first_half, 1)[0]
            second_trend = np.polyfit(range(len(second_half)), second_half, 1)[0]
            
            if first_trend > 0.01 and second_trend < -0.01:
                patterns.append({
                    "type": "reversal_bearish",
                    "confidence": 0.7,
                    "strength": abs(first_trend) + abs(second_trend)
                })
            elif first_trend < -0.01 and second_trend > 0.01:
                patterns.append({
                    "type": "reversal_bullish",
                    "confidence": 0.7,
                    "strength": abs(first_trend) + abs(second_trend)
                })
        
        return patterns
    
    def analyze_current_waveform(self) -> Dict[str, Any]:
        """Analyze current waveform state with enhanced mathematical analysis"""
        
        with self.lock:
            if len(self.price_history) < self.analysis_window:
                return {"error": "Insufficient data for analysis"}
            
            # Get current data points
            data_points = list(self.price_history)[-self.analysis_window:]
            data_array = np.array(data_points)
            
            # Create waveform data
            waveform_id = f"waveform_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            waveform_data = WaveformData(
                waveform_id=waveform_id,
                waveform_type=WaveformType.PRICE_WAVEFORM,
                timestamp=datetime.now(),
                data_points=data_points,
                metadata={"analysis_window": self.analysis_window},
                hash_signature=hashlib.sha256(str(self.price_history).encode()).hexdigest()[:16]
            )
            
            # Perform enhanced FFT analysis
            fft_result = self._perform_fft_analysis(data_points)
            
            # Calculate additional mathematical measures
            rho_coefficient = fft_result.get("rho_coefficient", 0.0)
            resonance_strength = self._calculate_waveform_resonance(data_array, fft_result["frequency_components"])
            entropy_complexity = self._calculate_entropy_complexity(data_array)
            
            # Detect patterns
            patterns = self._detect_patterns(data_points)
            
            # Update current state with new calculations
            self.current_waveform_state.update({
                "rho_coefficient": rho_coefficient,
                "resonance_strength": resonance_strength,
                "entropy_complexity": entropy_complexity,
                "current_acceleration": self._calculate_current_acceleration(data_array),
                "smoothed_acceleration": self._calculate_smoothed_acceleration(data_array),
                "current_velocity": self._calculate_current_velocity(data_array)
            })
            
            # Create enhanced analysis result
            analysis_id = f"analysis_{waveform_id}"
            analysis = WaveformAnalysis(
                analysis_id=analysis_id,
                waveform_id=waveform_id,
                timestamp=datetime.now(),
                frequency_components=fft_result["frequency_components"],
                amplitude_spectrum=fft_result["amplitude_spectrum"],
                phase_spectrum=fft_result["phase_spectrum"],
                dominant_frequencies=fft_result["dominant_frequencies"],
                pattern_matches=patterns,
                prediction_confidence=self._calculate_prediction_confidence(patterns, fft_result),
                next_state_probability=self._predict_next_state(patterns, fft_result),
                metadata={
                    "analysis_method": "enhanced_fft_pattern",
                    "rho_coefficient": rho_coefficient,
                    "resonance_strength": resonance_strength,
                    "entropy_complexity": entropy_complexity
                }
            )
            
            # Store results
            self.waveforms[waveform_id] = waveform_data
            self.analyses[analysis_id] = analysis
            
            # Save data
            self._save_waveform_data()
            
            return {
                "waveform_id": waveform_id,
                "analysis_id": analysis_id,
                "current_state": self.current_waveform_state,
                "patterns": patterns,
                "prediction_confidence": analysis.prediction_confidence,
                "next_state_probability": analysis.next_state_probability,
                "frequency_analysis": fft_result,
                "mathematical_measures": {
                    "rho_coefficient": rho_coefficient,
                    "resonance_strength": resonance_strength,
                    "entropy_complexity": entropy_complexity,
                    "current_acceleration": self.current_waveform_state["current_acceleration"],
                    "smoothed_acceleration": self.current_waveform_state["smoothed_acceleration"],
                    "current_velocity": self.current_waveform_state["current_velocity"]
                }
            }
    
    def _calculate_current_acceleration(self, data: np.ndarray) -> float:
        """Calculate current acceleration (second derivative) of the waveform."""
        try:
            if len(data) < 3:
                return 0.0
            
            # Calculate second derivative using finite differences
            dt = 1.0  # Assuming unit time steps
            acceleration = (data[-1] - 2 * data[-2] + data[-3]) / (dt ** 2)
            
            return acceleration
            
        except Exception as e:
            logger.error(f"Error calculating current acceleration: {e}")
            return 0.0
    
    def _calculate_smoothed_acceleration(self, data: np.ndarray) -> float:
        """Calculate smoothed acceleration using moving average."""
        try:
            if len(data) < 10:
                return 0.0
            
            # Calculate acceleration over a window
            window_size = min(10, len(data) // 2)
            accelerations = []
            
            for i in range(window_size, len(data)):
                if i >= 2:
                    acc = (data[i] - 2 * data[i-1] + data[i-2])
                    accelerations.append(acc)
            
            # Return smoothed acceleration
            return np.mean(accelerations) if accelerations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating smoothed acceleration: {e}")
            return 0.0
    
    def _calculate_current_velocity(self, data: np.ndarray) -> float:
        """Calculate current velocity (first derivative) of the waveform."""
        try:
            if len(data) < 2:
                return 0.0
            
            # Calculate first derivative using finite differences
            dt = 1.0  # Assuming unit time steps
            velocity = (data[-1] - data[-2]) / dt
            
            return velocity
            
        except Exception as e:
            logger.error(f"Error calculating current velocity: {e}")
            return 0.0
    
    def _perform_fft_analysis(self, data_points: List[float]) -> Dict[str, Any]:
        """Perform Fast Fourier Transform analysis"""
        
        # Convert to numpy array
        data = np.array(data_points)
        
        # Apply FFT
        fft_result = np.fft.fft(data)
        
        # Calculate frequency components
        freqs = np.fft.fftfreq(len(data))
        
        # Get amplitude spectrum
        amplitude_spectrum = np.abs(fft_result)
        
        # Get phase spectrum
        phase_spectrum = np.angle(fft_result)
        
        # Find dominant frequencies
        dominant_indices = np.argsort(amplitude_spectrum)[-5:]  # Top 5 frequencies
        dominant_frequencies = freqs[dominant_indices]
        
        # Calculate frequency components
        frequency_components = {
            "low_frequency": np.mean(amplitude_spectrum[freqs < 0.1]),
            "medium_frequency": np.mean(amplitude_spectrum[(freqs >= 0.1) & (freqs < 0.3)]),
            "high_frequency": np.mean(amplitude_spectrum[freqs >= 0.3]),
            "dominant_frequency": dominant_frequencies[0] if len(dominant_frequencies) > 0 else 0.0
        }
        
        # Calculate ρ (rho) - waveform alignment coefficient
        rho = self._calculate_rho_coefficient(data, amplitude_spectrum, phase_spectrum)
        
        return {
            "frequency_components": frequency_components,
            "amplitude_spectrum": amplitude_spectrum.tolist(),
            "phase_spectrum": phase_spectrum.tolist(),
            "dominant_frequencies": dominant_frequencies.tolist(),
            "rho_coefficient": rho
        }
    
    def _calculate_rho_coefficient(self, data: np.ndarray, amplitude_spectrum: np.ndarray, phase_spectrum: np.ndarray) -> float:
        """
        Calculate ρ (rho) coefficient for waveform alignment.
        
        ρ = |W(t_entry) / A|
        Where:
        - W(t_entry) = Waveform value at entry time
        - A = Average amplitude of the waveform
        
        This measures how well the current waveform aligns with expected patterns.
        """
        try:
            if len(data) == 0 or len(amplitude_spectrum) == 0:
                return 0.0
            
            # Calculate average amplitude
            avg_amplitude = np.mean(amplitude_spectrum)
            if avg_amplitude == 0:
                return 0.0
            
            # Get current waveform value (last data point)
            current_waveform_value = data[-1]
            
            # Calculate ρ coefficient
            rho = abs(current_waveform_value / avg_amplitude)
            
            # Normalize to [0, 1] range
            rho = min(1.0, rho)
            
            return rho
            
        except Exception as e:
            logger.error(f"Error calculating rho coefficient: {e}")
            return 0.0
    
    def _calculate_waveform_resonance(self, data: np.ndarray, frequency_components: Dict[str, float]) -> float:
        """
        Calculate waveform resonance strength.
        
        Resonance = Σ(amplitude_i * frequency_i) / Σ(amplitude_i)
        This measures how strongly the waveform resonates at its dominant frequencies.
        """
        try:
            if len(data) == 0:
                return 0.0
            
            # Get frequency components
            low_freq = frequency_components.get("low_frequency", 0.0)
            medium_freq = frequency_components.get("medium_frequency", 0.0)
            high_freq = frequency_components.get("high_frequency", 0.0)
            dominant_freq = frequency_components.get("dominant_frequency", 0.0)
            
            # Calculate weighted resonance
            amplitudes = [low_freq, medium_freq, high_freq]
            frequencies = [0.05, 0.2, 0.4]  # Representative frequencies
            
            if sum(amplitudes) == 0:
                return 0.0
            
            resonance = sum(amp * freq for amp, freq in zip(amplitudes, frequencies)) / sum(amplitudes)
            
            # Normalize to [0, 1] range
            resonance = min(1.0, resonance)
            
            return resonance
            
        except Exception as e:
            logger.error(f"Error calculating waveform resonance: {e}")
            return 0.0
    
    def _calculate_entropy_complexity(self, data: np.ndarray) -> float:
        """
        Calculate entropy-based complexity measure.
        
        Entropy = -Σ(p_i * log(p_i))
        Where p_i is the probability of each value in the data.
        """
        try:
            if len(data) == 0:
                return 0.0
            
            # Discretize data into bins
            hist, bin_edges = np.histogram(data, bins=min(20, len(data)//2))
            
            # Calculate probabilities
            probabilities = hist / np.sum(hist)
            probabilities = probabilities[probabilities > 0]  # Remove zero probabilities
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Normalize to [0, 1] range (max entropy for uniform distribution)
            max_entropy = np.log2(len(probabilities))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0.0
            
            return normalized_entropy
            
        except Exception as e:
            logger.error(f"Error calculating entropy complexity: {e}")
            return 0.0
    
    def _calculate_prediction_confidence(self, patterns: List[Dict[str, Any]], fft_result: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on patterns and frequency analysis"""
        
        confidence = 0.5  # Base confidence
        
        # Pattern confidence
        if patterns:
            pattern_confidence = np.mean([p.get("confidence", 0.0) for p in patterns])
            confidence += pattern_confidence * 0.3
        
        # Frequency stability
        freq_components = fft_result["frequency_components"]
        if freq_components["dominant_frequency"] > 0:
            confidence += 0.2
        
        # Volume momentum
        if self.current_waveform_state["volume_momentum"] > 0.1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _predict_next_state(self, patterns: List[Dict[str, Any]], fft_result: Dict[str, Any]) -> Dict[str, float]:
        """Predict next market state probabilities"""
        
        probabilities = {
            "bullish": 0.33,
            "bearish": 0.33,
            "sideways": 0.34
        }
        
        # Adjust based on patterns
        for pattern in patterns:
            if pattern["type"] == "uptrend":
                probabilities["bullish"] += 0.2
                probabilities["bearish"] -= 0.1
                probabilities["sideways"] -= 0.1
            elif pattern["type"] == "downtrend":
                probabilities["bearish"] += 0.2
                probabilities["bullish"] -= 0.1
                probabilities["sideways"] -= 0.1
            elif pattern["type"] == "reversal_bullish":
                probabilities["bullish"] += 0.15
                probabilities["bearish"] -= 0.1
                probabilities["sideways"] -= 0.05
            elif pattern["type"] == "reversal_bearish":
                probabilities["bearish"] += 0.15
                probabilities["bullish"] -= 0.1
                probabilities["sideways"] -= 0.05
        
        # Adjust based on momentum
        momentum = self.current_waveform_state["momentum_strength"]
        if momentum > 0.01:
            probabilities["bullish"] += 0.1
            probabilities["bearish"] -= 0.05
            probabilities["sideways"] -= 0.05
        elif momentum < -0.01:
            probabilities["bearish"] += 0.1
            probabilities["bullish"] -= 0.05
            probabilities["sideways"] -= 0.05
        
        # Normalize probabilities
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}
        
        return probabilities
    
    def get_waveform_statistics(self) -> Dict[str, Any]:
        """Get waveform statistics"""
        
        return {
            "total_waveforms": len(self.waveforms),
            "total_analyses": len(self.analyses),
            "current_state": self.current_waveform_state,
            "data_points": {
                "price_history": len(self.price_history),
                "volume_history": len(self.volume_history),
                "volatility_history": len(self.volatility_history),
                "momentum_history": len(self.momentum_history)
            },
            "analysis_window": self.analysis_window,
            "history_size": self.history_size
        }
    
    def start_background_analysis(self):
        """Start background analysis thread"""
        
        if self.running:
            return
        
        self.running = True
        self.analysis_thread = threading.Thread(target=self._background_analysis_loop)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def stop_background_analysis(self):
        """Stop background analysis thread"""
        
        self.running = False
        if self.analysis_thread:
            self.analysis_thread.join()
    
    def _background_analysis_loop(self):
        """Background analysis loop"""
        
        while self.running:
            try:
                # Perform periodic analysis
                if len(self.price_history) >= self.analysis_window:
                    self.analyze_current_waveform()
                
                # Sleep for analysis interval
                time.sleep(60)  # Analyze every minute
                
            except Exception as e:
                print(f"Error in background analysis: {e}")
                time.sleep(10)


def get_dlt_waveform_engine() -> DLTWaveformEngine:
    """Get singleton instance of DLT waveform engine"""
    if not hasattr(get_dlt_waveform_engine, '_instance'):
        get_dlt_waveform_engine._instance = DLTWaveformEngine()
    return get_dlt_waveform_engine._instance


# Example usage
if __name__ == "__main__":
    # Create DLT waveform engine
    engine = get_dlt_waveform_engine()
    
    # Simulate some market data
    for i in range(100):
        price = 50000.0 + np.random.normal(0, 1000)
        engine.update_tick_data(price)
        time.sleep(0.1)
    
    # Analyze current waveform
    analysis = engine.analyze_current_waveform()
    print("Waveform Analysis:")
    print(json.dumps(analysis, indent=2, default=str))
    
    # Get statistics
    stats = engine.get_waveform_statistics()
    print("\nWaveform Statistics:")
    print(json.dumps(stats, indent=2, default=str))
