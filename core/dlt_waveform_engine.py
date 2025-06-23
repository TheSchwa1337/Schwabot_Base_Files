#!/usr/bin/env python3
"""
DLT Waveform Engine - Schwabot UROS v1.0
========================================

Implements Distributed Ledger Technology (DLT) waveform analysis for trading signals.
Critical for processing blockchain data and market waveforms in real-time.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.signal.windows import hann

from core.type_defs import BitLevel, MatrixPhase, MatrixControllerType

logger = logging.getLogger(__name__)


@dataclass
class WaveformData:
    """Represents waveform data for analysis."""
    timestamp: datetime
    data: np.ndarray
    sample_rate: float
    frequency_components: Optional[np.ndarray] = None
    power_spectrum: Optional[np.ndarray] = None
    phase_spectrum: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaveformAnalysis:
    """Represents waveform analysis results."""
    analysis_id: str
    waveform_data: WaveformData
    dominant_frequency: float
    power_spectrum_peak: float
    bandwidth: float
    signal_to_noise_ratio: float
    correlation_coefficient: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DLTWaveformEngine:
    """
    Implements DLT waveform analysis for trading signals.
    Processes blockchain data and market waveforms in real-time.
    """
    
    def __init__(self, history_size: int = 100):
        """Initialize the DLT waveform engine."""
        self.history_size = history_size
        self.waveform_history: List[WaveformData] = []
        self.analysis_history: List[WaveformAnalysis] = []
        self.correlation_cache: Dict[str, float] = {}
        
        # Analysis parameters
        self.fft_size = 1024
        self.overlap_factor = 0.5
        self.window_type = 'hann'
        self.min_frequency = 0.1  # Hz
        self.max_frequency = 1000.0  # Hz
        
        # Performance tracking
        self.total_waveforms_processed = 0
        self.avg_processing_time = 0.0
        self.correlation_threshold = 0.7
        
        logger.info("DLT Waveform Engine initialized")
    
    def process_waveform_data(
        self,
        signal_id: str,
        data: np.ndarray,
        sample_rate: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WaveformAnalysis:
        """Process waveform data and return analysis results."""
        start_time = time.time()
        
        # Create waveform data object
        waveform_data = WaveformData(
            timestamp=datetime.now(),
            data=data,
            sample_rate=sample_rate,
            metadata=metadata or {}
        )
        
        # Compute FFT analysis
        frequency_components, power_spectrum, phase_spectrum = self._compute_fft(
            data, sample_rate
        )
        
        waveform_data.frequency_components = frequency_components
        waveform_data.power_spectrum = power_spectrum
        waveform_data.phase_spectrum = phase_spectrum
        
        # Analyze waveform characteristics
        dominant_frequency = self._find_dominant_frequency(frequency_components, power_spectrum)
        power_spectrum_peak = np.max(power_spectrum) if len(power_spectrum) > 0 else 0.0
        bandwidth = self._calculate_bandwidth(frequency_components, power_spectrum)
        signal_to_noise_ratio = self._calculate_snr(power_spectrum)
        
        # Calculate correlation with historical data
        correlation_coefficient = self._calculate_correlation(waveform_data)
        
        # Create analysis result
        analysis = WaveformAnalysis(
            analysis_id=f"analysis_{int(time.time() * 1000)}",
            waveform_data=waveform_data,
            dominant_frequency=dominant_frequency,
            power_spectrum_peak=power_spectrum_peak,
            bandwidth=bandwidth,
            signal_to_noise_ratio=signal_to_noise_ratio,
            correlation_coefficient=correlation_coefficient,
            metadata={"signal_id": signal_id}
        )
        
        # Update history
        self._update_history(waveform_data, analysis)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_performance_metrics(processing_time)
        
        logger.debug(f"Processed waveform {signal_id}: dominant_freq={dominant_frequency:.2f}Hz")
        return analysis
    
    def _compute_fft(
        self, data: np.ndarray, sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute FFT of the waveform data."""
        # Apply window function
        window = hann(len(data))
        windowed_data = data * window
        
        # Compute FFT
        fft_result = fft(windowed_data, n=self.fft_size)
        frequencies = fftfreq(self.fft_size, 1.0 / sample_rate)
        
        # Calculate power spectrum
        power_spectrum = np.abs(fft_result) ** 2
        
        # Calculate phase spectrum
        phase_spectrum = np.angle(fft_result)
        
        # Filter frequency range
        freq_mask = (frequencies >= self.min_frequency) & (frequencies <= self.max_frequency)
        filtered_frequencies = frequencies[freq_mask]
        filtered_power = power_spectrum[freq_mask]
        filtered_phase = phase_spectrum[freq_mask]
        
        return filtered_frequencies, filtered_power, filtered_phase
    
    def _find_dominant_frequency(
        self, frequencies: np.ndarray, power_spectrum: np.ndarray
    ) -> float:
        """Find the dominant frequency in the power spectrum."""
        if len(power_spectrum) == 0:
            return 0.0
        
        # Find peak in power spectrum
        peak_index = np.argmax(power_spectrum)
        dominant_frequency = frequencies[peak_index] if peak_index < len(frequencies) else 0.0
        
        return dominant_frequency
    
    def _calculate_bandwidth(
        self, frequencies: np.ndarray, power_spectrum: np.ndarray
    ) -> float:
        """Calculate the bandwidth of the signal."""
        if len(power_spectrum) == 0:
            return 0.0
        
        # Find -3dB point (half power)
        peak_power = np.max(power_spectrum)
        half_power = peak_power / 2.0
        
        # Find frequencies where power is above half power
        above_threshold = power_spectrum >= half_power
        if np.any(above_threshold):
            min_freq = frequencies[above_threshold][0]
            max_freq = frequencies[above_threshold][-1]
            bandwidth = max_freq - min_freq
        else:
            bandwidth = 0.0
        
        return bandwidth
    
    def _calculate_snr(self, power_spectrum: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(power_spectrum) == 0:
            return 0.0
        
        # Simple SNR calculation: peak power / mean power
        peak_power = np.max(power_spectrum)
        mean_power = np.mean(power_spectrum)
        
        if mean_power > 0:
            snr = peak_power / mean_power
        else:
            snr = 0.0
        
        return snr
    
    def _calculate_correlation(self, waveform_data: WaveformData) -> float:
        """Calculate correlation with historical waveforms."""
        if not self.waveform_history:
            return 0.0
        
        # Use the most recent waveform for correlation
        recent_waveform = self.waveform_history[-1]
        
        # Ensure data lengths match
        min_length = min(len(waveform_data.data), len(recent_waveform.data))
        if min_length == 0:
            return 0.0
        
        # Calculate correlation coefficient
        data1 = waveform_data.data[:min_length]
        data2 = recent_waveform.data[:min_length]
        
        correlation = np.corrcoef(data1, data2)[0, 1]
        
        # Handle NaN values
        if np.isnan(correlation):
            correlation = 0.0
        
        return correlation
    
    def _update_history(self, waveform_data: WaveformData, analysis: WaveformAnalysis) -> None:
        """Update waveform and analysis history."""
        self.waveform_history.append(waveform_data)
        self.analysis_history.append(analysis)
        
        # Maintain history size
        if len(self.waveform_history) > self.history_size:
            self.waveform_history = self.waveform_history[-self.history_size:]
            self.analysis_history = self.analysis_history[-self.history_size:]
    
    def _update_performance_metrics(self, processing_time: float) -> None:
        """Update performance tracking metrics."""
        self.total_waveforms_processed += 1
        
        # Update average processing time
        alpha = 0.1
        self.avg_processing_time = alpha * processing_time + (1 - alpha) * self.avg_processing_time
    
    def detect_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in waveform data."""
        patterns = []
        
        if len(self.analysis_history) < 2:
            return patterns
        
        # Analyze recent analyses for patterns
        recent_analyses = self.analysis_history[-10:]  # Last 10 analyses
        
        # Frequency pattern detection
        frequencies = [analysis.dominant_frequency for analysis in recent_analyses]
        if len(frequencies) >= 3:
            freq_std = np.std(frequencies)
            if freq_std < 1.0:  # Low frequency variation
                patterns.append({
                    "type": "stable_frequency",
                    "frequency_std": freq_std,
                    "avg_frequency": np.mean(frequencies),
                    "confidence": 1.0 - freq_std / 10.0
                })
        
        # Power pattern detection
        powers = [analysis.power_spectrum_peak for analysis in recent_analyses]
        if len(powers) >= 3:
            power_trend = np.polyfit(range(len(powers)), powers, 1)[0]
            if abs(power_trend) > 0.1:  # Significant trend
                patterns.append({
                    "type": "power_trend",
                    "trend_slope": power_trend,
                    "trend_direction": "increasing" if power_trend > 0 else "decreasing",
                    "confidence": min(1.0, abs(power_trend))
                })
        
        # Correlation pattern detection
        correlations = [analysis.correlation_coefficient for analysis in recent_analyses]
        if len(correlations) >= 3:
            avg_correlation = np.mean(correlations)
            if avg_correlation > self.correlation_threshold:
                patterns.append({
                    "type": "high_correlation",
                    "avg_correlation": avg_correlation,
                    "correlation_std": np.std(correlations),
                    "confidence": avg_correlation
                })
        
        return patterns
    
    def get_waveform_statistics(self) -> Dict[str, Any]:
        """Get comprehensive waveform statistics."""
        total_analyses = len(self.analysis_history)
        
        if total_analyses == 0:
            return {
                "total_waveforms_processed": 0,
                "average_processing_time": 0.0,
                "total_analyses": 0
            }
        
        # Frequency statistics
        frequencies = [analysis.dominant_frequency for analysis in self.analysis_history]
        avg_frequency = np.mean(frequencies)
        freq_std = np.std(frequencies)
        
        # Power statistics
        powers = [analysis.power_spectrum_peak for analysis in self.analysis_history]
        avg_power = np.mean(powers)
        max_power = np.max(powers)
        
        # SNR statistics
        snrs = [analysis.signal_to_noise_ratio for analysis in self.analysis_history]
        avg_snr = np.mean(snrs)
        
        # Correlation statistics
        correlations = [analysis.correlation_coefficient for analysis in self.analysis_history]
        avg_correlation = np.mean(correlations)
        
        return {
            "total_waveforms_processed": self.total_waveforms_processed,
            "average_processing_time": self.avg_processing_time,
            "total_analyses": total_analyses,
            "average_frequency": avg_frequency,
            "frequency_std": freq_std,
            "average_power": avg_power,
            "max_power": max_power,
            "average_snr": avg_snr,
            "average_correlation": avg_correlation,
            "history_size": self.history_size
        }
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Generate trading signals based on waveform analysis."""
        signals = []
        
        if not self.analysis_history:
            return signals
        
        # Get latest analysis
        latest_analysis = self.analysis_history[-1]
        
        # High frequency signal
        if latest_analysis.dominant_frequency > 100.0:
            signal = {
                "type": "high_frequency_signal",
                "frequency": latest_analysis.dominant_frequency,
                "confidence": min(1.0, latest_analysis.dominant_frequency / 200.0),
                "strength": min(1.0, latest_analysis.dominant_frequency / 500.0),
                "timestamp": latest_analysis.timestamp,
                "metadata": {"power_spectrum_peak": latest_analysis.power_spectrum_peak}
            }
            signals.append(signal)
        
        # High SNR signal
        if latest_analysis.signal_to_noise_ratio > 10.0:
            signal = {
                "type": "high_snr_signal",
                "snr": latest_analysis.signal_to_noise_ratio,
                "confidence": min(1.0, latest_analysis.signal_to_noise_ratio / 20.0),
                "strength": min(1.0, latest_analysis.signal_to_noise_ratio / 50.0),
                "timestamp": latest_analysis.timestamp,
                "metadata": {"bandwidth": latest_analysis.bandwidth}
            }
            signals.append(signal)
        
        # High correlation signal
        if latest_analysis.correlation_coefficient > 0.8:
            signal = {
                "type": "high_correlation_signal",
                "correlation": latest_analysis.correlation_coefficient,
                "confidence": latest_analysis.correlation_coefficient,
                "strength": latest_analysis.correlation_coefficient,
                "timestamp": latest_analysis.timestamp,
                "metadata": {"dominant_frequency": latest_analysis.dominant_frequency}
            }
            signals.append(signal)
        
        # Pattern-based signals
        patterns = self.detect_patterns()
        for pattern in patterns:
            if pattern["confidence"] > 0.7:
                signal = {
                    "type": f"pattern_{pattern['type']}",
                    "pattern_type": pattern["type"],
                    "confidence": pattern["confidence"],
                    "strength": pattern["confidence"],
                    "timestamp": datetime.now(),
                    "metadata": pattern
                }
                signals.append(signal)
        
        return signals


def main() -> None:
    """Main function for testing the DLT waveform engine."""
    # Initialize engine
    engine = DLTWaveformEngine()
    
    # Generate test signals
    sample_rate = 1000.0  # 1 kHz
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal 1: Sine wave
    signal1 = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave
    
    # Signal 2: Chirp signal
    signal2 = signal.chirp(t, f0=10, f1=100, t1=duration, method='linear')
    
    # Process waveforms
    analysis1 = engine.process_waveform_data("signal1", signal1, sample_rate)
    analysis2 = engine.process_waveform_data("signal2", signal2, sample_rate)
    
    # Get statistics
    stats = engine.get_waveform_statistics()
    print(f"Waveform statistics: {stats}")
    
    # Detect patterns
    patterns = engine.detect_patterns()
    print(f"Detected patterns: {len(patterns)}")
    
    # Get trading signals
    signals = engine.get_trading_signals()
    print(f"Generated {len(signals)} trading signals")


if __name__ == "__main__":
    main() 