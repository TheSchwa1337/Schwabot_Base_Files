#!/usr/bin/env python3
"""
Spectral Transform Engine - Schwabot Mathematical Framework
=========================================================

Implements spectral analysis, wavelet transforms, and entropy calculations
for the DLT Waveform Engine. Provides frequency domain analysis for
trading signal processing and pattern recognition.

Mathematical foundations:
- FFT for frequency decomposition
- Continuous Wavelet Transform (CWT) for time-frequency analysis  
- Spectral entropy for signal complexity measurement
- Power spectral density for market oscillation detection

Based on SxN-Math specifications and Windows-compatible architecture.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Optional, Union, Any
from decimal import Decimal, getcontext
import logging
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import pywt

# Set high precision for financial calculations
getcontext().prec = 18

# Type definitions for clarity
Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
ComplexVector = npt.NDArray[np.complex128]

logger = logging.getLogger(__name__)


class SpectralTransform:
    """
    Core spectral analysis engine for trading signals
    
    Provides frequency domain analysis, wavelet decomposition,
    and entropy-based signal characterization.
    """
    
    def __init__(self, sample_rate: float = 1.0):
        self.sample_rate = sample_rate
        self.epsilon = 1e-12  # Numerical stability constant
        logger.info("SpectralTransform engine initialized")
    
    def fft_transform(self, time_series: Vector) -> Tuple[ComplexVector, Vector]:
        """
        Fast Fourier Transform with frequency bins
        
        Args:
            time_series: Input signal in time domain
            
        Returns:
            (fft_coefficients, frequencies)
        """
        try:
            # Apply window to reduce spectral leakage
            windowed_signal = time_series * np.hanning(len(time_series))
            
            # Compute FFT
            fft_coeffs = fft(windowed_signal)
            frequencies = fftfreq(len(time_series), d=1/self.sample_rate)
            
            logger.debug(f"FFT computed for {len(time_series)} samples")
            return fft_coeffs, frequencies
            
        except Exception as e:
            logger.error(f"FFT computation failed: {e}")
            raise
    
    def power_spectral_density(self, time_series: Vector) -> Tuple[Vector, Vector]:
        """
        Compute Power Spectral Density using Welch's method
        
        Args:
            time_series: Input signal
            
        Returns:
            (frequencies, power_density)
        """
        try:
            # Use Welch's method for robust PSD estimation
            freqs, psd = signal.welch(
                time_series,
                fs=self.sample_rate,
                window='hann',
                nperseg=min(len(time_series)//4, 256),
                overlap=None
            )
            
            return freqs, psd
            
        except Exception as e:
            logger.error(f"PSD computation failed: {e}")
            raise
    
    def continuous_wavelet_transform(self, time_series: Vector, 
                                   scales: Optional[Vector] = None,
                                   wavelet: str = 'morl') -> Tuple[Matrix, Vector]:
        """
        Continuous Wavelet Transform for time-frequency analysis
        
        Args:
            time_series: Input signal
            scales: Wavelet scales (auto-generated if None)
            wavelet: Wavelet type ('morl', 'mexh', 'cgau1')
            
        Returns:
            (cwt_coefficients, scales_used)
        """
        try:
            if scales is None:
                # Auto-generate logarithmic scale distribution
                scales = np.logspace(0, np.log10(len(time_series)/4), 32)
            
            # Compute CWT
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)
            
            logger.debug(f"CWT computed with {len(scales)} scales")
            return coefficients, scales
            
        except Exception as e:
            logger.error(f"CWT computation failed: {e}")
            raise
    
    def spectral_entropy(self, time_series: Vector, base: float = 2.0) -> float:
        """
        Calculate spectral entropy as measure of signal complexity
        
        Implements: H = -Σ p_i * log_base(p_i) where p_i = |X(f)|² / Σ|X(f)|²
        
        Args:
            time_series: Input signal
            base: Logarithm base (2 for bits, e for nats)
            
        Returns:
            Spectral entropy value
        """
        try:
            # Compute power spectrum
            fft_coeffs, _ = self.fft_transform(time_series)
            power_spectrum = np.abs(fft_coeffs) ** 2
            
            # Only use positive frequencies (real signals are symmetric)
            half_len = len(power_spectrum) // 2
            power_spectrum = power_spectrum[:half_len]
            
            # Normalize to probability distribution
            total_power = np.sum(power_spectrum) + self.epsilon
            probabilities = power_spectrum / total_power
            
            # Remove zero probabilities for stable log computation
            probabilities = probabilities[probabilities > self.epsilon]
            
            # Calculate entropy
            if base == 2.0:
                entropy_val = -np.sum(probabilities * np.log2(probabilities))
            else:
                entropy_val = -np.sum(probabilities * np.log(probabilities) / np.log(base))
            
            return float(entropy_val)
            
        except Exception as e:
            logger.error(f"Spectral entropy calculation failed: {e}")
            return 0.0
    
    def dominant_frequency(self, time_series: Vector) -> float:
        """
        Find dominant frequency component in signal
        
        Args:
            time_series: Input signal
            
        Returns:
            Dominant frequency in Hz
        """
        try:
            freqs, psd = self.power_spectral_density(time_series)
            
            # Find frequency with maximum power (excluding DC component)
            if len(freqs) > 1:
                # Skip DC component (index 0)
                max_idx = np.argmax(psd[1:]) + 1
                dominant_freq = freqs[max_idx]
            else:
                dominant_freq = 0.0
            
            return float(dominant_freq)
            
        except Exception as e:
            logger.error(f"Dominant frequency detection failed: {e}")
            return 0.0
    
    def bandpower(self, time_series: Vector, freq_range: Tuple[float, float]) -> float:
        """
        Calculate power in specific frequency band
        
        Args:
            time_series: Input signal
            freq_range: (low_freq, high_freq) tuple
            
        Returns:
            Power in specified band
        """
        try:
            freqs, psd = self.power_spectral_density(time_series)
            
            # Find indices corresponding to frequency range
            low_freq, high_freq = freq_range
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Integrate power in band
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            
            return float(band_power)
            
        except Exception as e:
            logger.error(f"Band power calculation failed: {e}")
            return 0.0
    
    def signal_to_noise_ratio(self, time_series: Vector, 
                            signal_band: Tuple[float, float],
                            noise_band: Tuple[float, float]) -> float:
        """
        Calculate SNR between signal and noise frequency bands
        
        Args:
            time_series: Input signal
            signal_band: (low, high) frequency range for signal
            noise_band: (low, high) frequency range for noise
            
        Returns:
            SNR in dB
        """
        try:
            signal_power = self.bandpower(time_series, signal_band)
            noise_power = self.bandpower(time_series, noise_band)
            
            if noise_power > self.epsilon:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = float('inf')
            
            return float(snr_db)
            
        except Exception as e:
            logger.error(f"SNR calculation failed: {e}")
            return 0.0


class DLTWaveformEngine:
    """
    DLT (Discrete Linear Transform) Waveform Engine
    
    Specialized for trading signal analysis with entropy-based
    pattern detection and waveform characterization.
    """
    
    def __init__(self):
        self.spectral = SpectralTransform()
        self.waveform_memory: Dict[str, Any] = {}
        logger.info("DLT Waveform Engine initialized")
    
    def analyze_waveform(self, signal: Vector, signal_id: str = "") -> Dict[str, Any]:
        """
        Comprehensive waveform analysis for trading signals
        
        Args:
            signal: Input price/volume time series
            signal_id: Identifier for caching results
            
        Returns:
            Analysis dictionary with spectral characteristics
        """
        try:
            analysis = {
                'signal_length': len(signal),
                'spectral_entropy': self.spectral.spectral_entropy(signal),
                'dominant_frequency': self.spectral.dominant_frequency(signal),
                'signal_energy': float(np.sum(signal ** 2)),
                'peak_frequency_power': 0.0,
                'frequency_spread': 0.0,
                'waveform_complexity': 0.0
            }
            
            # Frequency domain analysis
            freqs, psd = self.spectral.power_spectral_density(signal)
            if len(psd) > 1:
                analysis['peak_frequency_power'] = float(np.max(psd))
                analysis['frequency_spread'] = float(np.std(freqs[psd > np.mean(psd)]))
            
            # Waveform complexity measure
            cwt_coeffs, scales = self.spectral.continuous_wavelet_transform(signal)
            analysis['waveform_complexity'] = float(np.std(np.abs(cwt_coeffs)))
            
            # Cache results if ID provided
            if signal_id:
                self.waveform_memory[signal_id] = analysis
            
            logger.debug(f"Waveform analysis completed for signal length {len(signal)}")
            return analysis
            
        except Exception as e:
            logger.error(f"Waveform analysis failed: {e}")
            return {'error': str(e)}
    
    def entropy_threshold_trigger(self, signal: Vector, 
                                threshold: float = 2.0) -> bool:
        """
        Entropy-based trigger for ghost swap detection
        
        Args:
            signal: Input signal to analyze
            threshold: Entropy threshold for trigger
            
        Returns:
            True if entropy exceeds threshold
        """
        try:
            entropy = self.spectral.spectral_entropy(signal)
            return entropy > threshold
            
        except Exception as e:
            logger.error(f"Entropy trigger evaluation failed: {e}")
            return False


# Main functions for external API
def fft(series: Vector) -> ComplexVector:
    """Simple FFT wrapper for external use"""
    transform = SpectralTransform()
    coeffs, _ = transform.fft_transform(series)
    return coeffs


def cwt(series: Vector, wave: str = 'morl') -> Matrix:
    """Simple CWT wrapper for external use"""
    transform = SpectralTransform()
    coeffs, _ = transform.continuous_wavelet_transform(series, wavelet=wave)
    return coeffs


def spectral_entropy(series: Vector, base: float = 2.0) -> float:
    """Simple spectral entropy wrapper for external use"""
    transform = SpectralTransform()
    return transform.spectral_entropy(series, base)


def main() -> None:
    """Test and demonstration function"""
    # Generate test signal
    t = np.linspace(0, 1, 1000)
    test_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    test_signal += 0.1 * np.random.randn(len(t))
    
    # Test spectral analysis
    engine = DLTWaveformEngine()
    results = engine.analyze_waveform(test_signal, "test_signal")
    
    logger.info(f"Test completed: {results}")
    print("SpectralTransform module test completed successfully")


if __name__ == "__main__":
    main() 