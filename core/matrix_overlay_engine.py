"""
Matrix Overlay Engine
===================

Implements long-range harmonic regression and matrix overlay analysis for shell state evolution.
Integrates with ZygotShell and QuantumCellularRiskMonitor for comprehensive pattern analysis.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from pathlib import Path

from .zygot_shell import ZygotShellState
from .quantum_cellular_risk_monitor import GANAnomalyMetrics

logger = logging.getLogger(__name__)

@dataclass
class HarmonicPattern:
    """Represents a harmonic pattern in shell state evolution"""
    frequency: float
    amplitude: float
    phase: float
    confidence: float
    start_time: float
    end_time: Optional[float] = None
    pattern_type: str = "unknown"
    metadata: Dict[str, Any] = None

@dataclass
class MatrixOverlay:
    """Represents a matrix overlay state"""
    timestamp: float
    shell_states: List[ZygotShellState]
    harmonic_patterns: List[HarmonicPattern]
    anomaly_metrics: List[GANAnomalyMetrics]
    overlay_score: float
    stability_index: float
    is_active: bool

class MatrixOverlayEngine:
    """Engine for analyzing long-range harmonic patterns in shell states"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_pattern_length = config.get('min_pattern_length', 10)
        self.max_pattern_length = config.get('max_pattern_length', 100)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.overlay_history: List[MatrixOverlay] = []
        
        # Pattern tracking
        self.active_patterns: List[HarmonicPattern] = []
        self.pattern_history: List[HarmonicPattern] = []
        
        # FFT parameters
        self.fft_window = config.get('fft_window', 50)
        self.freq_threshold = config.get('freq_threshold', 0.1)
        
    def analyze_shell_states(self, states: List[ZygotShellState]) -> MatrixOverlay:
        """
        Analyze shell states for harmonic patterns.
        
        Args:
            states: List of shell states to analyze
            
        Returns:
            MatrixOverlay object
        """
        if not states:
            return MatrixOverlay(
                timestamp=datetime.now().timestamp(),
                shell_states=[],
                harmonic_patterns=[],
                anomaly_metrics=[],
                overlay_score=0.0,
                stability_index=0.0,
                is_active=False
            )
            
        # Extract time series
        timestamps = [s.timestamp for s in states]
        alignment_scores = [s.alignment_score for s in states]
        drift_resonances = [s.drift_resonance for s in states]
        
        # Detect harmonic patterns
        patterns = self._detect_harmonic_patterns(
            timestamps,
            alignment_scores,
            drift_resonances
        )
        
        # Update active patterns
        self._update_active_patterns(patterns, timestamps[-1])
        
        # Compute overlay metrics
        overlay_score = self._compute_overlay_score(patterns)
        stability_index = self._compute_stability_index(states)
        
        # Create overlay
        overlay = MatrixOverlay(
            timestamp=timestamps[-1],
            shell_states=states,
            harmonic_patterns=patterns,
            anomaly_metrics=[],  # TODO: Integrate with GAN anomaly metrics
            overlay_score=overlay_score,
            stability_index=stability_index,
            is_active=len(patterns) > 0
        )
        
        self.overlay_history.append(overlay)
        return overlay
        
    def _detect_harmonic_patterns(self,
                                timestamps: List[float],
                                alignment_scores: List[float],
                                drift_resonances: List[float]) -> List[HarmonicPattern]:
        """
        Detect harmonic patterns in time series data.
        
        Args:
            timestamps: List of timestamps
            alignment_scores: List of alignment scores
            drift_resonances: List of drift resonance values
            
        Returns:
            List of detected harmonic patterns
        """
        patterns = []
        
        # Combine signals
        combined_signal = np.array(alignment_scores) + np.array(drift_resonances)
        
        # Apply FFT
        fft = np.fft.fft(combined_signal)
        freqs = np.fft.fftfreq(len(combined_signal))
        
        # Find significant frequencies
        significant_freqs = np.where(np.abs(fft) > self.freq_threshold * np.max(np.abs(fft)))[0]
        
        for freq_idx in significant_freqs:
            freq = freqs[freq_idx]
            if freq == 0:
                continue
                
            # Extract pattern parameters
            amplitude = np.abs(fft[freq_idx])
            phase = np.angle(fft[freq_idx])
            
            # Compute confidence
            confidence = min(amplitude / np.max(np.abs(fft)), 1.0)
            
            if confidence >= self.confidence_threshold:
                pattern = HarmonicPattern(
                    frequency=float(freq),
                    amplitude=float(amplitude),
                    phase=float(phase),
                    confidence=float(confidence),
                    start_time=timestamps[0],
                    pattern_type=self._classify_pattern_type(freq, amplitude, phase)
                )
                patterns.append(pattern)
                
        return patterns
        
    def _update_active_patterns(self, new_patterns: List[HarmonicPattern], current_time: float):
        """Update active pattern tracking"""
        # Close expired patterns
        for pattern in self.active_patterns:
            if pattern.end_time is None:
                pattern.end_time = current_time
                self.pattern_history.append(pattern)
                
        # Update active patterns
        self.active_patterns = new_patterns
        
    def _compute_overlay_score(self, patterns: List[HarmonicPattern]) -> float:
        """Compute overall overlay score"""
        if not patterns:
            return 0.0
            
        # Weight patterns by confidence and amplitude
        scores = [p.confidence * p.amplitude for p in patterns]
        return float(np.mean(scores))
        
    def _compute_stability_index(self, states: List[ZygotShellState]) -> float:
        """Compute stability index for shell states"""
        if not states:
            return 0.0
            
        # Use alignment and drift resonance
        alignments = [s.alignment_score for s in states]
        drifts = [s.drift_resonance for s in states]
        
        # Combine metrics
        stability = np.mean(alignments) * np.mean(drifts)
        return float(np.clip(stability, 0.0, 1.0))
        
    def _classify_pattern_type(self, freq: float, amplitude: float, phase: float) -> str:
        """Classify pattern type based on parameters"""
        if freq < 0.1:
            return "long_term"
        elif freq < 0.3:
            return "medium_term"
        else:
            return "short_term"
            
    def export_overlay_analysis(self, filepath: str):
        """Export overlay analysis to JSON"""
        if not self.overlay_history:
            return
            
        # Convert to JSON-serializable format
        analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "overlays": [{
                "timestamp": o.timestamp,
                "overlay_score": o.overlay_score,
                "stability_index": o.stability_index,
                "is_active": o.is_active,
                "patterns": [{
                    "frequency": p.frequency,
                    "amplitude": p.amplitude,
                    "phase": p.phase,
                    "confidence": p.confidence,
                    "pattern_type": p.pattern_type
                } for p in o.harmonic_patterns]
            } for o in self.overlay_history[-10:]]  # Last 10 overlays
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2) 