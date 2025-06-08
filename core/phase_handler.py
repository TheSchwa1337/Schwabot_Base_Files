"""
Phase Handler
============

Handles tensor phase alignment, phase drift detection, and signal phase correction logic.
Integrates with matrix fault resolution, shell resonance, and GAN drift diagnostics.

Core Features:
- Warp phase handling
- Plasma phase drift detection
- Profit-tier navigation alignment
- Klein bottle phase transitions
- Shell resonance tracking
"""

import numpy as np
from scipy.signal import hilbert
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PhaseMetrics:
    """Phase measurement metrics"""
    velocity: float
    curvature: float
    drift: float
    alignment: float
    lock_index: float
    warp_score: float
    plasma_energy: float

class PhaseHandler:
    def __init__(self, 
                 warp_threshold: float = 0.5,
                 plasma_threshold: float = 0.4,
                 profit_threshold: float = 0.95):
        self.warp_threshold = warp_threshold
        self.plasma_threshold = plasma_threshold
        self.profit_threshold = profit_threshold
        self.phase_history: List[np.ndarray] = []
        self.anchor_phases: List[np.ndarray] = []

    def compute_phase_velocity(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous phase velocity ω(t) = dφ(t)/dt
        Used for warp phase detection.
        """
        phase = np.unwrap(np.angle(hilbert(signal)))
        velocity = np.gradient(phase)
        return velocity

    def compute_phase_curvature(self, signal: np.ndarray) -> float:
        """
        Compute phase curvature for warp spectrum spread.
        S_warp = std(d²φ(t)/dt²)
        """
        phase = np.unwrap(np.angle(hilbert(signal)))
        curvature = np.gradient(np.gradient(phase))
        return np.std(curvature)

    def detect_warp_phase(self, signal: np.ndarray) -> Tuple[bool, float]:
        """
        Detect warp phase using velocity deviation and curvature.
        Returns (is_warped, warp_score)
        """
        velocity = self.compute_phase_velocity(signal)
        velocity_deviation = np.max(velocity) - np.min(velocity)
        curvature = self.compute_phase_curvature(signal)
        
        warp_score = 0.5 * (velocity_deviation / self.warp_threshold + 
                           curvature / self.warp_threshold)
        
        return warp_score > 1.0, warp_score

    def detect_plasma_drift(self, signal: np.ndarray) -> Tuple[float, float]:
        """
        Detect plasma phase drift using envelope and phase velocity.
        Returns (drift_score, plasma_energy)
        """
        # Compute Hilbert envelope
        envelope = np.abs(hilbert(signal))
        plasma_energy = np.sum(np.abs(np.gradient(envelope)))
        
        # Compute phase drift
        phase = np.unwrap(np.angle(hilbert(signal)))
        drift = np.std(np.gradient(phase))
        
        return drift, plasma_energy

    def align_phase_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute phase alignment score between vectors.
        P_align = ⟨φ_entry, φ_anchor⟩ / (||φ_entry|| ||φ_anchor||)
        """
        dot = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / (norm_product + 1e-8)

    def compute_phase_lock_index(self, phases: List[np.ndarray]) -> float:
        """
        Compute phase lock index across multiple signals.
        PLI = 1 - |∑(n=1 to N) e^(i(φ_n - φ̄))| / N
        """
        if not phases:
            return 0.0
            
        mean_phase = np.mean(phases, axis=0)
        sync_sum = np.sum([np.exp(1j*(p - mean_phase)) for p in phases])
        return 1 - np.abs(sync_sum) / len(phases)

    def resolve_phase_shift(self, signal: np.ndarray, 
                          reference: np.ndarray) -> np.ndarray:
        """
        Align signal phase to reference using cross-correlation.
        """
        if len(signal) != len(reference):
            raise ValueError("Signal and reference must be same length")
            
        correlation = np.correlate(reference, signal, mode='full')
        shift = correlation.argmax() - len(signal) + 1
        return np.roll(signal, shift)

    def compute_profit_band_duration(self, signal: np.ndarray,
                                   entry_idx: int,
                                   exit_idx: int) -> float:
        """
        Compute duration in resonant profit band.
        τ_profit = t_exit - t_entry
        """
        if entry_idx >= exit_idx:
            return 0.0
            
        phase = np.unwrap(np.angle(hilbert(signal)))
        band_phase = phase[entry_idx:exit_idx]
        
        # Check phase stability in band
        phase_std = np.std(band_phase)
        if phase_std > self.profit_threshold:
            return 0.0
            
        return exit_idx - entry_idx

    def update_anchor_phases(self, phase: np.ndarray,
                           profit_score: float) -> None:
        """
        Update anchor phases for profit-tier navigation.
        """
        if profit_score > self.profit_threshold:
            self.anchor_phases.append(phase)
            if len(self.anchor_phases) > 10:  # Keep last 10 profitable phases
                self.anchor_phases.pop(0)

    def get_phase_metrics(self, signal: np.ndarray) -> PhaseMetrics:
        """
        Compute comprehensive phase metrics for a signal.
        """
        velocity = self.compute_phase_velocity(signal)
        curvature = self.compute_phase_curvature(signal)
        drift, plasma_energy = self.detect_plasma_drift(signal)
        
        # Compute alignment with anchor phases
        alignment = 0.0
        if self.anchor_phases:
            alignments = [self.align_phase_vectors(signal, anchor) 
                        for anchor in self.anchor_phases]
            alignment = np.max(alignments)
            
        # Compute phase lock index
        self.phase_history.append(signal)
        if len(self.phase_history) > 10:
            self.phase_history.pop(0)
        lock_index = self.compute_phase_lock_index(self.phase_history)
        
        # Compute warp score
        _, warp_score = self.detect_warp_phase(signal)
        
        return PhaseMetrics(
            velocity=np.mean(velocity),
            curvature=curvature,
            drift=drift,
            alignment=alignment,
            lock_index=lock_index,
            warp_score=warp_score,
            plasma_energy=plasma_energy
        )

    def detect_phase_anomaly(self, signal: np.ndarray,
                            reference: np.ndarray) -> float:
        """
        Detect phase anomalies using GAN-style comparison.
        Returns anomaly score [0,1]
        """
        # Align phases
        aligned = self.resolve_phase_shift(signal, reference)
        
        # Compute phase differences
        phase_diff = np.unwrap(np.angle(hilbert(aligned))) - \
                    np.unwrap(np.angle(hilbert(reference)))
        
        # Normalize difference
        anomaly = np.mean(np.abs(phase_diff)) / np.pi
        return min(anomaly, 1.0)

    def compute_klein_bottle_phase(self, signal: np.ndarray) -> float:
        """
        Compute Klein bottle phase transition score.
        Higher score indicates non-orientable phase relationships.
        """
        phase = np.unwrap(np.angle(hilbert(signal)))
        gradient = np.gradient(phase)
        
        # Detect phase singularities
        singularities = np.where(np.abs(gradient) > np.pi)[0]
        if len(singularities) == 0:
            return 0.0
            
        # Compute transition score
        transition_score = len(singularities) / len(signal)
        return min(transition_score, 1.0)
