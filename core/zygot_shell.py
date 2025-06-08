"""
Zygot Shell Core
===============

Implements the Zygot-Zalgo alignment system and Drift Shell 144.44 harmonic framework.
Integrates with fractal core for recursive state tracking and matrix fault resolution.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import time
from datetime import datetime
import json
import os
from pathlib import Path
import uuid
import logging
import pywt
import hashlib
import torch
from .gan_filter import GANFilter, GANConfig

# Constants
DRIFT_SHELL_BOUNDARY = 144.44
ZYGOT_SUBHARMONIC_A = 36.11
ZYGOT_SUBHARMONIC_B = 72.22
PRE_COLLAPSE_ZONE = 108.33
GOLDEN_ECHO_PHASE = 1.618

@dataclass
class ZygotShellState:
    """
    Represents a Zygot shell state with drift, alignment metrics,
    and Euler-based trigger parameters.
    """
    vector: np.ndarray
    timestamp: float
    shell_radius: float
    zalgo_field: Optional[np.ndarray] = None
    alignment_score: float = 0.0
    drift_resonance: float = 0.0
    phase_angle: float = 0.0
    entropy: float = 0.0
    shell_type: str = "stable"  # stable, resonant, collapse
    uuid: uuid.UUID = field(default_factory=uuid.uuid4)
    symbolic_anchor: Optional[str] = None
    post_euler_field: Optional[np.ndarray] = None
    entropy_type: str = "wavelet"  # wavelet, shannon, tsallis
    is_anomaly: bool = False
    anomaly_confidence: float = 0.0

    def __repr__(self):
        return (f"ZygotShellState(vector={self.vector}, timestamp={self.timestamp}, "
                f"shell_radius={self.shell_radius}, alignment_score={self.alignment_score}, "
                f"drift_resonance={self.drift_resonance}, phase_angle={self.phase_angle}, "
                f"entropy={self.entropy}, shell_type={self.shell_type}, uuid={self.uuid}, "
                f"symbolic_anchor={self.symbolic_anchor}, post_euler_field={self.post_euler_field}, "
                f"entropy_type={self.entropy_type}, is_anomaly={self.is_anomaly}, "
                f"anomaly_confidence={self.anomaly_confidence})")

@dataclass
class ZalgoField:
    """Represents a Zalgo entropy field state"""
    field_vector: np.ndarray
    entropy_level: float
    collapse_risk: float
    ghost_signals: List[float]
    timestamp: float

class ZygotShell:
    """Core implementation of the Zygot-Zalgo alignment system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shell_states: List[ZygotShellState] = []
        self.symbolic_ledger = {}
        self.ledger_sync_interval = config.get('ledger_sync_interval', 100)
        self.state_counter = 0
        
        # Initialize GAN filter
        gan_config = config.get('gan_config', {})
        self.gan_filter = GANFilter(gan_config, input_dim=config.get('vector_dim', 32))
        
        # Load pre-trained model if available
        model_path = config.get('gan_model_path')
        if model_path and os.path.exists(model_path):
            self.gan_filter.load_model(model_path)
            
        self.decay_power = 2.0
        self.terms = 64
        self.dimension = 3
        self.state_history: List[ZygotShellState] = []
        self.zalgo_fields: List[ZalgoField] = []
        self.last_collapse_time: Optional[float] = None
        self.collapse_threshold = 0.85
        self.harmonic_boot_frequency = 0.125  # Hz
        
        # Euler-based trigger parameters
        self.euler_phase = 0.0
        self.euler_identity = np.exp(1j * np.pi) + 1
        self.post_euler_field = None
        
        # Initialize shell parameters
        self.shell_radius = DRIFT_SHELL_BOUNDARY
        self.alignment_threshold = 0.7
        self.drift_resonance_threshold = 0.5
        
    def compute_stability_index(self, volume_signal: float, volatility_map: float, config: Optional[Dict] = None) -> float:
        """
        Compute stability index based on volume and volatility signals.
        
        Args:
            volume_signal: Volume signal value
            volatility_map: Volatility map value
            config: Optional configuration parameters
            
        Returns:
            Stability index [0,1]
        """
        if config is None:
            config = {}
            
        # Extract parameters with defaults
        alpha_v = config.get('alpha_v', 15.8)
        alpha_s = config.get('alpha_s', 18.3)
        alpha_c = config.get('alpha_c', 0.714)
        
        # Compute stability components
        volume_stability = math.exp(-alpha_v * abs(volume_signal - 1.0))
        volatility_stability = math.exp(-alpha_s * abs(volatility_map - 1.0))
        
        # Combine into stability index
        stability = (volume_stability * volatility_stability) ** alpha_c
        
        return max(0.0, min(1.0, stability))
        
    def compute_unified_entropy(self, vector: np.ndarray, entropy_type: str = "wavelet") -> float:
        """
        Compute unified entropy using specified method.
        
        Args:
            vector: Input vector
            entropy_type: Type of entropy calculation (wavelet, shannon, tsallis)
            
        Returns:
            Entropy value [0,1]
        """
        if entropy_type == "wavelet":
            # Use wavelet decomposition
            coeffs = pywt.wavedec(vector, 'haar')
            total_energy = sum(np.sum(np.abs(coeff) ** 2) for coeff in coeffs)
            return float(np.clip(-np.log(total_energy + 1e-6), 0.0, self.config.get('entropy_clip', 1.0)))
        
        elif entropy_type == "shannon":
            # Use Shannon entropy
            hist, _ = np.histogram(vector, bins=50, density=True)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist)))
        
        elif entropy_type == "tsallis":
            # Use Tsallis entropy with q=2
            hist, _ = np.histogram(vector, bins=50, density=True)
            hist = hist[hist > 0]
            return float((1 - np.sum(hist**2)) / (2-1))
        
        else:
            raise ValueError(f"Unknown entropy type: {entropy_type}")
        
    def compute_drift_resonance(self, phase_angle: float, entropy: float) -> float:
        """
        Compute drift resonance based on phase angle and entropy.
        
        Args:
            phase_angle: Current phase angle
            entropy: Current entropy value
            
        Returns:
            Drift resonance score [0,1]
        """
        # Compute phase drift from π
        phase_drift = abs(phase_angle - math.pi)
        
        # Use unified entropy calculation
        entropy_drift = abs(entropy - 1.0)
        
        # Combine into resonance score with improved weighting
        resonance = 1.0 - (
            0.6 * phase_drift / math.pi +
            0.4 * entropy_drift
        )
        
        return max(0.0, min(1.0, resonance))
        
    def classify_shell_state(self, drift_resonance: float, alignment_score: float) -> str:
        """
        Classify shell state based on drift resonance and alignment.
        
        Args:
            drift_resonance: Current drift resonance
            alignment_score: Current alignment score
            
        Returns:
            Shell state classification
        """
        if drift_resonance >= self.drift_resonance_threshold and alignment_score >= self.alignment_threshold:
            return "stable"
        elif drift_resonance >= 0.3 and alignment_score >= 0.3:
            return "resonant"
        else:
            return "collapse"
            
    def compute_zalgo_field(self, vector: np.ndarray, entropy: float) -> ZalgoField:
        """
        Compute Zalgo entropy field for a given vector.
        
        Args:
            vector: Input vector
            entropy: Current entropy value
            
        Returns:
            ZalgoField object
        """
        # Compute field vector using Euler-based transformation
        field_vector = np.exp(1j * (math.pi + self.euler_phase)) * vector
        
        # Compute collapse risk
        collapse_risk = self._calculate_collapse_risk(vector)
        
        # Generate ghost signals
        ghost_signals = self._generate_ghost_signals(vector, entropy)
        
        return ZalgoField(
            field_vector=field_vector,
            entropy_level=entropy,
            collapse_risk=collapse_risk,
            ghost_signals=ghost_signals,
            timestamp=time.time()
        )
        
    def _calculate_collapse_risk(self, vector: np.ndarray) -> float:
        """Calculate risk of shell collapse"""
        if not self.state_history:
            return 0.0
            
        # Get latest state
        latest_state = self.state_history[-1]
        
        # Calculate vector coherence
        coherence = np.correlate(vector, latest_state.vector, mode='valid')[0]
        
        # Calculate entropy stability
        entropy_stability = 1.0 - abs(latest_state.entropy - 1.0)
        
        # Combine metrics
        collapse_risk = (1.0 - coherence) * 0.7 + (1.0 - entropy_stability) * 0.3
        
        return min(max(collapse_risk, 0.0), 1.0)
        
    def _generate_ghost_signals(self, vector: np.ndarray, entropy: float) -> List[float]:
        """Generate ghost signals for entropy injection"""
        # Use golden ratio for signal spacing
        signals = []
        for i in range(3):
            phase = i * GOLDEN_ECHO_PHASE
            signal = np.sin(2 * math.pi * phase) * entropy
            signals.append(float(signal))
        return signals
        
    def sync_symbolic_ledger(self, state: ZygotShellState) -> None:
        """
        Sync symbolic anchor to ledger with metadata.
        
        Args:
            state: Current shell state
        """
        if not state.symbolic_anchor:
            return
            
        # Create ledger entry
        entry = {
            'timestamp': state.timestamp,
            'uuid': str(state.uuid),
            'alignment_score': state.alignment_score,
            'drift_resonance': state.drift_resonance,
            'entropy': state.entropy,
            'shell_type': state.shell_type,
            'vector_hash': hashlib.sha256(state.vector.tobytes()).hexdigest()
        }
        
        # Add to ledger
        if state.symbolic_anchor not in self.symbolic_ledger:
            self.symbolic_ledger[state.symbolic_anchor] = []
        self.symbolic_ledger[state.symbolic_anchor].append(entry)
        
        # Trim old entries if needed
        if len(self.symbolic_ledger[state.symbolic_anchor]) > 1000:
            self.symbolic_ledger[state.symbolic_anchor] = self.symbolic_ledger[state.symbolic_anchor][-1000:]
            
    def get_symbolic_anchor_history(self, anchor: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history for a symbolic anchor.
        
        Args:
            anchor: Symbolic anchor to query
            limit: Optional limit on number of entries
            
        Returns:
            List of ledger entries
        """
        if anchor not in self.symbolic_ledger:
            return []
            
        entries = self.symbolic_ledger[anchor]
        if limit:
            return entries[-limit:]
        return entries
        
    def process_shell_state(self, state: ZygotShellState) -> ZygotShellState:
        """
        Process a shell state, including anomaly detection.
        
        Args:
            state: Input shell state
            
        Returns:
            Processed shell state
        """
        # Convert state vector to tensor
        state_tensor = torch.FloatTensor(state.vector).unsqueeze(0)
        
        # Detect anomalies
        is_anomaly, confidence = self.gan_filter.detect_anomaly(state_tensor)
        
        # Update state with anomaly information
        state.is_anomaly = is_anomaly
        state.anomaly_confidence = confidence
        
        # Process state as normal
        processed_state = self.process_shell_state(state)
        
        # Update GAN if needed
        if self.config.get('gan_online_training', False):
            self._update_gan(processed_state)
            
        return processed_state
        
    def _update_gan(self, state: ZygotShellState) -> None:
        """
        Update GAN model with new state.
        
        Args:
            state: New shell state
        """
        # Convert state to training format
        state_tensor = torch.FloatTensor(state.vector).unsqueeze(0)
        
        # Update GAN with single sample
        self.gan_filter.train_step(state_tensor, self.compute_unified_entropy)
        
        # Save model periodically
        if self.state_counter % self.config.get('gan_save_interval', 1000) == 0:
            self.gan_filter.save_model()
            
    def train_gan(self, 
                 train_data: List[ZygotShellState],
                 validation_data: Optional[List[ZygotShellState]] = None) -> None:
        """
        Train GAN model on historical shell states.
        
        Args:
            train_data: List of training shell states
            validation_data: Optional list of validation shell states
        """
        # Convert states to training format
        train_vectors = np.array([state.vector for state in train_data])
        if validation_data:
            val_vectors = np.array([state.vector for state in validation_data])
        else:
            val_vectors = None
            
        # Train GAN
        self.gan_filter.train(
            train_vectors,
            self.compute_unified_entropy,
            validation_data=val_vectors
        )
        
        # Save trained model
        self.gan_filter.save_model()
        
    def get_recent_states(self, count: int = 3) -> List[ZygotShellState]:
        """Get most recent shell states"""
        return self.state_history[-count:]
        
    def export_shell_map(self, filepath: str):
        """Export shell state map to JSON"""
        if not self.state_history:
            return
            
        # Get last 5 states
        recent_states = self.state_history[-5:]
        
        # Convert to JSON-serializable format
        shell_map = {
            "timestamp": datetime.utcnow().isoformat(),
            "states": [{
                "timestamp": state.timestamp,
                "shell_type": state.shell_type,
                "alignment_score": state.alignment_score,
                "drift_resonance": state.drift_resonance,
                "entropy": state.entropy,
                "phase_angle": state.phase_angle,
                "symbolic_anchor": state.symbolic_anchor
            } for state in recent_states]
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(shell_map, f, indent=2)
            
    def plot_shell_evolution(self, timestamps: List[float], alignment_scores: List[float], 
                           drift_resonances: List[float], label: str = "Shell Evolution"):
        """Plot shell evolution metrics"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, alignment_scores, label="Alignment")
        plt.plot(timestamps, drift_resonances, label="Drift Resonance")
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.title(label)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class ZygotControlHooks:
    @staticmethod
    def inject_ghost_hash(shell: ZygotShellState, ghost_id: str) -> ZygotShellState:
        shell.uuid = uuid.uuid5(uuid.NAMESPACE_DNS, ghost_id)
        return shell

    @staticmethod
    def symbolic_encode(shell: ZygotShellState) -> str:
        base = int(shell.alignment_score * 1000)
        return f"e1a7-{base:04x}-{str(shell.uuid)[:8]}"

    @staticmethod
    def trigger_safety_warning(message: str):
        logging.warning(f"[ZygotSafetyWarning] {message}") 