"""
Zygote Propagation Engine
=======================

Implements the propagation layer between Zygot shell genesis and trade logic activation.
Integrates with quantum cellular risk monitor, phase handler, and GAN filter for comprehensive
pattern recognition and strategy mapping.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch
from collections import deque

from .quantum_cellular_risk_monitor import QuantumRiskMetrics
from .phase_handler import PhaseHandler
from .gan_filter import GANFilter
from .zygot_shell import ZygotShellState
from .strategy_mapper import StrategyMapper
from utils.logger_bridge import log_propagation_event  # Optional: For JSON trail logging

logger = logging.getLogger(__name__)

# Constants
ARCHETYPE_LIBRARY = {
    "expansion_long": {
        "vector": np.array([0.9, 0.8, 0.1]),
        "phase_signature": np.array([0.1, 0.9, 0.2, 0.1]),
        "plasma_tolerance": 0.3,
        "warp_threshold": 0.6
    },
    "distribution_exit": {
        "vector": np.array([0.2, 0.3, 0.9]),
        "phase_signature": np.array([0.9, 0.1, 0.9, 0.1]),
        "plasma_tolerance": 0.5,
        "warp_threshold": 0.8
    },
    "accumulation_soft": {
        "vector": np.array([0.8, 0.9, 0.4]),
        "phase_signature": np.array([0.5, 0.5, 0.5, 0.5]),
        "plasma_tolerance": 0.4,
        "warp_threshold": 0.7
    },
    "neutral_hold": {
        "vector": np.array([0.5, 0.5, 0.5]),
        "phase_signature": np.array([0.5, 0.5, 0.5, 0.5]),
        "plasma_tolerance": 0.6,
        "warp_threshold": 0.9
    },
    "recursive_fold": {
        "vector": np.array([0.3, 0.9, 0.3]),
        "phase_signature": np.array([0.9, 0.1, 0.9, 0.1]),
        "plasma_tolerance": 0.1,
        "warp_threshold": 0.2
    }
}

@dataclass
class PropagationTrace:
    """Records propagation events for analysis and learning"""
    uuid: str
    timestamp: float
    propagation: Dict[str, Any]
    phase: Dict[str, float]
    plasma: Dict[str, float]
    klein_bottle_state: Optional[Dict[str, Any]] = None

class ZygotePropagationEngine:
    """
    Core propagation engine that translates Zygot shell genesis events into trade directives.
    Implements phase-aware archetype matching, temporal memory integration, and GAN anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phase_handler = PhaseHandler(config)
        self.gan_filter = GANFilter(config.get('gan_config', {}), input_dim=3)
        
        # Initialize trace history
        self.trace_history: List[PropagationTrace] = []
        self.shell_history: List[Dict] = []
        
        # Load pre-trained GAN model if available
        model_path = config.get('gan_model_path')
        if model_path and os.path.exists(model_path):
            self.gan_filter.load_model(model_path)
            
        # Initialize archetype library
        self.archetypes = ARCHETYPE_LIBRARY
        
        # Temporal memory
        self.memory_window = config.get('memory_window', 100)
        self.temporal_coherence_threshold = config.get('temporal_coherence_threshold', 0.6)
        
        # Phase tracking
        self.phase_velocity_history = deque(maxlen=self.memory_window)
        self.phase_stability_history = deque(maxlen=self.memory_window)
        
    def match_archetype_with_phase(
        self, 
        shell_vector: np.ndarray,
        phase_state: np.ndarray
    ) -> Tuple[str, float]:
        """
        Match shell vector to archetype with phase awareness.
        
        Args:
            shell_vector: Current shell vector
            phase_state: Current phase state
            
        Returns:
            Tuple of (matched_archetype, confidence)
        """
        best_match = None
        best_angle = float("inf")
        best_confidence = 0.0
        
        # Calculate phase velocity and stability
        phase_velocity = self.phase_handler.compute_phase_velocity(phase_state)
        phase_stability = np.std(phase_velocity)
        
        # Update phase history
        self.phase_velocity_history.append(phase_velocity)
        self.phase_stability_history.append(phase_stability)
        
        # Match against archetypes
        for name, archetype in self.archetypes.items():
            # Vector similarity
            vector_angle = np.arccos(
                np.dot(shell_vector, archetype["vector"]) / 
                (np.linalg.norm(shell_vector) * np.linalg.norm(archetype["vector"]) + 1e-8)
            )
            
            # Phase signature similarity
            phase_angle = np.arccos(
                np.dot(phase_state, archetype["phase_signature"]) /
                (np.linalg.norm(phase_state) * np.linalg.norm(archetype["phase_signature"]) + 1e-8)
            )
            
            # Combined angle (weighted)
            combined_angle = 0.7 * vector_angle + 0.3 * phase_angle
            
            if combined_angle < best_angle:
                best_match = name
                best_angle = combined_angle
                best_confidence = np.exp(-combined_angle)
                
                # Adjust confidence based on phase stability
                if phase_stability > 0.5:  # High phase drift
                    best_confidence *= 0.7
                    
        return best_match, best_confidence
        
    def _compute_temporal_coherence(
        self,
        shell_vector: np.ndarray,
        recent_shells: List[Dict]
    ) -> float:
        """
        Compute temporal coherence with recent shell history.
        
        Args:
            shell_vector: Current shell vector
            recent_shells: List of recent shell states
            
        Returns:
            Temporal coherence score [0,1]
        """
        if not recent_shells:
            return 1.0
            
        # Calculate vector similarities
        similarities = []
        for shell in recent_shells:
            if 'vector' in shell:
                similarity = np.dot(shell_vector, shell['vector']) / (
                    np.linalg.norm(shell_vector) * np.linalg.norm(shell['vector']) + 1e-8
                )
                similarities.append(similarity)
                
        # Weight by recency
        weights = np.exp(-np.arange(len(similarities)) / 5)
        weighted_similarity = np.average(similarities, weights=weights)
        
        return float(weighted_similarity)
        
    def _detect_quantum_anomaly(
        self,
        shell_vector: np.ndarray,
        best_angle: float
    ) -> Optional[Dict[str, Any]]:
        """
        Detect quantum anomalies in shell vectors.
        
        Args:
            shell_vector: Current shell vector
            best_angle: Best matching angle from archetype matching
            
        Returns:
            Anomaly detection result or None
        """
        if best_angle > np.pi/4:  # > 45 degrees from any archetype
            # Compute spectral anomaly score
            spectrum = np.fft.fft(shell_vector)
            spectral_entropy = -np.sum(np.abs(spectrum) * np.log(np.abs(spectrum) + 1e-8))
            
            if spectral_entropy > 2.0:
                return {
                    "type": "paradox_fractal",
                    "action": "isolate_and_analyze",
                    "spawn_new_archetype": True,
                    "spectral_entropy": float(spectral_entropy)
                }
                
        return None
        
    def _compute_klein_state(self, propagation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compute Klein bottle state for propagation result.
        
        Args:
            propagation_result: Current propagation result
            
        Returns:
            Klein bottle state or None
        """
        if propagation_result.get("matched_archetype") == "recursive_fold":
            return {
                "state": "klein_bottle",
                "confidence": propagation_result.get("confidence", 0.0),
                "phase_alignment": propagation_result.get("phase_alignment", 0.0)
            }
        return None
        
    def propagate(
        self,
        shell_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main propagation method that processes shell data and generates trade directives.
        
        Args:
            shell_data: Dictionary containing shell information
            
        Returns:
            Propagation directive
        """
        # Extract components
        shell_vector = shell_data["vector"]
        shell_uuid = shell_data["uuid"]
        profit_band = shell_data.get("profit_band", "NEUTRAL")
        phase_state = shell_data.get("phase_state", shell_vector)
        
        # Phase analysis
        phase_metrics = self.phase_handler.analyze_phase(phase_state)
        
        # Plasma detection
        plasma_metrics = self._detect_plasma(shell_vector)
        
        # GAN anomaly check
        anomaly_result = self.gan_filter.detect(shell_vector)
        
        # Archetype matching with phase awareness
        matched_archetype, confidence = self.match_archetype_with_phase(
            shell_vector, phase_state
        )
        
        # Check for quantum anomalies
        anomaly = self._detect_quantum_anomaly(shell_vector, confidence)
        
        # Generate propagation directive
        directive = {
            "uuid": shell_uuid,
            "matched_archetype": matched_archetype,
            "confidence": confidence,
            "profit_band": profit_band,
            "phase_metrics": phase_metrics,
            "plasma_metrics": plasma_metrics,
            "anomaly": anomaly,
            "timestamp": datetime.now().timestamp()
        }
        
        # Record trace
        trace = PropagationTrace(
            uuid=shell_uuid,
            timestamp=directive["timestamp"],
            propagation=directive,
            phase=phase_metrics,
            plasma=plasma_metrics,
            klein_bottle_state=self._compute_klein_state(directive)
        )
        self.trace_history.append(trace)
        
        # Update shell history
        self.shell_history.append(shell_data)
        if len(self.shell_history) > self.memory_window:
            self.shell_history.pop(0)
            
        return directive
        
    def _detect_plasma(self, shell_vector: np.ndarray) -> Dict[str, float]:
        """
        Detect plasma state in shell vector.
        
        Args:
            shell_vector: Current shell vector
            
        Returns:
            Plasma metrics
        """
        # Compute energy
        energy = np.sum(shell_vector ** 2)
        
        # Compute turbulence
        turbulence = np.std(np.diff(shell_vector))
        
        return {
            "energy": float(energy),
            "turbulence": float(turbulence)
        }
        
    def export_traces(self, filepath: str):
        """
        Export propagation traces to JSON file.
        
        Args:
            filepath: Path to export file
        """
        if not self.trace_history:
            return
            
        # Convert traces to JSON-serializable format
        traces = [{
            "uuid": trace.uuid,
            "timestamp": trace.timestamp,
            "propagation": trace.propagation,
            "phase": trace.phase,
            "plasma": trace.plasma,
            "klein_bottle_state": trace.klein_bottle_state
        } for trace in self.trace_history]
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(traces, f, indent=2)
            
    def get_propagation_summary(self) -> Dict[str, Any]:
        """
        Get summary of propagation activity.
        
        Returns:
            Propagation summary
        """
        return {
            "total_traces": len(self.trace_history),
            "recent_archetypes": [
                trace.propagation["matched_archetype"]
                for trace in self.trace_history[-5:]
            ],
            "average_confidence": np.mean([
                trace.propagation["confidence"]
                for trace in self.trace_history
            ]) if self.trace_history else 0.0,
            "anomaly_count": sum(
                1 for trace in self.trace_history
                if trace.propagation.get("anomaly") is not None
            )
        }

def evaluate_shell_propagation(shell_event: dict, runtime_context: dict = None):
    """
    Evaluates and propagates a shell event using the ZygotePropagationEngine.

    Args:
        shell_event (dict): The shell state emitted from a genesis trigger or shell vector sync.
        runtime_context (dict): Optional system context (tick count, current price, etc.)
    """
    zygote_engine = ZygotePropagationEngine(config_path='configs/propagation.yaml')
    strategy_mapper = StrategyMapper()

    result = zygote_engine.propagate(shell_event)

    if result['approved']:
        action_packet = result['action_packet']
        strategy_mapper.execute(action_packet)

        # Optional audit trail
        propagation_log = {
            "uuid": shell_event.get("uuid"),
            "strategy": action_packet.get("strategy_id"),
            "action": action_packet.get("action"),
            "confidence": result.get("confidence", 0.0),
            "timestamp": result.get("timestamp"),
            "matched_archetype": result.get("matched_archetype"),
            "context": runtime_context or {}
        }
        log_propagation_event(propagation_log)

# Call evaluate_shell_propagation(shell_event) after each _trigger_zygot_genesis or new shell vector lock. 