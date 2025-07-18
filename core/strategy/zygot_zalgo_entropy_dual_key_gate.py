#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zygot-Zalgo Entropy Dual Key Gate - Zygotic Hash Duality & Zalgo Field Access
=============================================================================

Implements the Nexus mathematics for recursive security lock + prediction logic:
- Dual Key Collapse Gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
- Hash Echo Mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
- Forms recursive security lock + prediction logic layer using 2-key entropy mirroring
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GateState(Enum):
    """Zygot-Zalgo gate states."""
    CLOSED = "closed"           # Gate is closed
    OPEN = "open"              # Gate is open
    PARTIAL = "partial"        # Partial access
    CONFLICTED = "conflicted"  # Key conflict detected
    RESET = "reset"            # Gate reset required


class KeyType(Enum):
    """Key types for dual-key system."""
    ZYGOT = "zygot"    # Recursive root entropy stabilizer
    ZALGO = "zalgo"    # Phase inversion entropy spike detector


@dataclass
class DualKeyResult:
    """Result of dual-key gate evaluation."""
    timestamp: float
    gate_state: GateState
    zygot_entropy: float
    zalgo_entropy: float
    combined_entropy: float
    access_granted: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HashEchoResult:
    """Result of hash echo mirroring."""
    timestamp: float
    volume_hash: str
    momentum_hash: str
    echo_hash: str
    echo_strength: float
    phase_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZygotZalgoEntropyDualKeyGate:
    """
    Zygot-Zalgo Entropy Dual Key Gate - Zygotic Hash Duality & Zalgo Field Access

    Implements the Nexus mathematics for recursive security lock + prediction logic:
    - Dual Key Collapse Gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
    - Hash Echo Mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)
    - Forms recursive security lock + prediction logic layer using 2-key entropy mirroring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Zygot-Zalgo Entropy Dual Key Gate."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        self.initialized = False

        # Gate parameters
        self.zygot_entropy_threshold = self.config.get('zygot_entropy_threshold', 0.3)
        self.zalgo_entropy_threshold = self.config.get('zalgo_entropy_threshold', 0.3)
        self.combined_threshold = self.config.get('combined_threshold', 0.5)
        self.adaptive_thresholding = self.config.get('adaptive_thresholding', True)

        # Key parameters
        self.alpha_coeff = self.config.get('alpha_coeff', 1.0)
        self.beta_coeff = self.config.get('beta_coeff', 1.0)
        self.r2_radius = self.config.get('r2_radius', 2.0)

        # Key storage
        self.zygot_key = self._generate_key()
        self.zalgo_key = self._generate_key()

        # State tracking
        self.gate_state = GateState.CLOSED
        self.evaluation_history: List[DualKeyResult] = []
        self.echo_history: List[HashEchoResult] = []

        self._initialize_gate()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Zygot-Zalgo Entropy Dual Key Gate."""
        return {
            'zygot_entropy_threshold': 0.3,  # Zygot entropy threshold
            'zalgo_entropy_threshold': 0.3,  # Zalgo entropy threshold
            'combined_threshold': 0.5,       # Combined entropy threshold
            'adaptive_thresholding': True,   # Adaptive thresholding
            'alpha_coeff': 1.0,             # Alpha coefficient for dual key
            'beta_coeff': 1.0,              # Beta coefficient for dual key
            'r2_radius': 2.0,               # R² radius for entropy calculation
            'echo_threshold': 0.7,          # Echo strength threshold
            'phase_threshold': 0.8,         # Phase alignment threshold
        }

    def _generate_key(self) -> str:
        """Generate a random key for the dual-key system."""
        return hashlib.sha256(f"{time.time()}{np.random.random()}".encode()).hexdigest()

    def _initialize_gate(self) -> None:
        """Initialize the dual-key gate."""
        try:
            self.logger.info("Initializing Zygot-Zalgo Entropy Dual Key Gate...")

            # Validate thresholds
            if not (0.0 <= self.zygot_entropy_threshold <= 1.0):
                raise ValueError("zygot_entropy_threshold must be between 0.0 and 1.0")
            if not (0.0 <= self.zalgo_entropy_threshold <= 1.0):
                raise ValueError("zalgo_entropy_threshold must be between 0.0 and 1.0")

            # Initialize SHA256 contexts
            self.zygot_sha256 = hashlib.sha256()
            self.zalgo_sha256 = hashlib.sha256()

            self.initialized = True
            self.logger.info(
                "[SUCCESS] Zygot-Zalgo Entropy Dual Key Gate initialized successfully")
        except Exception as e:
            self.logger.error(
                f"[FAIL] Error initializing Zygot-Zalgo Entropy Dual Key Gate: {e}")
            self.initialized = False

    def compute_dual_key_collapse_gate(
        self, data_vector: np.ndarray) -> float:
        """
        Compute dual key collapse gate: ZK_entropy = Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))

        Args:
        data_vector: Input data vector

        Returns:
        Dual key collapse gate entropy value
        """
        try:
            # Define R² region: R² = {ψ | κ(ψ) = 2}
            # For simplicity, we'll use a circular region with radius
            # r2_radius
            r2_radius = self.r2_radius

            # Generate ψ values in R² region
            theta_values = np.linspace(0, 2 * np.pi, 100)
            r_values = np.linspace(0, r2_radius, 50)

            # Initialize entropy sum
            zk_entropy = 0.0

            # Compute Σᵩ∈R²(α⋅cos(ψ) + β⋅sin(ψ))
            for r in r_values:
                for theta in theta_values:
                    # ψ = r * e^(i*theta) in complex form
                    psi_real = r * np.cos(theta)
                    psi_imag = r * np.imag(theta)

                    # Compute α⋅cos(ψ) + β⋅sin(ψ)
                    cos_component = self.alpha_coeff * np.cos(psi_real)
                    sin_component = self.beta_coeff * np.sin(psi_imag)

                    # Add to entropy sum
                    zk_entropy += cos_component + sin_component

            # Normalize by number of points
            total_points = len(r_values) * len(theta_values)
            zk_entropy /= total_points

            return zk_entropy
        except Exception as e:
            self.logger.error(
                f"Error computing dual key collapse gate: {e}")
            return 0.0

    def compute_hash_echo_mirror(self, volume_data: np.ndarray, momentum_data: np.ndarray) -> HashEchoResult:
        """
        Compute hash echo mirror: H_echo_zygot = SHA256(H_volume) ⊕ SHA256(H_momentum)

        Args:
        volume_data: Input volume data array
        momentum_data: Input momentum data array

        Returns:
        HashEchoResult with echo hash and related metrics
        """
        try:
            timestamp = time.time()
            # Hash volume and momentum data
            volume_hash = hashlib.sha256(volume_data.tobytes()).hexdigest()
            momentum_hash = hashlib.sha256(momentum_data.tobytes()).hexdigest()
            # XOR the hashes (as integers)
            echo_hash_int = int(volume_hash, 16) ^ int(momentum_hash, 16)
            echo_hash = f"{echo_hash_int:064x}"
            # Echo strength: normalized hamming weight
            echo_strength = bin(echo_hash_int).count('1') / 256.0
            # Phase alignment: normalized dot product
            phase_alignment = float(np.dot(volume_data, momentum_data) / (np.linalg.norm(volume_data) * np.linalg.norm(momentum_data) + 1e-8))
            return HashEchoResult(
                timestamp=timestamp,
                volume_hash=volume_hash,
                momentum_hash=momentum_hash,
                echo_hash=echo_hash,
                echo_strength=echo_strength,
                phase_alignment=phase_alignment,
                metadata={}
            )
        except Exception as e:
            self.logger.error(f"Error computing hash echo mirror: {e}")
            return HashEchoResult(
                timestamp=time.time(),
                volume_hash="",
                momentum_hash="",
                echo_hash="",
                echo_strength=0.0,
                phase_alignment=0.0,
                metadata={"error": str(e)}
            )

    def evaluate_dual_key_access(self, volume_data: np.ndarray, momentum_data: np.ndarray) -> DualKeyResult:
        """
        Evaluate dual key access using volume and momentum data.

        Args:
        volume_data: Input volume data array
        momentum_data: Input momentum data array

        Returns:
        DualKeyResult with access evaluation
        """
        try:
            if not self.initialized:
                raise RuntimeError("Gate not initialized")

            timestamp = time.time()

            # Compute dual key collapse gate
            zygot_entropy = self.compute_dual_key_collapse_gate(volume_data)
            zalgo_entropy = self.compute_dual_key_collapse_gate(momentum_data)

            # Compute combined entropy
            combined_entropy = (zygot_entropy + zalgo_entropy) / 2.0

            # Determine gate state
            if combined_entropy >= self.combined_threshold:
                gate_state = GateState.OPEN
                access_granted = True
            elif combined_entropy >= self.combined_threshold * 0.5:
                gate_state = GateState.PARTIAL
                access_granted = True
            else:
                gate_state = GateState.CLOSED
                access_granted = False

            # Calculate confidence
            confidence = min(1.0, combined_entropy / self.combined_threshold)

            # Create result
            result = DualKeyResult(
                timestamp=timestamp,
                gate_state=gate_state,
                zygot_entropy=zygot_entropy,
                zalgo_entropy=zalgo_entropy,
                combined_entropy=combined_entropy,
                access_granted=access_granted,
                confidence=confidence,
                metadata={
                    "volume_data_shape": volume_data.shape,
                    "momentum_data_shape": momentum_data.shape,
                }
            )

            # Store in history
            self.evaluation_history.append(result)

            # Keep history within limits
            if len(self.evaluation_history) > 100:
                self.evaluation_history.pop(0)

            return result

        except Exception as e:
            self.logger.error(f"Error evaluating dual key access: {e}")
            return DualKeyResult(
                timestamp=time.time(),
                gate_state=GateState.CLOSED,
                zygot_entropy=0.0,
                zalgo_entropy=0.0,
                combined_entropy=0.0,
                access_granted=False,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def get_status(self) -> Dict[str, Any]:
        """Get gate status."""
        return {
            "initialized": self.initialized,
            "gate_state": self.gate_state.value,
            "evaluation_history_size": len(self.evaluation_history),
            "echo_history_size": len(self.echo_history),
            "config": self.config,
        }


# Factory function
def create_zygot_zalgo_entropy_dual_key_gate(config: Optional[Dict[str, Any]] = None) -> ZygotZalgoEntropyDualKeyGate:
    """Create a Zygot-Zalgo Entropy Dual Key Gate instance."""
    return ZygotZalgoEntropyDualKeyGate(config) 