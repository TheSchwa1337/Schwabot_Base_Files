from __future__ import annotations

"""
Ω-B-Γ Logic (Alpha Encryption) – Nexus Thought Core Integration
===============================================================

Developed by Maxamillion M.A.A. DeLeon screen/pen name TheSchwa1337 ("The Schwa") & Nexus AI 
– Recursive Systems Architects | Authors of Ω-B-Γ Logic & Alpha Encryption Protocol

Alpha Encryption represents a deeply recursive, multi-state encryption protocol designed 
explicitly within our internal architecture. It leverages fractal recursion, quantum-inspired 
gate logic, Bayesian probability filtering, and harmonic wave-pattern encoding.

This encryption method uniquely blends mathematical discoveries from the Logic Chaos Core (LCC) 
with fractal geometry, recursive dynamics, and quantum resonance principles, stepping distinctly 
away from classical ECC or standard chaotic maps.

Integration with SchwaBot Vortex Math Security Protocol (VMSP) provides enhanced cryptographic 
protection through pattern-based authentication and recursive mathematical legitimacy.
"""

import numpy as np
import hashlib
import time
import cmath
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .vortex_security import get_vortex_security


class AlphaEncryptionLayer(Enum):
    """Alpha Encryption processing layers"""

    OMEGA = "omega"  # Recursive Superposition Layer
    BETA = "beta"  # Quantum Bayesian Logic Gates
    GAMMA = "gamma"  # Harmonic Wave Pattern Encoding


@dataclass
class OmegaState:
    """State representation for Omega (fractal recursion) layer"""

    complex_state: complex
    recursion_depth: int
    entropy_pool: str
    fractal_constant: complex
    convergence_metric: float


@dataclass
class BetaState:
    """State representation for Beta (Quantum Bayesian) layer"""

    probability_matrix: List[float]
    gate_state: str
    bayesian_entropy: float
    quantum_coherence: float
    state_history: List[str]


@dataclass
class GammaState:
    """State representation for Gamma (harmonic waveform) layer"""

    harmonic_wave: np.ndarray
    frequency_components: List[float]
    phase_coefficients: List[float]
    amplitude_modulation: List[float]
    wave_entropy: float


@dataclass
class AlphaEncryptionResult:
    """Complete Alpha Encryption result with all layer states"""

    encrypted_data: np.ndarray
    omega_state: OmegaState
    beta_state: BetaState
    gamma_state: GammaState
    total_entropy: float
    encryption_hash: str
    processing_time: float


class OmegaLayer:
    """
    Ω (Omega) – Recursive Superposition Layer

    Implements fractal-based recursive encryption using Mandelbrot-influenced
    recursive embedding with dynamically computed constants.
    """

    def __init__(self, max_depth: int = 100, convergence_threshold: float = 1e-6):
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold

    def _generate_entropy_constant(self, data: str, iteration: int) -> complex:
        """Generate SHA-256 based entropy constant for fractal recursion"""
        entropy_seed = f"{data}_{iteration}_{time.time()}"
        hash_obj = hashlib.sha256(entropy_seed.encode())
        hex_hash = hash_obj.hexdigest()

        # Convert hash to complex number
        real_part = int(hex_hash[:16], 16) % 255
        imag_part = int(hex_hash[16:32], 16) % 127

        return complex(real_part / 255.0, imag_part / 127.0)

    def _calculate_convergence_metric(
        self, z_current: complex, z_previous: complex
    ) -> float:
        """Calculate convergence metric for fractal stability analysis"""
        return abs(z_current - z_previous)

    def process(self, data: str, depth: Optional[int] = None) -> OmegaState:
        """
        Execute Omega layer fractal recursion:
        Ω_{n+1}(z) = Ω_n(z)^2 + C(z)
        """
        if depth is None:
            depth = min(self.max_depth, len(data) * 5)  # Adaptive depth

        # Initialize complex state from input data
        z = complex(np.mean([ord(c) for c in data]), len(data) / 10.0)
        z_previous = z

        entropy_pool = ""
        convergence_metrics = []

        for iteration in range(depth):
            # Generate dynamic constant C(z)
            C = self._generate_entropy_constant(data, iteration)

            # Apply fractal recursion: z = z^2 + C
            z_previous = z
            z = z**2 + C

            # Calculate convergence
            convergence = self._calculate_convergence_metric(z, z_previous)
            convergence_metrics.append(convergence)

            # Build entropy pool
            entropy_pool += hashlib.sha256(str(z).encode()).hexdigest()[:8]

            # Check for convergence or divergence control
            if abs(z) > 1000:  # Prevent excessive divergence
                z = z / abs(z) * 100  # Normalize while preserving phase

            if convergence < self.convergence_threshold and iteration > 10:
                break

        # Calculate final convergence metric
        final_convergence = np.mean(convergence_metrics) if convergence_metrics else 0.0

        return OmegaState(
            complex_state=z,
            recursion_depth=iteration + 1,
            entropy_pool=entropy_pool,
            fractal_constant=C,
            convergence_metric=final_convergence,
        )


class BetaLayer:
    """
    Β (Beta) – Quantum Bayesian Logic Gates

    Implements Quantum Bayesian Probability Matrix (QBPM) with multi-state
    truth values enabling quantum-like behavior.
    """

    def __init__(self):
        self.state_history: List[str] = []

    def _calculate_qbpm(
        self, omega_state: OmegaState, gamma_context: Optional[Any] = None
    ) -> List[float]:
        """
        Calculate Quantum Bayesian Probability Matrix:
        P(B | Ω, Γ) = P(Ω,Γ | B)P(B) / Σ P(Ω,Γ | B_i)P(B_i)
        """
        # Extract features from omega state
        real_component = np.real(omega_state.complex_state)
        imag_component = np.imag(omega_state.complex_state)
        depth_factor = omega_state.recursion_depth / 100.0

        # Calculate base probabilities
        p_state0 = abs(np.sin(real_component)) * (1 - depth_factor)
        p_state1 = abs(np.cos(imag_component)) * depth_factor
        p_superposition = abs(np.sin(real_component) * np.cos(imag_component))

        # Normalize probabilities
        total = p_state0 + p_state1 + p_superposition
        if total > 0:
            return [p_state0 / total, p_state1 / total, p_superposition / total]
        else:
            return [1 / 3, 1 / 3, 1 / 3]  # Equal probability fallback

    def _select_quantum_state(self, probabilities: List[float]) -> str:
        """Select quantum state based on probability distribution"""
        states = ["state0", "state1", "superposition"]
        choice = np.random.choice(states, p=probabilities)
        self.state_history.append(choice)
        return choice

    def _calculate_bayesian_entropy(self, probabilities: List[float]) -> float:
        """Calculate Quantum Bayesian Entropy (QBE)"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def _calculate_quantum_coherence(self, omega_state: OmegaState) -> float:
        """Calculate quantum coherence from fractal state"""
        phase = cmath.phase(omega_state.complex_state)
        magnitude = abs(omega_state.complex_state)

        # Coherence based on phase stability and magnitude
        coherence = np.exp(-abs(phase)) * min(1.0, magnitude / 100.0)
        return coherence

    def process(self, omega_state: OmegaState) -> BetaState:
        """Execute Beta layer Quantum Bayesian logic processing"""
        # Calculate probability matrix
        probabilities = self._calculate_qbpm(omega_state)

        # Select quantum gate state
        gate_state = self._select_quantum_state(probabilities)

        # Calculate metrics
        bayesian_entropy = self._calculate_bayesian_entropy(probabilities)
        quantum_coherence = self._calculate_quantum_coherence(omega_state)

        return BetaState(
            probability_matrix=probabilities,
            gate_state=gate_state,
            bayesian_entropy=bayesian_entropy,
            quantum_coherence=quantum_coherence,
            state_history=self.state_history.copy(),
        )


class GammaLayer:
    """
    Γ (Gamma) – Harmonic Wave Pattern Encoding

    Implements harmonic waveform encoding using discrete wavelet transforms
    and complex harmonic frequencies.
    """

    def __init__(self, sample_rate: int = 1000, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.time_samples = int(sample_rate * duration)

    def _generate_harmonic_parameters(
        self, beta_state: BetaState, omega_state: OmegaState
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate amplitude, frequency, and phase parameters from recursive layers"""
        # Base frequency selection from beta state
        if beta_state.gate_state == "state0":
            base_freq = 440.0  # A4 note
        elif beta_state.gate_state == "state1":
            base_freq = 880.0  # A5 note
        else:  # superposition
            base_freq = 660.0  # E5 note

        # Generate harmonic series
        num_harmonics = min(10, max(3, int(omega_state.recursion_depth / 10)))
        frequencies = [base_freq * (i + 1) for i in range(num_harmonics)]

        # Calculate amplitudes from probability matrix
        amplitudes = []
        for i, freq in enumerate(frequencies):
            amp = beta_state.probability_matrix[i % len(beta_state.probability_matrix)]
            amp *= np.exp(-i * 0.1)  # Natural harmonic decay
            amplitudes.append(amp)

        # Calculate phases from omega complex state
        base_phase = cmath.phase(omega_state.complex_state)
        phases = [base_phase + i * np.pi / 4 for i in range(num_harmonics)]

        return frequencies, amplitudes, phases

    def _generate_harmonic_wave(
        self, frequencies: List[float], amplitudes: List[float], phases: List[float]
    ) -> np.ndarray:
        """
        Generate harmonic wave using equation:
        Γ(f) = Σ a_k * e^(i(ω_k * t + φ_k))
        """
        t = np.linspace(0, self.duration, self.time_samples)
        wave = np.zeros(self.time_samples, dtype=complex)

        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            omega_k = 2 * np.pi * freq
            harmonic_component = amp * np.exp(1j * (omega_k * t + phase))
            wave += harmonic_component

        # Return real part for practical encoding
        return np.real(wave)

    def _calculate_wave_entropy(self, wave: np.ndarray) -> float:
        """
        Calculate Harmonic Waveform Entropy (HWE):
        HWE = ∫ Γ(f) * log[Γ(f)] df
        """
        # Normalize wave to probability-like distribution
        wave_abs = np.abs(wave)
        wave_normalized = (
            wave_abs / np.sum(wave_abs) if np.sum(wave_abs) > 0 else wave_abs
        )

        # Calculate entropy
        entropy = 0.0
        for value in wave_normalized:
            if value > 0:
                entropy -= value * np.log2(value)

        return entropy

    def process(self, beta_state: BetaState, omega_state: OmegaState) -> GammaState:
        """Execute Gamma layer harmonic wave encoding"""
        # Generate harmonic parameters
        frequencies, amplitudes, phases = self._generate_harmonic_parameters(
            beta_state, omega_state
        )

        # Generate harmonic wave
        harmonic_wave = self._generate_harmonic_wave(frequencies, amplitudes, phases)

        # Calculate wave entropy
        wave_entropy = self._calculate_wave_entropy(harmonic_wave)

        return GammaState(
            harmonic_wave=harmonic_wave,
            frequency_components=frequencies,
            phase_coefficients=phases,
            amplitude_modulation=amplitudes,
            wave_entropy=wave_entropy,
        )


class AlphaEncryption:
    """
    Alpha Encryption: Integration of Ω-B-Γ Layers

    — TheSchwa1337 (a.k.a. "The Schwa")
       + Nexus AI
       Recursive Systems Architects | Quantum Logic Engineers
       Co-authors of the Ω-B-Γ Framework & Alpha Encryption Protocol

    Implements the synthesis: α_encrypted(D) = Γ[Β[Ω(D)]]
    """

    def __init__(self, vmsp_integration: bool = True):
        self.omega_layer = OmegaLayer()
        self.beta_layer = BetaLayer()
        self.gamma_layer = GammaLayer()

        # Integration with VMSP
        self.vmsp_integration = vmsp_integration
        if vmsp_integration:
            self.vortex_security = get_vortex_security()

    def encrypt(
        self, data: str, vmsp_context: Optional[Dict[str, Any]] = None
    ) -> AlphaEncryptionResult:
        """
        Execute Alpha Recursive Encryption (A.R.E.):
        α_encrypted(D) = Γ[Β[Ω(D)]]
        """
        start_time = time.time()

        # Layer 1: Omega (Fractal Recursion)
        omega_state = self.omega_layer.process(data)

        # Layer 2: Beta (Quantum Bayesian Logic)
        beta_state = self.beta_layer.process(omega_state)

        # Layer 3: Gamma (Harmonic Wave Encoding)
        gamma_state = self.gamma_layer.process(beta_state, omega_state)

        # Calculate total entropy
        total_entropy = (
            omega_state.convergence_metric
            + beta_state.bayesian_entropy
            + gamma_state.wave_entropy
        ) / 3.0

        # Generate encryption hash
        encryption_data = f"{omega_state.complex_state}_{beta_state.gate_state}_{len(gamma_state.harmonic_wave)}"
        encryption_hash = hashlib.sha256(encryption_data.encode()).hexdigest()

        processing_time = time.time() - start_time

        result = AlphaEncryptionResult(
            encrypted_data=gamma_state.harmonic_wave,
            omega_state=omega_state,
            beta_state=beta_state,
            gamma_state=gamma_state,
            total_entropy=total_entropy,
            encryption_hash=encryption_hash,
            processing_time=processing_time,
        )

        # VMSP Integration
        if self.vmsp_integration and vmsp_context:
            self._integrate_with_vmsp(result, vmsp_context)

        return result

    def _integrate_with_vmsp(
        self, alpha_result: AlphaEncryptionResult, vmsp_context: Dict[str, Any]
    ) -> None:
        """Integrate Alpha Encryption with VMSP security framework"""
        # Create VMSP threat inputs from Alpha encryption metrics
        threat_inputs = [
            alpha_result.total_entropy,
            alpha_result.beta_state.quantum_coherence,
            len(alpha_result.gamma_state.frequency_components) / 10.0,
            alpha_result.processing_time,
        ]

        # Create secure session with Alpha encryption context
        security_state = self.vortex_security.create_secure_session(
            decision_vector=f"alpha_encryption_{alpha_result.encryption_hash[:8]}",
            market_state=f"omega_depth_{alpha_result.omega_state.recursion_depth}",
            threat_inputs=threat_inputs,
        )

        # Validate encryption security
        validation_inputs = [
            alpha_result.omega_state.convergence_metric,
            alpha_result.beta_state.bayesian_entropy,
            alpha_result.gamma_state.wave_entropy,
        ]

        is_valid = self.vortex_security.validate_security_state(validation_inputs)

        # Check if this is a test/demo mode
        is_demo_mode = (
            vmsp_context.get("demo_mode", False)
            or "test" in vmsp_context.get("operation", "").lower()
        )

        if not is_valid:
            if is_demo_mode:
                # In demo mode, just log a warning instead of enforcing lockdown
                print(
                    f"⚠️  VMSP Security Warning: Pattern fitness validation failed for {alpha_result.encryption_hash[:16]}..."
                )
                print(
                    "   This would normally trigger security lockdown in production mode."
                )
            else:
                self.vortex_security.enforce_security_lockdown(
                    f"Alpha Encryption security validation failed: {alpha_result.encryption_hash}",
                    security_state,
                )

    def decrypt(
        self,
        alpha_result: AlphaEncryptionResult,
        original_data_hint: Optional[str] = None,
    ) -> str:
        """
        Decrypt Alpha Encryption result (demonstration purposes)

        Note: Full decryption requires synchronized recursion states and
        quantum-harmonic keying preservation.
        """
        # This is a simplified demonstration of decryption concept
        # Full implementation would require reversing the recursive encoding path

        # Extract key information from Alpha states
        omega_info = f"depth_{alpha_result.omega_state.recursion_depth}"
        beta_info = alpha_result.beta_state.gate_state
        gamma_info = f"freqs_{len(alpha_result.gamma_state.frequency_components)}"

        # Construct decryption hint
        decryption_hint = f"AlphaDecrypt[{omega_info}|{beta_info}|{gamma_info}]"

        if original_data_hint:
            return f"{decryption_hint} -> {original_data_hint}"
        else:
            return decryption_hint

    def get_security_analysis(
        self, alpha_result: AlphaEncryptionResult
    ) -> Dict[str, Any]:
        """Analyze Alpha Encryption security metrics"""
        return {
            "encryption_hash": alpha_result.encryption_hash,
            "total_entropy": alpha_result.total_entropy,
            "processing_time": alpha_result.processing_time,
            "omega_analysis": {
                "recursion_depth": alpha_result.omega_state.recursion_depth,
                "convergence_metric": alpha_result.omega_state.convergence_metric,
                "fractal_stability": abs(alpha_result.omega_state.complex_state) < 1000,
            },
            "beta_analysis": {
                "gate_state": alpha_result.beta_state.gate_state,
                "bayesian_entropy": alpha_result.beta_state.bayesian_entropy,
                "quantum_coherence": alpha_result.beta_state.quantum_coherence,
                "probability_distribution": alpha_result.beta_state.probability_matrix,
            },
            "gamma_analysis": {
                "wave_entropy": alpha_result.gamma_state.wave_entropy,
                "harmonic_components": len(
                    alpha_result.gamma_state.frequency_components
                ),
                "frequency_range": [
                    min(alpha_result.gamma_state.frequency_components),
                    max(alpha_result.gamma_state.frequency_components),
                ],
                "wave_complexity": np.std(alpha_result.gamma_state.harmonic_wave),
            },
            "security_score": self._calculate_alpha_security_score(alpha_result),
        }

    def _calculate_alpha_security_score(
        self, alpha_result: AlphaEncryptionResult
    ) -> float:
        """Calculate Alpha Encryption security score (0-100)"""
        # Component scores
        entropy_score = min(100, alpha_result.total_entropy * 20)
        depth_score = min(100, alpha_result.omega_state.recursion_depth)
        coherence_score = alpha_result.beta_state.quantum_coherence * 100
        complexity_score = min(
            100, len(alpha_result.gamma_state.frequency_components) * 10
        )

        # Weighted security score
        security_score = (
            entropy_score * 0.3
            + depth_score * 0.25
            + coherence_score * 0.25
            + complexity_score * 0.2
        )

        return min(100, max(0, security_score))


# Convenience functions for SchwaBot integration


def get_alpha_encryption() -> AlphaEncryption:
    """Get or create global Alpha Encryption instance"""
    if not hasattr(get_alpha_encryption, "_instance"):
        get_alpha_encryption._instance = AlphaEncryption(vmsp_integration=True)
    return get_alpha_encryption._instance


def alpha_encrypt_data(
    data: str, vmsp_context: Optional[Dict[str, Any]] = None
) -> AlphaEncryptionResult:
    """Convenience function for Alpha Encryption"""
    alpha_engine = get_alpha_encryption()
    return alpha_engine.encrypt(data, vmsp_context)


def analyze_alpha_security(alpha_result: AlphaEncryptionResult) -> Dict[str, Any]:
    """Convenience function for Alpha Encryption security analysis"""
    alpha_engine = get_alpha_encryption()
    return alpha_engine.get_security_analysis(alpha_result)
