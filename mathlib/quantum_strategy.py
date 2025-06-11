import numpy as np
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from scipy.linalg import expm
import random

@dataclass
class QuantumState:
    """Represents a quantum state vector"""
    amplitudes: np.ndarray  # Complex amplitudes
    basis_states: List[Any]  # Basis states
    
    def normalize(self) -> None:
        """Normalize the state vector"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm
            
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution over basis states"""
        return np.abs(self.amplitudes) ** 2

class QuantumStrategyEngine:
    def __init__(
        self,
        basis_strategies: List[Callable],
        initial_state: Optional[QuantumState] = None
    ):
        """
        Initialize the quantum strategy engine
        
        Args:
            basis_strategies: List of basis strategy functions
            initial_state: Initial quantum state (optional)
        """
        self.basis_strategies = basis_strategies
        self.num_strategies = len(basis_strategies)
        
        if initial_state is None:
            # Initialize with uniform superposition
            amplitudes = np.ones(self.num_strategies, dtype=complex) / np.sqrt(self.num_strategies)
            self.state = QuantumState(amplitudes, basis_strategies)
        else:
            self.state = initial_state
            self.state.normalize()
            
    def evolve_superposition(self, market_observable: np.ndarray, dt: float = 0.01) -> None:
        """
        Evolve the quantum state using Schrödinger equation
        
        Args:
            market_observable: Market observation matrix
            dt: Time step for evolution
        """
        # Construct Hamiltonian from market observable
        H = self._construct_hamiltonian(market_observable)
        
        # Compute time evolution operator
        U = expm(-1j * H * dt)
        
        # Evolve state
        self.state.amplitudes = U @ self.state.amplitudes
        self.state.normalize()
        
    def _construct_hamiltonian(self, market_observable: np.ndarray) -> np.ndarray:
        """
        Construct Hamiltonian operator from market observable
        
        Args:
            market_observable: Market observation matrix
            
        Returns:
            Hamiltonian matrix
        """
        # Ensure market observable is Hermitian
        H = (market_observable + market_observable.T) / 2
        
        # Add small diagonal terms for stability
        H += np.eye(self.num_strategies) * 1e-6
        
        return H
        
    def collapse_to_strategy(self, market_state: Any) -> Callable:
        """
        Collapse quantum superposition to a classical strategy
        
        Args:
            market_state: Current market state
            
        Returns:
            Selected strategy function
        """
        # Compute strategy probabilities
        probs = self.state.get_probabilities()
        
        # Sample strategy based on probabilities
        strategy_idx = np.random.choice(self.num_strategies, p=probs)
        
        return self.basis_strategies[strategy_idx]
        
    def measure_strategy_observable(self, observable: np.ndarray) -> float:
        """
        Measure expectation value of a strategy observable
        
        Args:
            observable: Hermitian operator representing the observable
            
        Returns:
            Expectation value
        """
        # Ensure observable is Hermitian
        obs = (observable + observable.T) / 2
        
        # Compute expectation value
        return np.real(self.state.amplitudes.conj() @ obs @ self.state.amplitudes)
        
    def entangle_strategies(
        self,
        other_engine: 'QuantumStrategyEngine',
        entanglement_strength: float = 0.5
    ) -> None:
        """
        Entangle this strategy engine with another
        
        Args:
            other_engine: Another quantum strategy engine
            entanglement_strength: Strength of entanglement (0 to 1)
        """
        # Create entangled state
        combined_amplitudes = np.kron(
            self.state.amplitudes,
            other_engine.state.amplitudes
        )
        
        # Apply entanglement operator
        entangler = self._create_entanglement_operator(
            self.num_strategies,
            other_engine.num_strategies,
            entanglement_strength
        )
        
        # Update states
        entangled_state = entangler @ combined_amplitudes
        self.state.amplitudes = entangled_state[:self.num_strategies]
        other_engine.state.amplitudes = entangled_state[self.num_strategies:]
        
        # Normalize both states
        self.state.normalize()
        other_engine.state.normalize()
        
    def _create_entanglement_operator(
        self,
        dim1: int,
        dim2: int,
        strength: float
    ) -> np.ndarray:
        """Create entanglement operator between two quantum systems"""
        # Create controlled-NOT like operator
        op = np.eye(dim1 * dim2)
        for i in range(min(dim1, dim2)):
            idx1 = i * dim2 + i
            idx2 = i * dim2 + ((i + 1) % dim2)
            op[idx1, idx1] = np.sqrt(1 - strength)
            op[idx1, idx2] = np.sqrt(strength)
            op[idx2, idx1] = np.sqrt(strength)
            op[idx2, idx2] = np.sqrt(1 - strength)
        return op
        
    def get_strategy_coherence(self) -> float:
        """
        Compute coherence of the strategy superposition
        
        Returns:
            Coherence measure (0 to 1)
        """
        # Compute off-diagonal elements of density matrix
        rho = np.outer(self.state.amplitudes, self.state.amplitudes.conj())
        off_diag = np.abs(rho - np.diag(np.diag(rho)))
        
        # Normalize by maximum possible coherence
        max_coherence = self.num_strategies - 1
        return np.sum(off_diag) / max_coherence if max_coherence > 0 else 0
        
    def apply_strategy_rotation(self, angle: float, axis: int = 0) -> None:
        """
        Apply rotation to strategy superposition
        
        Args:
            angle: Rotation angle in radians
            axis: Axis of rotation (strategy index)
        """
        # Create rotation operator
        R = np.eye(self.num_strategies, dtype=complex)
        R[axis, axis] = np.exp(1j * angle)
        
        # Apply rotation
        self.state.amplitudes = R @ self.state.amplitudes
        self.state.normalize() 