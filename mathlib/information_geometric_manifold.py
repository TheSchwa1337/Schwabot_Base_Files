import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import entropy
from scipy.optimize import minimize

@dataclass
class MarketState:
    """Represents a point on the market state manifold"""
    parameters: np.ndarray  # Natural parameters θ
    distribution: np.ndarray  # Probability distribution p(x|θ)
    
class InformationGeometricManifold:
    def __init__(self, dim: int, metric_type: str = 'fisher'):
        """
        Initialize the information geometric manifold
        
        Args:
            dim: Dimension of the parameter space
            metric_type: Type of metric to use ('fisher' or 'wasserstein')
        """
        self.dim = dim
        self.metric_type = metric_type
        self.epsilon = 1e-6  # For numerical stability
        
    def compute_fisher_metric(self, state: MarketState) -> np.ndarray:
        """
        Compute the Fisher information metric at a given state
        
        Args:
            state: Current market state
            
        Returns:
            g_ij: Fisher metric tensor
        """
        # Compute log-likelihood gradients
        log_p = np.log(state.distribution + self.epsilon)
        grad_log_p = np.gradient(log_p)
        
        # Compute Fisher metric as expectation of outer product of gradients
        g_ij = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                g_ij[i,j] = np.mean(grad_log_p[i] * grad_log_p[j])
                
        return g_ij
    
    def natural_gradient_step(
        self,
        current_state: MarketState,
        target_state: MarketState,
        learning_rate: float = 0.01
    ) -> MarketState:
        """
        Perform natural gradient descent step on the manifold
        
        Args:
            current_state: Current market state
            target_state: Target market state
            learning_rate: Step size for gradient descent
            
        Returns:
            Updated market state
        """
        # Compute Fisher metric at current state
        g_ij = self.compute_fisher_metric(current_state)
        
        # Compute gradient of KL divergence
        grad_kl = self._compute_kl_gradient(current_state, target_state)
        
        # Compute natural gradient using Fisher metric
        g_inv = np.linalg.inv(g_ij + self.epsilon * np.eye(self.dim))
        natural_grad = g_inv @ grad_kl
        
        # Update parameters using natural gradient
        new_params = current_state.parameters - learning_rate * natural_grad
        
        # Project back to valid parameter space if needed
        new_params = self._project_parameters(new_params)
        
        # Create new state with updated parameters
        new_dist = self._compute_distribution(new_params)
        return MarketState(parameters=new_params, distribution=new_dist)
    
    def _compute_kl_gradient(
        self,
        current: MarketState,
        target: MarketState
    ) -> np.ndarray:
        """Compute gradient of KL divergence between states"""
        ratio = target.distribution / (current.distribution + self.epsilon)
        grad_log_p = np.gradient(np.log(current.distribution + self.epsilon))
        return -np.mean(ratio[:, np.newaxis] * grad_log_p, axis=0)
    
    def _project_parameters(self, params: np.ndarray) -> np.ndarray:
        """Project parameters to valid space"""
        # Add any necessary constraints here
        return np.clip(params, -10, 10)  # Example: clip to reasonable range
    
    def _compute_distribution(self, params: np.ndarray) -> np.ndarray:
        """Compute probability distribution from parameters"""
        # Example: softmax transformation
        exp_params = np.exp(params - np.max(params))
        return exp_params / np.sum(exp_params)
    
    def geodesic_distance(self, state1: MarketState, state2: MarketState) -> float:
        """
        Compute geodesic distance between two states
        
        Args:
            state1: First market state
            state2: Second market state
            
        Returns:
            Geodesic distance
        """
        # Compute Fisher metrics at both states
        g1 = self.compute_fisher_metric(state1)
        g2 = self.compute_fisher_metric(state2)
        
        # Compute average metric along geodesic
        g_avg = (g1 + g2) / 2
        
        # Compute parameter difference
        delta = state2.parameters - state1.parameters
        
        # Compute geodesic distance
        return np.sqrt(delta.T @ g_avg @ delta)
    
    def detect_regime_change(
        self,
        state_sequence: List[MarketState],
        threshold: float = 0.1
    ) -> List[int]:
        """
        Detect regime changes by monitoring geodesic deviations
        
        Args:
            state_sequence: Sequence of market states
            threshold: Threshold for regime change detection
            
        Returns:
            Indices where regime changes occur
        """
        regime_changes = []
        
        for i in range(1, len(state_sequence)):
            # Compute geodesic distance between consecutive states
            dist = self.geodesic_distance(state_sequence[i-1], state_sequence[i])
            
            # If distance exceeds threshold, mark as regime change
            if dist > threshold:
                regime_changes.append(i)
                
        return regime_changes 