"""
Basket Tensor Feedback
=====================

Implements the tensor feedback system for Schwabot's recursive trading intelligence.
Manages SHA-tensor relationships and trust scoring.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import hashlib

from .profit_tensor import ProfitTensorStore
from .bitmap_engine import BitmapEngine

@dataclass
class TensorMetrics:
    """Container for tensor-specific metrics"""
    mean_profit: float
    std_dev_profit: float
    mean_thermal_cost: float
    sortino_ratio: float
    cvar: float  # Conditional Value at Risk
    success_rate: float
    last_update: datetime
    usage_count: int = 0

class BasketTensorFeedback:
    """Manages tensor feedback and trust scoring for baskets"""
    
    def __init__(self, rho_coefficients: Optional[Dict[int, float]] = None):
        self.tensor_store = ProfitTensorStore()
        self.bitmap_engine = BitmapEngine()
        
        # Initialize trust coefficients for each bit depth
        self.rho_coefficients = rho_coefficients or {
            4: 1.0,   # 4-bit
            8: 1.0,   # 8-bit
            16: 1.0,  # 16-bit
            42: 1.0,  # 42-bit
            81: 1.0   # 81-bit
        }
        
        # Track tensor metrics
        self.tensor_metrics: Dict[str, TensorMetrics] = {}
        
        # Track bit depth performance
        self.bit_depth_stats: Dict[int, Dict[str, float]] = {
            depth: {
                'success_count': 0,
                'total_count': 0,
                'avg_profit': 0.0,
                'avg_thermal': 0.0
            }
            for depth in self.rho_coefficients.keys()
        }

    def compute_trust_score(self, market_signature: dict) -> float:
        """Compute trust score based on market signature and tensor feedback"""
        # Get current bitmap and generate SHA key
        bitmap = self.infer_bitmap(market_signature)
        sha_key = self.bitmap_engine.generate_sha_key(bitmap)
        
        # Get tensor feedback
        tensor = self.tensor_store.lookup(sha_key)
        if tensor is None:
            return 0.5  # Default trust score for unknown tensors
            
        # Calculate trust score based on tensor metrics
        metrics = self._get_or_create_tensor_metrics(sha_key, tensor)
        
        # Weight different aspects of trust
        weights = {
            'mean_profit': 0.3,
            'sortino_ratio': 0.2,
            'success_rate': 0.2,
            'cvar': 0.15,
            'thermal_efficiency': 0.15
        }
        
        # Calculate thermal efficiency (inverse of thermal cost)
        thermal_efficiency = 1.0 / (metrics.mean_thermal_cost + 1e-9)
        
        # Combine weighted scores
        trust_score = (
            weights['mean_profit'] * np.tanh(metrics.mean_profit) +
            weights['sortino_ratio'] * np.tanh(metrics.sortino_ratio) +
            weights['success_rate'] * metrics.success_rate +
            weights['cvar'] * (1.0 - np.tanh(metrics.cvar)) +  # Lower CVaR is better
            weights['thermal_efficiency'] * np.tanh(thermal_efficiency)
        )
        
        return float(np.clip(trust_score, 0.0, 1.0))

    def infer_bitmap(self, market_signature: dict) -> List[bool]:
        """Infer bitmap from market signature"""
        entropy = market_signature.get('entropy', [0.2]*5)
        _, probs = self.bitmap_engine.entropy_to_bitmap_transform(entropy)
        return [p > 0.4 for p in probs]

    def _get_or_create_tensor_metrics(self, sha_key: str, tensor: np.ndarray) -> TensorMetrics:
        """Get or create tensor metrics for a SHA key"""
        if sha_key not in self.tensor_metrics:
            # Calculate initial metrics
            profits = tensor[:, 0] if tensor.ndim > 1 else tensor
            thermal_costs = tensor[:, 1] if tensor.ndim > 1 else np.zeros_like(profits)
            
            self.tensor_metrics[sha_key] = TensorMetrics(
                mean_profit=float(np.mean(profits)),
                std_dev_profit=float(np.std(profits)),
                mean_thermal_cost=float(np.mean(thermal_costs)),
                sortino_ratio=self._calculate_sortino_ratio(profits),
                cvar=self._calculate_cvar(profits),
                success_rate=float(np.mean(profits > 0)),
                last_update=datetime.now()
            )
            
        return self.tensor_metrics[sha_key]

    def _calculate_sortino_ratio(self, profits: np.ndarray, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio for profit series"""
        excess_returns = profits - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return float('inf')
            
        return float(np.mean(excess_returns) / downside_std)

    def _calculate_cvar(self, profits: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        if len(profits) == 0:
            return 0.0
            
        var = np.percentile(profits, (1 - confidence_level) * 100)
        return float(np.mean(profits[profits <= var]))

    def update_tensor_metrics(self, sha_key: str, profit: float, thermal_cost: float, bit_depth: int):
        """Update tensor metrics with new profit and thermal cost"""
        if sha_key not in self.tensor_metrics:
            return
            
        metrics = self.tensor_metrics[sha_key]
        metrics.usage_count += 1
        
        # Update bit depth statistics
        stats = self.bit_depth_stats[bit_depth]
        stats['total_count'] += 1
        if profit > 0:
            stats['success_count'] += 1
            
        # Update averages using exponential moving average
        alpha = 0.1  # Learning rate
        stats['avg_profit'] = (1 - alpha) * stats['avg_profit'] + alpha * profit
        stats['avg_thermal'] = (1 - alpha) * stats['avg_thermal'] + alpha * thermal_cost
        
        # Update rho coefficient based on performance
        success_rate = stats['success_count'] / stats['total_count']
        self.rho_coefficients[bit_depth] = np.clip(
            self.rho_coefficients[bit_depth] * (1 + (success_rate - 0.5) * 0.1),
            0.1,  # Minimum trust
            2.0   # Maximum trust
        )

    def get_bit_depth_activation(self, market_signature: dict) -> Dict[int, float]:
        """Get activation probabilities for each bit depth"""
        # Calculate utility scores for each bit depth
        utilities = {}
        for bit_depth in self.rho_coefficients.keys():
            # Generate SHA key for this bit depth
            bitmap = self.infer_bitmap(market_signature)
            sha_key = self.bitmap_engine.generate_sha_key(bitmap)
            
            # Get tensor feedback
            tensor = self.tensor_store.lookup(sha_key)
            if tensor is not None:
                metrics = self._get_or_create_tensor_metrics(sha_key, tensor)
                utilities[bit_depth] = (
                    metrics.mean_profit * 0.4 +
                    metrics.sortino_ratio * 0.3 +
                    metrics.success_rate * 0.3
                )
            else:
                utilities[bit_depth] = 0.0
                
        # Apply softmax with rho coefficients
        exp_utilities = {
            depth: np.exp(self.rho_coefficients[depth] * util)
            for depth, util in utilities.items()
        }
        sum_exp = sum(exp_utilities.values())
        
        return {
            depth: float(util / sum_exp)
            for depth, util in exp_utilities.items()
        }

    def get_tensor_stats(self) -> Dict:
        """Get statistics about tensor performance"""
        return {
            'total_tensors': len(self.tensor_metrics),
            'bit_depth_stats': self.bit_depth_stats,
            'rho_coefficients': self.rho_coefficients,
            'avg_trust_score': np.mean([
                self.compute_trust_score({'entropy': [0.2]*5})
                for _ in range(100)  # Sample 100 random market states
            ])
        }

# Example usage
if __name__ == "__main__":
    feedback = BasketTensorFeedback()
    
    # Test market signature
    market_sig = {
        'entropy': [0.2, 0.3, 0.4, 0.5, 0.6],
        'profit_gradient': 0.002,
        'thermal_state': 0.5
    }
    
    # Get trust score
    trust = feedback.compute_trust_score(market_sig)
    print(f"Trust score: {trust:.4f}")
    
    # Get bit depth activation
    activations = feedback.get_bit_depth_activation(market_sig)
    print("\nBit depth activations:")
    for depth, prob in activations.items():
        print(f"{depth}-bit: {prob:.4f}") 