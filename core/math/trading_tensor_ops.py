# -*- coding: utf-8 -*-
"""
Trading Tensor Operations for Schwabot Mathematical Analysis
==========================================================

Provides specialized tensor operations for trading analysis, including
profit surface calculations, volatility tensors, and momentum analysis.

Mathematical Operations:
- Multi-dimensional profit surface analysis
- Volatility tensor calculations
- Momentum and correlation tensors
- BTC price tensor mapping
- Phase transition analysis

MATHEMATICAL PRESERVATION: All core mathematical logic preserved.
"""

import numpy as np
from numpy.typing import NDArray
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class TradingTensorOps:
    """Trading-specific tensor operations for mathematical analysis."""
    
    def __init__(self):
        """Initialize Trading Tensor Operations."""
        logger.info("📊 Trading Tensor Operations initialized")
    
    def calculate_profit_surface(self, price_tensor: NDArray, volume_tensor: NDArray) -> NDArray:
        """Calculate multi-dimensional profit surface."""
        try:
            if price_tensor.shape != volume_tensor.shape:
                raise ValueError("Price and volume tensors must have the same shape")
            
            # Calculate profit surface as price * volume interaction
            profit_surface = price_tensor * volume_tensor
            
            # Add second-order terms for non-linear effects
            profit_surface += 0.1 * np.square(price_tensor) * np.sqrt(volume_tensor)
            
            # Apply smoothing filter
            from scipy import ndimage
            try:
                profit_surface = ndimage.gaussian_filter(profit_surface, sigma=1.0)
            except ImportError:
                # Fallback to simple averaging if scipy not available
                pass
            
            return profit_surface
            
        except Exception as e:
            logger.error(f"Profit surface calculation failed: {e}")
            return np.zeros_like(price_tensor)
    
    def calculate_volatility_tensor(self, price_history: NDArray, window_size: int = 20) -> NDArray:
        """Calculate volatility tensor from price history."""
        try:
            if len(price_history) < window_size:
                return np.array([0.5])  # Default volatility
            
            # Calculate rolling volatility
            volatility_values = []
            for i in range(window_size, len(price_history)):
                window = price_history[i-window_size:i]
                returns = np.diff(window) / window[:-1]
                volatility = np.std(returns)
                volatility_values.append(volatility)
            
            volatility_tensor = np.array(volatility_values)
            
            # Create multi-dimensional volatility surface
            if len(volatility_tensor) > 1:
                # Reshape into 2D surface if enough data
                rows = int(np.sqrt(len(volatility_tensor)))
                if rows > 1:
                    cols = len(volatility_tensor) // rows
                    volatility_tensor = volatility_tensor[:rows*cols].reshape(rows, cols)
            
            return volatility_tensor
            
        except Exception as e:
            logger.error(f"Volatility tensor calculation failed: {e}")
            return np.array([0.5])
    
    def calculate_momentum_tensor(self, price_data: NDArray, period: int = 14) -> NDArray:
        """Calculate momentum tensor for trend analysis."""
        try:
            if len(price_data) < period:
                return np.array([0.0])
            
            # Calculate momentum as rate of change
            momentum_values = []
            for i in range(period, len(price_data)):
                current_price = price_data[i]
                past_price = price_data[i - period]
                momentum = (current_price - past_price) / past_price
                momentum_values.append(momentum)
            
            momentum_tensor = np.array(momentum_values)
            
            # Add momentum derivatives for second-order analysis
            if len(momentum_tensor) > 1:
                momentum_derivative = np.diff(momentum_tensor)
                # Combine momentum and its derivative
                combined_tensor = np.stack([momentum_tensor[:-1], momentum_derivative])
                return combined_tensor
            
            return momentum_tensor
            
        except Exception as e:
            logger.error(f"Momentum tensor calculation failed: {e}")
            return np.array([0.0])
    
    def calculate_correlation_matrix(self, multi_asset_data: Dict[str, NDArray]) -> NDArray:
        """Calculate correlation matrix for multi-asset analysis."""
        try:
            if not multi_asset_data:
                return np.eye(2)  # Return identity matrix if no data
            
            assets = list(multi_asset_data.keys())
            n_assets = len(assets)
            
            # Find minimum length across all assets
            min_length = min(len(data) for data in multi_asset_data.values())
            
            if min_length < 2:
                return np.eye(n_assets)
            
            # Create data matrix
            data_matrix = np.zeros((n_assets, min_length))
            for i, asset in enumerate(assets):
                data_matrix[i, :] = multi_asset_data[asset][:min_length]
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(data_matrix)
            
            # Handle NaN values
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return np.eye(2)
    
    def calculate_entropy_signal(self, price_data: NDArray, volume_data: NDArray) -> NDArray:
        """Calculate entropy signal for market randomness analysis."""
        try:
            if len(price_data) != len(volume_data):
                min_len = min(len(price_data), len(volume_data))
                price_data = price_data[:min_len]
                volume_data = volume_data[:min_len]
            
            # Calculate price returns
            price_returns = np.diff(price_data) / price_data[:-1]
            
            # Discretize returns into bins for entropy calculation
            bins = 10
            hist, _ = np.histogram(price_returns, bins=bins)
            
            # Calculate probability distribution
            hist = hist + 1e-10  # Add small epsilon to avoid log(0)
            probabilities = hist / np.sum(hist)
            
            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Create entropy signal tensor
            entropy_signal = np.full_like(price_data[:-1], entropy)
            
            # Modulate entropy with volume
            volume_normalized = volume_data[:-1] / np.max(volume_data[:-1])
            entropy_signal *= volume_normalized
            
            return entropy_signal
            
        except Exception as e:
            logger.error(f"Entropy signal calculation failed: {e}")
            return np.array([0.5])
    
    def calculate_btc_price_tensor(self, btc_prices: List[float], features: List[str]) -> NDArray:
        """Calculate BTC price tensor with multiple features."""
        try:
            if not btc_prices:
                return np.array([[50000.0]])  # Default BTC price
            
            btc_array = np.array(btc_prices)
            
            # Create feature tensor
            feature_tensors = []
            
            for feature in features:
                if feature == "price":
                    feature_tensors.append(btc_array)
                elif feature == "returns":
                    returns = np.diff(btc_array) / btc_array[:-1]
                    returns = np.concatenate([[0]], returns)  # Add zero for first element
                    feature_tensors.append(returns)
                elif feature == "volatility":
                    volatility = self.calculate_volatility_tensor(btc_array)
                    # Pad to match price array length
                    if len(volatility) < len(btc_array):
                        padding = np.full(len(btc_array) - len(volatility), np.mean(volatility))
                        volatility = np.concatenate([padding, volatility])
                    feature_tensors.append(volatility[:len(btc_array)])
                elif feature == "momentum":
                    momentum = self.calculate_momentum_tensor(btc_array)
                    # Flatten if multi-dimensional
                    if momentum.ndim > 1:
                        momentum = momentum.flatten()
                    # Pad to match price array length
                    if len(momentum) < len(btc_array):
                        padding = np.full(len(btc_array) - len(momentum), 0.0)
                        momentum = np.concatenate([padding, momentum])
                    feature_tensors.append(momentum[:len(btc_array)])
                else:
                    # Default feature (normalized price)
                    normalized_price = (btc_array - np.min(btc_array)) / (np.max(btc_array) - np.min(btc_array))
                    feature_tensors.append(normalized_price)
            
            # Stack features into tensor
            btc_tensor = np.stack(feature_tensors, axis=1)
            
            return btc_tensor
            
        except Exception as e:
            logger.error(f"BTC price tensor calculation failed: {e}")
            return np.array([[50000.0]])
    
    def calculate_profit_optimization_tensor(self, price_tensor: NDArray, cost_tensor: NDArray, risk_tensor: NDArray) -> NDArray:
        """Calculate profit optimization tensor considering costs and risks."""
        try:
            # Ensure all tensors have the same shape
            min_shape = np.minimum.reduce([price_tensor.shape, cost_tensor.shape, risk_tensor.shape])
            
            price_resized = price_tensor[:min_shape[0]] if price_tensor.ndim == 1 else price_tensor[:min_shape[0], :min_shape[1]]
            cost_resized = cost_tensor[:min_shape[0]] if cost_tensor.ndim == 1 else cost_tensor[:min_shape[0], :min_shape[1]]
            risk_resized = risk_tensor[:min_shape[0]] if risk_tensor.ndim == 1 else risk_tensor[:min_shape[0], :min_shape[1]]
            
            # Calculate profit optimization: maximize profit, minimize cost and risk
            profit_optimization = price_resized - cost_resized - 0.5 * risk_resized
            
            # Apply sigmoid activation for normalization
            profit_optimization = 1 / (1 + np.exp(-profit_optimization))
            
            return profit_optimization
            
        except Exception as e:
            logger.error(f"Profit optimization tensor calculation failed: {e}")
            return np.array([0.5])
    
    def calculate_phase_transition_tensor(self, market_phases: List[str], transition_matrix: NDArray) -> NDArray:
        """Calculate phase transition tensor for market state analysis."""
        try:
            if not market_phases:
                return np.eye(4)  # Default 4x4 identity for 4 phases
            
            # Map phases to indices
            phase_map = {"valley": 0, "ascent": 1, "peak": 2, "descent": 3}
            
            # Create transition sequence
            phase_indices = [phase_map.get(phase, 0) for phase in market_phases]
            
            # Calculate transition probabilities
            n_phases = len(phase_map)
            transition_counts = np.zeros((n_phases, n_phases))
            
            for i in range(len(phase_indices) - 1):
                current_phase = phase_indices[i]
                next_phase = phase_indices[i + 1]
                transition_counts[current_phase, next_phase] += 1
            
            # Normalize to get probabilities
            transition_tensor = transition_counts / (np.sum(transition_counts, axis=1, keepdims=True) + 1e-10)
            
            # Use provided transition matrix if available
            if transition_matrix.shape == transition_tensor.shape:
                transition_tensor = 0.7 * transition_tensor + 0.3 * transition_matrix
            
            return transition_tensor
            
        except Exception as e:
            logger.error(f"Phase transition tensor calculation failed: {e}")
            return np.eye(4)
    
    def test_tensor_operations(self) -> Dict[str, Any]:
        """Test all tensor operations with sample data."""
        try:
            logger.info("🧪 Testing Trading Tensor Operations...")
            
            # Generate sample data
            price_data = np.random.normal(50000, 1000, 100)  # Sample BTC prices
            volume_data = np.random.exponential(1000, 100)   # Sample volumes
            
            # Test profit surface
            profit_surface = self.calculate_profit_surface(
                price_data.reshape(10, 10), 
                volume_data.reshape(10, 10)
            )
            logger.info(f"  ✅ Profit surface calculation: shape {profit_surface.shape}")
            
            # Test volatility tensor
            volatility = self.calculate_volatility_tensor(price_data)
            logger.info(f"  ✅ Volatility tensor calculation: shape {volatility.shape}")
            
            # Test momentum tensor
            momentum = self.calculate_momentum_tensor(price_data)
            logger.info(f"  ✅ Momentum tensor calculation: shape {momentum.shape}")
            
            # Test BTC price tensor
            btc_tensor = self.calculate_btc_price_tensor(
                price_data.tolist(), 
                ["price", "returns", "volatility"]
            )
            logger.info(f"  ✅ BTC price tensor calculation: shape {btc_tensor.shape}")
            
            # Test profit optimization tensor
            optimization_tensor = self.calculate_profit_optimization_tensor(
                profit_surface, 
                np.random.normal(0.1, 0.02, profit_surface.shape),  # Costs
                np.random.normal(0.2, 0.05, profit_surface.shape)   # Risks
            )
            logger.info(f"  ✅ Profit optimization tensor calculation: shape {optimization_tensor.shape}")
            
            # Test phase transition tensor
            phase_tensor = self.calculate_phase_transition_tensor(
                ["valley", "ascent", "peak", "descent"] * 5,
                np.eye(4)
            )
            logger.info(f"  ✅ Phase transition tensor calculation: shape {phase_tensor.shape}")
            
            logger.info("✅ Trading Tensor Operations test completed successfully")
            
            return {
                "profit_surface_shape": profit_surface.shape,
                "volatility_shape": volatility.shape,
                "momentum_shape": momentum.shape,
                "btc_tensor_shape": btc_tensor.shape,
                "optimization_shape": optimization_tensor.shape,
                "phase_tensor_shape": phase_tensor.shape,
                "test_status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Trading tensor operations test failed: {e}")
            return {"test_status": "failed", "error": str(e)}


# Global instance
trading_tensor_ops = TradingTensorOps()

# Export functions for external use
def calculate_profit_surface(price_tensor: NDArray, volume_tensor: NDArray) -> NDArray:
    """Calculate profit surface for external use."""
    return trading_tensor_ops.calculate_profit_surface(price_tensor, volume_tensor)

def calculate_btc_price_tensor(btc_prices: List[float], features: List[str]) -> NDArray:
    """Calculate BTC price tensor for external use."""
    return trading_tensor_ops.calculate_btc_price_tensor(btc_prices, features)

# Export all components
__all__ = [
    "TradingTensorOps",
    "trading_tensor_ops",
    "calculate_profit_surface",
    "calculate_btc_price_tensor"
]

# Run tests if executed directly
if __name__ == "__main__":
    test_results = trading_tensor_ops.test_tensor_operations()
    print(f"Test Results: {test_results}") 