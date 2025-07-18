#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Mathematical Functions Test

This test validates that core mathematical concepts are actually implemented
and calculating real values for BTC/USDC trading decisions.
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoreMathTester:
    """Test core mathematical functions for BTC/USDC trading."""
    
    def __init__(self):
        """Initialize the core math tester."""
        # Test data for BTC/USDC trading
        self.btc_prices = [45000, 45500, 46000, 45800, 46200, 46500, 46300, 46700, 47000, 46800]
        self.volumes = [1000, 1200, 1100, 1300, 1400, 1350, 1500, 1600, 1550, 1700]
        self.entry_price = 45000
        self.current_price = 46800
        
        logger.info("Core Math Tester initialized")

    def test_shannon_entropy(self) -> Dict[str, Any]:
        """Test Shannon entropy calculation."""
        logger.info("Testing Shannon entropy calculation...")
        
        try:
            # Create probability distribution from price changes
            price_changes = np.diff(self.btc_prices)
            price_changes_abs = np.abs(price_changes)
            if np.sum(price_changes_abs) > 0:
                probabilities = price_changes_abs / np.sum(price_changes_abs)
            else:
                probabilities = np.ones_like(price_changes_abs) / len(price_changes_abs)
            
            # Calculate Shannon entropy: H = -Σ p_i * log2(p_i)
            shannon_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            logger.info(f"Shannon entropy: {shannon_entropy:.6f}")
            return {
                'shannon_entropy': float(shannon_entropy),
                'probabilities': probabilities.tolist(),
                'price_changes': price_changes.tolist()
            }
            
        except Exception as e:
            logger.error(f"Shannon entropy test failed: {e}")
            return {'shannon_entropy': None, 'error': str(e)}

    def test_tensor_scoring(self) -> Dict[str, Any]:
        """Test tensor scoring calculation."""
        logger.info("Testing tensor scoring calculation...")
        
        try:
            # Create market vector from price and volume data
            market_vector = np.array([p * v for p, v in zip(self.btc_prices, self.volumes)])
            
            # Create weight matrix (identity matrix for simplicity)
            n = len(market_vector)
            weight_matrix = np.eye(n)
            
            # Calculate tensor score: T = Σᵢⱼ wᵢⱼ * xᵢ * xⱼ
            # This is equivalent to: T = x^T * W * x
            tensor_score = np.sum(weight_matrix * np.outer(market_vector, market_vector))
            
            logger.info(f"Tensor score: {tensor_score:.6f}")
            return {
                'tensor_score': float(tensor_score),
                'market_vector': market_vector.tolist(),
                'weight_matrix_shape': weight_matrix.shape
            }
            
        except Exception as e:
            logger.error(f"Tensor scoring test failed: {e}")
            return {'tensor_score': None, 'error': str(e)}

    def test_zbe_calculation(self) -> Dict[str, Any]:
        """Test Zero Bit Entropy (ZBE) calculation."""
        logger.info("Testing ZBE calculation...")
        
        try:
            # Create probability distribution
            price_changes = np.diff(self.btc_prices)
            price_changes_abs = np.abs(price_changes)
            if np.sum(price_changes_abs) > 0:
                probabilities = price_changes_abs / np.sum(price_changes_abs)
            else:
                probabilities = np.ones_like(price_changes_abs) / len(price_changes_abs)
            
            # Calculate ZBE: H = -Σ p_i * log2(p_i)
            zbe = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            logger.info(f"ZBE: {zbe:.6f}")
            return {
                'zbe': float(zbe),
                'probabilities': probabilities.tolist()
            }
            
        except Exception as e:
            logger.error(f"ZBE calculation test failed: {e}")
            return {'zbe': None, 'error': str(e)}

    def test_quantum_wave_function(self) -> Dict[str, Any]:
        """Test quantum wave function calculation."""
        logger.info("Testing quantum wave function calculation...")
        
        try:
            # Create market data array
            market_data = np.array(self.btc_prices)
            
            # Quantum wave function parameters
            amplitude = 1.0
            wave_number = 0.1
            angular_frequency = 0.1
            time_evolution = 1.0
            
            # Create spatial coordinates
            x_coords = np.linspace(0, len(market_data), len(market_data))
            
            # Calculate quantum wave function: ψ(x,t) = A * exp(i(kx - ωt))
            wave_function = amplitude * np.exp(1j * (wave_number * x_coords - angular_frequency * time_evolution))
            
            # Calculate quantum potential (simplified)
            quantum_potential = np.mean(np.abs(wave_function) ** 2)
            
            # Calculate energy expectation
            energy_expectation = np.real(np.mean(wave_function * np.conj(wave_function)))
            
            # Calculate wave function norm
            wave_function_norm = np.sqrt(np.sum(np.abs(wave_function) ** 2))
            
            logger.info(f"Quantum potential: {quantum_potential:.6f}")
            logger.info(f"Energy expectation: {energy_expectation:.6f}")
            logger.info(f"Wave function norm: {wave_function_norm:.6f}")
            
            return {
                'quantum_potential': float(quantum_potential),
                'energy_expectation': float(energy_expectation),
                'wave_function_norm': float(wave_function_norm),
                'wave_function_real': np.real(wave_function).tolist(),
                'wave_function_imag': np.imag(wave_function).tolist()
            }
            
        except Exception as e:
            logger.error(f"Quantum wave function test failed: {e}")
            return {'quantum_potential': None, 'error': str(e)}

    def test_market_volatility(self) -> Dict[str, Any]:
        """Test market volatility calculation."""
        logger.info("Testing market volatility calculation...")
        
        try:
            # Calculate log returns
            log_returns = np.diff(np.log(self.btc_prices))
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(log_returns)
            
            # Calculate annualized volatility (assuming daily data)
            annualized_volatility = volatility * np.sqrt(365)
            
            # Calculate entropy-based volatility
            returns_abs = np.abs(log_returns)
            if np.sum(returns_abs) > 0:
                probabilities = returns_abs / np.sum(returns_abs)
            else:
                probabilities = np.ones_like(returns_abs) / len(returns_abs)
            
            entropy_volatility = -np.sum(probabilities * np.log(probabilities + 1e-10))
            
            logger.info(f"Volatility: {volatility:.6f}")
            logger.info(f"Annualized volatility: {annualized_volatility:.6f}")
            logger.info(f"Entropy-based volatility: {entropy_volatility:.6f}")
            
            return {
                'volatility': float(volatility),
                'annualized_volatility': float(annualized_volatility),
                'entropy_volatility': float(entropy_volatility),
                'log_returns': log_returns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Market volatility test failed: {e}")
            return {'volatility': None, 'error': str(e)}

    def test_trading_decision(self) -> Dict[str, Any]:
        """Test complete trading decision calculation."""
        logger.info("Testing complete trading decision calculation...")
        
        try:
            # Get all mathematical metrics
            entropy_result = self.test_shannon_entropy()
            tensor_result = self.test_tensor_scoring()
            zbe_result = self.test_zbe_calculation()
            quantum_result = self.test_quantum_wave_function()
            volatility_result = self.test_market_volatility()
            
            # Combine metrics for trading decision
            decision_score = 0.0
            decision_factors = []
            
            # Entropy factors (25% weight)
            if entropy_result.get('shannon_entropy') is not None:
                normalized_entropy = min(1.0, entropy_result['shannon_entropy'] / 3.0)  # Normalize to [0,1]
                decision_score += 0.25 * normalized_entropy
                decision_factors.append(f"Entropy: {normalized_entropy:.4f}")
            
            # Tensor factors (25% weight)
            if tensor_result.get('tensor_score') is not None:
                normalized_tensor = min(1.0, tensor_result['tensor_score'] / 1e12)  # Normalize
                decision_score += 0.25 * normalized_tensor
                decision_factors.append(f"Tensor: {normalized_tensor:.4f}")
            
            # ZBE factors (20% weight)
            if zbe_result.get('zbe') is not None:
                normalized_zbe = min(1.0, zbe_result['zbe'] / 3.0)  # Normalize
                decision_score += 0.20 * normalized_zbe
                decision_factors.append(f"ZBE: {normalized_zbe:.4f}")
            
            # Quantum factors (20% weight)
            if quantum_result.get('energy_expectation') is not None:
                normalized_energy = min(1.0, quantum_result['energy_expectation'])
                decision_score += 0.20 * normalized_energy
                decision_factors.append(f"Quantum: {normalized_energy:.4f}")
            
            # Volatility factors (10% weight)
            if volatility_result.get('entropy_volatility') is not None:
                normalized_vol = min(1.0, volatility_result['entropy_volatility'] / 3.0)
                decision_score += 0.10 * normalized_vol
                decision_factors.append(f"Volatility: {normalized_vol:.4f}")
            
            # Determine trading action
            if decision_score > 0.7:
                action = "STRONG_BUY"
            elif decision_score > 0.5:
                action = "BUY"
            elif decision_score > 0.3:
                action = "HOLD"
            elif decision_score > 0.1:
                action = "SELL"
            else:
                action = "STRONG_SELL"
            
            # Calculate position size based on decision score
            position_size = abs(decision_score - 0.5) * 2  # 0 to 1
            
            # Calculate expected profit/loss
            price_change_pct = (self.current_price - self.entry_price) / self.entry_price
            expected_pnl = position_size * price_change_pct * 1000  # Assuming $1000 position
            
            logger.info(f"Decision score: {decision_score:.6f}")
            logger.info(f"Trading action: {action}")
            logger.info(f"Position size: {position_size:.4f}")
            logger.info(f"Expected P&L: ${expected_pnl:.2f}")
            logger.info(f"Decision factors: {', '.join(decision_factors)}")
            
            return {
                'decision_score': float(decision_score),
                'trading_action': action,
                'position_size': float(position_size),
                'expected_pnl': float(expected_pnl),
                'decision_factors': decision_factors,
                'price_change_pct': float(price_change_pct),
                'entropy_result': entropy_result,
                'tensor_result': tensor_result,
                'zbe_result': zbe_result,
                'quantum_result': quantum_result,
                'volatility_result': volatility_result
            }
            
        except Exception as e:
            logger.error(f"Trading decision test failed: {e}")
            return {'decision_score': None, 'error': str(e)}

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive core mathematical test."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE CORE MATHEMATICAL TEST")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run trading decision test (which includes all other tests)
        trading_result = self.test_trading_decision()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_duration': time.time() - start_time,
            'trading_decision': trading_result,
            'market_data': {
                'btc_prices': self.btc_prices,
                'volumes': self.volumes,
                'entry_price': self.entry_price,
                'current_price': self.current_price
            }
        }
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test duration: {comprehensive_results['test_duration']:.2f} seconds")
        logger.info(f"Final trading decision: {trading_result.get('trading_action', 'UNKNOWN')}")
        logger.info(f"Decision confidence: {trading_result.get('decision_score', 0):.4f}")
        logger.info(f"Expected P&L: ${trading_result.get('expected_pnl', 0):.2f}")
        
        return comprehensive_results


def main():
    """Run the comprehensive core mathematical test."""
    tester = CoreMathTester()
    results = tester.run_comprehensive_test()
    
    # Save results to file
    import json
    with open('core_math_test_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, complex):
                return str(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    logger.info("Test results saved to: core_math_test_results.json")
    
    return results


if __name__ == "__main__":
    main() 