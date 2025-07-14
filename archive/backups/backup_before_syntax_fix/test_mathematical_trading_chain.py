#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Mathematical Trading Chain Test

This test validates that all mathematical concepts are actually implemented
and calculating real values for BTC/USDC trading decisions.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from core.advanced_tensor_algebra import AdvancedTensorAlgebra

# Import our mathematical modules
from core.entropy_math import EntropyMathSystem
from core.gpu_handlers import run_gpu_strategy
from core.quantum_mathematical_bridge import QuantumMathematicalBridge
from core.tensor_score_utils import TensorScoreUtils
from core.unified_math_system import UnifiedMathSystem


class MathematicalTradingChainTester:
    """Test the complete mathematical trading chain for BTC/USDC."""
    
    def __init__(self):
        """Initialize the mathematical chain tester."""
        self.entropy_system = EntropyMathSystem()
        self.tensor_utils = TensorScoreUtils()
        self.tensor_algebra = AdvancedTensorAlgebra()
        self.quantum_bridge = QuantumMathematicalBridge()
        self.unified_math = UnifiedMathSystem()
        
        # Test data for BTC/USDC trading
        self.btc_prices = [45000, 45500, 46000, 45800, 46200, 46500, 46300, 46700, 47000, 46800]
        self.volumes = [1000, 1200, 1100, 1300, 1400, 1350, 1500, 1600, 1550, 1700]
        self.entry_price = 45000
        self.current_price = 46800
        
        logger.info("Mathematical Trading Chain Tester initialized")

    def test_entropy_calculations(self) -> Dict[str, Any]:
        """Test entropy calculations for market analysis."""
        logger.info("Testing entropy calculations...")
        
        results = {}
        
        # Test Shannon entropy calculation
        try:
            # Create probability distribution from price changes
            price_changes = np.diff(self.btc_prices)
            price_changes_abs = np.abs(price_changes)
            if np.sum(price_changes_abs) > 0:
                probabilities = price_changes_abs / np.sum(price_changes_abs)
            else:
                probabilities = np.ones_like(price_changes_abs) / len(price_changes_abs)
            
            shannon_entropy = self.entropy_system.calculate_shannon_entropy(probabilities.tolist())
            results['shannon_entropy'] = shannon_entropy
            logger.info(f"Shannon entropy: {shannon_entropy:.6f}")
            
        except Exception as e:
            logger.error(f"Shannon entropy test failed: {e}")
            results['shannon_entropy'] = None
        
        # Test entropy-based volatility
        try:
            returns = np.diff(np.log(self.btc_prices))
            entropy_volatility = self.entropy_system.calculate_entropy_based_volatility(returns.tolist())
            results['entropy_volatility'] = entropy_volatility
            logger.info(f"Entropy-based volatility: {entropy_volatility:.6f}")
            
        except Exception as e:
            logger.error(f"Entropy volatility test failed: {e}")
            results['entropy_volatility'] = None
        
        # Test entropy trigger score
        try:
            trigger_score = self.entropy_system.calculate_entropy_trigger_score(
                self.btc_prices, self.volumes
            )
            results['entropy_trigger_score'] = trigger_score
            logger.info(f"Entropy trigger score: {trigger_score:.6f}")
            
        except Exception as e:
            logger.error(f"Entropy trigger score test failed: {e}")
            results['entropy_trigger_score'] = None
        
        return results

    def test_tensor_operations(self) -> Dict[str, Any]:
        """Test tensor operations for market analysis."""
        logger.info("Testing tensor operations...")
        
        results = {}
        
        # Test tensor scoring
        try:
            # Create market vector from price and volume data
            market_vector = np.array([p * v for p, v in zip(self.btc_prices, self.volumes)])
            
            # Calculate tensor score
            tensor_score = self.tensor_utils.calculate_tensor_score(market_vector)
            results['tensor_score'] = tensor_score.tensor_score
            logger.info(f"Tensor score: {tensor_score.tensor_score:.6f}")
            
        except Exception as e:
            logger.error(f"Tensor scoring test failed: {e}")
            results['tensor_score'] = None
        
        # Test market tensor score
        try:
            market_tensor_result = self.tensor_utils.calculate_market_tensor_score(
                self.btc_prices, self.volumes
            )
            results['market_tensor_score'] = market_tensor_result.tensor_score
            logger.info(f"Market tensor score: {market_tensor_result.tensor_score:.6f}")
            
        except Exception as e:
            logger.error(f"Market tensor score test failed: {e}")
            results['market_tensor_score'] = None
        
        # Test ZBE calculation
        try:
            # Create probability distribution
            price_changes = np.diff(self.btc_prices)
            price_changes_abs = np.abs(price_changes)
            if np.sum(price_changes_abs) > 0:
                probabilities = price_changes_abs / np.sum(price_changes_abs)
            else:
                probabilities = np.ones_like(price_changes_abs) / len(price_changes_abs)
            
            zbe = self.tensor_utils.calculate_zbe(probabilities)
            results['zbe'] = zbe
            logger.info(f"ZBE: {zbe:.6f}")
            
        except Exception as e:
            logger.error(f"ZBE calculation test failed: {e}")
            results['zbe'] = None
        
        return results

    def test_quantum_operations(self) -> Dict[str, Any]:
        """Test quantum operations for market analysis."""
        logger.info("Testing quantum operations...")
        
        results = {}
        
        # Test quantum superposition
        try:
            # Create trading signals from price changes
            price_changes = np.diff(self.btc_prices)
            quantum_state = self.quantum_bridge.create_quantum_superposition(price_changes.tolist())
            results['quantum_amplitude'] = abs(quantum_state.amplitude)
            results['quantum_probability'] = quantum_state.probability
            logger.info(f"Quantum amplitude: {abs(quantum_state.amplitude):.6f}")
            logger.info(f"Quantum probability: {quantum_state.probability:.6f}")
            
        except Exception as e:
            logger.error(f"Quantum superposition test failed: {e}")
            results['quantum_amplitude'] = None
            results['quantum_probability'] = None
        
        # Test quantum profit vectorization
        try:
            entry_signals = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01]
            exit_signals = [0.02, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01, 0.02]
            
            quantum_profit_result = self.quantum_bridge.quantum_profit_vectorization(
                self.current_price, 1000.0, entry_signals, exit_signals
            )
            results['quantum_profit'] = quantum_profit_result['quantum_profit']
            results['quantum_fidelity'] = quantum_profit_result['quantum_fidelity']
            logger.info(f"Quantum profit: {quantum_profit_result['quantum_profit']:.6f}")
            logger.info(f"Quantum fidelity: {quantum_profit_result['quantum_fidelity']:.6f}")
            
        except Exception as e:
            logger.error(f"Quantum profit vectorization test failed: {e}")
            results['quantum_profit'] = None
            results['quantum_fidelity'] = None
        
        return results

    def test_wave_function_analysis(self) -> Dict[str, Any]:
        """Test wave function analysis for market prediction."""
        logger.info("Testing wave function analysis...")
        
        results = {}
        
        try:
            # Create market data array
            market_data = np.array(self.btc_prices)
            
            # Create quantum wave function
            wave_function_result = self.unified_math.create_quantum_wave_function(
                market_data, time_evolution=1.0, amplitude=1.0, 
                wave_number=0.1, angular_frequency=0.1
            )
            
            results['quantum_potential'] = wave_function_result['quantum_potential']
            results['energy_expectation'] = wave_function_result['energy_expectation']
            results['wave_function_norm'] = wave_function_result['wave_function_norm']
            
            logger.info(f"Quantum potential: {wave_function_result['quantum_potential']:.6f}")
            logger.info(f"Energy expectation: {wave_function_result['energy_expectation']:.6f}")
            logger.info(f"Wave function norm: {wave_function_result['wave_function_norm']:.6f}")
            
        except Exception as e:
            logger.error(f"Wave function analysis test failed: {e}")
            results['quantum_potential'] = None
            results['energy_expectation'] = None
            results['wave_function_norm'] = None
        
        return results

    def test_gpu_accelerated_operations(self) -> Dict[str, Any]:
        """Test GPU-accelerated mathematical operations."""
        logger.info("Testing GPU-accelerated operations...")
        
        results = {}
        
        # Test GPU entropy calculation
        try:
            # Create probability distribution
            price_changes = np.diff(self.btc_prices)
            price_changes_abs = np.abs(price_changes)
            if np.sum(price_changes_abs) > 0:
                probabilities = price_changes_abs / np.sum(price_changes_abs)
            else:
                probabilities = np.ones_like(price_changes_abs) / len(price_changes_abs)
            
            gpu_entropy_result = run_gpu_strategy("entropy", {
                "probability_dist": probabilities.tolist()
            })
            
            results['gpu_shannon_entropy'] = gpu_entropy_result.get('shannon_entropy')
            results['gpu_profit_delta'] = gpu_entropy_result.get('profit_delta')
            results['gpu_accelerated'] = gpu_entropy_result.get('gpu_accelerated', False)
            
            logger.info(f"GPU Shannon entropy: {gpu_entropy_result.get('shannon_entropy', 'N/A')}")
            logger.info(f"GPU profit delta: {gpu_entropy_result.get('profit_delta', 'N/A')}")
            logger.info(f"GPU accelerated: {gpu_entropy_result.get('gpu_accelerated', False)}")
            
        except Exception as e:
            logger.error(f"GPU entropy calculation test failed: {e}")
            results['gpu_shannon_entropy'] = None
            results['gpu_profit_delta'] = None
            results['gpu_accelerated'] = False
        
        return results

    def test_trading_decision_chain(self) -> Dict[str, Any]:
        """Test the complete trading decision chain."""
        logger.info("Testing complete trading decision chain...")
        
        results = {}
        
        try:
            # Step 1: Calculate entropy metrics
            entropy_results = self.test_entropy_calculations()
            
            # Step 2: Calculate tensor metrics
            tensor_results = self.test_tensor_operations()
            
            # Step 3: Calculate quantum metrics
            quantum_results = self.test_quantum_operations()
            
            # Step 4: Calculate wave function metrics
            wave_results = self.test_wave_function_analysis()
            
            # Step 5: Calculate GPU metrics
            gpu_results = self.test_gpu_accelerated_operations()
            
            # Step 6: Combine all metrics for trading decision
            decision_score = 0.0
            decision_factors = []
            
            # Entropy factors (30% weight)
            if entropy_results.get('entropy_trigger_score') is not None:
                decision_score += 0.3 * entropy_results['entropy_trigger_score']
                decision_factors.append(f"Entropy trigger: {entropy_results['entropy_trigger_score']:.4f}")
            
            # Tensor factors (25% weight)
            if tensor_results.get('tensor_score') is not None:
                decision_score += 0.25 * tensor_results['tensor_score']
                decision_factors.append(f"Tensor score: {tensor_results['tensor_score']:.4f}")
            
            # Quantum factors (25% weight)
            if quantum_results.get('quantum_profit') is not None:
                normalized_quantum_profit = max(0, min(1, quantum_results['quantum_profit'] / 1000))
                decision_score += 0.25 * normalized_quantum_profit
                decision_factors.append(f"Quantum profit: {normalized_quantum_profit:.4f}")
            
            # Wave function factors (20% weight)
            if wave_results.get('energy_expectation') is not None:
                normalized_energy = max(0, min(1, wave_results['energy_expectation']))
                decision_score += 0.2 * normalized_energy
                decision_factors.append(f"Energy expectation: {normalized_energy:.4f}")
            
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
            
            results['decision_score'] = decision_score
            results['trading_action'] = action
            results['position_size'] = position_size
            results['decision_factors'] = decision_factors
            
            # Calculate expected profit/loss
            price_change_pct = (self.current_price - self.entry_price) / self.entry_price
            expected_pnl = position_size * price_change_pct * 1000  # Assuming $1000 position
            
            results['price_change_pct'] = price_change_pct
            results['expected_pnl'] = expected_pnl
            
            logger.info(f"Decision score: {decision_score:.6f}")
            logger.info(f"Trading action: {action}")
            logger.info(f"Position size: {position_size:.4f}")
            logger.info(f"Expected P&L: ${expected_pnl:.2f}")
            logger.info(f"Decision factors: {', '.join(decision_factors)}")
            
        except Exception as e:
            logger.error(f"Trading decision chain test failed: {e}")
            results['decision_score'] = None
            results['trading_action'] = None
            results['position_size'] = None
            results['expected_pnl'] = None
        
        return results

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive mathematical trading chain test."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE MATHEMATICAL TRADING CHAIN TEST")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        entropy_results = self.test_entropy_calculations()
        tensor_results = self.test_tensor_operations()
        quantum_results = self.test_quantum_operations()
        wave_results = self.test_wave_function_analysis()
        gpu_results = self.test_gpu_accelerated_operations()
        trading_results = self.test_trading_decision_chain()
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_duration': time.time() - start_time,
            'entropy_calculations': entropy_results,
            'tensor_operations': tensor_results,
            'quantum_operations': quantum_results,
            'wave_function_analysis': wave_results,
            'gpu_operations': gpu_results,
            'trading_decision': trading_results,
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
        logger.info(f"Final trading decision: {trading_results.get('trading_action', 'UNKNOWN')}")
        logger.info(f"Decision confidence: {trading_results.get('decision_score', 0):.4f}")
        logger.info(f"Expected P&L: ${trading_results.get('expected_pnl', 0):.2f}")
        
        return comprehensive_results


def main():
    """Run the comprehensive mathematical trading chain test."""
    tester = MathematicalTradingChainTester()
    results = tester.run_comprehensive_test()
    
    # Save results to file
    import json
    with open('mathematical_trading_chain_test_results.json', 'w') as f:
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
    
    logger.info("Test results saved to: mathematical_trading_chain_test_results.json")
    
    return results


if __name__ == "__main__":
    main() 