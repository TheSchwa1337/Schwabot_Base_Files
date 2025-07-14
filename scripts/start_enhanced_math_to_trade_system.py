#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Enhanced Math-to-Trade System Startup Script
===============================================

This script initializes and starts the complete enhanced mathematical trading system
with all mathematical modules (immune, entropy, math, tensor_algebra) properly configured
for market entry/exit/hold decisions.

Features:
- Loads all mathematical systems with proper configuration
- Initializes market decision engines for entry/exit/hold patterns
- Sets up real-time market data processing
- Configures risk management and safety features
- Starts the complete math-to-trade integration system

Author: Schwabot Team
Date: 2025-01-02
"""

import asyncio
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_math_to_trade.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EnhancedMathToTradeSystem:
    """
    Complete enhanced math-to-trade system with all mathematical modules.
    
    Integrates immune system, entropy system, math system, and tensor algebra
    for comprehensive market analysis and decision making.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced math-to-trade system."""
        self.config_path = config_path or "config/enhanced_math_to_trade_config.yaml"
        self.config = self._load_configuration()
        self.logger = logging.getLogger(__name__)
        
        # Initialize all mathematical systems
        self.immune_system = None
        self.entropy_system = None
        self.math_system = None
        self.tensor_algebra_system = None
        
        # Market data and signals
        self.market_data_history: List[Dict[str, Any]] = []
        self.signal_history: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        
        # System state
        self.is_running = False
        self.is_initialized = False
        
        self.logger.info("Enhanced Math-to-Trade System created")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration from YAML file."""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'immune_system': {
                'entry_confidence_threshold': 0.7,
                'exit_confidence_threshold': 0.6,
                'hold_confidence_threshold': 0.5
            },
            'entropy_system': {
                'low_entropy_threshold': 0.3,
                'high_entropy_threshold': 0.7,
                'extreme_entropy_threshold': 0.9
            },
            'math_system': {
                'consensus_threshold': 0.6,
                'min_agreement_count': 3
            },
            'tensor_algebra': {
                'tensor_contraction_threshold': 0.7,
                'fourier_signal_threshold': 0.6
            },
            'enhanced_integration': {
                'consensus_required': True,
                'consensus_threshold': 0.6,
                'min_agreement_modules': 3
            },
            'trading': {
                'enabled': False,  # Disable for safety
                'default_pair': 'BTC/USD'
            }
        }
    
    async def initialize_systems(self):
        """Initialize all mathematical systems."""
        try:
            self.logger.info("Initializing mathematical systems...")
            
            # Initialize immune system
            from core.immune import ImmuneSystemFactory, ImmuneSystemConfig
            immune_config = ImmuneSystemConfig(**self.config.get('immune_system', {}))
            self.immune_system = ImmuneSystemFactory.create_with_params(**immune_config.__dict__)
            self.logger.info("✅ Immune system initialized")
            
            # Initialize entropy system
            from core.entropy import EntropySystemFactory, EntropySystemConfig
            entropy_config = EntropySystemConfig(**self.config.get('entropy_system', {}))
            self.entropy_system = EntropySystemFactory.create_with_params(**entropy_config.__dict__)
            self.logger.info("✅ Entropy system initialized")
            
            # Initialize math system
            from core.math import MathSystemFactory, MathSystemConfig
            math_config = MathSystemConfig(**self.config.get('math_system', {}))
            self.math_system = MathSystemFactory.create_with_params(**math_config.__dict__)
            self.logger.info("✅ Math system initialized")
            
            # Initialize tensor algebra system
            from core.math.tensor_algebra import TensorAlgebraFactory, TensorAlgebraConfig
            tensor_config = TensorAlgebraConfig(**self.config.get('tensor_algebra', {}))
            self.tensor_algebra_system = TensorAlgebraFactory.create_with_params(**tensor_config.__dict__)
            self.logger.info("✅ Tensor algebra system initialized")
            
            # Initialize enhanced integration
            from core.enhanced_math_to_trade_integration import EnhancedMathToTradeIntegration
            self.enhanced_integration = EnhancedMathToTradeIntegration(self.config)
            self.logger.info("✅ Enhanced integration initialized")
            
            self.is_initialized = True
            self.logger.info("🎉 All mathematical systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing systems: {e}")
            self.is_initialized = False
            raise
    
    async def process_market_data(self, price: float, volume: float, 
                                price_history: Optional[List[float]] = None,
                                volume_history: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Process market data through all mathematical systems.
        
        Args:
            price: Current market price
            volume: Current market volume
            price_history: Historical price data
            volume_history: Historical volume data
            
        Returns:
            Comprehensive analysis result with decisions from all systems
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Systems not initialized")
            
            # Prepare historical data
            if price_history is None:
                price_history = [price] * 20  # Default history
            if volume_history is None:
                volume_history = [volume] * 20  # Default history
            
            price_array = np.array(price_history)
            volume_array = np.array(volume_history)
            
            # Process through all mathematical systems
            results = {}
            
            # 1. Immune system analysis
            immune_signal = self.immune_system.analyze_market_signal(price, volume)
            results['immune_system'] = {
                'decision': immune_signal.market_decision.value,
                'confidence': immune_signal.confidence,
                'risk_score': immune_signal.risk_score,
                'immune_response': immune_signal.immune_response.value,
                'qsc_collapse_state': immune_signal.qsc_collapse_state.value
            }
            
            # 2. Entropy system analysis
            entropy_signal = self.entropy_system.analyze_market_entropy(
                price_array, volume_array, price, volume
            )
            results['entropy_system'] = {
                'decision': entropy_signal.decision.value,
                'confidence': entropy_signal.confidence,
                'risk_level': entropy_signal.risk_level,
                'entropy_state': entropy_signal.entropy_state.value,
                'shannon_entropy': entropy_signal.entropy_metrics.shannon_entropy,
                'drift_direction': entropy_signal.drift_direction
            }
            
            # 3. Math system analysis
            math_signal = self.math_system.analyze_market_mathematics(
                price_array, volume_array, price, volume
            )
            results['math_system'] = {
                'decision': math_signal.decision.value,
                'confidence': math_signal.confidence,
                'risk_level': math_signal.risk_level,
                'tensor_state': math_signal.tensor_state.value,
                'eigenvalue_score': math_signal.eigenvalue_score,
                'tensor_norm': math_signal.tensor_norm
            }
            
            # 4. Tensor algebra analysis
            tensor_signal = self.tensor_algebra_system.analyze_market_tensors(
                price_array, volume_array, price, volume
            )
            results['tensor_algebra'] = {
                'decision': tensor_signal.decision.value,
                'confidence': tensor_signal.confidence,
                'risk_level': tensor_signal.risk_level,
                'tensor_state': tensor_signal.tensor_state.value,
                'eigenvalue_magnitude': tensor_signal.eigenvalue_magnitude,
                'fourier_magnitude': tensor_signal.fourier_magnitude
            }
            
            # 5. Enhanced integration analysis
            enhanced_signal = await self.enhanced_integration.process_market_data_comprehensive(
                price=price, volume=volume, asset_pair="BTC/USD"
            )
            results['enhanced_integration'] = {
                'signal_type': enhanced_signal.signal_type.value if enhanced_signal else 'UNKNOWN',
                'confidence': enhanced_signal.confidence if enhanced_signal else 0.0,
                'strength': enhanced_signal.strength if enhanced_signal else 0.0,
                'mathematical_score': enhanced_signal.mathematical_score if enhanced_signal else 0.0
            }
            
            # 6. Consensus analysis
            consensus_result = self._analyze_consensus(results)
            results['consensus'] = consensus_result
            
            # 7. Final decision
            final_decision = self._make_final_decision(results)
            results['final_decision'] = final_decision
            
            # Store results
            self.signal_history.append(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return {
                'error': str(e),
                'final_decision': {
                    'action': 'WAIT',
                    'confidence': 0.0,
                    'reason': 'Error in processing'
                }
            }
    
    def _analyze_consensus(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus across all mathematical systems."""
        try:
            # Extract decisions from all systems
            decisions = []
            confidences = []
            
            for system_name, system_result in results.items():
                if system_name in ['immune_system', 'entropy_system', 'math_system', 'tensor_algebra']:
                    if 'decision' in system_result:
                        decisions.append(system_result['decision'])
                    if 'confidence' in system_result:
                        confidences.append(system_result['confidence'])
            
            # Calculate consensus metrics
            consensus_threshold = self.config.get('enhanced_integration', {}).get('consensus_threshold', 0.6)
            min_agreement = self.config.get('enhanced_integration', {}).get('min_agreement_modules', 3)
            
            # Count agreement on decisions
            decision_counts = {}
            for decision in decisions:
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Find most common decision
            if decision_counts:
                most_common_decision = max(decision_counts, key=decision_counts.get)
                agreement_count = decision_counts[most_common_decision]
                agreement_ratio = agreement_count / len(decisions)
            else:
                most_common_decision = 'WAIT'
                agreement_count = 0
                agreement_ratio = 0.0
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Determine consensus status
            has_consensus = (agreement_ratio >= consensus_threshold and 
                           agreement_count >= min_agreement)
            
            return {
                'has_consensus': has_consensus,
                'consensus_decision': most_common_decision,
                'agreement_ratio': agreement_ratio,
                'agreement_count': agreement_count,
                'total_systems': len(decisions),
                'average_confidence': avg_confidence,
                'consensus_threshold': consensus_threshold,
                'min_agreement_required': min_agreement
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing consensus: {e}")
            return {
                'has_consensus': False,
                'consensus_decision': 'WAIT',
                'agreement_ratio': 0.0,
                'agreement_count': 0,
                'total_systems': 0,
                'average_confidence': 0.0,
                'error': str(e)
            }
    
    def _make_final_decision(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Make final trading decision based on all system results."""
        try:
            consensus = results.get('consensus', {})
            
            if not consensus.get('has_consensus', False):
                return {
                    'action': 'WAIT',
                    'confidence': consensus.get('average_confidence', 0.0),
                    'reason': 'No consensus across mathematical systems'
                }
            
            consensus_decision = consensus.get('consensus_decision', 'WAIT')
            avg_confidence = consensus.get('average_confidence', 0.0)
            
            # Map consensus decision to trading action
            decision_mapping = {
                'ENTER_LONG': 'BUY',
                'ENTER_SHORT': 'SELL',
                'EXIT_LONG': 'SELL',
                'EXIT_SHORT': 'BUY',
                'HOLD': 'HOLD',
                'WAIT': 'WAIT',
                'EMERGENCY_EXIT': 'EMERGENCY_SELL'
            }
            
            action = decision_mapping.get(consensus_decision, 'WAIT')
            
            # Determine reason
            if consensus_decision in ['ENTER_LONG', 'ENTER_SHORT']:
                reason = f"Consensus entry signal from {consensus.get('agreement_count')} systems"
            elif consensus_decision in ['EXIT_LONG', 'EXIT_SHORT']:
                reason = f"Consensus exit signal from {consensus.get('agreement_count')} systems"
            elif consensus_decision == 'HOLD':
                reason = f"Consensus hold signal from {consensus.get('agreement_count')} systems"
            elif consensus_decision == 'EMERGENCY_EXIT':
                reason = f"Emergency exit signal from {consensus.get('agreement_count')} systems"
            else:
                reason = "Waiting for stronger consensus"
            
            return {
                'action': action,
                'confidence': avg_confidence,
                'reason': reason,
                'consensus_decision': consensus_decision,
                'agreement_count': consensus.get('agreement_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error making final decision: {e}")
            return {
                'action': 'WAIT',
                'confidence': 0.0,
                'reason': f'Error in decision making: {str(e)}'
            }
    
    async def run_system(self):
        """Run the enhanced math-to-trade system."""
        try:
            if not self.is_initialized:
                await self.initialize_systems()
            
            self.is_running = True
            self.logger.info("🚀 Enhanced Math-to-Trade System started")
            
            # Main system loop
            while self.is_running:
                try:
                    # Simulate market data (replace with real data feed)
                    current_price = 50000.0 + (time.time() % 1000) * 0.1
                    current_volume = 1000.0 + (time.time() % 100) * 10
                    
                    # Process market data
                    results = await self.process_market_data(
                        price=current_price,
                        volume=current_volume
                    )
                    
                    # Log results
                    final_decision = results.get('final_decision', {})
                    self.logger.info(f"📊 Market Analysis - Price: {current_price:.2f}, "
                                   f"Decision: {final_decision.get('action', 'UNKNOWN')}, "
                                   f"Confidence: {final_decision.get('confidence', 0.0):.3f}")
                    
                    # Store market data
                    self.market_data_history.append({
                        'timestamp': time.time(),
                        'price': current_price,
                        'volume': current_volume
                    })
                    
                    # Keep only recent history
                    if len(self.market_data_history) > 1000:
                        self.market_data_history = self.market_data_history[-1000:]
                    
                    # Wait before next iteration
                    await asyncio.sleep(1.0)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 System interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(5.0)  # Wait before retrying
            
        except Exception as e:
            self.logger.error(f"Error running system: {e}")
        finally:
            self.is_running = False
            self.logger.info("🛑 Enhanced Math-to-Trade System stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'signal_count': len(self.signal_history),
            'market_data_count': len(self.market_data_history),
            'immune_system_status': self.immune_system.get_system_status() if self.immune_system else None,
            'entropy_system_status': self.entropy_system.get_system_status() if self.entropy_system else None,
            'math_system_status': self.math_system.get_system_status() if self.math_system else None,
            'tensor_algebra_status': self.tensor_algebra_system.get_system_status() if self.tensor_algebra_system else None,
            'recent_decisions': [r.get('final_decision', {}) for r in self.signal_history[-10:]]
        }
    
    def stop_system(self):
        """Stop the system."""
        self.is_running = False
        self.logger.info("🛑 System stop requested")


async def main():
    """Main function to start the enhanced math-to-trade system."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Initialize and start the system
        system = EnhancedMathToTradeSystem()
        
        # Start the system
        await system.run_system()
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
    except Exception as e:
        logger.error(f"❌ System failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Import numpy for array operations
    import numpy as np
    
    # Run the system
    asyncio.run(main()) 