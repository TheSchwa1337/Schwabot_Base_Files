#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Math Integration Bridge for Schwabot
====================================
Connects the mathematical logic engine with all existing Schwabot systems.
Provides unified interface for mathematical operations across the trading system.
"""

import logging


import logging


import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import core mathematical functions
from core.math_logic_engine import (
    bitmap_fold,
    clonal_expansion_coefficient,
    drift_chain_weight,
    echo_trigger_zone,
    entropy_drift,
    hash_priority_score,
    mutation_rate,
    orbital_energy,
    phase_rotation,
    rebuy_probability,
    should_enter,
    should_exit,
    sigmoid,
    strategy_hash_evolution,
    vault_mass,
    vault_reentry_delay,
)

logger = logging.getLogger(__name__)


class IntegrationMode(Enum):
    """Integration modes for mathematical operations."""
    STRATEGY_BIT_MAPPER = "strategy_bit_mapper"
    TCELL_SURVIVAL = "tcell_survival"
    VAULT_ORBITAL_BRIDGE = "vault_orbital_bridge"
    ENTROPY_DECAY = "entropy_decay"
    SYMBOLIC_REGISTRY = "symbolic_registry"
    UNIFIED_PIPELINE = "unified_pipeline"


@dataclass
class MathIntegrationResult:
    """Result of mathematical integration operation."""
    success: bool
    operation: str
    result: Any
    confidence: float
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MathIntegrationBridge:
    """Bridge between mathematical logic engine and Schwabot systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize math integration bridge."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.operation_stats: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        
        # Integration state
        self.active_integrations: Dict[str, bool] = {}
        self.last_operation_time: Dict[str, float] = {}
        
        self.logger.info("✅ Math Integration Bridge initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'enable_strategy_integration': True,
            'enable_tcell_integration': True,
            'enable_vault_integration': True,
            'enable_entropy_integration': True,
            'enable_symbolic_integration': True,
            'enable_pipeline_integration': True,
            'max_execution_time': 1.0,  # seconds
            'confidence_threshold': 0.7,
            'error_retry_count': 3,
            'enable_performance_tracking': True,
            'validation_enabled': True
        }
    
    def integrate_with_strategy_bit_mapper(self, asset: str, market_data: Dict[str, Any],
                                         strategy_params: Dict[str, Any]) -> MathIntegrationResult:
        """Integrate mathematical logic with strategy bit mapper."""
        start_time = time.time()
        
        try:
            # Extract required data
            psi = np.array(market_data.get('price_history', []))
            phi = np.array(market_data.get('volume_history', []))
            xi = np.array(market_data.get('entropy_history', []))
            
            if len(psi) == 0 or len(phi) == 0 or len(xi) == 0:
                return MathIntegrationResult(
                    success=False,
                    operation="strategy_bit_mapper_integration",
                    result=None,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    error_message="Insufficient market data"
                )
            
            # Calculate entropy drift
            drift_value = entropy_drift(psi, phi, xi, n=8)
            
            # Calculate orbital energy
            omega_mean = np.mean(psi[-8:]) if len(psi) >= 8 else np.mean(psi)
            phi_mean = np.mean(phi[-8:]) if len(phi) >= 8 else np.mean(phi)
            xi_mean = np.mean(xi[-8:]) if len(xi) >= 8 else np.mean(xi)
            
            energy, orbital_state = orbital_energy(omega_mean, phi_mean, xi_mean)
            
            # Calculate entry/exit signals
            tcell_activation = strategy_params.get('tcell_activation', 0.5)
            clonal_coeff = strategy_params.get('clonal_coefficient', 0.5)
            rebuy_prob = rebuy_probability(omega_mean, xi_mean, phi_mean, 0.1)
            echo_zone = echo_trigger_zone(xi_mean, phi_mean, omega_mean, omega_mean)
            
            should_enter_signal = should_enter(tcell_activation, clonal_coeff, rebuy_prob, echo_zone)
            should_exit_signal = should_exit(tcell_activation, clonal_coeff, rebuy_prob, echo_zone)
            
            result = {
                'drift_value': drift_value,
                'orbital_energy': energy,
                'orbital_state': orbital_state,
                'should_enter': should_enter_signal,
                'should_exit': should_exit_signal,
                'rebuy_probability': rebuy_prob,
                'echo_trigger_zone': echo_zone
            }
            
            confidence = min(1.0, (abs(drift_value) + energy + rebuy_prob) / 3.0)
            
            return MathIntegrationResult(
                success=True,
                operation="strategy_bit_mapper_integration",
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={'asset': asset, 'orbital_state': orbital_state}
            )
            
        except Exception as e:
            self.logger.error(f"Error in strategy bit mapper integration: {e}")
            return MathIntegrationResult(
                success=False,
                operation="strategy_bit_mapper_integration",
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def integrate_with_tcell_survival(self, tcell_data: Dict[str, Any],
                                    market_conditions: Dict[str, Any]) -> MathIntegrationResult:
        """Integrate mathematical logic with T-cell survival engine."""
        start_time = time.time()
        
        try:
            # Extract T-cell parameters
            strategy_hash = tcell_data.get('strategy_hash', '')
            survival_score = tcell_data.get('survival_score', 0.5)
            roi = tcell_data.get('roi', 0.0)
            
            # Calculate clonal expansion coefficient
            tcell_activation = tcell_data.get('activation', 0.5)
            xi_weight = tcell_data.get('xi_weight', 0.3)
            clonal_coeff = clonal_expansion_coefficient(tcell_activation, roi, xi_weight)
            
            # Calculate mutation rate
            phi = market_conditions.get('volatility', 0.1)
            volatility = market_conditions.get('market_volatility', 0.1)
            mutation_rate_val = mutation_rate(roi, phi, volatility)
            
            # Evolve strategy hash
            delta_roi = tcell_data.get('delta_roi', 0.0)
            entropy_deviation = market_conditions.get('entropy_deviation', 0.0)
            new_strategy_hash = strategy_hash_evolution(strategy_hash, delta_roi, entropy_deviation)
            
            # Calculate hash priority score
            asset_drift_alignment = market_conditions.get('asset_drift_alignment', 0.5)
            hps = hash_priority_score(roi, clonal_coeff, xi_weight, asset_drift_alignment)
            
            result = {
                'clonal_expansion_coefficient': clonal_coeff,
                'mutation_rate': mutation_rate_val,
                'new_strategy_hash': new_strategy_hash,
                'hash_priority_score': hps,
                'survival_improvement': clonal_coeff - survival_score
            }
            
            confidence = min(1.0, (clonal_coeff + hps + (1 - mutation_rate_val)) / 3.0)
            
            return MathIntegrationResult(
                success=True,
                operation="tcell_survival_integration",
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={'strategy_hash': strategy_hash[:16]}
            )
            
        except Exception as e:
            self.logger.error(f"Error in T-cell survival integration: {e}")
            return MathIntegrationResult(
                success=False,
                operation="tcell_survival_integration",
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def integrate_with_vault_orbital_bridge(self, vault_data: Dict[str, Any],
                                          orbital_params: Dict[str, Any]) -> MathIntegrationResult:
        """Integrate mathematical logic with vault orbital bridge."""
        start_time = time.time()
        
        try:
            # Extract vault parameters
            xi_values = vault_data.get('xi_values', [0.5])
            phi_values = vault_data.get('phi_values', [0.5])
            roi_values = vault_data.get('roi_values', [0.0])
            holding_weights = vault_data.get('holding_weights', [1.0])
            
            # Calculate vault mass
            vault_mass_val = vault_mass(xi_values, phi_values, roi_values, holding_weights)
            
            # Calculate vault re-entry delay
            xi_exit = vault_data.get('xi_exit', 0.5)
            phi_entry = vault_data.get('phi_entry', 0.5)
            tick_entropy = orbital_params.get('tick_entropy', 1.0)
            reentry_delay = vault_reentry_delay(xi_exit, phi_entry, vault_mass_val, tick_entropy)
            
            # Calculate phase rotation
            xi = orbital_params.get('xi', 0.5)
            phi = orbital_params.get('phi', 0.5)
            omega = orbital_params.get('omega', 0.5)
            phase_rot = phase_rotation(xi, phi, omega, period=16)
            
            # Calculate orbital energy
            energy, orbital_state = orbital_energy(omega, phi, xi)
            
            result = {
                'vault_mass': vault_mass_val,
                'reentry_delay': reentry_delay,
                'phase_rotation': phase_rot,
                'orbital_energy': energy,
                'orbital_state': orbital_state,
                'vault_pressure': vault_mass_val / max(len(xi_values), 1)
            }
            
            confidence = min(1.0, (energy + (1.0 / (reentry_delay + 1)) + (1 - vault_mass_val)) / 3.0)
            
            return MathIntegrationResult(
                success=True,
                operation="vault_orbital_bridge_integration",
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={'orbital_state': orbital_state}
            )
            
        except Exception as e:
            self.logger.error(f"Error in vault orbital bridge integration: {e}")
            return MathIntegrationResult(
                success=False,
                operation="vault_orbital_bridge_integration",
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def integrate_with_entropy_decay(self, entropy_data: Dict[str, Any],
                                   time_params: Dict[str, Any]) -> MathIntegrationResult:
        """Integrate mathematical logic with entropy decay system."""
        start_time = time.time()
        
        try:
            # Extract entropy parameters
            psi = np.array(entropy_data.get('psi_history', []))
            phi = np.array(entropy_data.get('phi_history', []))
            xi = np.array(entropy_data.get('xi_history', []))
            
            if len(psi) == 0 or len(phi) == 0 or len(xi) == 0:
                return MathIntegrationResult(
                    success=False,
                    operation="entropy_decay_integration",
                    result=None,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    error_message="Insufficient entropy data"
                )
            
            # Calculate entropy drift with different window sizes
            drift_short = entropy_drift(psi, phi, xi, n=4)
            drift_medium = entropy_drift(psi, phi, xi, n=8)
            drift_long = entropy_drift(psi, phi, xi, n=16)
            
            # Calculate bitmap folding for memory compression
            bitmap_data = entropy_data.get('bitmap_history', [0, 0, 0])
            folded_bitmap = bitmap_fold(bitmap_data, k=3)
            
            # Calculate cross-asset drift if available
            omega_a = np.array(entropy_data.get('omega_a', []))
            omega_b = np.array(entropy_data.get('omega_b', []))
            cross_asset_drift = 0.0
            
            if len(omega_a) > 0 and len(omega_b) > 0:
                delta_t = time_params.get('delta_t', 0)
                roi_weight = time_params.get('roi_weight', 0.5)
                xi_score = time_params.get('xi_score', 0.5)
                cross_asset_drift = drift_chain_weight(omega_a, omega_b, delta_t, roi_weight, xi_score)
            
            result = {
                'drift_short': drift_short,
                'drift_medium': drift_medium,
                'drift_long': drift_long,
                'folded_bitmap': folded_bitmap,
                'cross_asset_drift': cross_asset_drift,
                'drift_trend': 'increasing' if drift_long > drift_short else 'decreasing'
            }
            
            confidence = min(1.0, (abs(drift_medium) + abs(cross_asset_drift) + 0.5) / 2.5)
            
            return MathIntegrationResult(
                success=True,
                operation="entropy_decay_integration",
                result=result,
                confidence=confidence,
                execution_time=time.time() - start_time,
                metadata={'drift_trend': result['drift_trend']}
            )
            
        except Exception as e:
            self.logger.error(f"Error in entropy decay integration: {e}")
            return MathIntegrationResult(
                success=False,
                operation="entropy_decay_integration",
                result=None,
                confidence=0.0,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrations."""
        return {
            'active_integrations': self.active_integrations,
            'operation_stats': {k: len(v) for k, v in self.operation_stats.items()},
            'error_counts': self.error_counts,
            'last_operation_times': self.last_operation_time,
            'config': self.config
        }
    
    def validate_mathematical_operations(self) -> Dict[str, bool]:
        """Validate all mathematical operations."""
        validation_results = {}
        
        try:
            # Test entropy drift
            test_psi = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            test_phi = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
            test_xi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            
            drift_result = entropy_drift(test_psi, test_phi, test_xi, n=3)
            validation_results['entropy_drift'] = isinstance(drift_result, float)
            
            # Test orbital energy
            energy, state = orbital_energy(0.5, 0.5, 0.5)
            validation_results['orbital_energy'] = isinstance(energy, float) and state in ['s', 'p', 'd', 'f']
            
            # Test vault mass
            mass_result = vault_mass([0.5], [0.5], [0.1], [1.0])
            validation_results['vault_mass'] = isinstance(mass_result, float)
            
            # Test bitmap folding
            fold_result = bitmap_fold([1, 2, 3], k=2)
            validation_results['bitmap_fold'] = isinstance(fold_result, int)
            
            # Test sigmoid
            sigmoid_result = sigmoid(0.0)
            validation_results['sigmoid'] = isinstance(sigmoid_result, float) and 0 <= sigmoid_result <= 1
            
        except Exception as e:
            self.logger.error(f"Error in mathematical validation: {e}")
            validation_results['validation_error'] = False
        
        return validation_results


# Factory function
def create_math_integration_bridge(config: Optional[Dict[str, Any]] = None) -> MathIntegrationBridge:
    """Create a math integration bridge instance."""
    return MathIntegrationBridge(config) 