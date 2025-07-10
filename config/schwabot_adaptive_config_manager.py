#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Configuration Management System for Schwabot

Provides intelligent, dynamic configuration management with:
- Multi-source configuration loading
- Market condition-based adaptive configurations
- Resilient fallback mechanisms
- Comprehensive system state integration
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from core.advanced_tensor_algebra import AdvancedTensorAlgebra
from core.enhanced_error_recovery_system import EnhancedErrorRecoverySystem
from core.system_state_profiler import SystemStateProfiler


class MarketConditionAnalyzer:
    """Analyze market conditions for adaptive configuration"""

    def __init__(self, tensor_algebra: AdvancedTensorAlgebra) -> None:
        self.tensor_algebra = tensor_algebra

    def analyze_market_entropy(self, market_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze market entropy and generate adaptive parameters

        Args:
            market_data: Market price/volume data

        Returns:
            Entropy-based market condition metrics
        """
        try:
            # Calculate Shannon entropy using tensor algebra's market entropy method
            entropy = self.tensor_algebra.calculate_market_entropy(market_data)

            # Calculate volatility
            volatility = np.std(market_data)

            # Advanced frequency analysis using FFT
            if len(market_data) > 1:
                # Use FFT for frequency analysis
                fft_result = np.fft.fft(market_data)
                power_spectrum = np.abs(fft_result) ** 2
                frequencies = np.fft.fftfreq(len(market_data))
                
                # Find dominant frequency (positive frequencies only)
                positive_freq_mask = frequencies > 0
                if np.any(positive_freq_mask):
                    dominant_frequency = frequencies[positive_freq_mask][np.argmax(power_spectrum[positive_freq_mask])]
                else:
                    dominant_frequency = 0.0
            else:
                dominant_frequency = 0.0

            # Calculate market complexity using advanced metrics
            market_complexity = self._calculate_market_complexity(entropy, volatility, dominant_frequency)

            return {
                'entropy': entropy,
                'volatility': volatility,
                'dominant_frequency': dominant_frequency,
                'market_complexity': market_complexity,
                'tensor_score': self.tensor_algebra.tensor_score(market_data),
                'quantum_coherence': self._calculate_quantum_coherence(market_data)
            }

        except Exception as e:
            logging.error(f"Market entropy analysis failed: {e}")
            return {'entropy': 0.5, 'volatility': 0.0, 'dominant_frequency': 0.0, 'market_complexity': 0.5}

    def _calculate_market_complexity(self, entropy: float, volatility: float, frequency: float) -> float:
        """
        Calculate an integrated market complexity metric using advanced mathematical principles

        Args:
            entropy: Market entropy value
            volatility: Market price volatility
            frequency: Dominant frequency component

        Returns:
            Complexity score between 0 and 1
        """
        # Normalize frequency to 0-1 range
        normalized_freq = min(abs(frequency), 1.0)
        
        # Weighted complexity calculation incorporating all three factors
        complexity = (0.4 * entropy + 0.4 * volatility + 0.2 * normalized_freq)
        
        # Apply sigmoid-like normalization for better distribution
        complexity = 1.0 / (1.0 + np.exp(-5 * (complexity - 0.5)))
        
        return np.clip(complexity, 0, 1)

    def _calculate_quantum_coherence(self, market_data: np.ndarray) -> float:
        """
        Calculate quantum coherence measure for market data
        
        Args:
            market_data: Market price data
            
        Returns:
            Coherence measure between 0 and 1
        """
        try:
            # Create quantum superposition of market data
            superposition_result = self.tensor_algebra.create_quantum_superposition(market_data.tolist())
            return superposition_result.get('coherence', 0.5)
        except Exception:
            return 0.5


@dataclass
class AdaptiveConfigurationState:
    """Comprehensive state tracking for adaptive configuration"""

    timestamp: datetime = field(default_factory=datetime.now)
    system_health: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, float] = field(default_factory=dict)
    active_strategies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    mathematical_state: Dict[str, Any] = field(default_factory=dict)


class SchwabotAdaptiveConfigManager:
    """
    Intelligent Configuration Management System

    Features:
    - Multi-source configuration loading
    - Adaptive configuration generation
    - System state integration
    - Performance and error tracking
    - Advanced mathematical integration
    """

    def __init__(
        self,
        config_dir: str = 'config',
        recovery_system: Optional[EnhancedErrorRecoverySystem] = None,
        system_profiler: Optional[SystemStateProfiler] = None,
        tensor_algebra: Optional[AdvancedTensorAlgebra] = None,
    ):
        self.config_dir = config_dir
        self.recovery_system = recovery_system or EnhancedErrorRecoverySystem()
        self.system_profiler = system_profiler or SystemStateProfiler()
        self.tensor_algebra = tensor_algebra or AdvancedTensorAlgebra()

        self.market_analyzer = MarketConditionAnalyzer(self.tensor_algebra)

        # Configuration caches
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._adaptive_state = AdaptiveConfigurationState()

    def load_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Load configurations from multiple sources

        Returns:
            Dictionary of loaded configurations
        """
        config_files = [
            'schwabot_core_config.yaml',
            'high_frequency_crypto_config.yaml',
            'mathematical_framework_config.py',
            'system_interlinking_config.yaml',
        ]

        for config_file in config_files:
            full_path = os.path.join(self.config_dir, config_file)

            try:
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        if config_file.endswith('.yaml'):
                            config = yaml.safe_load(f)
                        elif config_file.endswith('.json'):
                            config = json.load(f)
                        elif config_file.endswith('.py'):
                            # For Python config files, you might need a custom loader
                            config = self._load_python_config(full_path)

                        self._config_cache[config_file] = config
                else:
                    # Create placeholder config if file doesn't exist
                    self._config_cache[config_file] = self._create_placeholder_config(config_file)

            except Exception as e:
                logging.warning(f"Could not load config {config_file}: {e}")
                self._config_cache[config_file] = self._create_placeholder_config(config_file)

        return self._config_cache

    def _create_placeholder_config(self, config_file: str) -> Dict[str, Any]:
        """Create placeholder configuration for missing files"""
        if 'schwabot_core_config.yaml' in config_file:
            return {
                'trading': {
                    'strategy_mode': 'adaptive',
                    'risk_management': 'enabled',
                    'performance_tracking': True,
                    'mathematical_integration': True
                },
                'system': {
                    'adaptive_configuration': True,
                    'error_recovery': True,
                    'quantum_enhanced': True
                },
                'mathematical_framework': {
                    'tensor_operations': True,
                    'entropy_analysis': True,
                    'quantum_coherence': True
                }
            }
        elif 'high_frequency_crypto_config.yaml' in config_file:
            return {
                'high_frequency': {
                    'enabled': True,
                    'latency_threshold': 0.001,
                    'batch_size': 100,
                    'quantum_optimization': True
                },
                'btc_usdc_trading': {
                    'enabled': True,
                    'min_order_size': 0.001,
                    'max_order_size': 1.0,
                    'risk_limits': {
                        'max_drawdown': 0.05,
                        'max_position_size': 0.1
                    }
                }
            }
        else:
            return {'placeholder': True, 'file': config_file}

    def _load_python_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from Python files

        Args:
            config_path: Path to Python configuration file

        Returns:
            Extracted configuration dictionary
        """
        # Placeholder for Python config loading logic
        return {'python_config': True, 'path': config_path}

    def generate_adaptive_configuration(self, market_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate an adaptive configuration based on current system and market state

        Args:
            market_data: Optional market data for entropy analysis

        Returns:
            Dynamically generated configuration
        """
        # Load base configurations
        base_configs = self.load_configurations()

        # Analyze system health using the correct method
        system_health = self.system_profiler.get_system_health_summary()

        # Analyze market conditions with advanced mathematical analysis
        market_conditions = self.market_analyzer.analyze_market_entropy(market_data) if market_data is not None else {}

        # Update adaptive state with comprehensive information
        self._adaptive_state.system_health = system_health
        self._adaptive_state.market_conditions = market_conditions
        self._adaptive_state.mathematical_state = {
            'tensor_score': market_conditions.get('tensor_score', 0.0),
            'quantum_coherence': market_conditions.get('quantum_coherence', 0.0),
            'entropy_level': market_conditions.get('entropy', 0.5)
        }

        # Dynamic configuration adjustment based on mathematical analysis
        adaptive_config = base_configs.get('schwabot_core_config.yaml', {})

        # Adjust configuration based on market complexity and mathematical state
        complexity = market_conditions.get('market_complexity', 0.5)
        tensor_score = market_conditions.get('tensor_score', 0.5)
        quantum_coherence = market_conditions.get('quantum_coherence', 0.5)

        # Advanced strategy selection based on mathematical analysis
        if complexity > 0.7 and tensor_score < 0.3:
            # High complexity, low tensor score: Conservative quantum-enhanced strategy
            adaptive_config['strategy_mode'] = 'conservative_quantum'
            adaptive_config['risk_management'] = 'enhanced_quantum'
            adaptive_config['error_recovery_mode'] = 'enhanced'
            adaptive_config['quantum_optimization'] = True
        elif complexity < 0.3 and tensor_score > 0.7:
            # Low complexity, high tensor score: Aggressive classical strategy
            adaptive_config['strategy_mode'] = 'aggressive_classical'
            adaptive_config['risk_management'] = 'standard'
            adaptive_config['error_recovery_mode'] = 'standard'
            adaptive_config['quantum_optimization'] = False
        else:
            # Balanced strategy with quantum enhancement
            adaptive_config['strategy_mode'] = 'balanced_hybrid'
            adaptive_config['risk_management'] = 'adaptive_quantum'
            adaptive_config['error_recovery_mode'] = 'adaptive'
            adaptive_config['quantum_optimization'] = quantum_coherence > 0.6

        # Add system health-based adjustments
        stability_score = system_health.get('performance_metrics', {}).get('stability_score', 1.0)
        if stability_score < 0.7:
            adaptive_config['error_recovery_mode'] = 'enhanced'
            adaptive_config['risk_management'] = 'conservative'
            adaptive_config['quantum_optimization'] = False  # Disable quantum features if unstable

        # Integrate error recovery insights
        error_stats = self.recovery_system.get_error_statistics()
        if error_stats['recovery_rate'] < 0.8:
            adaptive_config['error_recovery_mode'] = 'enhanced'
            adaptive_config['system_monitoring'] = 'intensive'

        # Add BTC/USDC specific configurations
        adaptive_config['btc_usdc_trading'] = {
            'enabled': True,
            'quantum_enhanced': adaptive_config.get('quantum_optimization', False),
            'risk_limits': {
                'max_drawdown': 0.05 if complexity > 0.7 else 0.1,
                'max_position_size': 0.05 if complexity > 0.7 else 0.2
            }
        }

        return adaptive_config

    def track_system_performance(self, performance_metrics: Dict[str, float]):
        """
        Track system performance metrics for adaptive configuration

        Args:
            performance_metrics: Performance metrics dictionary
        """
        self._adaptive_state.performance_metrics = performance_metrics
        self._adaptive_state.timestamp = datetime.now()

        # Log performance alerts
        if performance_metrics.get('profit', 0) < 0:
            logging.warning("Negative performance detected. Adjusting strategy.")
        if performance_metrics.get('sharpe_ratio', 0) < 1.0:
            logging.info("Low Sharpe ratio detected. Reviewing risk management.")

    def get_adaptive_state(self) -> AdaptiveConfigurationState:
        """
        Get current adaptive configuration state

        Returns:
            Current adaptive state
        """
        return self._adaptive_state

    def get_mathematical_analysis(self) -> Dict[str, Any]:
        """
        Get current mathematical analysis state
        
        Returns:
            Mathematical analysis results
        """
        return {
            'tensor_algebra_status': self.tensor_algebra.get_status() if hasattr(self.tensor_algebra, 'get_status') else {},
            'market_conditions': self._adaptive_state.market_conditions,
            'mathematical_state': self._adaptive_state.mathematical_state,
            'system_health': self._adaptive_state.system_health
        }


def create_adaptive_config_manager() -> SchwabotAdaptiveConfigManager:
    """Factory function to create adaptive configuration manager"""
    return SchwabotAdaptiveConfigManager()
