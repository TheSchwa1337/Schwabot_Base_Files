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

# Import core system components for integration
from core.enhanced_error_recovery_system import EnhancedErrorRecoverySystem
from core.system_state_profiler import SystemStateProfiler


class MarketConditionAnalyzer:
    """Analyze market conditions for adaptive configuration"""

    def __init__(self,   tensor_algebra: AdvancedTensorAlgebra) -> None:
        self.tensor_algebra = tensor_algebra

    def analyze_market_entropy(self,   market_data: np.ndarray) -> Dict[str, float]:
        """
        Analyze market entropy and generate adaptive parameters

        Args:
            market_data: Market price/volume data

        Returns:
            Entropy-based market condition metrics
        """
        try:
            # Calculate Shannon entropy
            entropy = self.tensor_algebra.entropy_modulation.calculate_shannon_entropy(market_data)

            # Spectral analysis for market cycles
            frequencies, power_spectrum = self.tensor_algebra.spectral_analysis.fourier_spectrum(market_data)

            # Determine market volatility and cyclicity
            volatility = np.std(market_data)
            dominant_frequency = frequencies[np.argmax(power_spectrum)]

            return {
                'entropy': entropy,
                'volatility': volatility,
                'dominant_frequency': dominant_frequency,
                'market_complexity': self._calculate_market_complexity(entropy, volatility),
            }

        except Exception as e:
            logging.error(f"Market entropy analysis failed: {e}")
            return {'entropy': 0.5, 'volatility': 0.0, 'dominant_frequency': 0.0, 'market_complexity': 0.5}

    def _calculate_market_complexity(self,   entropy: float, volatility: float) -> float:
        """
        Calculate an integrated market complexity metric

        Args:
            entropy: Market entropy value
            volatility: Market price volatility

        Returns:
            Complexity score between 0 and 1
        """
        return np.clip((entropy + volatility) / 2, 0, 1)


@dataclass
class AdaptiveConfigurationState:
    """Comprehensive state tracking for adaptive configuration"""

    timestamp: datetime = field(default_factory=datetime.now)
    system_health: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, float] = field(default_factory=dict)
    active_strategies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)


class SchwabotAdaptiveConfigManager:
    """
    Intelligent Configuration Management System

    Features:
    - Multi-source configuration loading
    - Adaptive configuration generation
    - System state integration
    - Performance and error tracking
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
                with open(full_path, 'r') as f:
                    if config_file.endswith('.yaml'):
                        config = yaml.safe_load(f)
                    elif config_file.endswith('.json'):
                        config = json.load(f)
                    elif config_file.endswith('.py'):
                        # For Python config files, you might need a custom loader
                        config = self._load_python_config(full_path)

                    self._config_cache[config_file] = config

            except Exception as e:
                logging.warning(f"Could not load config {config_file}: {e}")

        return self._config_cache

    def _load_python_config(self,   config_path: str) -> Dict[str, Any]:
        """
        Load configuration from Python files

        Args:
            config_path: Path to Python configuration file

        Returns:
            Extracted configuration dictionary
        """
        # Placeholder for Python config loading logic
        return {}

    def generate_adaptive_configuration(self,   market_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate an adaptive configuration based on current system and market state

        Args:
            market_data: Optional market data for entropy analysis

        Returns:
            Dynamically generated configuration
        """
        # Load base configurations
        base_configs = self.load_configurations()

        # Analyze system health
        system_health = self.system_profiler.get_system_profile()

        # Analyze market conditions
        market_conditions = self.market_analyzer.analyze_market_entropy(market_data) if market_data is not None else {}

        # Update adaptive state
        self._adaptive_state.system_health = system_health.__dict__
        self._adaptive_state.market_conditions = market_conditions

        # Dynamic configuration adjustment
        adaptive_config = base_configs.get('schwabot_core_config.yaml', {})

        # Adjust configuration based on market complexity
        complexity = market_conditions.get('market_complexity', 0.5)

        if complexity > 0.7:
            # High complexity: More conservative strategy
            adaptive_config.update(
                base_configs.get('high_frequency_crypto_config.yaml', {}).get('high_complexity_settings', {})
            )
        elif complexity < 0.3:
            # Low complexity: More aggressive strategy
            adaptive_config.update(
                base_configs.get('high_frequency_crypto_config.yaml', {}).get('low_complexity_settings', {})
            )

        # Integrate error recovery insights
        error_stats = self.recovery_system.get_error_statistics()
        if error_stats['recovery_rate'] < 0.8:
            # Increase resilience if recovery rate is low
            adaptive_config['error_recovery_mode'] = 'enhanced'

        return adaptive_config

    def track_system_performance(self, performance_metrics: Dict[str, float]):
        """
        Track and log system performance metrics

        Args:
            performance_metrics: Dictionary of performance metrics
        """
        self._adaptive_state.performance_metrics.update(performance_metrics)

        # Optional: Persist performance data or trigger alerts
        if performance_metrics.get('profit', 0) < 0:
            logging.warning("Negative performance detected. Adjusting strategy.")

    def get_adaptive_state(self) -> AdaptiveConfigurationState:
        """
        Retrieve the current adaptive configuration state

        Returns:
            Comprehensive adaptive state
        """
        return self._adaptive_state


# Convenience function for creating the adaptive config manager
def create_adaptive_config_manager() -> SchwabotAdaptiveConfigManager:
    """
    Create a fully initialized Schwabot Adaptive Configuration Manager

    Returns:
        Configured SchwabotAdaptiveConfigManager instance
    """
    return SchwabotAdaptiveConfigManager()
