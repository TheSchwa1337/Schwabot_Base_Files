#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test Suite for Schwabot Adaptive Configuration Management
"""

import numpy as np
import pytest

from config.schwabot_adaptive_config_manager import MarketConditionAnalyzer, SchwabotAdaptiveConfigManager
from core.advanced_tensor_algebra import AdvancedTensorAlgebra
from core.enhanced_error_recovery_system import EnhancedErrorRecoverySystem
from core.system_state_profiler import SystemStateProfiler


class TestAdaptiveConfigurationSystem:
    @pytest.fixture
    def adaptive_config_manager(self):
        """Create an instance of the adaptive configuration manager"""
        return SchwabotAdaptiveConfigManager()
    
    def test_configuration_loading(self, adaptive_config_manager):
        """Test that configurations are loaded successfully"""
        configs = adaptive_config_manager.load_configurations()
        
        assert 'schwabot_core_config.yaml' in configs
        assert 'high_frequency_crypto_config.yaml' in configs
    
    def test_market_condition_analysis(self):
        """Test market condition analysis capabilities"""
        tensor_algebra = AdvancedTensorAlgebra()
        market_analyzer = MarketConditionAnalyzer(tensor_algebra)
        
        # Simulate market data
        market_data = np.random.normal(0, 1, 1000)
        
        market_conditions = market_analyzer.analyze_market_entropy(market_data)
        
        assert 'entropy' in market_conditions
        assert 'volatility' in market_conditions
        assert 'dominant_frequency' in market_conditions
        assert 'market_complexity' in market_conditions
        
        assert 0 <= market_conditions['entropy'] <= 1
        assert 0 <= market_conditions['market_complexity'] <= 1
    
    def test_adaptive_configuration_generation(self, adaptive_config_manager):
        """Test generation of adaptive configurations"""
        # Simulate low complexity market data
        low_complexity_data = np.random.normal(0, 0.1, 1000)
        low_complexity_config = adaptive_config_manager.generate_adaptive_configuration(low_complexity_data)
        
        assert 'strategy_mode' in low_complexity_config
        assert low_complexity_config.get('strategy_mode') == 'aggressive'
        
        # Simulate high complexity market data
        high_complexity_data = np.random.normal(0, 2, 1000)
        high_complexity_config = adaptive_config_manager.generate_adaptive_configuration(high_complexity_data)
        
        assert 'strategy_mode' in high_complexity_config
        assert high_complexity_config.get('strategy_mode') == 'conservative'
    
    def test_performance_tracking(self, adaptive_config_manager):
        """Test system performance tracking"""
        performance_metrics = {
            'profit': 1000.0,
            'sharpe_ratio': 2.5,
            'max_drawdown': -0.1
        }
        
        adaptive_config_manager.track_system_performance(performance_metrics)
        
        adaptive_state = adaptive_config_manager.get_adaptive_state()
        
        assert adaptive_state.performance_metrics == performance_metrics
    
    def test_error_recovery_integration(self, adaptive_config_manager):
        """Test integration with error recovery system"""
        # Simulate low recovery rate
        adaptive_config_manager.recovery_system.error_stats = {
            'total_errors': 10,
            'recovered_errors': 2,
            'recovery_rate': 0.2
        }
        
        adaptive_config = adaptive_config_manager.generate_adaptive_configuration()
        
        assert 'error_recovery_mode' in adaptive_config
        assert adaptive_config['error_recovery_mode'] == 'enhanced'

def main():
    """Run the test suite"""
    pytest.main([__file__])

if __name__ == '__main__':
    main() 