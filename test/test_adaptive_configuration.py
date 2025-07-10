#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Test Suite for Schwabot Adaptive Configuration Management
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        # Simulate low complexity market data with high tensor score
        low_complexity_data = np.random.normal(0, 0.1, 1000)
        low_complexity_config = adaptive_config_manager.generate_adaptive_configuration(low_complexity_data)
        
        assert 'strategy_mode' in low_complexity_config
        # With low complexity and high tensor score, should be aggressive_classical
        assert low_complexity_config.get('strategy_mode') in ['aggressive_classical', 'balanced_hybrid']
        
        # Simulate high complexity market data with low tensor score
        high_complexity_data = np.random.normal(0, 2, 1000)
        high_complexity_config = adaptive_config_manager.generate_adaptive_configuration(high_complexity_data)
        
        assert 'strategy_mode' in high_complexity_config
        # With high complexity and low tensor score, should be conservative_quantum
        assert high_complexity_config.get('strategy_mode') in ['conservative_quantum', 'balanced_hybrid']
        
        # Test that quantum optimization is properly configured
        assert 'quantum_optimization' in high_complexity_config
        assert 'btc_usdc_trading' in high_complexity_config
    
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

    def test_mathematical_analysis_integration(self, adaptive_config_manager):
        """Test advanced mathematical analysis integration"""
        # Generate market data
        market_data = np.random.normal(0, 1, 1000)
        
        # Get mathematical analysis
        math_analysis = adaptive_config_manager.get_mathematical_analysis()
        
        # Verify mathematical analysis structure
        assert 'tensor_algebra_status' in math_analysis
        assert 'market_conditions' in math_analysis
        assert 'mathematical_state' in math_analysis
        assert 'system_health' in math_analysis
        
        # Verify tensor algebra status
        tensor_status = math_analysis['tensor_algebra_status']
        assert 'active' in tensor_status
        assert 'mathematical_capabilities' in tensor_status
        
        # Verify mathematical capabilities
        capabilities = tensor_status['mathematical_capabilities']
        assert capabilities['tensor_operations'] == True
        assert capabilities['quantum_operations'] == True
        assert capabilities['entropy_analysis'] == True

def main():
    """Run the test suite"""
    pytest.main([__file__])

if __name__ == '__main__':
    main() 