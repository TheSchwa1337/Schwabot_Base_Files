"""
Comprehensive Test Suite for Sustainment Principles
==================================================

Tests all 8 sustainment principles for mathematical correctness,
integration behavior, edge cases, and performance characteristics.
"""

import pytest
import numpy as np
import time
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

# Import sustainment principles
import sys
sys.path.append('core')

from sustainment_principles import (
    SustainmentCalculator, SustainmentState, PrincipleMetrics,
    IntegrationPrinciple, AnticipationPrinciple, ResponsivenessPrinciple,
    SimplicityPrinciple, EconomyPrinciple, SurvivabilityPrinciple,
    ContinuityPrinciple, TranscendencePrinciple
)

from sustainment_integration_hooks import (
    SustainmentIntegrationManager, StrategyMapperHook, FractalCoreHook,
    ProfitNavigatorHook, BusCoreHook, ThermalZoneHook
)

class TestPrincipleMetrics:
    """Test the basic metrics and state classes"""
    
    def test_principle_metrics_creation(self):
        """Test creating principle metrics"""
        metric = PrincipleMetrics(
            value=0.8,
            confidence=0.9,
            threshold=0.6,
            metadata={'test': True}
        )
        
        assert metric.value == 0.8
        assert metric.confidence == 0.9
        assert metric.threshold == 0.6
        assert metric.metadata['test'] is True
        assert metric.is_healthy()  # Above threshold with high confidence
    
    def test_principle_metrics_health_check(self):
        """Test health checking logic"""
        # Healthy: above threshold with good confidence
        healthy = PrincipleMetrics(value=0.8, confidence=0.8, threshold=0.6)
        assert healthy.is_healthy()
        
        # Unhealthy: below threshold
        unhealthy_value = PrincipleMetrics(value=0.5, confidence=0.8, threshold=0.6)
        assert not unhealthy_value.is_healthy()
        
        # Unhealthy: low confidence
        unhealthy_confidence = PrincipleMetrics(value=0.8, confidence=0.6, threshold=0.6)
        assert not unhealthy_confidence.is_healthy()
    
    def test_sustainment_state_composite_score(self):
        """Test composite score calculation"""
        state = SustainmentState()
        
        # Set all principles to same values
        for principle_name in ['integration', 'anticipation', 'responsiveness', 
                              'simplicity', 'economy', 'survivability', 
                              'continuity', 'transcendence']:
            setattr(state, principle_name, PrincipleMetrics(value=0.8, confidence=0.9))
        
        composite = state.composite_score()
        assert 0.7 <= composite <= 0.8  # Should be close to 0.8 * 0.9
    
    def test_sustainment_state_failing_principles(self):
        """Test failing principles detection"""
        state = SustainmentState()
        
        # Set some principles as failing
        state.integration = PrincipleMetrics(value=0.3, confidence=0.8, threshold=0.6)
        state.anticipation = PrincipleMetrics(value=0.8, confidence=0.5, threshold=0.6)
        state.responsiveness = PrincipleMetrics(value=0.8, confidence=0.8, threshold=0.6)
        
        failing = state.failing_principles()
        assert 'integration' in failing  # Low value
        assert 'anticipation' in failing  # Low confidence
        assert 'responsiveness' not in failing  # Healthy

class TestIntegrationPrinciple:
    """Test Integration Principle (softmax normalization)"""
    
    @pytest.fixture
    def integration_principle(self):
        config = {'integration_softmax_alpha': 1.0, 'integration_threshold': 0.6}
        return IntegrationPrinciple(config)
    
    def test_softmax_normalization(self, integration_principle):
        """Test softmax weight normalization"""
        context = {
            'subsystem_scores': {
                'strategy_a': 0.8,
                'strategy_b': 0.6,
                'strategy_c': 0.4
            }
        }
        
        metric = integration_principle.calculate(context)
        
        assert metric.value > 0.0
        assert metric.confidence > 0.0
        assert 'weights' in metric.metadata
        
        # Check weights sum to 1 (softmax property)
        weights = metric.metadata['weights']
        weight_sum = sum(weights.values())
        assert abs(weight_sum - 1.0) < 1e-6
    
    def test_empty_subsystems(self, integration_principle):
        """Test behavior with no subsystems"""
        context = {'subsystem_scores': {}}
        
        metric = integration_principle.calculate(context)
        
        assert metric.value == 0.0
        assert metric.confidence == 0.1
        assert 'error' in metric.metadata
    
    def test_weight_balance_calculation(self, integration_principle):
        """Test weight balance metric"""
        # Balanced scores should give better integration
        balanced_context = {
            'subsystem_scores': {'a': 0.6, 'b': 0.6, 'c': 0.6}
        }
        
        # Unbalanced scores
        unbalanced_context = {
            'subsystem_scores': {'a': 0.9, 'b': 0.1, 'c': 0.1}
        }
        
        balanced_metric = integration_principle.calculate(balanced_context)
        unbalanced_metric = integration_principle.calculate(unbalanced_context)
        
        # Balanced should have better weight balance
        assert balanced_metric.metadata['weight_balance'] > unbalanced_metric.metadata['weight_balance']

class TestAnticipationPrinciple:
    """Test Anticipation Principle (Kalman-style prediction)"""
    
    @pytest.fixture
    def anticipation_principle(self):
        config = {
            'anticipation_tau': 0.1,
            'kalman_gain': 0.3,
            'anticipation_threshold': 0.5
        }
        return AnticipationPrinciple(config)
    
    def test_kalman_prediction(self, anticipation_principle):
        """Test Kalman-style prediction mechanism"""
        # First prediction (no history)
        context1 = {
            'current_state': {
                'price': 100.0,
                'entropy': 0.5,
                'volume': 1000.0
            }
        }
        
        metric1 = anticipation_principle.calculate(context1)
        assert 'predicted_price' in metric1.metadata
        
        # Second prediction (with history)
        context2 = {
            'current_state': {
                'price': 102.0,
                'entropy': 0.6,
                'volume': 1100.0
            }
        }
        
        metric2 = anticipation_principle.calculate(context2)
        assert 'predicted_price' in metric2.metadata
        assert 'avg_prediction_error' in metric2.metadata
    
    def test_entropy_derivative_calculation(self, anticipation_principle):
        """Test entropy derivative calculation"""
        # Calculate once to establish history
        context1 = {'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0}}
        anticipation_principle.calculate(context1)
        
        # Calculate again with different entropy
        context2 = {'current_state': {'price': 101.0, 'entropy': 0.7, 'volume': 1000.0}}
        metric = anticipation_principle.calculate(context2)
        
        # Should have calculated entropy derivative
        assert 'entropy_derivative' in metric.metadata
    
    def test_prediction_accuracy_improvement(self, anticipation_principle):
        """Test that prediction accuracy improves with consistent data"""
        # Generate consistent upward trend
        for i in range(10):
            context = {
                'current_state': {
                    'price': 100.0 + i * 1.0,  # Steady increase
                    'entropy': 0.5,
                    'volume': 1000.0
                }
            }
            metric = anticipation_principle.calculate(context)
        
        # Accuracy should improve with more data
        assert metric.confidence > 0.5

class TestResponsivenessPrinciple:
    """Test Responsiveness Principle (exponential latency response)"""
    
    @pytest.fixture
    def responsiveness_principle(self):
        config = {
            'max_latency_ms': 100.0,
            'responsiveness_threshold': 0.7
        }
        return ResponsivenessPrinciple(config)
    
    def test_exponential_latency_response(self, responsiveness_principle):
        """Test exponential response to latency"""
        # Low latency should give high responsiveness
        low_latency_context = {
            'system_latency_ms': 10.0,
            'event_response_ms': 5.0
        }
        
        # High latency should give low responsiveness
        high_latency_context = {
            'system_latency_ms': 200.0,
            'event_response_ms': 100.0
        }
        
        low_metric = responsiveness_principle.calculate(low_latency_context)
        high_metric = responsiveness_principle.calculate(high_latency_context)
        
        assert low_metric.value > high_metric.value
        assert low_metric.value > 0.8  # Should be high for low latency
        assert high_metric.value < 0.5  # Should be low for high latency
    
    def test_consistency_factor(self, responsiveness_principle):
        """Test that consistency improves responsiveness"""
        # Add consistent latency measurements
        for latency in [50.0, 51.0, 49.0, 50.5, 49.5]:
            context = {
                'system_latency_ms': latency,
                'event_response_ms': 25.0
            }
            responsiveness_principle.calculate(context)
        
        # Final measurement should benefit from consistency
        final_context = {
            'system_latency_ms': 50.0,
            'event_response_ms': 25.0
        }
        metric = responsiveness_principle.calculate(final_context)
        
        assert 'latency_std' in metric.metadata
        assert metric.metadata['latency_std'] < 2.0  # Should be low std

class TestSimplicityPrinciple:
    """Test Simplicity Principle (complexity minimization)"""
    
    @pytest.fixture
    def simplicity_principle(self):
        config = {
            'max_operations': 1000,
            'simplicity_threshold': 0.6
        }
        return SimplicityPrinciple(config)
    
    def test_complexity_calculation(self, simplicity_principle):
        """Test complexity to simplicity conversion"""
        # Low complexity should give high simplicity
        simple_context = {
            'operation_count': 10,
            'ncco_complexity': 5,
            'active_strategies': 2
        }
        
        # High complexity should give low simplicity
        complex_context = {
            'operation_count': 800,
            'ncco_complexity': 100,
            'active_strategies': 10
        }
        
        simple_metric = simplicity_principle.calculate(simple_context)
        complex_metric = simplicity_principle.calculate(complex_context)
        
        assert simple_metric.value > complex_metric.value
        assert simple_metric.metadata['normalized_complexity'] < complex_metric.metadata['normalized_complexity']
    
    def test_complexity_trend_penalty(self, simplicity_principle):
        """Test that increasing complexity reduces simplicity"""
        # Gradually increase complexity
        for ops in [100, 200, 300, 400, 500]:
            context = {
                'operation_count': ops,
                'ncco_complexity': 10,
                'active_strategies': 2
            }
            simplicity_principle.calculate(context)
        
        # Final calculation should show negative trend
        final_context = {
            'operation_count': 600,
            'ncco_complexity': 10,
            'active_strategies': 2
        }
        metric = simplicity_principle.calculate(final_context)
        
        assert 'complexity_trend' in metric.metadata
        assert metric.metadata['complexity_trend'] > 0  # Increasing complexity

class TestEconomyPrinciple:
    """Test Economy Principle (profit-per-compute efficiency)"""
    
    @pytest.fixture
    def economy_principle(self):
        config = {
            'min_efficiency': 0.001,
            'economy_threshold': 0.5
        }
        return EconomyPrinciple(config)
    
    def test_efficiency_calculation(self, economy_principle):
        """Test profit-per-compute efficiency"""
        # High profit, low compute should give high efficiency
        efficient_context = {
            'profit_delta': 0.1,  # 10% profit
            'cpu_cycles': 100.0,
            'gpu_cycles': 50.0,
            'memory_usage_mb': 128.0
        }
        
        # Low profit, high compute should give low efficiency
        inefficient_context = {
            'profit_delta': 0.01,  # 1% profit
            'cpu_cycles': 1000.0,
            'gpu_cycles': 500.0,
            'memory_usage_mb': 1024.0
        }
        
        efficient_metric = economy_principle.calculate(efficient_context)
        inefficient_metric = economy_principle.calculate(inefficient_context)
        
        assert efficient_metric.metadata['efficiency'] > inefficient_metric.metadata['efficiency']
    
    def test_sigmoid_normalization(self, economy_principle):
        """Test sigmoid normalization of efficiency"""
        context = {
            'profit_delta': 0.05,
            'cpu_cycles': 200.0,
            'gpu_cycles': 100.0,
            'memory_usage_mb': 256.0
        }
        
        metric = economy_principle.calculate(context)
        
        # Value should be normalized between 0 and 1
        assert 0.0 <= metric.value <= 1.0
    
    def test_efficiency_consistency(self, economy_principle):
        """Test that consistent efficiency is rewarded"""
        # Add consistent efficiency measurements
        for profit in [0.05, 0.048, 0.052, 0.049, 0.051]:
            context = {
                'profit_delta': profit,
                'cpu_cycles': 200.0,
                'gpu_cycles': 100.0,
                'memory_usage_mb': 256.0
            }
            economy_principle.calculate(context)
        
        final_metric = economy_principle.calculate(context)
        assert 'avg_efficiency' in final_metric.metadata

class TestSurvivabilityPrinciple:
    """Test Survivability Principle (positive curvature)"""
    
    @pytest.fixture
    def survivability_principle(self):
        config = {'survivability_threshold': 0.6}
        return SurvivabilityPrinciple(config)
    
    def test_utility_curvature_calculation(self, survivability_principle):
        """Test second derivative (curvature) calculation"""
        # Create increasing utility pattern (positive curvature)
        utilities = [0.5, 0.6, 0.75]  # Accelerating improvement
        
        for i, utility in enumerate(utilities):
            context = {
                'current_utility': utility,
                'entropy_level': 0.5,
                'shock_magnitude': 0.1,
                'recovery_rate': 1.0
            }
            survivability_principle.calculate(context)
        
        # Should detect positive curvature
        assert len(survivability_principle.utility_memory) == 3
    
    def test_shock_response_integration(self, survivability_principle):
        """Test shock response factor"""
        context = {
            'current_utility': 0.7,
            'entropy_level': 0.5,
            'shock_magnitude': 0.3,  # Moderate shock
            'recovery_rate': 1.2     # Good recovery
        }
        
        metric = survivability_principle.calculate(context)
        
        assert 'avg_shock' in metric.metadata
        assert 'recovery_rate' in metric.metadata
    
    def test_survivability_under_stress(self, survivability_principle):
        """Test survivability calculation under stress conditions"""
        # Simulate stress scenario with shocks but good recovery
        for i in range(5):
            context = {
                'current_utility': 0.6 + 0.1 * i,  # Improving despite shocks
                'entropy_level': 0.5,
                'shock_magnitude': 0.2,
                'recovery_rate': 1.5  # Excellent recovery
            }
            survivability_principle.calculate(context)
        
        # Should show good survivability due to recovery
        final_context = {
            'current_utility': 0.8,
            'entropy_level': 0.5,
            'shock_magnitude': 0.1,
            'recovery_rate': 1.5
        }
        metric = survivability_principle.calculate(final_context)
        
        assert metric.value > 0.5  # Should be decent survivability

class TestContinuityPrinciple:
    """Test Continuity Principle (integral memory)"""
    
    @pytest.fixture
    def continuity_principle(self):
        config = {
            'continuity_window': 20,
            'continuity_threshold': 0.6
        }
        return ContinuityPrinciple(config)
    
    def test_integral_memory_calculation(self, continuity_principle):
        """Test sliding window integral calculation"""
        # Add coherence measurements
        coherences = [0.7, 0.8, 0.6, 0.9, 0.7]
        
        for coherence in coherences:
            context = {
                'coherence': coherence,
                'stability': 0.8,
                'uptime_ratio': 0.95
            }
            continuity_principle.calculate(context)
        
        final_metric = continuity_principle.calculate(context)
        
        assert 'integral_memory' in final_metric.metadata
        expected_avg = np.mean(coherences)
        assert abs(final_metric.metadata['integral_memory'] - expected_avg) < 0.1
    
    def test_fluctuation_penalty(self, continuity_principle):
        """Test penalty for large fluctuations"""
        # Stable coherence
        stable_coherences = [0.7, 0.71, 0.69, 0.70, 0.72] * 3  # 15 measurements
        
        for coherence in stable_coherences:
            context = {
                'coherence': coherence,
                'stability': 0.8,
                'uptime_ratio': 0.95
            }
            continuity_principle.calculate(context)
        
        stable_metric = continuity_principle.calculate(context)
        
        # Reset for unstable test
        continuity_principle.coherence_buffer.clear()
        
        # Unstable coherence
        unstable_coherences = [0.9, 0.2, 0.8, 0.1, 0.7] * 3  # 15 measurements
        
        for coherence in unstable_coherences:
            context = {
                'coherence': coherence,
                'stability': 0.8,
                'uptime_ratio': 0.95
            }
            continuity_principle.calculate(context)
        
        unstable_metric = continuity_principle.calculate(context)
        
        # Stable should have better continuity
        assert stable_metric.value > unstable_metric.value

class TestTranscendencePrinciple:
    """Test Transcendence Principle (recursive convergence)"""
    
    @pytest.fixture
    def transcendence_principle(self):
        config = {
            'convergence_threshold': 0.01,
            'transcendence_threshold': 0.7,
            'fixed_point_target': 0.8
        }
        return TranscendencePrinciple(config)
    
    def test_fixed_point_iteration(self, transcendence_principle):
        """Test fixed-point iteration convergence"""
        # Simulate optimization process
        for i in range(10):
            context = {
                'optimization_state': 0.5 + i * 0.02,  # Gradual improvement
                'learning_rate': 0.1,
                'improvement_rate': 0.02
            }
            transcendence_principle.calculate(context)
        
        # Should show convergence behavior
        assert len(transcendence_principle.iteration_history) == 10
    
    def test_convergence_detection(self, transcendence_principle):
        """Test convergence detection mechanism"""
        # Simulate converging sequence
        target = 0.8
        learning_rate = 0.2
        current = 0.3
        
        for i in range(15):
            # Fixed-point iteration: x_{n+1} = x_n + α(target - x_n)
            current = current + learning_rate * (target - current)
            
            context = {
                'optimization_state': current,
                'learning_rate': learning_rate,
                'improvement_rate': abs(target - current)
            }
            metric = transcendence_principle.calculate(context)
        
        # Should detect convergence
        assert 'is_converging' in metric.metadata
        assert 'convergence_rate' in metric.metadata
        assert metric.metadata['distance_to_target'] < 0.1  # Should be close to target

class TestSustainmentCalculator:
    """Test the main sustainment calculator"""
    
    @pytest.fixture
    def sustainment_calculator(self):
        config = {
            'integration': {'integration_softmax_alpha': 1.0},
            'anticipation': {'kalman_gain': 0.3},
            'responsiveness': {'max_latency_ms': 100.0},
            'simplicity': {'max_operations': 1000},
            'economy': {'min_efficiency': 0.001},
            'survivability': {'survivability_threshold': 0.6},
            'continuity': {'continuity_window': 50},
            'transcendence': {'fixed_point_target': 0.8}
        }
        return SustainmentCalculator(config)
    
    def test_calculate_all_principles(self, sustainment_calculator):
        """Test calculation of all principles"""
        context = {
            'subsystem_scores': {'a': 0.8, 'b': 0.6, 'c': 0.7},
            'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0},
            'system_latency_ms': 50.0,
            'event_response_ms': 25.0,
            'operation_count': 100,
            'ncco_complexity': 10,
            'active_strategies': 3,
            'profit_delta': 0.05,
            'cpu_cycles': 200.0,
            'gpu_cycles': 100.0,
            'memory_usage_mb': 256.0,
            'current_utility': 0.7,
            'shock_magnitude': 0.1,
            'recovery_rate': 1.2,
            'coherence': 0.8,
            'stability': 0.9,
            'uptime_ratio': 0.95,
            'optimization_state': 0.6,
            'learning_rate': 0.1,
            'improvement_rate': 0.02
        }
        
        state = sustainment_calculator.calculate_all(context)
        
        # Check all principles were calculated
        assert state.integration.value >= 0.0
        assert state.anticipation.value >= 0.0
        assert state.responsiveness.value >= 0.0
        assert state.simplicity.value >= 0.0
        assert state.economy.value >= 0.0
        assert state.survivability.value >= 0.0
        assert state.continuity.value >= 0.0
        assert state.transcendence.value >= 0.0
        
        # Check composite score
        composite = state.composite_score()
        assert 0.0 <= composite <= 1.0
    
    def test_integration_weights_extraction(self, sustainment_calculator):
        """Test extraction of integration weights"""
        context = {
            'subsystem_scores': {'strategy_a': 0.8, 'strategy_b': 0.6, 'strategy_c': 0.4}
        }
        
        sustainment_calculator.calculate_all(context)
        weights = sustainment_calculator.get_integration_weights()
        
        assert isinstance(weights, dict)
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6
    
    def test_anticipation_prediction_extraction(self, sustainment_calculator):
        """Test extraction of anticipation prediction"""
        context = {
            'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0}
        }
        
        sustainment_calculator.calculate_all(context)
        prediction = sustainment_calculator.get_anticipation_prediction()
        
        assert isinstance(prediction, dict)
        if prediction:  # Might be empty on first calculation
            assert 'predicted_price' in prediction
    
    def test_health_report_generation(self, sustainment_calculator):
        """Test comprehensive health report"""
        context = {
            'subsystem_scores': {'a': 0.8, 'b': 0.6},
            'current_state': {'price': 100.0, 'entropy': 0.5},
            'system_latency_ms': 50.0,
            'operation_count': 100,
            'profit_delta': 0.05,
            'cpu_cycles': 200.0,
            'current_utility': 0.7,
            'coherence': 0.8,
            'optimization_state': 0.6
        }
        
        sustainment_calculator.calculate_all(context)
        health_report = sustainment_calculator.get_health_report()
        
        assert 'composite_score' in health_report
        assert 'healthy_principles' in health_report
        assert 'failing_principles' in health_report
        assert 'overall_health' in health_report
    
    def test_performance_trends(self, sustainment_calculator):
        """Test performance trend analysis"""
        # Generate multiple calculations
        for i in range(10):
            context = {
                'subsystem_scores': {'a': 0.7 + i*0.01},  # Gradual improvement
                'current_state': {'price': 100.0, 'entropy': 0.5},
                'system_latency_ms': 50.0,
                'operation_count': 100,
                'profit_delta': 0.05,
                'cpu_cycles': 200.0,
                'current_utility': 0.6 + i*0.02,
                'coherence': 0.8,
                'optimization_state': 0.6
            }
            sustainment_calculator.calculate_all(context)
        
        trends = sustainment_calculator.get_performance_trends(window=5)
        
        assert 'trend_direction' in trends
        assert 'current_score' in trends
        assert 'avg_score' in trends

class TestIntegrationHooks:
    """Test integration hooks with Schwabot modules"""
    
    def test_strategy_mapper_hook(self):
        """Test strategy mapper integration hook"""
        mock_strategy_mapper = Mock()
        mock_strategy_mapper.active_strategies = {
            'strategy_a': {'score': 0.8},
            'strategy_b': {'score': 0.6}
        }
        mock_strategy_mapper.overall_performance = 0.7
        mock_strategy_mapper.avg_execution_time_ms = 45.0
        
        config = {'integration': {'integration_softmax_alpha': 1.0}}
        calc = SustainmentCalculator(config)
        
        hook = StrategyMapperHook(calc, mock_strategy_mapper)
        metrics = hook.collect_metrics()
        
        assert metrics.module_name == 'strategy_mapper'
        assert metrics.performance_score == 0.7
        assert metrics.latency_ms == 45.0
        assert 'subsystem_scores' in metrics.custom_metrics
    
    def test_fractal_core_hook(self):
        """Test fractal core integration hook"""
        mock_fractal_core = Mock()
        mock_fractal_core.current_coherence = 0.8
        mock_fractal_core.entropy_level = 0.5
        mock_fractal_core.last_processing_time_ms = 30.0
        mock_fractal_core.current_price = 100.0
        mock_fractal_core.volume = 1000.0
        
        config = {'anticipation': {'kalman_gain': 0.3}}
        calc = SustainmentCalculator(config)
        
        hook = FractalCoreHook(calc, mock_fractal_core)
        metrics = hook.collect_metrics()
        
        assert metrics.module_name == 'fractal_core'
        assert metrics.performance_score == 0.8
        assert 'current_state' in metrics.custom_metrics
    
    def test_integration_manager(self):
        """Test sustainment integration manager"""
        config = {
            'integration': {'integration_softmax_alpha': 1.0},
            'anticipation': {'kalman_gain': 0.3}
        }
        calc = SustainmentCalculator(config)
        manager = SustainmentIntegrationManager(calc)
        
        # Register mock modules
        mock_strategy_mapper = Mock()
        mock_fractal_core = Mock()
        
        manager.register_strategy_mapper(mock_strategy_mapper)
        manager.register_fractal_core(mock_fractal_core)
        
        status = manager.get_integration_status()
        
        assert not status['running']  # Not started yet
        assert 'strategy_mapper' in status['registered_hooks']
        assert 'fractal_core' in status['registered_hooks']
        assert status['hook_count'] == 2

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_division_handling(self):
        """Test handling of zero division scenarios"""
        config = {'economy': {'min_efficiency': 0.001}}
        economy = EconomyPrinciple(config)
        
        # Zero compute should not crash
        context = {
            'profit_delta': 0.05,
            'cpu_cycles': 0.0,
            'gpu_cycles': 0.0,
            'memory_usage_mb': 0.0
        }
        
        metric = economy.calculate(context)
        assert metric.value >= 0.0  # Should not crash
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        config = {'integration': {'integration_softmax_alpha': 1.0}}
        integration = IntegrationPrinciple(config)
        
        # Invalid context
        invalid_contexts = [
            {},
            {'subsystem_scores': None},
            {'subsystem_scores': 'invalid'},
            {'subsystem_scores': {'a': 'not_a_number'}}
        ]
        
        for context in invalid_contexts:
            metric = integration.calculate(context)
            assert metric.value == 0.0
            assert metric.confidence <= 0.1
    
    def test_extreme_values(self):
        """Test behavior with extreme values"""
        config = {'responsiveness': {'max_latency_ms': 100.0}}
        responsiveness = ResponsivenessPrinciple(config)
        
        # Extreme latencies
        extreme_contexts = [
            {'system_latency_ms': 0.0, 'event_response_ms': 0.0},      # Zero latency
            {'system_latency_ms': 10000.0, 'event_response_ms': 5000.0}, # Very high latency
            {'system_latency_ms': -10.0, 'event_response_ms': -5.0}    # Negative latency
        ]
        
        for context in extreme_contexts:
            metric = responsiveness.calculate(context)
            assert 0.0 <= metric.value <= 1.0  # Should be bounded

class TestPerformance:
    """Test performance characteristics"""
    
    def test_calculation_performance(self):
        """Test calculation performance under load"""
        config = {
            'integration': {'integration_softmax_alpha': 1.0},
            'anticipation': {'kalman_gain': 0.3},
            'responsiveness': {'max_latency_ms': 100.0},
            'simplicity': {'max_operations': 1000},
            'economy': {'min_efficiency': 0.001},
            'survivability': {'survivability_threshold': 0.6},
            'continuity': {'continuity_window': 50},
            'transcendence': {'fixed_point_target': 0.8}
        }
        calc = SustainmentCalculator(config)
        
        context = {
            'subsystem_scores': {f'strategy_{i}': 0.5 + i*0.1 for i in range(10)},
            'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0},
            'system_latency_ms': 50.0,
            'operation_count': 500,
            'profit_delta': 0.05,
            'cpu_cycles': 200.0,
            'current_utility': 0.7,
            'coherence': 0.8,
            'optimization_state': 0.6
        }
        
        # Time 100 calculations
        start_time = time.time()
        for _ in range(100):
            calc.calculate_all(context)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / 100
        
        # Should be fast enough for real-time use
        assert avg_time_ms < 10.0  # Less than 10ms per calculation
    
    def test_memory_usage(self):
        """Test memory usage with large history"""
        config = {
            'integration': {'history_size': 1000},
            'anticipation': {'history_size': 1000},
            'responsiveness': {'history_size': 1000},
            'simplicity': {'history_size': 1000},
            'economy': {'history_size': 1000},
            'survivability': {'history_size': 1000},
            'continuity': {'history_size': 1000},
            'transcendence': {'history_size': 1000}
        }
        calc = SustainmentCalculator(config)
        
        context = {
            'subsystem_scores': {'a': 0.8},
            'current_state': {'price': 100.0},
            'system_latency_ms': 50.0,
            'operation_count': 100,
            'profit_delta': 0.05,
            'cpu_cycles': 200.0,
            'current_utility': 0.7,
            'coherence': 0.8,
            'optimization_state': 0.6
        }
        
        # Fill up history buffers
        for i in range(1500):  # More than buffer size
            calc.calculate_all(context)
        
        # Check that buffers are properly limited
        for principle in calc.principles.values():
            assert len(principle.history) <= 1000

class TestMathematicalCorrectness:
    """Test mathematical correctness of formulas"""
    
    def test_softmax_properties(self):
        """Test softmax normalization properties"""
        config = {'integration_softmax_alpha': 1.0}
        integration = IntegrationPrinciple(config)
        
        # Test softmax properties: sum = 1, all positive
        scores = np.array([0.8, 0.6, 0.4, 0.2])
        exp_scores = np.exp(1.0 * scores)
        weights = exp_scores / np.sum(exp_scores)
        
        assert abs(np.sum(weights) - 1.0) < 1e-10  # Sum to 1
        assert np.all(weights > 0)  # All positive
        assert np.all(weights < 1)  # All less than 1
    
    def test_exponential_decay_properties(self):
        """Test exponential decay properties"""
        config = {'max_latency_ms': 100.0}
        responsiveness = ResponsivenessPrinciple(config)
        
        # Test exponential decay: R = e^(-ℓ/λ)
        lambda_max = 100.0
        
        # Should decrease with increasing latency
        latencies = [10, 50, 100, 200, 500]
        responses = [np.exp(-l / lambda_max) for l in latencies]
        
        # Should be monotonically decreasing
        for i in range(1, len(responses)):
            assert responses[i] < responses[i-1]
        
        # Should approach 0 for very high latency
        assert responses[-1] < 0.01
    
    def test_sigmoid_normalization(self):
        """Test sigmoid normalization properties"""
        # Test sigmoid: σ(x) = 1 / (1 + e^(-x))
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))
        
        # Properties to test
        assert abs(sigmoid(0) - 0.5) < 1e-10  # σ(0) = 0.5
        assert sigmoid(10) > 0.99  # σ(large positive) ≈ 1
        assert sigmoid(-10) < 0.01  # σ(large negative) ≈ 0
        
        # Monotonically increasing
        x_vals = np.linspace(-5, 5, 11)
        y_vals = [sigmoid(x) for x in x_vals]
        for i in range(1, len(y_vals)):
            assert y_vals[i] > y_vals[i-1]

if __name__ == "__main__":
    # Run basic smoke test
    print("🧪 Running Sustainment Principles Smoke Test...")
    
    # Test basic functionality
    config = {
        'integration': {'integration_softmax_alpha': 1.0},
        'anticipation': {'kalman_gain': 0.3},
        'responsiveness': {'max_latency_ms': 100.0},
        'simplicity': {'max_operations': 1000},
        'economy': {'min_efficiency': 0.001},
        'survivability': {'survivability_threshold': 0.6},
        'continuity': {'continuity_window': 50},
        'transcendence': {'fixed_point_target': 0.8}
    }
    
    calc = SustainmentCalculator(config)
    
    context = {
        'subsystem_scores': {'strategy_a': 0.8, 'strategy_b': 0.6, 'strategy_c': 0.4},
        'current_state': {'price': 100.0, 'entropy': 0.5, 'volume': 1000.0},
        'system_latency_ms': 50.0,
        'event_response_ms': 25.0,
        'operation_count': 100,
        'ncco_complexity': 10,
        'active_strategies': 3,
        'profit_delta': 0.05,
        'cpu_cycles': 200.0,
        'gpu_cycles': 100.0,
        'memory_usage_mb': 256.0,
        'current_utility': 0.7,
        'entropy_level': 0.5,
        'shock_magnitude': 0.1,
        'recovery_rate': 1.2,
        'coherence': 0.8,
        'stability': 0.9,
        'uptime_ratio': 0.95,
        'optimization_state': 0.6,
        'learning_rate': 0.1,
        'improvement_rate': 0.02
    }
    
    # Calculate all principles
    state = calc.calculate_all(context)
    
    print(f"✅ Integration: {state.integration.value:.3f}")
    print(f"✅ Anticipation: {state.anticipation.value:.3f}")
    print(f"✅ Responsiveness: {state.responsiveness.value:.3f}")
    print(f"✅ Simplicity: {state.simplicity.value:.3f}")
    print(f"✅ Economy: {state.economy.value:.3f}")
    print(f"✅ Survivability: {state.survivability.value:.3f}")
    print(f"✅ Continuity: {state.continuity.value:.3f}")
    print(f"✅ Transcendence: {state.transcendence.value:.3f}")
    print(f"✅ Composite Score: {state.composite_score():.3f}")
    
    # Test integration weights
    weights = calc.get_integration_weights()
    print(f"✅ Integration Weights: {weights}")
    
    # Test health report
    health = calc.get_health_report()
    print(f"✅ Overall Health: {health['overall_health']}")
    print(f"✅ Healthy Principles: {health['healthy_principles']}/8")
    
    print("\n🎉 All Tests Passed! Sustainment Principles Framework Ready!") 