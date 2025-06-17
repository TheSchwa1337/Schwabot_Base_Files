"""
Test Strategy Sustainment Validator Integration
==============================================

Tests the 8-principle sustainment framework validation system
and its integration with Schwabot's core mathematical frameworks.
"""

import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import logging

# Disable logging during tests
logging.disable(logging.CRITICAL)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy_sustainment_validator import (
    StrategySustainmentValidator, 
    StrategyMetrics, 
    SustainmentPrinciple,
    ValidationResult,
    validate_strategy_quick
)

class TestStrategySustainmentValidator(unittest.TestCase):
    """Test the complete sustainment validation framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = {
            'overall_threshold': 0.75,
            'weight_survivability': 1.5,
            'weight_transcendence': 2.0,
            'threshold_survivability': 0.85,
            'confidence_config': {
                'stability_epsilon': 0.01,
                'confidence_threshold': 0.6
            }
        }
        self.validator = StrategySustainmentValidator(self.config)
    
    def test_excellent_strategy_validation(self):
        """Test validation of a high-quality strategy"""
        excellent_metrics = StrategyMetrics(
            # Integration metrics - excellent
            entropy_coherence=0.95,
            system_harmony=0.90,
            module_alignment=0.88,
            
            # Anticipation metrics - excellent
            lead_time_prediction=0.92,
            pattern_recognition_depth=0.89,
            signal_forecast_accuracy=0.94,
            
            # Responsiveness metrics - excellent
            latency=0.02,  # Very low latency
            adaptation_speed=0.91,
            market_reaction_time=0.03,
            
            # Simplicity metrics - good
            logic_complexity=0.20,  # Low complexity
            operation_count=150,    # Reasonable operation count
            decision_tree_depth=4,  # Simple decision tree
            
            # Economy metrics - excellent
            profit_efficiency=0.89,
            resource_utilization=0.87,
            cost_benefit_ratio=0.93,
            
            # Survivability metrics - excellent
            drawdown_resistance=0.92,
            risk_adjusted_return=0.86,
            volatility_tolerance=0.88,
            
            # Continuity metrics - excellent
            pattern_memory_depth=0.91,
            state_persistence=0.89,
            cycle_completion_rate=0.94,
            
            # Transcendence metrics - excellent
            emergent_signal_score=0.87,
            adaptive_learning_rate=0.85,
            optimization_convergence=0.90
        )
        
        result = self.validator.validate_strategy(excellent_metrics, "excellent_strategy")
        
        # Should pass all validations
        self.assertEqual(result.overall_status, "PASS")
        self.assertTrue(result.execution_approved)
        self.assertGreater(result.overall_score, 0.85)
        self.assertGreater(result.weighted_score, 0.85)
        self.assertGreater(result.confidence, 0.8)
        
        # All principles should pass
        for principle, score in result.principle_scores.items():
            self.assertTrue(score.passed, f"{principle.value} should pass")
            self.assertGreater(score.score, 0.7, f"{principle.value} score too low")
    
    def test_poor_strategy_validation(self):
        """Test validation of a poor-quality strategy"""
        poor_metrics = StrategyMetrics(
            # Integration metrics - poor
            entropy_coherence=0.35,
            system_harmony=0.40,
            module_alignment=0.30,
            
            # Anticipation metrics - poor
            lead_time_prediction=0.25,
            pattern_recognition_depth=0.30,
            signal_forecast_accuracy=0.28,
            
            # Responsiveness metrics - poor
            latency=0.8,  # High latency
            adaptation_speed=0.20,
            market_reaction_time=0.9,
            
            # Simplicity metrics - poor (too complex)
            logic_complexity=0.85,
            operation_count=2000,
            decision_tree_depth=25,
            
            # Economy metrics - poor
            profit_efficiency=0.15,
            resource_utilization=0.25,
            cost_benefit_ratio=0.20,
            
            # Survivability metrics - poor
            drawdown_resistance=0.10,
            risk_adjusted_return=0.05,
            volatility_tolerance=0.15,
            
            # Continuity metrics - poor
            pattern_memory_depth=0.20,
            state_persistence=0.25,
            cycle_completion_rate=0.30,
            
            # Transcendence metrics - poor
            emergent_signal_score=0.15,
            adaptive_learning_rate=0.10,
            optimization_convergence=0.20
        )
        
        result = self.validator.validate_strategy(poor_metrics, "poor_strategy")
        
        # Should fail validation
        self.assertEqual(result.overall_status, "FAIL")
        self.assertFalse(result.execution_approved)
        self.assertLess(result.overall_score, 0.5)
        self.assertGreater(len(result.recommendations), 5)  # Should have many recommendations
        
        # Critical principles should fail
        survivability_score = result.principle_scores[SustainmentPrinciple.SURVIVABILITY]
        economy_score = result.principle_scores[SustainmentPrinciple.ECONOMY]
        self.assertFalse(survivability_score.passed)
        self.assertFalse(economy_score.passed)
    
    def test_marginal_strategy_validation(self):
        """Test validation of a marginal strategy (should get WARNING)"""
        marginal_metrics = StrategyMetrics(
            # Mixed quality metrics - some good, some poor
            entropy_coherence=0.72,
            system_harmony=0.68,
            module_alignment=0.70,
            
            lead_time_prediction=0.65,
            pattern_recognition_depth=0.71,
            signal_forecast_accuracy=0.69,
            
            latency=0.15,
            adaptation_speed=0.74,
            market_reaction_time=0.12,
            
            logic_complexity=0.45,
            operation_count=600,
            decision_tree_depth=8,
            
            profit_efficiency=0.76,  # Good economy
            resource_utilization=0.78,
            cost_benefit_ratio=0.74,
            
            drawdown_resistance=0.87,  # Good survivability
            risk_adjusted_return=0.82,
            volatility_tolerance=0.85,
            
            pattern_memory_depth=0.73,
            state_persistence=0.71,
            cycle_completion_rate=0.76,
            
            emergent_signal_score=0.68,
            adaptive_learning_rate=0.72,
            optimization_convergence=0.74
        )
        
        result = self.validator.validate_strategy(marginal_metrics, "marginal_strategy")
        
        # Should get warning but still be approved
        self.assertIn(result.overall_status, ["WARNING", "PASS"])
        self.assertTrue(result.execution_approved)  # Should still be approved
        self.assertGreater(result.overall_score, 0.6)
        self.assertLessEqual(len(result.recommendations), 5)  # Some recommendations
    
    def test_principle_weights_impact(self):
        """Test that principle weights properly impact overall scoring"""
        base_metrics = StrategyMetrics(
            entropy_coherence=0.8,
            profit_efficiency=0.8,
            drawdown_resistance=0.8,
            emergent_signal_score=0.9,  # High transcendence (weight=2.0)
            latency=0.1,
            logic_complexity=0.3,
            pattern_memory_depth=0.8,
            # Set reasonable defaults for other metrics
            **{attr: 0.7 for attr in [
                'system_harmony', 'module_alignment', 'lead_time_prediction',
                'pattern_recognition_depth', 'signal_forecast_accuracy',
                'adaptation_speed', 'market_reaction_time', 'operation_count',
                'decision_tree_depth', 'resource_utilization', 'cost_benefit_ratio',
                'risk_adjusted_return', 'volatility_tolerance', 'state_persistence',
                'cycle_completion_rate', 'adaptive_learning_rate', 'optimization_convergence'
            ]}
        )
        
        result = self.validator.validate_strategy(base_metrics)
        
        # Weighted score should be higher than unweighted due to high transcendence
        # (transcendence has weight 2.0, so high score there boosts weighted average)
        self.assertGreaterEqual(result.weighted_score, result.overall_score)
    
    def test_critical_principle_failure(self):
        """Test that critical principle failure prevents execution"""
        # Good strategy except for survivability
        metrics_bad_survivability = StrategyMetrics(
            entropy_coherence=0.9,
            profit_efficiency=0.9,
            drawdown_resistance=0.1,  # Very poor survivability
            risk_adjusted_return=0.1,
            volatility_tolerance=0.1,
            latency=0.05,
            logic_complexity=0.2,
            pattern_memory_depth=0.9,
            emergent_signal_score=0.9,
            # Set other metrics to good values
            **{attr: 0.8 for attr in [
                'system_harmony', 'module_alignment', 'lead_time_prediction',
                'pattern_recognition_depth', 'signal_forecast_accuracy',
                'adaptation_speed', 'market_reaction_time', 'resource_utilization',
                'cost_benefit_ratio', 'state_persistence', 'cycle_completion_rate',
                'adaptive_learning_rate', 'optimization_convergence'
            ]},
            operation_count=100,
            decision_tree_depth=3
        )
        
        result = self.validator.validate_strategy(metrics_bad_survivability)
        
        # Should fail due to critical principle failure
        self.assertEqual(result.overall_status, "FAIL")
        self.assertFalse(result.execution_approved)
        
        survivability_score = result.principle_scores[SustainmentPrinciple.SURVIVABILITY]
        self.assertFalse(survivability_score.passed)
    
    def test_performance_tracking(self):
        """Test performance tracking functionality"""
        # Run several validations
        for i in range(5):
            metrics = StrategyMetrics(
                entropy_coherence=0.8 + i * 0.02,
                profit_efficiency=0.75 + i * 0.03,
                drawdown_resistance=0.85 + i * 0.01,
                latency=0.1 - i * 0.01,
                logic_complexity=0.3,
                pattern_memory_depth=0.8,
                emergent_signal_score=0.7 + i * 0.04,
                # Set reasonable defaults
                **{attr: 0.7 + i * 0.02 for attr in [
                    'system_harmony', 'module_alignment', 'lead_time_prediction',
                    'pattern_recognition_depth', 'signal_forecast_accuracy',
                    'adaptation_speed', 'market_reaction_time', 'resource_utilization',
                    'cost_benefit_ratio', 'risk_adjusted_return', 'volatility_tolerance',
                    'state_persistence', 'cycle_completion_rate', 'adaptive_learning_rate',
                    'optimization_convergence'
                ]},
                operation_count=100,
                decision_tree_depth=3
            )
            
            self.validator.validate_strategy(metrics, f"test_strategy_{i}")
        
        # Check performance summary
        summary = self.validator.get_performance_summary()
        
        self.assertEqual(summary['total_validations'], 5)
        self.assertIn('pass_rate', summary)
        self.assertIn('average_score', summary)
        self.assertIn('principle_averages', summary)
        self.assertGreater(summary['average_score'], 0.5)
        
        # Should track improvement trend
        self.assertIn(summary['recent_trend'], ['improving', 'stable', 'insufficient_data'])
    
    def test_adaptive_threshold_adjustment(self):
        """Test adaptive threshold adjustment"""
        original_threshold = self.validator.principle_thresholds[SustainmentPrinciple.ECONOMY]
        
        # Adjust threshold
        new_threshold = 0.80
        self.validator.adjust_thresholds(SustainmentPrinciple.ECONOMY, new_threshold)
        
        self.assertEqual(self.validator.principle_thresholds[SustainmentPrinciple.ECONOMY], new_threshold)
        
        # Adjust weight
        original_weight = self.validator.principle_weights[SustainmentPrinciple.ECONOMY]
        new_weight = 1.5
        self.validator.adjust_weights(SustainmentPrinciple.ECONOMY, new_weight)
        
        self.assertEqual(self.validator.principle_weights[SustainmentPrinciple.ECONOMY], new_weight)
    
    def test_quick_validation_function(self):
        """Test the convenience quick validation function"""
        # Test passing validation
        result = validate_strategy_quick(
            entropy_coherence=0.85,
            profit_efficiency=0.80,
            drawdown_resistance=0.90,
            latency=0.05
        )
        self.assertTrue(result)
        
        # Test failing validation  
        result = validate_strategy_quick(
            entropy_coherence=0.30,
            profit_efficiency=0.20,
            drawdown_resistance=0.15,
            latency=0.8
        )
        self.assertFalse(result)
    
    @patch('core.strategy_sustainment_validator.CollapseConfidenceEngine')
    def test_confidence_integration(self, mock_confidence_engine):
        """Test integration with CollapseConfidenceEngine"""
        # Mock confidence engine response
        mock_collapse_state = Mock()
        mock_collapse_state.confidence = 0.85
        mock_confidence_engine.return_value.calculate_collapse_confidence.return_value = mock_collapse_state
        
        metrics = StrategyMetrics(
            entropy_coherence=0.8,
            profit_efficiency=0.8,
            drawdown_resistance=0.8,
            emergent_signal_score=0.7,
            **{attr: 0.7 for attr in [
                'system_harmony', 'module_alignment', 'lead_time_prediction',
                'pattern_recognition_depth', 'signal_forecast_accuracy',
                'adaptation_speed', 'market_reaction_time', 'resource_utilization',
                'cost_benefit_ratio', 'risk_adjusted_return', 'volatility_tolerance',
                'state_persistence', 'cycle_completion_rate', 'adaptive_learning_rate',
                'optimization_convergence'
            ]},
            latency=0.1,
            logic_complexity=0.3,
            pattern_memory_depth=0.8,
            operation_count=100,
            decision_tree_depth=3
        )
        
        validator = StrategySustainmentValidator(self.config)
        result = validator.validate_strategy(metrics)
        
        # Should use the mocked confidence
        self.assertEqual(result.confidence, 0.85)
    
    def test_fractal_integration_scoring(self):
        """Test scoring components that integrate with fractal core"""
        # This would normally integrate with real fractal core
        # For testing, we verify the scoring logic handles missing/mock data gracefully
        
        metrics = StrategyMetrics(
            entropy_coherence=0.8,
            pattern_recognition_depth=0.7,
            **{attr: 0.7 for attr in [
                'system_harmony', 'module_alignment', 'lead_time_prediction',
                'signal_forecast_accuracy', 'adaptation_speed', 'market_reaction_time',
                'resource_utilization', 'cost_benefit_ratio', 'risk_adjusted_return',
                'volatility_tolerance', 'state_persistence', 'cycle_completion_rate',
                'adaptive_learning_rate', 'optimization_convergence', 'profit_efficiency',
                'drawdown_resistance', 'emergent_signal_score'
            ]},
            latency=0.1,
            logic_complexity=0.3,
            pattern_memory_depth=0.8,
            operation_count=100,
            decision_tree_depth=3
        )
        
        result = self.validator.validate_strategy(metrics)
        
        # Should complete without errors even with mock fractal integration
        self.assertIsInstance(result, ValidationResult)
        self.assertIn(result.overall_status, ["PASS", "WARNING", "FAIL"])
    
    def test_recommendation_generation(self):
        """Test that appropriate recommendations are generated"""
        # Strategy with specific weaknesses
        weak_responsiveness_metrics = StrategyMetrics(
            entropy_coherence=0.8,
            profit_efficiency=0.8,
            drawdown_resistance=0.8,
            latency=0.9,  # Very poor responsiveness
            adaptation_speed=0.2,
            market_reaction_time=0.8,
            **{attr: 0.7 for attr in [
                'system_harmony', 'module_alignment', 'lead_time_prediction',
                'pattern_recognition_depth', 'signal_forecast_accuracy',
                'resource_utilization', 'cost_benefit_ratio', 'risk_adjusted_return',
                'volatility_tolerance', 'state_persistence', 'cycle_completion_rate',
                'adaptive_learning_rate', 'optimization_convergence', 'emergent_signal_score'
            ]},
            logic_complexity=0.3,
            pattern_memory_depth=0.8,
            operation_count=100,
            decision_tree_depth=3
        )
        
        result = self.validator.validate_strategy(weak_responsiveness_metrics)
        
        # Should generate specific recommendations for responsiveness
        recommendations_text = ' '.join(result.recommendations)
        self.assertIn('latency', recommendations_text.lower())


class TestIntegrationWithExistingSystems(unittest.TestCase):
    """Test integration with existing Schwabot systems"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.validator = StrategySustainmentValidator()
    
    def test_strategy_metrics_from_real_data(self):
        """Test building strategy metrics from realistic Schwabot data"""
        # Simulate realistic tick signature and market conditions
        mock_tick_signature = Mock()
        mock_tick_signature.correlation_score = 0.82
        mock_tick_signature.signal_strength = 0.78
        mock_tick_signature.profit_tier = "GOLD"
        mock_tick_signature.tick_id = "test_tick_123"
        
        mock_market_conditions = {
            'volatility': 0.15,
            'thermal_budget': 0.85,
            'profit_tier_success_rate': 0.72,
            'recent_profit_correlation': 0.68,
            'anomaly_count': 1
        }
        
        # Build metrics like the strategy execution mapper would
        strategy_metrics = StrategyMetrics(
            entropy_coherence=mock_tick_signature.correlation_score,
            system_harmony=mock_market_conditions['thermal_budget'],
            module_alignment=min(mock_tick_signature.signal_strength, 1.0),
            
            lead_time_prediction=mock_tick_signature.signal_strength,
            pattern_recognition_depth=0.65,  # Simulated pattern depth
            signal_forecast_accuracy=mock_market_conditions['recent_profit_correlation'],
            
            latency=0.05,
            adaptation_speed=1.0 - mock_market_conditions['volatility'],
            market_reaction_time=0.1,
            
            logic_complexity=0.3,
            operation_count=8,  # Number of strategy types
            decision_tree_depth=3,
            
            profit_efficiency=mock_tick_signature.signal_strength * mock_market_conditions['profit_tier_success_rate'],
            resource_utilization=mock_market_conditions['thermal_budget'],
            cost_benefit_ratio=mock_tick_signature.correlation_score,
            
            drawdown_resistance=1.0 - mock_market_conditions['volatility'],
            risk_adjusted_return=mock_tick_signature.signal_strength * (1.0 - mock_market_conditions['volatility']),
            volatility_tolerance=min(1.0 - mock_market_conditions['volatility'], 1.0),
            
            pattern_memory_depth=0.8,
            state_persistence=0.8,
            cycle_completion_rate=0.9,
            
            emergent_signal_score=mock_tick_signature.signal_strength * mock_tick_signature.correlation_score,
            adaptive_learning_rate=0.7,
            optimization_convergence=0.8
        )
        
        result = self.validator.validate_strategy(strategy_metrics, "integration_test")
        
        # Should produce reasonable validation results
        self.assertIsInstance(result, ValidationResult)
        self.assertGreater(result.overall_score, 0.5)
        self.assertGreater(result.confidence, 0.3)
        self.assertIn(result.overall_status, ["PASS", "WARNING", "FAIL"])


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2) 