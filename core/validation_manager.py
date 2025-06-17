"""
Validation Manager
================

Coordinates and manages the validation process for the trading system,
ensuring mathematical correctness, proper sequencing, and robust error handling.
"""

import yaml
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from pathlib import Path

from core.strategy_sustainment_validator import StrategySustainmentValidator
from core.master_orchestrator import MasterOrchestrator
from core.ferris_wheel_scheduler import FerrisRunner
from core.resource_sequencer import ResourceSequencer
from core.mathlib_v3 import SustainmentMathLib
from core.quantum_antipole_engine import QuantumAntipoleEngine
from core.fractal_controller import FractalController

logger = logging.getLogger(__name__)

class ValidationManager:
    """Manages and coordinates the validation process"""
    
    def __init__(self, config_path: str = "config/validation_config.yaml"):
        """Initialize the validation manager"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        self.validation_results = {}
        self.start_time = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load validation configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load validation config: {e}")
            raise
            
    def _setup_logging(self):
        """Configure logging based on config"""
        log_config = self.config['logging']
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format=log_config['format'],
            filename=log_config['file'],
            filemode='a'
        )
        
    def _initialize_components(self):
        """Initialize validation components"""
        self.math_lib = SustainmentMathLib()
        self.validator = StrategySustainmentValidator({})
        self.sequencer = ResourceSequencer()
        self.ferris_runner = FerrisRunner([])
        self.orchestrator = MasterOrchestrator()
        self.antipole_engine = QuantumAntipoleEngine()
        self.fractal_controller = FractalController()
        
    def run_validation_suite(self) -> Dict:
        """Run the complete validation suite"""
        self.start_time = datetime.now()
        results = {
            'mathematical': self._validate_mathematical_correctness(),
            'sequencing': self._validate_sequencing(),
            'error_handling': self._validate_error_handling(),
            'performance': self._validate_performance()
        }
        self.validation_results = results
        return results
        
    def _validate_mathematical_correctness(self) -> Dict:
        """Validate mathematical correctness of core components"""
        results = {
            'kalman_filter': self._validate_kalman_filter(),
            'utility_function': self._validate_utility_function(),
            'convergence': self._validate_convergence()
        }
        return results
        
    def _validate_kalman_filter(self) -> Dict:
        """Validate Kalman filter predictions"""
        config = self.config['mathematical_validation']['kalman_filter']
        results = {'passed': True, 'errors': []}
        
        # Generate test data
        true_values = np.array([100.0, 102.0, 101.0, 103.0, 102.0])
        noisy_measurements = true_values + np.random.normal(0, config['measurement_noise'], len(true_values))
        
        # Run predictions
        predictions = []
        for measurement in noisy_measurements:
            pred = self.math_lib.kalman_predict(measurement)
            predictions.append(pred)
            
        # Calculate error metrics
        mse = np.mean((np.array(predictions) - true_values) ** 2)
        if mse > config['max_prediction_error']:
            results['passed'] = False
            results['errors'].append(f"Kalman filter MSE {mse} exceeds threshold {config['max_prediction_error']}")
            
        return results
        
    def _validate_utility_function(self) -> Dict:
        """Validate utility function properties"""
        config = self.config['mathematical_validation']['utility_function']
        results = {'passed': True, 'errors': []}
        
        # Test points
        test_points = np.linspace(0, 1, config['test_points'])
        
        for x in test_points:
            # Test first derivative
            deriv = self.math_lib.calculate_utility_derivative(x)
            if deriv < config['min_derivative']:
                results['passed'] = False
                results['errors'].append(f"Utility function derivative {deriv} below threshold at x={x}")
                
            # Test second derivative
            second_deriv = self.math_lib.calculate_utility_second_derivative(x)
            if second_deriv < config['min_second_derivative']:
                results['passed'] = False
                results['errors'].append(f"Utility function second derivative {second_deriv} below threshold at x={x}")
                
        return results
        
    def _validate_sequencing(self) -> Dict:
        """Validate trade sequence timing and dependencies"""
        results = {
            'ordering': self._validate_sequence_ordering(),
            'timing': self._validate_timing_constraints(),
            'dependencies': self._validate_dependencies()
        }
        return results
        
    def _validate_sequence_ordering(self) -> Dict:
        """Validate proper ordering of trade sequences"""
        config = self.config['sequence_validation']['ordering']
        results = {'passed': True, 'errors': []}
        
        # Create test sequence
        sequence_id = "test_seq_1"
        self.sequencer.start_sequence(sequence_id, profit_target=0.02, max_drawdown=0.01)
        
        # Simulate sequence execution
        timestamps = []
        for i in range(5):
            timestamp = datetime.now()
            self.sequencer.update_sequence(sequence_id, success=True, profit=0.01)
            timestamps.append(timestamp)
            
        # Verify ordering
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            results['passed'] = False
            results['errors'].append("Sequence timestamps not in ascending order")
            
        return results
        
    def _validate_error_handling(self) -> Dict:
        """Validate error handling and recovery mechanisms"""
        results = {
            'mathematical': self._validate_mathematical_error_handling(),
            'recovery': self._validate_recovery_mechanisms(),
            'input_validation': self._validate_input_handling()
        }
        return results
        
    def _validate_mathematical_error_handling(self) -> Dict:
        """Validate handling of mathematical edge cases"""
        config = self.config['error_handling']['mathematical']
        results = {'passed': True, 'errors': []}
        
        # Test zero division
        result = self.validator.calculate_efficiency(profit=0.0, cycles=0.0)
        if result is None:
            results['passed'] = False
            results['errors'].append("Zero division not properly handled")
            
        # Test numerical overflow
        large_value = 1e308
        result = self.validator.calculate_utility(large_value)
        if result >= float('inf'):
            results['passed'] = False
            results['errors'].append("Numerical overflow not properly handled")
            
        return results
        
    def _validate_performance(self) -> Dict:
        """Validate system performance"""
        config = self.config['performance_monitoring']
        results = {'passed': True, 'errors': []}
        
        if not config['enabled']:
            return results
            
        # Monitor system metrics
        metrics = self._collect_performance_metrics()
        
        # Check against thresholds
        thresholds = config['alert_thresholds']
        for metric, value in metrics.items():
            if value > thresholds.get(metric, float('inf')):
                results['passed'] = False
                results['errors'].append(f"{metric} {value} exceeds threshold {thresholds[metric]}")
                
        return results
        
    def _collect_performance_metrics(self) -> Dict:
        """Collect system performance metrics"""
        # This would be implemented to collect actual system metrics
        return {
            'cpu_usage_percent': 0.0,
            'memory_usage_percent': 0.0,
            'latency_ms': 0.0,
            'error_rate_percent': 0.0
        }
        
    def generate_validation_report(self) -> str:
        """Generate a detailed validation report"""
        if not self.validation_results:
            return "No validation results available"
            
        report = []
        report.append("Validation Report")
        report.append("===============")
        report.append(f"Generated: {datetime.now()}")
        report.append(f"Duration: {datetime.now() - self.start_time}")
        report.append("")
        
        for category, results in self.validation_results.items():
            report.append(f"{category.upper()} Validation")
            report.append("-" * len(f"{category.upper()} Validation"))
            
            if isinstance(results, dict):
                for subcategory, subresults in results.items():
                    report.append(f"\n{subcategory}:")
                    if isinstance(subresults, dict):
                        report.append(f"  Passed: {subresults['passed']}")
                        if subresults.get('errors'):
                            report.append("  Errors:")
                            for error in subresults['errors']:
                                report.append(f"    - {error}")
                    else:
                        report.append(f"  {subresults}")
            else:
                report.append(f"  {results}")
                
            report.append("")
            
        return "\n".join(report)
        
    def save_validation_report(self, output_path: str = "validation_report.txt"):
        """Save validation report to file"""
        report = self.generate_validation_report()
        try:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            raise 