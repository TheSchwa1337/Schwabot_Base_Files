"""
Unitizer Module
==============

This module provides unit testing and validation functionality for the Schwabot system.
"""

from core.math_core import UnifiedMathematicalProcessor, MATHEMATICAL_CONSTANTS
from core.dormant_engine import DormantStateLearningEngine, DormantState
import numpy as np
import unittest
from typing import List, Dict, Any

class Unitizer:
    def __init__(self):
        self.math_processor = UnifiedMathematicalProcessor()
        self.dormant_engine = DormantStateLearningEngine(model_type='rf')
        
    def validate_units(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate units through the mathematical validation system"""
        # Run mathematical analysis
        math_results = self.math_processor.run_complete_analysis()
        
        # Process dormant states
        dormant_states = [
            DormantState(
                state_id=i,
                features=np.array([d['feature1'], d['feature2'], d['feature3']]),
                label=f"D{i}",
                confidence=0.95,
                timestamp=np.datetime64('now').astype(float)
            )
            for i, d in enumerate(data)
        ]
        
        # Train and predict
        self.dormant_engine.train(dormant_states)
        
        # Generate validation summary
        summary = {
            'math_validation': math_results,
            'dormant_states': len(dormant_states),
            'timestamp': np.datetime64('now').astype(str)
        }
        
        return summary

class TestUnitizer(unittest.TestCase):
    def setUp(self):
        self.unitizer = Unitizer()
        
    def test_validation(self):
        # Create sample test data
        test_data = [
            {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},
            {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5},
            {'feature1': 0.8, 'feature2': 1.8, 'feature3': 2.8}
        ]
        
        # Run validation
        results = self.unitizer.validate_units(test_data)
        
        # Assert results
        self.assertIn('math_validation', results)
        self.assertIn('dormant_states', results)
        self.assertIn('timestamp', results)
        self.assertEqual(results['dormant_states'], len(test_data))

# Example usage
if __name__ == "__main__":
    # Create sample unit data
    sample_data = [
        {'feature1': 1.0, 'feature2': 2.0, 'feature3': 3.0},
        {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5},
        {'feature1': 0.8, 'feature2': 1.8, 'feature3': 2.8}
    ]
    
    # Validate units
    unitizer = Unitizer()
    results = unitizer.validate_units(sample_data)
    
    # Print results
    print("\nUnit Validation Results:")
    print("=" * 40)
    print(f"Total dormant states validated: {results['dormant_states']}")
    print(f"Validation timestamp: {results['timestamp']}")
    print("\nMathematical Validation Summary:")
    print("-" * 30)
    for key, value in results['math_validation'].items():
        print(f"{key}: {value}")
        
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 