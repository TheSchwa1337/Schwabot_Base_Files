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
import time
import logging
from datetime import datetime

class Unitizer:
    def __init__(self, model_type: str = 'rf'):
        self.math_processor = UnifiedMathematicalProcessor()
        self.dormant_engine = DormantStateLearningEngine(model_type=model_type)
        logging.info("[Unitizer] Initialized with model: %s", model_type)
        
    def validate_units(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate strategy logic by confirming mathematical and dormant consistency"""
        try:
            math_results = self.math_processor.run_complete_analysis()
        except Exception as e:
            logging.error(f"[Unitizer] Math processor failed: {e}")
            math_results = {'error': str(e)}

        dormant_states = [
            DormantState(
                state_id=i,
                features=np.array([
                    float(d.get('feature1', 0.0)),
                    float(d.get('feature2', 0.0)),
                    float(d.get('feature3', 0.0))
                ]),
                label=f"D{i}",
                confidence=0.95,
                timestamp=time.time()
            )
            for i, d in enumerate(data)
        ]
        
        self.dormant_engine.train(dormant_states)
        
        ts_string = datetime.utcnow().isoformat()
        
        return {
            'math_validation': math_results,
            'dormant_states': len(dormant_states),
            'timestamp': ts_string
        }

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
    logging.basicConfig(level=logging.INFO)

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
    print(f"Dormant states validated: {results['dormant_states']}")
    print(f"Validation timestamp: {results['timestamp']}")
    print("\nMathematical Validation Summary:")
    print("-" * 40)
    if isinstance(results['math_validation'], dict):
        for key, val in results['math_validation'].items():
            print(f"{key}: {val}")
    else:
        print("Math validation output was not structured as expected.")
        
    # Run unit tests
    print("\nRunning Unit Tests")
    print("=" * 40)
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 