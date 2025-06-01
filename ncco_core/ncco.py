"""
NCCO Core Module
===============

This module provides the core NCCO (Neural Control and Coordination Object) functionality
for the Schwabot system.
"""

from core.math_core import UnifiedMathematicalProcessor, MATHEMATICAL_CONSTANTS
from core.dormant_engine import DormantStateLearningEngine, DormantState
import numpy as np

class NCCO:
    def __init__(self, id: int, price_delta: float, base_price: float, bit_mode: int, score: float = 0.0, pre_commit_id: str = None):
        self.id = id
        self.price_delta = price_delta
        self.base_price = base_price
        self.bit_mode = bit_mode
        self.score = score
        self.pre_commit_id = pre_commit_id

    def __repr__(self):
        return f"NCCO(id={self.id}, price_delta={self.price_delta}, base_price={self.base_price}, bit_mode={self.bit_mode}, score={self.score}, pre_commit_id={self.pre_commit_id})"

# Example usage
if __name__ == "__main__":
    # Initialize mathematical processor
    processor = UnifiedMathematicalProcessor()
    results = processor.run_complete_analysis()
    report = processor.generate_summary_report(results)
    print(report)
    
    # Initialize dormant state engine
    engine = DormantStateLearningEngine(model_type='rf')
    
    # Create sample data
    data = [
        DormantState(
            state_id=0,
            features=np.array([1.0, 2.0, 3.0]),
            label="D0",
            confidence=0.95,
            timestamp=np.datetime64('now').astype(float)
        )
    ]
    
    # Train and predict
    engine.train(data)
    new_features = np.array([[0.5, 0.3, 0.7]])
    prediction, confidence = engine.predict(new_features)
    print(f"Predicted state: {prediction} (confidence: {confidence:.3f})") 