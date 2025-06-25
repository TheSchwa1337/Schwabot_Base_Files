# Import safe print for Windows compatibility
try:
    from .utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug
import math
except ImportError:
    try:
#         from core.utils.windows_cli_compatibility import safe_print, info, warn, error, success, debug  # F811: duplicate import
    except ImportError:
def safe_print(message):
    print(message)
def info(message):
    print(f"[INFO] {message}")
def warn(message):
    print(f"[WARN] {message}")
def error(message):
    print(f"[ERROR] {message}")
def success(message):
    print(f"[SUCCESS] {message}")
def debug(message):
    print(f"[DEBUG] {message}")
from core.unified_math_system import unified_math
"""
Schwabot Vector Validator
=========================

Reinforcement learning engine that adjusts path weighting, trigger tolerances,
and hash/volume response curves based on failed and successful trades.

This component feeds on:
- Failed vector data (backtest false positives)
- Successful trades
- Known bad vectors from settings controller

It then adjusts:
- Path weighting
- Trigger tolerances
- Hash/volume response curves
- Matrix routing preferences
"""

import json
# from core.unified_math_system import unified_math  # F811: duplicate import
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from .settings_controller import get_settings_controller


@dataclass
class Vector:
    """Represents a trading vector with all associated data"""
    vector_id: str
    matrix_id: str
    tick_id: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    success: bool
    profit_loss: float
    confidence: float
    hash_signature: str
    volume_data: Dict[str, float]
    ghost_signal_strength: float
    entropy_level: float
    failure_type: Optional[str] = None
    reinforcement_weight: float = 1.0


@dataclass
class ValidationResult:
    """Result of vector validation"""
    is_valid: bool
    confidence_score: float
    adjusted_weight: float
    recommended_action: str
    failure_reason: Optional[str] = None
    reinforcement_notes: List[str] = None


class VectorValidator:
    """Reinforcement learning engine for vector validation"""

    def __init__(self):
        self.settings_controller = get_settings_controller()

        # Vector history for learning
        self.vector_history: List[Vector] = []
        self.successful_vectors: List[Vector] = []
        self.failed_vectors: List[Vector] = []

        # Learning parameters
        self.learning_rate = 0.05
        self.memory_decay = 0.95
        self.min_confidence_threshold = 0.6
        self.max_confidence_threshold = 0.95

        # Path performance tracking
        self.path_performance: Dict[str, Dict[str, float]] = {}
        self.matrix_performance: Dict[str, Dict[str, float]] = {}

        # Response curve adjustments
        self.hash_response_curves: Dict[str, List[float]] = {}
        self.volume_response_curves: Dict[str, List[float]] = {}

        # Initialize response curves
        self._initialize_response_curves()

    def _initialize_response_curves(self):
        """Initialize hash and volume response curves"""
        # Hash response curve (confidence vs hash similarity)
        self.hash_response_curves["default"] = [0.1, 0.3, 0.5, 0.7, 0.9]

        # Volume response curve (confidence vs volume ratio)
        self.volume_response_curves["default"] = [0.2, 0.4, 0.6, 0.8, 1.0]

    def validate_vector(self, vector_data: Dict[str, Any]) -> ValidationResult:
        """Validate a trading vector using reinforcement learning"""
        # Create vector object
        vector = self._create_vector_from_data(vector_data)

        # Check if it's a known bad vector
        if self.settings_controller.is_bad_vector(vector.hash_signature, vector.matrix_id):
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                adjusted_weight=0.0,
                recommended_action="avoid",
                failure_reason="known_bad_vector",
                reinforcement_notes=["Vector matches known bad vector pattern"]
            )

        # Calculate base confidence
        base_confidence = self._calculate_base_confidence(vector)

        # Apply reinforcement learning adjustments
        adjusted_confidence = self._apply_reinforcement_adjustments(vector, base_confidence)

        # Get path weight
        path_weight = self.settings_controller.get_matrix_weight(vector.matrix_id)

        # Determine validity
        is_valid = adjusted_confidence >= self.min_confidence_threshold

        # Determine recommended action
        if is_valid:
            recommended_action = "execute"
        elif adjusted_confidence > 0.4:
            recommended_action = "monitor"
        else:
            recommended_action = "avoid"

        # Create validation result
        result = ValidationResult(
            is_valid=is_valid,
            confidence_score=adjusted_confidence,
            adjusted_weight=path_weight * adjusted_confidence,
            recommended_action=recommended_action,
            reinforcement_notes=self._generate_reinforcement_notes(vector, adjusted_confidence)
        )

        # Update learning data
        self._update_learning_data(vector, result)

        return result

    def _create_vector_from_data(self, vector_data: Dict[str, Any]) -> Vector:
        """Create a Vector object from input data"""
        # Generate hash signature
        hash_input = f"{vector_data.get('matrix_id', '')}{vector_data.get('tick_id', 0)}{vector_data.get('entry_price', 0)}"
        hash_signature = hashlib.sha256(hash_input.encode()).hexdigest()

        return Vector(
            vector_id=vector_data.get('vector_id', f"vec_{hash_signature[:8]}"),
            matrix_id=vector_data.get('matrix_id', 'SFS8-A5'),
            tick_id=vector_data.get('tick_id', 0),
            entry_price=vector_data.get('entry_price', 0.0),
            exit_price=vector_data.get('exit_price', 0.0),
            entry_time=datetime.fromisoformat(vector_data.get('entry_time', datetime.now().isoformat())),
            exit_time=datetime.fromisoformat(vector_data.get('exit_time', datetime.now().isoformat())),
            success=vector_data.get('success', True),
            profit_loss=vector_data.get('profit_loss', 0.0),
            confidence=vector_data.get('confidence', 0.5),
            hash_signature=hash_signature,
            volume_data=vector_data.get('volume_data', {}),
            ghost_signal_strength=vector_data.get('ghost_signal_strength', 0.5),
            entropy_level=vector_data.get('entropy_level', 0.5),
            failure_type=vector_data.get('failure_type'),
            reinforcement_weight=vector_data.get('reinforcement_weight', 1.0)
        )

    def _calculate_base_confidence(self, vector: Vector) -> float:
        """Calculate base confidence score for a vector"""
        # Start with vector's base confidence
        confidence = vector.confidence

        # Adjust based on ghost signal strength
        ghost_adjustment = vector.ghost_signal_strength * 0.3
        confidence += ghost_adjustment

        # Adjust based on entropy level
        entropy_adjustment = (1.0 - vector.entropy_level) * 0.2
        confidence += entropy_adjustment

        # Adjust based on volume data
        volume_adjustment = self._calculate_volume_adjustment(vector.volume_data)
        confidence += volume_adjustment

        # Ensure confidence is within bounds
        confidence = unified_math.max(0.0, unified_math.min(1.0, confidence))

        return confidence

    def _calculate_volume_adjustment(self, volume_data: Dict[str, float]) -> float:
        """Calculate confidence adjustment based on volume data"""
        if not volume_data:
            return 0.0

        # Calculate volume ratio
        current_volume = volume_data.get('current', 0.0)
        avg_volume = volume_data.get('average', 1.0)

        if avg_volume == 0:
            return 0.0

        volume_ratio = current_volume / avg_volume

        # Apply volume response curve
        if volume_ratio < 0.5:
            return -0.1
        elif volume_ratio < 1.0:
            return 0.0
        elif volume_ratio < 2.0:
            return 0.1
        else:
            return 0.2

    def _apply_reinforcement_adjustments(self, vector: Vector, base_confidence: float) -> float:
        """Apply reinforcement learning adjustments to confidence"""
        adjusted_confidence = base_confidence

        # Get matrix performance history
        matrix_perf = self.matrix_performance.get(vector.matrix_id, {})
        success_rate = matrix_perf.get('success_rate', 0.5)

        # Adjust based on matrix success rate
        matrix_adjustment = (success_rate - 0.5) * 0.2
        adjusted_confidence += matrix_adjustment

        # Adjust based on path performance
        path_perf = self.path_performance.get(vector.matrix_id, {})
        path_success_rate = path_perf.get('success_rate', 0.5)

        path_adjustment = (path_success_rate - 0.5) * 0.15
        adjusted_confidence += path_adjustment

        # Apply reinforcement weight
        adjusted_confidence *= vector.reinforcement_weight

        # Ensure confidence is within bounds
        adjusted_confidence = unified_math.max(0.0, unified_math.min(1.0, adjusted_confidence))

        return adjusted_confidence

    def _generate_reinforcement_notes(self, vector: Vector, confidence: float) -> List[str]:
        """Generate reinforcement learning notes"""
        notes = []

        # Matrix performance note
        matrix_perf = self.matrix_performance.get(vector.matrix_id, {})
        if matrix_perf:
            success_rate = matrix_perf.get('success_rate', 0.5)
            notes.append(f"Matrix {vector.matrix_id} success rate: {success_rate:.2f}")

        # Path performance note
        path_perf = self.path_performance.get(vector.matrix_id, {})
        if path_perf:
            path_success_rate = path_perf.get('success_rate', 0.5)
            notes.append(f"Path success rate: {path_success_rate:.2f}")

        # Ghost signal note
        if vector.ghost_signal_strength > 0.7:
            notes.append("Strong ghost signal detected")
        elif vector.ghost_signal_strength < 0.3:
            notes.append("Weak ghost signal")

        # Entropy note
        if vector.entropy_level > 0.8:
            notes.append("High entropy - increased uncertainty")
        elif vector.entropy_level < 0.2:
            notes.append("Low entropy - stable conditions")

        return notes

    def _update_learning_data(self, vector: Vector, result: ValidationResult):
        """Update learning data with new vector information"""
        # Add to history
        self.vector_history.append(vector)

        # Categorize vector
        if vector.success:
            self.successful_vectors.append(vector)
        else:
            self.failed_vectors.append(vector)

        # Update matrix performance
        self._update_matrix_performance(vector)

        # Update path performance
        self._update_path_performance(vector)

        # Update settings controller
        self.settings_controller.update_matrix_weights(vector.matrix_id, vector.success)

        # Add to bad vectors if failed
        if not vector.success and vector.failure_type:
            self.settings_controller.add_bad_vector(
                vector.hash_signature,
                vector.tick_id,
                vector.failure_type,
                vector.matrix_id,
                result.confidence_score
            )

    def _update_matrix_performance(self, vector: Vector):
        """Update matrix performance statistics"""
        matrix_id = vector.matrix_id

        if matrix_id not in self.matrix_performance:
            self.matrix_performance[matrix_id] = {
                'total_trades': 0,
                'successful_trades': 0,
                'success_rate': 0.5,
                'avg_profit': 0.0,
                'avg_confidence': 0.5
            }

        perf = self.matrix_performance[matrix_id]
        perf['total_trades'] += 1

        if vector.success:
            perf['successful_trades'] += 1

        perf['success_rate'] = perf['successful_trades'] / perf['total_trades']

        # Update average profit
        current_avg = perf['avg_profit']
        perf['avg_profit'] = (current_avg * (perf['total_trades'] - 1) + vector.profit_loss) / perf['total_trades']

        # Update average confidence
        current_avg_conf = perf['avg_confidence']
        perf['avg_confidence'] = (current_avg_conf * (perf['total_trades'] - 1) + vector.confidence) / perf['total_trades']

    def _update_path_performance(self, vector: Vector):
        """Update path performance statistics"""
        matrix_id = vector.matrix_id

        if matrix_id not in self.path_performance:
            self.path_performance[matrix_id] = {
                'total_trades': 0,
                'successful_trades': 0,
                'success_rate': 0.5,
                'avg_profit': 0.0,
                'avg_confidence': 0.5
            }

        perf = self.path_performance[matrix_id]
        perf['total_trades'] += 1

        if vector.success:
            perf['successful_trades'] += 1

        perf['success_rate'] = perf['successful_trades'] / perf['total_trades']

        # Update average profit
        current_avg = perf['avg_profit']
        perf['avg_profit'] = (current_avg * (perf['total_trades'] - 1) + vector.profit_loss) / perf['total_trades']

        # Update average confidence
        current_avg_conf = perf['avg_confidence']
        perf['avg_confidence'] = (current_avg_conf * (perf['total_trades'] - 1) + vector.confidence) / perf['total_trades']

    def update_vector_weights(self, bad_vectors: List[Vector], good_vectors: List[Vector]):
        """Update vector weights based on bad and good vectors"""
        # Update weights for bad vectors
        for vector in bad_vectors:
            self.settings_controller.update_matrix_weights(vector.matrix_id, False)

            # Add to bad vectors map if not already present
            if not self.settings_controller.is_bad_vector(vector.hash_signature, vector.matrix_id):
                self.settings_controller.add_bad_vector(
                    vector.hash_signature,
                    vector.tick_id,
                    vector.failure_type or "unknown",
                    vector.matrix_id,
                    vector.confidence
                )

        # Update weights for good vectors
        for vector in good_vectors:
            self.settings_controller.update_matrix_weights(vector.matrix_id, True)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all matrices and paths"""
        summary = {
            'matrix_performance': self.matrix_performance,
            'path_performance': self.path_performance,
            'total_vectors': len(self.vector_history),
            'successful_vectors': len(self.successful_vectors),
            'failed_vectors': len(self.failed_vectors),
            'overall_success_rate': len(self.successful_vectors) / unified_math.max(len(self.vector_history), 1),
            'matrix_weights': self.settings_controller.matrix_path_weights,
            'known_bad_vectors': len(self.settings_controller.known_bad_vectors)
        }

        return summary

    def adjust_response_curves(self, matrix_id: str, success_rate: float):
        """Adjust response curves based on matrix performance"""
        if success_rate > 0.7:
            # Increase sensitivity for successful matrices
            self.hash_response_curves[matrix_id] = [0.05, 0.2, 0.4, 0.6, 0.8]
            self.volume_response_curves[matrix_id] = [0.1, 0.3, 0.5, 0.7, 0.9]
        elif success_rate < 0.3:
            # Decrease sensitivity for failing matrices
            self.hash_response_curves[matrix_id] = [0.2, 0.4, 0.6, 0.8, 0.95]
            self.volume_response_curves[matrix_id] = [0.3, 0.5, 0.7, 0.85, 0.95]

    def save_learning_data(self, filepath: str = "learning_data.json"):
        """Save learning data to file"""
        data = {
            'vector_history': [asdict(v) for v in self.vector_history],
            'matrix_performance': self.matrix_performance,
            'path_performance': self.path_performance,
            'hash_response_curves': self.hash_response_curves,
            'volume_response_curves': self.volume_response_curves,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_learning_data(self, filepath: str = "learning_data.json"):
        """Load learning data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load vector history
            self.vector_history = [Vector(**v) for v in data.get('vector_history', [])]

            # Load performance data
            self.matrix_performance = data.get('matrix_performance', {})
            self.path_performance = data.get('path_performance', {})

            # Load response curves
            self.hash_response_curves = data.get('hash_response_curves', {})
            self.volume_response_curves = data.get('volume_response_curves', {})

            # Rebuild successful/failed vectors lists
            self.successful_vectors = [v for v in self.vector_history if v.success]
            self.failed_vectors = [v for v in self.vector_history if not v.success]

        except FileNotFoundError:
            safe_print(f"Learning data file {filepath} not found. Starting with empty data.")
        except Exception as e:
            safe_print(f"Error loading learning data: {e}")


# Global vector validator instance
vector_validator = VectorValidator()


def get_vector_validator() -> VectorValidator:
    """Get the global vector validator instance"""
    return vector_validator


if __name__ == "__main__":
    # Test the vector validator
    validator = VectorValidator()

    safe_print("=== Schwabot Vector Validator Test ===")

    # Test vector data
    test_vector_data = {
        'vector_id': 'test_vec_001',
        'matrix_id': 'SFS8-A5',
        'tick_id': 12345,
        'entry_price': 50000.0,
        'exit_price': 50100.0,
        'entry_time': datetime.now().isoformat(),
        'exit_time': datetime.now().isoformat(),
        'success': True,
        'profit_loss': 100.0,
        'confidence': 0.8,
        'volume_data': {'current': 1000000, 'average': 800000},
        'ghost_signal_strength': 0.7,
        'entropy_level': 0.3
    }

    # Validate vector
    result = validator.validate_vector(test_vector_data)

    safe_print(f"Vector ID: {test_vector_data['vector_id']}")
    safe_print(f"Valid: {result.is_valid}")
    safe_print(f"Confidence: {result.confidence_score:.3f}")
    safe_print(f"Adjusted Weight: {result.adjusted_weight:.3f}")
    safe_print(f"Recommended Action: {result.recommended_action}")
    safe_print(f"Reinforcement Notes: {result.reinforcement_notes}")

    # Get performance summary
    summary = validator.get_performance_summary()
    safe_print("\nPerformance Summary:")
    safe_print(f"Total Vectors: {summary['total_vectors']}")
    safe_print(f"Success Rate: {summary['overall_success_rate']:.2%}")
    safe_print(f"Matrix Weights: {summary['matrix_weights']}")

    safe_print("Vector validator test completed!")
