"""
Dormant State Learning Engine
============================

This module provides the core functionality for learning and predicting dormant states
in the Schwabot system, using both Random Forest and Neural Network approaches.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

@dataclass
class DormantState:
    """Container for dormant state data"""
    state_id: int
    features: np.ndarray
    label: str
    confidence: float
    timestamp: float

class DormantStateLearningEngine:
    """Unified engine for dormant state learning and prediction"""
    
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the learning engine
        
        Args:
            model_type: Type of model to use ('rf' for Random Forest or 'nn' for Neural Network)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'nn':
            self.model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
        else:
            raise ValueError("model_type must be either 'rf' or 'nn'")
    
    def preprocess_data(self, data: List[DormantState]) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data for learning"""
        try:
            X = np.array([d.features for d in data])
            y = np.array([d.label for d in data])
            
            self.X = self.scaler.fit_transform(X)
            self.y = y
            
            return self.X, self.y
        except Exception as e:
            logging.error(f"Error preprocessing data: {e}")
            raise
    
    def train(self, data: List[DormantState]) -> None:
        """Train the model with input data"""
        try:
            X, y = self.preprocess_data(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Log training results
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            logging.info(f"Training score: {train_score:.3f}")
            logging.info(f"Testing score: {test_score:.3f}")
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict dormant state based on features"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get prediction and probability
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled).max()
            
            return prediction, probability
            
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = [
        DormantState(
            state_id=0,
            features=np.array([1.0, 2.0, 3.0]),
            label="D0",
            confidence=0.95,
            timestamp=np.datetime64('now').astype(float)
        ),
        # Add more sample data...
    ]
    
    # Initialize and train model
    engine = DormantStateLearningEngine(model_type='rf')
    engine.train(data)
    
    # Make prediction
    new_features = np.array([[0.5, 0.3, 0.7]])
    prediction, confidence = engine.predict(new_features)
    print(f"Predicted state: {prediction} (confidence: {confidence:.3f})")

"""
Dormant Logic Engine (DLE)
=========================

Implements the dormant node system that activates based on golden sequences
and braid angle resonance. Each node represents a potential trading opportunity
that becomes active when specific pattern conditions are met.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .cursor_engine import Cursor, CursorState
from .braid_pattern_engine import BraidPattern, BraidPatternEngine

@dataclass
class DormantNode:
    """Represents a dormant trading node"""
    node_id: str
    triplet: str
    activation_threshold: float
    resonance_angle: float
    golden_sequence: List[Tuple[int, float]]
    metadata: Dict
    is_active: bool = False
    last_activation: Optional[float] = None

class DormantEngine:
    """Manages dormant nodes and their activation"""
    
    def __init__(self):
        self.nodes: Dict[str, DormantNode] = {}
        self.active_nodes: List[str] = []
        self.golden_sequences: Dict[str, List[Tuple[int, float]]] = {}
        self.resonance_history: List[Tuple[str, float]] = []
        
    def register_node(self, node_id: str, triplet: str, 
                     activation_threshold: float = 0.8,
                     resonance_angle: float = 0.0,
                     golden_sequence: Optional[List[Tuple[int, float]]] = None,
                     metadata: Optional[Dict] = None) -> None:
        """Register a new dormant node"""
        self.nodes[node_id] = DormantNode(
            node_id=node_id,
            triplet=triplet,
            activation_threshold=activation_threshold,
            resonance_angle=resonance_angle,
            golden_sequence=golden_sequence or [],
            metadata=metadata or {},
            is_active=False
        )
        
    def check_activation(self, cursor_state: CursorState, 
                        pattern: Optional[BraidPattern] = None) -> List[str]:
        """Check for node activations based on current state"""
        activated_nodes = []
        
        for node_id, node in self.nodes.items():
            if not node.is_active:
                # Check golden sequence match
                if self._check_golden_sequence(node, cursor_state):
                    # Check resonance angle
                    if self._check_resonance(node, cursor_state.braid_angle):
                        # Activate node
                        node.is_active = True
                        node.last_activation = cursor_state.timestamp
                        self.active_nodes.append(node_id)
                        activated_nodes.append(node_id)
                        
        return activated_nodes
    
    def _check_golden_sequence(self, node: DormantNode, 
                             cursor_state: CursorState) -> bool:
        """Check if current state matches golden sequence"""
        if not node.golden_sequence:
            return False
            
        # Get recent cursor history
        recent_states = cursor_state.history[-len(node.golden_sequence):]
        
        # Check sequence match
        for (expected_delta, expected_angle), state in zip(node.golden_sequence, recent_states):
            if (state.delta_idx != expected_delta or 
                abs(state.braid_angle - expected_angle) > 5.0):
                return False
                
        return True
    
    def _check_resonance(self, node: DormantNode, current_angle: float) -> bool:
        """Check if current angle resonates with node's resonance angle"""
        angle_diff = abs(current_angle - node.resonance_angle) % 360.0
        return angle_diff < 5.0 or angle_diff > 355.0
    
    def deactivate_node(self, node_id: str) -> None:
        """Deactivate a node"""
        if node_id in self.nodes:
            self.nodes[node_id].is_active = False
            if node_id in self.active_nodes:
                self.active_nodes.remove(node_id)
    
    def get_active_nodes(self) -> List[DormantNode]:
        """Get list of currently active nodes"""
        return [self.nodes[node_id] for node_id in self.active_nodes]
    
    def get_node_history(self, node_id: str) -> List[Tuple[float, bool]]:
        """Get activation history for a node"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            return [(node.last_activation, node.is_active)]
        return []
    
    def clear_history(self) -> None:
        """Clear all node history"""
        self.resonance_history.clear()
        for node in self.nodes.values():
            node.last_activation = None
            node.is_active = False
        self.active_nodes.clear()

# Example usage
if __name__ == "__main__":
    # Initialize DLE
    dle = DormantEngine()
    
    # Register test nodes
    dle.register_node(
        node_id="NODE_001",
        triplet="172",
        activation_threshold=0.8,
        resonance_angle=45.0,
        golden_sequence=[(1, 45.0), (2, 90.0), (1, 135.0)],
        metadata={"type": "entry"}
    )
    
    # Test activation
    cursor_state = CursorState(
        triplet=(0.1, 0.2, 0.3),
        delta_idx=1,
        braid_angle=45.0,
        timestamp=1234567890.0
    )
    
    activated = dle.check_activation(cursor_state)
    print(f"Activated nodes: {activated}") 