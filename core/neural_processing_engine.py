#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Processing Engine
========================

Advanced neural processing system for the Schwabot trading engine.

Features:
- Multiple neural network architectures (CNN, LSTM, Transformer)
- Reinforcement learning agent for action selection
- Ensemble prediction with confidence weighting
- GPU acceleration support
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import PyTorch components
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - neural processing will be limited")

# Try to import scikit-learn components
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - some metrics will be approximated")

logger = logging.getLogger(__name__)


@dataclass
class NeuralPrediction:
    """Neural network prediction result"""
    prediction: float
    confidence: float
    probability_distribution: np.ndarray
    feature_importance: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None


@dataclass
class TrainingMetrics:
    """Training metrics for neural networks"""
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    epoch: int


# Only define PyTorch-based neural network classes if PyTorch is available
if TORCH_AVAILABLE:

    class PricePatternCNN(nn.Module):
        """Convolutional Neural Network for price pattern recognition"""

        def __init__(self, input_channels: int = 1, sequence_length: int = 100, num_classes: int = 3) -> None:
            super(PricePatternCNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

            # Batch normalization
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)

            # Dropout for regularization
            self.dropout = nn.Dropout(0.3)

            # Adaptive pooling
            self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

            # Fully connected layers
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)

            # Activation functions
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # Convolutional layers with batch norm and activation
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))

            # Adaptive pooling
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)

            # Fully connected layers
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.fc3(x)

            return x

    class TradingLSTM(nn.Module):
        """LSTM Network for temporal sequence modeling in trading"""

        def __init__(
            self,
            input_size: int = 10,
            hidden_size: int = 128,
            num_layers: int = 3,
            output_size: int = 1,
        ):
            super(TradingLSTM, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

            # Attention mechanism
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)

            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, output_size)

            # Dropout and activation
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, x):
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)

            # Apply attention mechanism
            attention_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

            # Use the last output
            x = attention_out[:, -1, :]

            # Fully connected layers
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.dropout(self.relu(self.fc2(x)))
            x = self.tanh(self.fc3(x))

            return x, attention_weights

    class TradingTransformer(nn.Module):
        """Transformer architecture for advanced trading signal processing"""

        def __init__(
            self,
            input_dim: int = 10,
            d_model: int = 256,
            nhead: int = 8,
            num_layers: int = 6,
            output_dim: int = 1,
        ):
            super(TradingTransformer, self).__init__()

            self.d_model = d_model

            # Input projection
            self.input_projection = nn.Linear(input_dim, d_model)

            # Positional encoding
            self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_projection = nn.Linear(d_model, output_dim)

            # Dropout and activation
            self.dropout = nn.Dropout(0.1)
            self.tanh = nn.Tanh()

        def forward(self, x):
            # Input projection
            x = self.input_projection(x)

            # Add positional encoding
            seq_len = x.size(1)
            pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0)
            x = x + pos_encoding

            # Transformer encoder
            x = self.transformer_encoder(x)

            # Global average pooling
            x = torch.mean(x, dim=1)

            # Output projection
            x = self.dropout(x)
            x = self.tanh(self.output_projection(x))

            return x


class NeuralProcessingEngine:
    """
    Neural Processing Engine for advanced trading signal processing.
    
    Provides:
    - Multiple neural network architectures
    - Training and inference capabilities
    - Ensemble prediction methods
    - Performance monitoring and optimization
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """Initialize the neural processing engine"""
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
        # Initialize models
        self.models = {}
        self.optimizers = {}
        self.training_history = {}
        
        # Performance tracking
        self.total_predictions = 0
        self.avg_prediction_time = 0.0
        self.model_accuracies = {}

        logger.info(f"🧠 Neural Processing Engine initialized on {self.device}")

    def create_model(self, model_type: str, model_config: Dict[str, Any]) -> str:
        """Create a neural network model"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - cannot create neural models")
                return None
                
            model_id = f"{model_type}_{len(self.models)}"
            
            if model_type == "cnn":
                model = PricePatternCNN(**model_config)
            elif model_type == "lstm":
                model = TradingLSTM(**model_config)
            elif model_type == "transformer":
                model = TradingTransformer(**model_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Move to device
            model = model.to(self.device)
            
            # Create optimizer
            optimizer = optim.Adam(model.parameters(), lr=model_config.get("learning_rate", 0.001))
            
            # Store model and optimizer
            self.models[model_id] = model
            self.optimizers[model_id] = optimizer
            self.training_history[model_id] = []
            
            logger.info(f"Created {model_type} model: {model_id}")
            return model_id

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            return None

    def train_model(self, model_id: str, train_data: np.ndarray, train_labels: np.ndarray, 
                   epochs: int = 10, batch_size: int = 32) -> TrainingMetrics:
        """Train a neural network model"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - cannot train neural models")
                return TrainingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
                
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            model = self.models[model_id]
            optimizer = self.optimizers[model_id]
            
            # Convert to PyTorch tensors
            train_tensor = torch.FloatTensor(train_data).to(self.device)
            label_tensor = torch.LongTensor(train_labels).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(train_tensor, label_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            model.train()
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_data, batch_labels in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_data)
                    loss = F.cross_entropy(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_total += batch_labels.size(0)
                    epoch_correct += (predicted == batch_labels).sum().item()
                
                # Calculate metrics
                accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                avg_loss = epoch_loss / len(dataloader)
                
                # Store training metrics
                metrics = TrainingMetrics(
                    loss=avg_loss,
                    accuracy=accuracy,
                    precision=0.0,  # Would need to calculate from predictions
                    recall=0.0,     # Would need to calculate from predictions
                    f1_score=0.0,   # Would need to calculate from predictions
                    epoch=epoch
                )
                
                self.training_history[model_id].append(metrics)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
            # Update model accuracy
            self.model_accuracies[model_id] = accuracy
            
            return metrics

        except Exception as e:
            logger.error(f"Error training model {model_id}: {e}")
            return TrainingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    def predict(self, model_id: str, input_data: np.ndarray) -> NeuralPrediction:
        """Make a prediction using a trained model"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - returning dummy prediction")
                return NeuralPrediction(0.0, 0.5, np.array([0.5, 0.5]), {})
                
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")

            model = self.models[model_id]
            model.eval()
            
            # Convert to PyTorch tensor
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward pass
                output = model(input_tensor)
                
                # Get prediction and confidence
                probabilities = F.softmax(output, dim=1)
                prediction, predicted_class = torch.max(probabilities, 1)
                confidence = prediction.item()
                
                # Convert to numpy
                prob_dist = probabilities.cpu().numpy().flatten()
                
                # Feature importance (simplified)
                feature_importance = {f"feature_{i}": 1.0/len(input_data) for i in range(len(input_data))}
                
                # Update statistics
                self.total_predictions += 1
                
                return NeuralPrediction(
                    prediction=predicted_class.item(),
                    confidence=confidence,
                    probability_distribution=prob_dist,
                    feature_importance=feature_importance
                )

        except Exception as e:
            logger.error(f"Error making prediction with model {model_id}: {e}")
            return NeuralPrediction(0.0, 0.0, np.array([]), {})

    def ensemble_predict(self, input_data: np.ndarray, model_ids: List[str] = None) -> NeuralPrediction:
        """Make ensemble prediction using multiple models"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - returning dummy ensemble prediction")
                return NeuralPrediction(0.0, 0.5, np.array([0.5, 0.5]), {})
                
            if model_ids is None:
                model_ids = list(self.models.keys())
            
            if not model_ids:
                raise ValueError("No models available for ensemble prediction")
            
            # Get predictions from all models
            predictions = []
            confidences = []
            prob_dists = []
            
            for model_id in model_ids:
                pred = self.predict(model_id, input_data)
                predictions.append(pred.prediction)
                confidences.append(pred.confidence)
                prob_dists.append(pred.probability_distribution)
            
            # Ensemble aggregation
            ensemble_prediction = int(np.mean(predictions))
            ensemble_confidence = np.mean(confidences)
            ensemble_prob_dist = np.mean(prob_dists, axis=0)
            
            # Weighted feature importance
            ensemble_feature_importance = {}
            for model_id in model_ids:
                pred = self.predict(model_id, input_data)
                for feature, importance in pred.feature_importance.items():
                    if feature not in ensemble_feature_importance:
                        ensemble_feature_importance[feature] = []
                    ensemble_feature_importance[feature].append(importance)
            
            # Average feature importance
            for feature in ensemble_feature_importance:
                ensemble_feature_importance[feature] = np.mean(ensemble_feature_importance[feature])
            
            return NeuralPrediction(
                prediction=ensemble_prediction,
                confidence=ensemble_confidence,
                probability_distribution=ensemble_prob_dist,
                feature_importance=ensemble_feature_importance
            )

        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return NeuralPrediction(0.0, 0.0, np.array([]), {})

    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        try:
            if model_id not in self.models:
                return {"error": "Model not found"}
            
            # Get training history
            history = self.training_history.get(model_id, [])
            
            if not history:
                return {"status": "no_training_history"}
            
            # Calculate metrics
            latest_metrics = history[-1]
            best_accuracy = max([m.accuracy for m in history]) if history else 0.0
            avg_loss = np.mean([m.loss for m in history]) if history else 0.0
            
            return {
                "model_id": model_id,
                "latest_accuracy": latest_metrics.accuracy,
                "best_accuracy": best_accuracy,
                "average_loss": avg_loss,
                "total_epochs": len(history),
                "device": str(self.device) if self.device else "cpu"
            }

        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"error": str(e)}

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
return {
            "total_models": len(self.models),
            "total_predictions": self.total_predictions,
            "device": str(self.device) if self.device else "cpu",
            "torch_available": TORCH_AVAILABLE,
            "sklearn_available": SKLEARN_AVAILABLE,
            "model_accuracies": self.model_accuracies.copy()
        }

    def save_model(self, model_id: str, filepath: str) -> bool:
        """Save a trained model to disk"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - cannot save model")
                return False
                
            if model_id not in self.models:
return False

            model = self.models[model_id]
            torch.save(model.state_dict(), filepath)
            logger.info(f"Model {model_id} saved to {filepath}")
return True

except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
return False

    def load_model(self, model_id: str, filepath: str) -> bool:
        """Load a trained model from disk"""
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available - cannot load model")
                return False
                
            if model_id not in self.models:
                return False
            
            model = self.models[model_id]
            model.load_state_dict(torch.load(filepath, map_location=self.device))
            logger.info(f"Model {model_id} loaded from {filepath}")
return True

except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
return False

    def cleanup_resources(self) -> None:
        """Clean up neural processing resources"""
        try:
            # Clear models
            self.models.clear()
            self.optimizers.clear()
            self.training_history.clear()
            
            # Clear CUDA cache if using GPU
            if TORCH_AVAILABLE and self.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("🧹 Neural Processing Engine resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        self.cleanup_resources()


def create_neural_processing_engine(use_gpu: bool = True) -> NeuralProcessingEngine:
    """Create a new neural processing engine instance"""
    return NeuralProcessingEngine(use_gpu=use_gpu)
