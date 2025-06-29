# core/entropy_feature_harvester.py

import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Assuming UnifiedEntropyEngine is in the same core directory
from core.entropy_engine import UnifiedEntropyEngine

logger = logging.getLogger(__name__)


class EntropyFeatureHarvester:


""""""
    Responsible for gathering, preprocessing, and extracting entropy - related features
    from historical and live trading data for GAN training and real - time feed.
It normalizes features and handles time - windowing of shell states.
""""""

    def __init__(self, entropy_engine: UnifiedEntropyEngine,):
            window_size: int = 50, feature_keys: Optional[List[str]] = None):
    """"""
    Initializes the EntropyFeatureHarvester.

    Args:
            entropy_engine(UnifiedEntropyEngine): An instance of the UnifiedEntropyEngine for entropy calculation.
            window_size(int): The size of the sliding window for time - series features.
        feature_keys (Optional[List[str]]): List of keys to extract from raw shell state data.
                                            If None, a default set is used.
    """"""
    self.entropy_engine = entropy_engine
    self.window_size = window_size
    self.raw_data_buffer: deque[Dict[str, Any]] = deque(maxlen = window_size)

        # Default feature keys if not provided
        self.feature_keys = feature_keys if feature_keys is not None else [)]
        "price", "volume", "entropy", "phase_angle", "drift_resonance", "volatility", "sentiment"
]
        logger.info(f"EntropyFeatureHarvester initialized with window size {window_size} and features: {self.feature_keys}")

    def add_shell_state_data(self, shell_state_data: Dict[str, Any]):
    """"""
    Adds a new raw shell state data point to the internal buffer.

    Args:
        shell_state_data (Dict[str, Any]): A dictionary containing various shell state metrics.
                                            Expected keys: 'price', 'volume', 'phase_angle', etc.
    """"""
        # Optionally compute entropy if not already part of the shell_state_data
        if "entropy" not in shell_state_data:
            # Assuming 'price' is available for entropy calculation
            price_data = np.array([shell_state_data.get("price", 0.0)]) # Convert to array for entropy engine
            try:
                # Default to Shannon entropy for raw data if not specified
            shell_state_data["entropy"] = self.entropy_engine.compute_entropy(price_data, "shannon")
            except Exception as e:
                logger.warning(f"Could not compute entropy for shell state data: {e}")
            shell_state_data["entropy"] = 0.0

            self.raw_data_buffer.append(shell_state_data)
            logger.debug(f"Added shell state data. Buffer size: {len(self.raw_data_buffer)}")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
    """"""
    Normalizes a numpy array of features using Min-Max scaling.

    Args:
        features (np.ndarray): The input features array.

    Returns:
        np.ndarray: Normalized features array.
    """"""
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)

        # Avoid division by zero for constant features
    range_val = max_val - min_val
    normalized_features = np.where(range_val != 0, (features - min_val) / range_val, 0.0)
    logger.debug("Features normalized.")
    return normalized_features

    def harvest_features(self) -> Optional[np.ndarray]:
    """"""
    Extracts and preprocesses features from the buffered shell states.
    This includes time-windowing and normalization.

    Returns:
        Optional[np.ndarray]: A flattened, normalized 1D vector of shell states
                                suitable for GAN input, or None if insufficient data.
    """"""
        if len(self.raw_data_buffer) < self.window_size:
        logger.warning(f"Insufficient data in buffer ({len(self.raw_data_buffer)} < {self.window_size}) to harvest features.")
        return None

        # Extract features for the current window
    window_data = list(self.raw_data_buffer)
    extracted_features_list: List[List[float]] = []

        for item in window_data:
        single_data_point_features: List[float] = []
            for key in self.feature_keys:
                value = item.get(key, 0.0) # Default to 0.0 if key is missing
                if isinstance(value, (int, float)):
                single_data_point_features.append(float(value))
                    else:
                # Handle non-numeric types, e.g., symbolic anchors. For GAN, they might need encoding.
                # For now, we'll log a warning and append 0.0'
                    logger.warning(f"Non-numeric value found for feature '{key}': {value}. Appending 0.0.")
                single_data_point_features.append(0.0)
                extracted_features_list.append(single_data_point_features)

                        if not extracted_features_list:
                return None

                    # Convert to numpy array for normalization
                features_array = np.array(extracted_features_list, dtype = np.float32)

                # Normalize features across the window
                normalized_features = self._normalize_features(features_array)

                    # Flatten the data for GAN input (as a 1D vector of shell states)
                flattened_features = normalized_features.flatten()
                logger.info(f"Harvested and normalized features. Shape: {flattened_features.shape}")
            return flattened_features

    def get_latest_shell_state_features(self) -> Optional[np.ndarray]:
    """"""
    Extracts features from the latest single shell state in the buffer.
        Useful for real-time feeding into the GAN for anomaly detection (inference).

    Returns:
            Optional[np.ndarray]: A flattened, normalized 1D vector for the latest shell state,
                                or None if buffer is empty.
    """"""
        if not self.raw_data_buffer:
        return None

    latest_item = self.raw_data_buffer[-1]
    single_data_point_features: List[float] = []

        for key in self.feature_keys:
        value = latest_item.get(key, 0.0)
            if isinstance(value, (int, float)):
            single_data_point_features.append(float(value))
                else:
                logger.warning(f"Non-numeric value found for feature '{key}' in latest shell state: {value}. Appending 0.0.")
            single_data_point_features.append(0.0)

                    if not single_data_point_features:
            return None

            features_array = np.array([single_data_point_features], dtype = np.float32)
            normalized_features = self._normalize_features(features_array)

        return normalized_features.flatten()

    def get_buffer_size(self) -> int:
    """"""
    Returns the current number of shell states in the internal buffer.
    """"""
    return len(self.raw_data_buffer)

    if __name__ == "__main__":
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Initialize Entropy Engine for the Harvester
entropy_engine_instance = UnifiedEntropyEngine()
harvester = EntropyFeatureHarvester(entropy_engine = entropy_engine_instance, window_size=5)

print("\n--- Simulating Shell State Data --- ")

    # Simulate some shell state data with some missing keys and non-numeric values
simulated_shell_states = [)]
    {"price": 100.0, "volume": 1000, "phase_angle": 0.1, "drift_resonance": 0.1, "volatility": 0.5, "sentiment": 0.6, "symbolic_anchor": "ABC"},
        {"price": 100.5, "volume": 1050, "phase_angle": 0.2, "drift_resonance": 0.2, "volatility": 0.6, "sentiment": 0.7},
            {"price": 101.0, "volume": 1100, "phase_angle": 0.3, "drift_resonance": 0.3, "volatility": 0.7, "sentiment": 0.8, "invalid_feature": "text"},
            {"price": 100.8, "volume": 1080, "phase_angle": 0.25, "drift_resonance": 0.25, "volatility": 0.65, "sentiment": 0.75},
            {"price": 101.2, "volume": 1120, "phase_angle": 0.35, "drift_resonance": 0.35, "volatility": 0.75, "sentiment": 0.85},
            {"price": 101.5, "volume": 1150, "phase_angle": 0.4, "drift_resonance": 0.4, "volatility": 0.8, "sentiment": 0.9},
]
        for i, state in enumerate(simulated_shell_states):
    harvester.add_shell_state_data(state)
    print(f"Buffer size: {harvester.get_buffer_size()}")
            if harvester.get_buffer_size() >= harvester.window_size:
        features = harvester.harvest_features()
                if features is not None:
                print(f"Harvested features (sample for {i+1} ticks): {features[:5]}... (shape: {features.shape})")
                    else:
                    print(f"Could not harvest features for {i+1} ticks.")

                print("\n--- Testing get_latest_shell_state_features ---")
                latest_features = harvester.get_latest_shell_state_features()
                        if latest_features is not None:
                    print(f"Latest shell state features: {latest_features[:5]}... (shape: {latest_features.shape})")
                            else:
                        print("No latest features available.")

                            # Test with insufficient data
                        empty_harvester = EntropyFeatureHarvester(entropy_engine = entropy_engine_instance, window_size=10)
                        empty_harvester.add_shell_state_data({"price": 50.0, "volume": 500, "phase_angle": 0.5})
                            print("\n--- Testing with insufficient data ---")
                        insufficient_features = empty_harvester.harvest_features()
                            print(f"Features with insufficient data: {insufficient_features}")